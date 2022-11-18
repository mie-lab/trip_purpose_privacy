import pickle
import json
import os
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from foursquare_privacy.utils.io import read_gdf_csv, read_poi_geojson
from foursquare_privacy.models.xgb import XGBWrapper
from foursquare_privacy.models.mlp import MLPWrapper
from foursquare_privacy.utils.user_distribution import get_user_dist_mae
from foursquare_privacy.utils.spatial_folds import user_or_venue_split
from foursquare_privacy.add_poi import POI_processor, get_embedding, get_closest_poi_feats
from foursquare_privacy.location_masking import LocationMasker

model_dict = {"xgb": {"model_class": XGBWrapper, "config": {}}, "mlp": {"model_class": MLPWrapper, "config": {}}}
# xgb config: tree_method="gpu_hist", gpu_id=0 if gpu available


def cross_validation(dataset, folds, models=[], save_name=None, load_name=None, save_feature_importance=False):
    if len(models) == 0 and load_name is None:
        print("Training models...")
    else:
        print("Apply trained models on obfuscated data")
    assert len(models) == 0 or len(models) == len(folds), "either pass no model or pass all train models"
    # get features and labels
    features = dataset[[col for col in dataset.columns if col.startswith("feat")]]
    # print("List of features", features.columns)
    uni_labels = np.unique(dataset["label"])
    # map labels to numbers
    label_mapping = {u: i for i, u in enumerate(uni_labels)}
    labels = dataset["label"].map(label_mapping)

    result_df = dataset[["user_id", "venue_id", "label", "ground_truth"]].copy()
    proba_columns = [f"proba_{u}" for u in uni_labels]
    result_df["prediction"] = -1

    for fold_num, fold_samples in enumerate(folds):
        test_x = features.loc[fold_samples]
        # test_y = labels.loc[fold_samples]
        train_x = features.drop(fold_samples)
        train_y = labels.drop(fold_samples)

        # Option 1: train mode --> models are not given
        if len(models) < len(folds) and load_name is None:
            # fit and predict
            model = ModelClass(model_config)  # smaller max depth did not help
            model.fit(train_x, train_y)
            if save_name is not None:
                model.save(f"{save_name}_fold{fold_num}")
                print("Saved model:", f"{save_name}_fold{fold_num}")
            models.append(model)
        # Option 2: test mode --> models given, just apply
        else:
            if load_name is not None:
                model = ModelClass(model_config)
                model.load(f"{load_name}_fold{fold_num}")
                print("Loaded model", f"{load_name}_fold{fold_num}")
            else:
                model = models[fold_num]
        y_pred_proba = model.predict(test_x)
        y_pred = np.argmax(y_pred_proba, axis=1)
        # y_pred = np.random.choice(np.arange(12), len(test_x))  # testing

        result_df.loc[fold_samples, proba_columns] = y_pred_proba
        result_df.loc[fold_samples, "prediction"] = y_pred
        # print(f"Accuracy fold {fold_num+1}:", sum(y_pred == test_y) / len(test_y))
    # save features importance
    if save_feature_importance:
        with open(os.path.join(out_dir, f"feature_importance_{masking}.json"), "w") as outfile:
            json.dump(models[0].feature_importance, outfile)
    return models, result_df


def print_results(result_df, name, out_dir):
    acc = accuracy_score(result_df["ground_truth"], result_df["prediction"])
    bal_acc = balanced_accuracy_score(result_df["ground_truth"], result_df["prediction"])
    user_mae = np.mean(get_user_dist_mae(result_df))
    print(name, "Acc:", round(acc, 3), "Bal. acc:", round(bal_acc, 3), "User-wise MAE", round(user_mae, 3))
    result_df.to_csv(os.path.join(out_dir, f"predictions_{name}.csv"))
    results_dict[name] = {"Accuracy": acc, "Balanced accuracy": bal_acc, "User-wise MAE": user_mae}


results_dict = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="data", type=str)
    parser.add_argument("-c", "--city", default="newyorkcity", type=str)
    parser.add_argument("-o", "--out_dir", default=os.path.join("outputs", "test"), type=str)
    parser.add_argument("-p", "--poi_data", default="foursquare", type=str)
    parser.add_argument("-m", "--model", default="xgb", type=str)
    parser.add_argument(
        "-x", "--embed_model_path", default="../z_inactive_projects/space2vec/spacegraph/model_dir/", type=str
    )
    parser.add_argument("-f", "--fold_mode", default="user", type=str)
    parser.add_argument("-k", "--kfold", default=5, type=int)
    parser.add_argument("-b", "--buffer_factor", default=1.5, type=float)
    # minimum buffer around the location, aside from the buffer factor
    parser.add_argument("--min_buffer", default=100, type=float)
    parser.add_argument("--lda", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--closestk", action="store_true")
    parser.add_argument("--inbuffer", action="store_true")
    parser.add_argument("--xgbdepth", default=6, type=int)
    args = parser.parse_args()

    city = args.city
    embed_model_path = os.path.join(args.embed_model_path, f"{args.poi_data}_{args.city}_16")

    out_dir_base = args.out_dir
    os.makedirs(out_dir_base, exist_ok=True)
    further_out_name = f"_{args.embed}_{args.lda}_{args.inbuffer}_{args.closestk}_{args.xgbdepth}_{args.kfold}"
    out_name = f"{args.model}_{args.poi_data}_{args.city}_{args.fold_mode}" + further_out_name
    out_dir = os.path.join(out_dir_base, out_name)
    print(out_dir_base, out_dir)
    if os.path.exists(out_dir):
        print("Warning: Output directory already exists, may be overwriting files")
    os.makedirs(out_dir, exist_ok=True)

    assert any([args.embed, args.inbuffer, args.closestk, args.lda]), "must define some spatial representation"

    # get model
    ModelClass = model_dict[args.model]["model_class"]
    model_config = model_dict[args.model]["config"]
    model_config["max_depth"] = args.xgbdepth

    # load data
    data_raw = read_gdf_csv(os.path.join(args.data_path, f"checkin_{city}_features.csv"))
    # convert to id arr
    uni_labels = np.unique(data_raw["label"])
    label_mapping = {elem: i for i, elem in enumerate(uni_labels)}
    data_raw["ground_truth"] = data_raw["label"].map(label_mapping)

    if args.poi_data == "both":
        pois_1 = read_poi_geojson(os.path.join(args.data_path, f"pois_{city}_foursquare.geojson"))
        pois_2 = read_poi_geojson(os.path.join(args.data_path, f"pois_{city}_osm.geojson"))
        pois = pd.concat((pois_2[["id", "poi_my_label", "poi_type", "geometry"]], pois_1))
        pois_1, pois_2 = None, None  # free space of variable
    else:
        pois = read_poi_geojson(os.path.join(args.data_path, f"pois_{city}_{args.poi_data}.geojson"))
    print(f"Using {len(pois)} pois")

    # Split data
    np.random.seed(42)
    folds = user_or_venue_split(data_raw, by=args.fold_mode, kfold=args.kfold)
    # print("Fold lengths", [len(f) for f in folds])

    # 1) USER-FEATURES: check the performance with solely the user features
    _, results_only_user = cross_validation(data_raw, folds)
    print_results(results_only_user, "temporal_features", out_dir)

    temporal_feats = [col for col in data_raw.columns if col.startswith("feat")]

    # obfuscate coordinates
    for masking in [0, 25, 50, 100, 200, 400, 800, 1200]:
        buffering = max(args.buffer_factor * masking, args.min_buffer)
        print(f"-------- Masking {masking} ---------")
        if masking == 0:
            data = data_raw.copy()
            models = []
        else:
            masker = LocationMasker(data_raw)
            data = masker(masking)
            # # double check the latitude difference
            # print(
            #     f"Average latitude difference (masking {masking})",
            #     (data_raw["latitude"] - data["latitude"]).abs().mean(),
            # )

        # 2) CLOSEST_POI - Use simply the nearest poi label
        spatial_joined = data.sjoin_nearest(pois, how="left")  # , distance_col="distance")
        spatial_joined["prediction"] = spatial_joined["poi_my_label"].map(label_mapping)
        print_results(spatial_joined, f"spatial_join_{masking}", out_dir)

        dataset = data.copy()

        if args.inbuffer:
            # get poi features
            poi_process = POI_processor(data, pois)
            poi_process(buffer=buffering)
            poi_features = poi_process.distance_count_features()
            dataset = dataset.merge(poi_features, left_on=["latitude", "longitude"], right_index=True, how="left")
            del poi_process

        if args.lda:
            poi_process = POI_processor(data, pois)
            poi_process(buffer=buffering)
            lda_features = poi_process.lda_features()
            dataset = dataset.merge(lda_features, left_on=["latitude", "longitude"], right_index=True, how="left")
            del poi_process

        if args.closestk:
            dataset = get_closest_poi_feats(dataset, pois)

        if args.embed:
            poi_pointset_path = os.path.join(args.data_path, f"space2vec_{args.poi_data}_{args.city}")
            dataset = get_embedding(dataset, poi_pointset_path, embed_model_path, 10)

        print("Percentage of rows with at least one NaN", dataset.isna().any(axis=1).sum() / len(dataset))
        dataset = dataset.fillna(0)

        _, result_df = cross_validation(dataset, folds, models=[], save_feature_importance=True)
        print_results(result_df, f"all_features_{masking}", out_dir)

        dataset_spatial = dataset.drop(temporal_feats, axis=1)
        _, result_df = cross_validation(dataset_spatial, folds, models=[])
        print_results(result_df, f"spatial_features_{masking}", out_dir)

        # # CODE to train only on non-obfuscated and test on others
        # if masking == 0:
        #     models, result_df = cross_validation(dataset, folds, models)
        # else:
        #     _, result_df = cross_validation(dataset, folds, models)
        # print_results(result_df, f"all_features_{masking}", out_dir)

    with open(os.path.join(out_dir, "results.json"), "w") as outfile:
        json.dump(results_dict, outfile)
