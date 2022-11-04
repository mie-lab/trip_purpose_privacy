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
from foursquare_privacy.utils.spatial_folds import spatial_split, venue_split
from foursquare_privacy.add_poi import POI_processor
from foursquare_privacy.location_masking import LocationMasker

model_dict = {"xgb": {"model_class": XGBWrapper, "config": {}}, "mlp": {"model_class": MLPWrapper, "config": {}}}
# xgb config: tree_method="gpu_hist", gpu_id=0 if gpu available


def cross_validation(dataset, folds, models=[], save_name=None, load_name=None):
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

    result_df = dataset[["user_id", "venue_id", "label"]].copy()
    result_df["ground_truth"] = labels.astype(int)
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
    return models, result_df


def print_results(result_df, name):
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
    parser.add_argument("-f", "--fold_mode", default="spatial", type=str)
    parser.add_argument("-k", "--kfold", default=4, type=int)
    parser.add_argument("-b", "--buffer_factor", default=1.5, type=float)
    parser.add_argument("-l", "--lda", action="store_true")
    args = parser.parse_args()

    city = args.city

    out_dir_base = args.out_dir
    os.makedirs(out_dir_base, exist_ok=True)
    out_name = f"{args.model}_{args.poi_data}_{args.city}_{args.fold_mode}"
    out_dir = os.path.join(out_dir_base, out_name)
    print(out_dir_base, out_dir)
    if os.path.exists(out_dir):
        print("Warning: Output directory already exists, may be overwriting files")
    os.makedirs(out_dir, exist_ok=True)

    # get model
    ModelClass = model_dict[args.model]["model_class"]
    model_config = model_dict[args.model]["config"]

    # load data
    data_raw = read_gdf_csv(os.path.join(args.data_path, f"checkin_{city}_features.csv"))
    pois = read_poi_geojson(os.path.join(args.data_path, f"pois_{city}_{args.poi_data}.geojson"))

    # Split data
    np.random.seed(42)
    if args.fold_mode == "spatial":
        folds = spatial_split(data_raw, args.kfold)
    elif args.fold_mode == "venue":
        folds = venue_split(data_raw, args.kfold)
    else:
        raise ValueError("fold_mode argument must be one of spatial, venue")
    # print("Fold lengths", [len(f) for f in folds])

    # 1) USER-FEATURES: check the performance with solely the user features
    _, results_only_user = cross_validation(data_raw, folds)
    print_results(results_only_user, "temporal_features")

    # # Uncomment to keep only the spacial features:
    # drop_cols = [col for col in data_raw.columns if col.startswith("feat")]
    # print("Dropping features: ", drop_cols)
    # data_raw.drop(drop_cols, axis=1, inplace=True)
    # print(data_raw.columns)

    # obfuscate coordinates
    for masking in [0, 25, 50, 100, 200, 400, 800, 1600]:
        buffering = max(args.buffer_factor * masking, 400)
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
        spatial_joined["ground_truth"] = spatial_joined["label"]
        spatial_joined["prediction"] = spatial_joined["poi_my_label"]
        print_results(spatial_joined, f"spatial_join_{masking}")

        # get poi features
        poi_process = POI_processor(data, pois)
        poi_process(buffer=buffering)
        poi_features = poi_process.distance_count_features()
        if args.lda:
            lda_features = poi_process.lda_features()
            assert len(poi_features) == len(lda_features)
            poi_features = poi_features.merge(lda_features, left_index=True, right_index=True)

        # version 2: together with user features
        dataset = data.merge(poi_features, left_on=["latitude", "longitude"], right_index=True, how="left")
        print("Percentage of rows with at least one NaN", dataset.isna().any(axis=1).sum() / len(dataset))
        dataset = dataset.fillna(0)
        # print("Merge user featuers and POI features", len(poi_features), len(data), len(dataset))
        # if any(pd.isna(dataset)):
        #     print("Attention: NaNs in data", sum(pd.isna(dataset)))

        _, result_df = cross_validation(dataset, folds, models=[])
        # # CODE to train only on non-obfuscated and test on others
        # if masking == 0:
        #     models, result_df = cross_validation(dataset, folds, models)
        # else:
        #     _, result_df = cross_validation(dataset, folds, models)
        print_results(result_df, f"all_features_{masking}")

    with open(os.path.join(out_dir, "results.json"), "w") as outfile:
        json.dump(results_dict, outfile)
