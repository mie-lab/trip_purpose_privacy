import json
from multiprocessing.sharedctypes import Value
import os
import pandas as pd
import xgboost
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from foursquare_privacy.utils.io import read_gdf_csv, read_poi_geojson
from foursquare_privacy.models.xgb import XGBWrapper
from foursquare_privacy.models.mlp import MLPWrapper
from foursquare_privacy.utils.user_distribution import get_user_dist_mae
from foursquare_privacy.utils.spatial_folds import spatial_split
from foursquare_privacy.add_poi import POI_processor
from foursquare_privacy.plotting import confusion_matrix, plot_confusion_matrix
from foursquare_privacy.location_masking import LocationMasker

model_dict = {"xgb": {"model_class": XGBWrapper, "config": {}}, "mlp": {"model_class": MLPWrapper, "config": {}}}
# xgb config: tree_method="gpu_hist", gpu_id=0 if gpu available


def cross_validation(dataset, folds, models=[]):
    if len(models) == 0:
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
        if len(models) < len(folds):
            # fit and predict
            model = ModelClass(model_config)  # smaller max depth did not help
            model.fit(train_x, train_y)
            models.append(model)
        # Option 2: test mode --> models given, just apply
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
    parser.add_argument("-o", "--out_name", default="1", type=str)
    parser.add_argument("-p", "--poi_data", default="foursquare", type=str)
    parser.add_argument("-m", "--model", default="xgb", type=str)
    parser.add_argument("-k", "--kfold", default=4, type=int)
    args = parser.parse_args()

    city = args.city

    out_name_full = f"{args.model}_{args.poi_data}_{args.city}_{args.out_name}"
    out_dir = os.path.join("outputs", out_name_full)
    os.makedirs(out_dir, exist_ok=True)

    # get model
    ModelClass = model_dict[args.model]["model_class"]
    model_config = model_dict[args.model]["config"]

    # load data
    data_raw = read_gdf_csv(os.path.join(args.data_path, f"foursquare_{city}_features.csv"))
    pois = read_poi_geojson(os.path.join(args.data_path, f"pois_{city}_{args.poi_data}.geojson"))

    # Split data
    np.random.seed(42)
    folds = spatial_split(data_raw, args.kfold)
    # print("Fold lengths", [len(f) for f in folds])

    # 1) USER-FEATURES: check the performance with solely the user features
    _, results_only_user = cross_validation(data_raw, folds)
    print_results(results_only_user, "user_only")

    # obfuscate coordinates
    for masking in [0, 25, 50, 100, 200]:
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
        poi_process()
        distance_features = poi_process.distance_count_features()
        lda_features = poi_process.lda_features()
        assert len(distance_features) == len(lda_features)
        poi_features = distance_features.merge(lda_features, left_index=True, right_index=True)

        # version 2: together with user features
        dataset = data.merge(poi_features, left_on=["latitude", "longitude"], right_index=True, how="left")
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
