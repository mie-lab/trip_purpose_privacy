import json
import os
import pandas as pd
import xgboost
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from foursquare_privacy.utils.io import read_gdf_csv, read_poi_geojson
from foursquare_privacy.utils.user_distribution import get_user_dist_mae
from foursquare_privacy.utils.spatial_folds import spatial_split
from foursquare_privacy.add_poi import POI_processor
from foursquare_privacy.plotting import confusion_matrix, plot_confusion_matrix
from foursquare_privacy.location_masking import LocationMasker

KFOLD = 4
OUT_NAME = "xgb_test"


def cross_validation(dataset):
    # get features and labels
    features = dataset[[col for col in dataset.columns if col.startswith("feat")]]
    # print("List of features", features.columns)
    uni_labels = np.unique(dataset["label"])
    # map labels to numbers
    label_mapping = {u: i for i, u in enumerate(uni_labels)}
    labels = dataset["label"].map(label_mapping)

    folds = spatial_split(dataset, KFOLD)
    # print("Fold lengths", [len(f) for f in folds])

    result_df = dataset[["user_id", "venue_id", "label"]].copy()
    result_df["ground_truth"] = labels.astype(int)
    result_df["prediction"] = -1

    for fold_num, fold_samples in enumerate(folds):
        test_x = features.loc[fold_samples]
        # test_y = labels.loc[fold_samples]
        train_x = features.drop(fold_samples)
        train_y = labels.drop(fold_samples)

        # fit and predict
        model = xgboost.XGBClassifier()  # smaller max depth did not help
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        # y_pred = np.random.choice(np.arange(12), len(test_x))  # testing

        result_df.loc[fold_samples, "prediction"] = y_pred
        # print(f"Accuracy fold {fold_num+1}:", sum(y_pred == test_y) / len(test_y))
    return result_df


def print_results(result_df, name):
    acc = accuracy_score(result_df["ground_truth"], result_df["prediction"])
    bal_acc = balanced_accuracy_score(result_df["ground_truth"], result_df["prediction"])
    user_mae = np.mean(get_user_dist_mae(result_df))
    print(name, "Acc:", round(acc, 3), "Bal. acc:", round(bal_acc, 3), "User-wise MAE", round(user_mae, 3))
    result_df.to_csv(os.path.join(out_dir, f"predictions_{name}.csv"))
    results_dict[name] = {"Accuracy": acc, "Balanced accuracy": bal_acc, "User-wise MAE": user_mae}


results_dict = {}

if __name__ == "__main__":
    city = "newyorkcity"

    out_dir = os.path.join("outputs", OUT_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # load data
    data_raw = read_gdf_csv(os.path.join("data", f"foursquare_{city}_features.csv"))
    pois = read_poi_geojson(os.path.join("data", f"pois_{city}_foursquare.geojson"))

    # 1) USER-FEATURES: check the performance with solely the user features
    results_only_user = cross_validation(data_raw)
    print_results(results_only_user, "user_only")

    # obfuscate coordinates
    for masking in [0, 25, 50, 100, 200]:
        print(f"-------- Masking {masking} ---------")
        if masking == 0:
            data = data_raw.copy()
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

        # plot_confusion_matrix(
        #     test_y, y_pred, col_names=uni_labels, out_path=os.path.join("figures", "xgb_poi_confusion.png"),
        # )
        result_df = cross_validation(dataset)
        print_results(result_df, f"all_features_{masking}")

    with open(os.path.join(out_dir, "results.json"), "w") as outfile:
        json.dump(results_dict, outfile)
