import os
import pandas as pd
import xgboost
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from foursquare_privacy.utils.io import read_gdf_csv
from foursquare_privacy.utils.user_distribution import get_user_dist_mae
from foursquare_privacy.utils.spatial_folds import spatial_split
from foursquare_privacy.add_poi import POI_processor
from foursquare_privacy.plotting import confusion_matrix, plot_confusion_matrix
from foursquare_privacy.location_masking import LocationMasker

KFOLD = 4
# MASKING = 50
OUT_NAME = "xgb_test"

if __name__ == "__main__":
    city = "newyorkcity"

    out_dir = os.path.join("outputs", OUT_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # load data
    data_raw = read_gdf_csv(os.path.join("data", f"foursquare_{city}_features.csv"))
    poi_path = os.path.join("data", f"pois_{city}_labelled.geojson")

    # obfuscate coordinates
    for masking in [0, 25, 50, 100, 200]:
        print(f"-------- Masking {masking} ---------")
        if masking == 0:
            data = data_raw.copy()
        else:
            masker = LocationMasker(data_raw)
            data = masker(masking)  # 50 meters
            print(
                f"Average latitude difference (masking {masking})",
                (data_raw["latitude"] - data["latitude"]).abs().mean(),
            )

        # get poi features
        poi_process = POI_processor(data, poi_path=poi_path)
        poi_process()
        distance_features = poi_process.distance_count_features()
        lda_features = poi_process.lda_features()
        assert len(distance_features) == len(lda_features)
        poi_features = distance_features.merge(lda_features, left_index=True, right_index=True)

        # version 1: solely POI features
        # label_per_loc = data.groupby(["latitude", "longitude"]).agg({"label": "first"})
        # dataset = poi_features.merge(label_per_loc, left_index=True, right_index=True, how="left")
        # dataset = pd.read_csv("data/poionly_dataset.csv")

        # version 2: together with user features
        dataset = data.merge(poi_features, left_on=["longitude", "latitude"], right_index=True, how="left")
        print("Merge user featuers and POI features", len(poi_features), len(data), len(dataset))

        # get features and labels
        features = dataset[[col for col in dataset.columns if col.startswith("feat")]]
        print("List of features", features.columns)
        uni_labels = np.unique(dataset["label"])
        # map labels to numbers
        label_mapping = {u: i for i, u in enumerate(uni_labels)}
        labels = dataset["label"].map(label_mapping)

        folds = spatial_split(dataset, KFOLD)
        print("Fold lengths", [len(f) for f in folds])

        result_df = dataset[["user_id", "venue_id", "label"]]
        result_df["ground_truth"] = labels
        result_df["prediction"] = pd.NA

        for fold_num, fold_samples in enumerate(folds):
            test_x = features.loc[fold_samples]
            test_y = labels.loc[fold_samples]
            train_x = features.drop(fold_samples)
            train_y = labels.drop(fold_samples)

            # # Simple train test split (0.9)
            # train_len = int(len(features) * 0.9)
            # train_x, test_x = features[:train_len], features[train_len:]
            # train_y, test_y = labels_argmax[:train_len], labels_argmax[train_len:]
            print("train and test data", train_x.shape, test_x.shape, train_y.shape, test_y.shape)

            # fit and predict
            model = xgboost.XGBClassifier()  # smaller max depth did not help
            model.fit(train_x, train_y)
            y_pred = model.predict(test_x)
            # y_pred = np.random.choice(np.arange(12), len(test_x)) # testing

            result_df.loc[fold_samples, "prediction"] = y_pred

            print(f"Accuracy fold {fold_num+1}:", sum(y_pred == test_y) / len(test_y))

        print(
            "Acc:",
            round(accuracy_score(result_df["ground_truth"], result_df["prediction"]), 3),
            "Bal. acc:",
            round(balanced_accuracy_score(result_df["ground_truth"], result_df["prediction"]), 3),
            "User-wise MAE",
            round(np.mean(get_user_dist_mae(result_df)), 3),
        )

        result_df.to_csv(os.path.join(out_dir, f"predictions_{masking}.csv"))
        # plot_confusion_matrix(
        #     test_y, y_pred, col_names=uni_labels, out_path=os.path.join("figures", "xgb_poi_confusion.png"),
        # )
