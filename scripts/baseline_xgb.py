import os
import pandas as pd
import xgboost
import numpy as np

from foursquare_privacy.utils.io import read_gdf_csv
from foursquare_privacy.add_poi import POI_processor
from foursquare_privacy.plotting import confusion_matrix, plot_confusion_matrix

if __name__ == "__main__":
    city = "tokyo"
    data = read_gdf_csv(os.path.join("data", f"foursquare_{city}.csv"))
    poi_path = os.path.join("data", f"pois_{city}_labelled.geojson")

    # get poi features
    poi_process = POI_processor(data, poi_path=poi_path)
    poi_process()
    distance_features = poi_process.distance_count_features()
    lda_features = poi_process.lda_features()
    assert len(distance_features) == len(lda_features)
    poi_features = distance_features.merge(lda_features, left_index=True, right_index=True)

    # version 1: solely POI features
    label_per_loc = data.groupby(["latitude", "longitude"]).agg({"label": "first"})
    features_and_labels = poi_features.merge(label_per_loc, left_index=True, right_index=True, how="left")
    labels_one_hot = pd.get_dummies(features_and_labels["label"], prefix="label")
    # get label and features as input data
    labels_argmax = np.argmax(np.array(labels_one_hot), axis=1)
    features = features_and_labels.drop("label", axis=1)

    # train test split
    train_len = int(len(features) * 0.9)
    train_x, test_x = features[:train_len], features[train_len:]
    train_y, test_y = labels_argmax[:train_len], labels_argmax[train_len:]
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

    # fit and predict
    model = xgboost.XGBClassifier()
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)

    print("Accuracy:", sum(y_pred == test_y) / len(test_y))

    plot_confusion_matrix(
        test_y,
        y_pred,
        col_names=np.unique(features_and_labels["label"]),
        out_path=os.path.join("figures", "xgb_poi_confusion.png"),
    )
