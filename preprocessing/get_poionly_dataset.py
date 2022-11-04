import os
from turtle import distance
from typing import final
import pandas as pd
import xgboost
import numpy as np

from foursquare_privacy.utils.io import read_gdf_csv
from foursquare_privacy.add_poi import POI_processor
from foursquare_privacy.plotting import confusion_matrix, plot_confusion_matrix

if __name__ == "__main__":
    distance_features, geom_with_poi, labels = [], [], []
    for city in ["tokyo", "newyorkcity"]:
        data = read_gdf_csv(os.path.join("data", f"checkin_{city}.csv"))
        poi_path = os.path.join("data", f"pois_{city}_labelled.geojson")

        # get poi features
        poi_process = POI_processor(data, poi_path=poi_path)
        poi_process()
        dist_feats = poi_process.distance_count_features()
        dist_feats["city"] = city
        distance_features.append(dist_feats)
        geom_with_poi.append(poi_process.geom_with_pois)
        grouped_labels_df = data.groupby(["latitude", "longitude"]).agg({"label": "first"})
        labels.append(grouped_labels_df)
    distance_features = pd.concat(distance_features)
    label_per_loc = pd.concat(labels)

    # for lda: use both of them together to get the same topics
    poi_process.geom_with_pois = pd.concat(geom_with_poi)
    lda_features = poi_process.lda_features()

    assert len(distance_features) == len(lda_features)
    distance_features.fillna(0, inplace=True)
    poi_features = distance_features.merge(lda_features, left_index=True, right_index=True)

    # version 1: solely POI features
    final_dataset = poi_features.merge(label_per_loc, left_index=True, right_index=True, how="left")

    print(final_dataset.shape)
    final_dataset.to_csv(os.path.join("data", "poionly_dataset.csv"))
