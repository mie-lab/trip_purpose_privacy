import os
import numpy as np
import geopandas as gpd

from foursquare_privacy.utils.io import read_gdf_csv, read_poi_geojson
from foursquare_privacy.plotting import plot_confusion_matrix

if __name__ == "__main__":
    # lod pois and data
    pois = read_poi_geojson(os.path.join("data", "pois_newyorkcity_labelled.geojson"))
    data = read_gdf_csv(os.path.join("data", "foursquare_nyc.csv"))

    # double check that all are zero
    # grouped_test = data.groupby(["longitude", "latitude"]).agg({"label": "nunique"})
    # assert grouped_test["label"].mean() == 1

    # group by longitude and latitude to get only one prediction per location
    data = gpd.GeoDataFrame(data.groupby(["longitude", "latitude"]).agg({"label": "first", "geometry": "first"}))
    data.set_geometry("geometry", inplace=True)
    data.set_crs(pois.crs, inplace=True)

    spatial_joined = data.sjoin_nearest(pois, how="left")  # , distance_col="distance")

    # remove duplicates (created because some pois have the same distance from a venue)
    spatial_joined = (
        spatial_joined.reset_index().groupby(["longitude", "latitude"]).agg({"label": "first", "poi_my_label": "first"})
    )

    print("Accuracy", sum(spatial_joined["label"] == spatial_joined["poi_my_label"]) / len(spatial_joined))

    possible = spatial_joined[spatial_joined["label"] != "work"]
    possible = possible[possible["label"] != "residential"]
    print("Accuracy without home and work", sum(possible["label"] == possible["poi_my_label"]) / len(possible))

    plot_confusion_matrix(
        spatial_joined["label"].values,
        spatial_joined["poi_my_label"].values,
        out_path=os.path.join("figures", "simpleclosest_poi_confusion.png"),
    )
