import os
import geopandas as gpd

from foursquare_privacy.utils.io import read_gdf_csv

if __name__ == "__main__":
    # lod pois and data
    pois = gpd.read_file(os.path.join("data", "pois_newyorkcity_labelled.geojson"))
    data = read_gdf_csv(os.path.join("data", "foursquare_nyc.csv"))

    # TODO: project geometries!!
    spatial_joined = data.sjoin_nearest(pois, distance_col="distance")

    print("Accuracy", sum(spatial_joined["label"] == spatial_joined["poi_my_label"]) / len(spatial_joined))

    possible = spatial_joined[spatial_joined["label"] != "work"]
    possible = possible[possible["label"] != "residential"]
    print("Accuracy without home and work", sum(possible["label"] == possible["poi_my_label"]) / len(possible))
