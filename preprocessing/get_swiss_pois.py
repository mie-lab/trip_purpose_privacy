from shapely import wkt
import os
import numpy as np
import json
import pandas as pd
import geopandas as gpd


if __name__ == "__main__":

    with open(os.path.join("data", "foursquare_taxonomy.json"), "r") as infile:
        poi_taxonomy = json.load(infile)

    pois_global = pd.read_csv(os.path.join("data", "dataset_TIST2015", "dataset_TIST2015_POIs.txt"), sep="\t", header=0)
    swiss_pois = pois_global[pois_global["US"] == "CH"]

    swiss_pois = swiss_pois.rename(
        columns={"40.733596": "latitude", "-74.003139": "longitude", "Jazz Club": "poi_type"}
    ).drop(["US", "3fd66200f964a52000e71ee3"], axis=1)
    swiss_pois["id"] = np.arange(len(swiss_pois))
    swiss_pois.set_index("id", inplace=True)

    swiss_pois.loc[swiss_pois["poi_type"].str.contains("Caf"), "poi_type"] = "Cafe"
    swiss_pois["poi_my_label"] = swiss_pois["poi_type"].map(poi_taxonomy)

    print("Number of POIs in Switzerland", len(swiss_pois))
    swiss_pois.dropna(inplace=True)
    print("After removing nans:", len(swiss_pois))
    # to gdf
    gdf = gpd.GeoDataFrame(
        swiss_pois, geometry=gpd.points_from_xy(swiss_pois["longitude"], swiss_pois["latitude"]), crs="epsg:4326"
    )
    gdf["geometry"] = gdf["geometry"].apply(wkt.dumps)

    # save
    gdf.to_csv(os.path.join("data", "foursquare_switzerland.csv"))
