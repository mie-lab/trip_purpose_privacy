from shapely import wkt
import os
import json
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from collections import Counter


def get_most_often(real_list):
    if isinstance(real_list, str):
        real_list = eval(real_list)
    real_set = set(real_list)
    real_set.discard("unknown")
    real_set.discard("errand")
    if len(real_set) == 0:
        real_set = {"unknown"}
    return max(real_set, key=real_list.count)


if __name__ == "__main__":
    # Get engine
    DBLOGIN_FILE = "../dblogin_mielab.json"
    with open(DBLOGIN_FILE) as json_file:
        LOGIN_DATA = json.load(json_file)
    engine_yumuv = create_engine("postgresql://{user}:{password}@{host}:{port}/yumuv".format(**LOGIN_DATA))

    sql = "SELECT user_id, started_at, finished_at, geom, location_id, purpose FROM\
         yumuv_graph_rep.staypoints where activity=true"
    sp_yumuv = gpd.read_postgis(sql, engine_yumuv, geom_col="geom")

    print("Number of all YUMUV staypoints (only activities!)", len(sp_yumuv))

    # clean
    sp_yumuv["poi_type"] = sp_yumuv["purpose"].apply(get_most_often)

    print("Number of staypoints per category")
    print(Counter(sp_yumuv["poi_type"]))

    # try to match foursquare categories
    mapping_to_categories = {
        "study": "Education",
        "sport": "Sports and Recreation",
        "eat": "Dining",
        "work": "Work",
        "leisure": "Leisure",
    }
    sp_yumuv["label"] = sp_yumuv["poi_type"].map(mapping_to_categories)
    sp_yumuv["label"] = sp_yumuv["label"].fillna(sp_yumuv["poi_type"])

    sp_filtered = sp_yumuv[~sp_yumuv["label"].isin(["unknown", "home", "wait"])]
    print("Remaining staypoints after filtering out home and unknown", len(sp_filtered) / len(sp_yumuv))

    # align with the foursquare columns
    sp_filtered["longitude"] = sp_filtered["geom"].x
    sp_filtered["latitude"] = sp_filtered["geom"].y
    sp_filtered = sp_filtered.rename(
        columns={"started_at": "local_time", "location_id": "venue_id", "geom": "geometry", "poi_type": "category"}
    ).drop(["purpose"], axis=1, errors="ignore")

    sp_filtered["geometry"] = sp_filtered["geometry"].apply(wkt.dumps)
    sp_filtered.to_csv(os.path.join("data", f"yumuv.csv"))
