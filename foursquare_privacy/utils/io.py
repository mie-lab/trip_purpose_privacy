import pandas as pd
import geopandas as gpd
from shapely import wkt


def txt_to_df(path, encoding="utf-8"):
    # read and change to dataframe
    f = open(path, "r", encoding=encoding)
    all_lines = f.readlines()

    rows = [row[:-2].split("\t") for row in all_lines]

    df = pd.DataFrame(rows)

    return df


def read_gdf_csv(path):
    data = pd.read_csv(path, index_col="id")
    data = gpd.GeoDataFrame(data)
    data["geometry"] = data["geometry"].apply(wkt.loads)
    data.crs = "EPSG:4326"
    data["local_time"] = pd.to_datetime(data["local_time"])
    data = data[data["label"] != "other"]
    return data
