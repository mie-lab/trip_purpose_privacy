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
    if "newyorkcity" in path or "nyc" in path:
        print("Projecting into NYC CRS")
        data.to_crs("EPSG:32118", inplace=True)
    elif "tokyo" in path or "tky" in path:
        print("Projecting into TKY CRS")
        data.to_crs("EPSG:30169", inplace=True)
    else:
        raise RuntimeError("Can only handly ny or tky")
    # adjust lon and lat to crs
    data["longitude"] = data["geometry"].x
    data["latitude"] = data["geometry"].y
    # add datetime
    data["local_time"] = pd.to_datetime(data["local_time"])
    data = data[data["label"] != "other"]
    return data


def read_poi_geojson(path):
    gdf = gpd.read_file(path)
    gdf.crs = "EPSG:4326"
    if "newyorkcity" in path or "nyc" in path:
        print("Projecting POIs into NYC CRS")
        gdf.to_crs("EPSG:32118", inplace=True)
    elif "tokyo" in path or "tky" in path:
        print("Projecting POIs into TKY CRS")
        gdf.to_crs("EPSG:30169", inplace=True)
    return gdf
