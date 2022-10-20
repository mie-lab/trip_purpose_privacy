import os
import pandas as pd
import geopandas as gpd
from shapely import wkt

if __name__ == "__main__":
    out_path = "data"
    for city in ["tokyo", "newyorkcity"]:
        data = pd.read_csv(os.path.join("data", f"foursquare_{city}.csv"))
        # group by geometry
        poi_foursquare = data.groupby(["latitude", "longitude"]).agg(
            {"label": "first", "category": "first", "geometry": "first"}
        )
        poi_foursquare = (
            poi_foursquare.reset_index()
            .drop(["latitude", "longitude"], axis=1)
            .rename(columns={"label": "poi_my_label", "category": "poi_type"})
        )
        poi_foursquare.index.name = "id"

        # to geodataframe
        poi_foursquare["geometry"] = poi_foursquare["geometry"].apply(wkt.loads)
        poi_foursquare = gpd.GeoDataFrame(poi_foursquare, geometry="geometry")
        poi_foursquare.crs = "EPSG:4326"

        poi_foursquare.to_file(os.path.join(out_path, f"pois_{city}_foursquare.geojson"), driver="GeoJSON")
