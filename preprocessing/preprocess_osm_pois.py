import os
import pandas as pd
import geopandas as gpd
from foursquare_privacy.utils.purpose_categories import osm_poi_mapping


if __name__ == "__main__":
    out_path = "data"
    for city in ["newyorkcity", "tokyo"]:
        pois = gpd.read_file(os.path.join("data", f"pois_{city}.geojson"))
        pois_simple = pois.drop(
            [
                "addr:city",
                "addr:country",
                "addr:full",
                "addr:housenumber",
                "addr:housename",
                "addr:postcode",
                "addr:place",
                "addr:street",
                "email",
                "operator",
                "phone",
                "ref",
                "url",
                "website",
                "wikipedia",
                "version",
                "id",
                "changeset",
                "amenity",  # all amenities were merged into the shop-labels (poi_type)
            ],
            axis=1,
            errors="ignore",
        )

        # remove parking stuff
        pois_simple = pois_simple[pois_simple["poi_type"] != "parking_space"]

        # map to my labels
        pois_simple["poi_my_label"] = pois_simple["poi_type"].map(osm_poi_mapping)
        pois_simple = pois_simple[~pd.isna(pois_simple["poi_my_label"])]

        # save only main columns
        pois_simple = pois_simple[["geometry", "poi_type", "poi_my_label"]]

        pois_simple.to_file(os.path.join(out_path, f"pois_{city}_osm.geojson"), driver="GeoJSON")

