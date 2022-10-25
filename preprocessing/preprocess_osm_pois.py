import os
import json
import pandas as pd
from pyrosm import OSM
from pyrosm import get_data

out_path = "data"

for city in ["newyorkcity", "tokyo"]:
    fp = get_data(city)
    # Initialize the OSM parser object
    osm = OSM(fp)

    # By default pyrosm reads all elements having "amenity", "shop" or "tourism" tag
    # Here, let's read only "amenity" and "shop" by applying a custom filter that
    # overrides the default filtering mechanism
    custom_filter = {
        "healthcare": True,
        "shop": True,
        "leisure": True,
        "amenity": True,
        "tourism": ["museum"],
        "building": ["religious", "transportation"],
        "public_transport": ["station"],
    }
    pois = osm.get_data_by_custom_criteria(custom_filter=custom_filter, keep_ways=False, keep_relations=False)
    print("Raw POIs length", len(pois))

    pois_simple = pois.drop(
        [
            "timestamp",
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
            "changeset",
            "opening_hours",
        ],
        axis=1,
        errors="ignore",
    )
    # remove parking stuff
    pois_simple = pois_simple[pois_simple["amenity"] != "parking"]
    pois_simple = pois_simple[pois_simple["amenity"] != "parking_space"]

    pois_simple.to_file(os.path.join(out_path, f"pois_{city}_osm_raw.geojson"), driver="GeoJSON")

    with open(os.path.join(out_path, "osm_poi_mapping.json"), "r") as infile:
        osm_poi_mapping = json.load(infile)

    # Combine everything in the poi type
    pois_simple["poi_type"] = pois_simple["amenity"]
    pois_simple["poi_type"] = pois_simple["poi_type"].fillna(pois_simple["leisure"])
    pois_simple.loc[
        pd.isna(pois_simple["poi_type"]) & (pois_simple["tags"].str.contains("healthcare").fillna(False)), "poi_type"
    ] = "healthcare"
    # add museums
    pois_simple.loc[
        pd.isna(pois_simple["poi_type"]) & (pois_simple["name"].str.contains("museum").fillna(False)), "poi_type"
    ] = "museum"
    pois_simple["poi_type"] = pois_simple["poi_type"].fillna(pois_simple["religion"])
    pois_simple["poi_type"] = pois_simple["poi_type"].fillna(pois_simple["public_transport"])
    pois_simple["poi_type"] = pois_simple["poi_type"].fillna(pois_simple["shop"])

    # reduce to relevant columns and dropn nans
    prev_len = len(pois_simple)
    pois_simple = pois_simple[["id", "lon", "lat", "geometry", "poi_type"]].dropna()
    print("Number of POIs that are dropped because they cannot be assigned a poi_type:", prev_len - len(pois_simple))

    # Add my labels
    prev_len = len(pois_simple)
    pois_simple["poi_my_label"] = pois_simple["poi_type"].map(osm_poi_mapping)
    pois_simple = pois_simple[~pd.isna(pois_simple["poi_my_label"])]
    print("Number of POIs that are dropped because they cannot be assigned a label:", prev_len - len(pois_simple))

    # save
    pois_simple.to_file(os.path.join(out_path, f"pois_{city}_osm.geojson"), driver="GeoJSON")
    print(f"Saving {len(pois_simple)} POIs")
