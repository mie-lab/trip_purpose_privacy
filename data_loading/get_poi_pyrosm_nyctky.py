import os
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
    custom_filter = {"amenity": True, "shop": True}
    pois = osm.get_pois(custom_filter=custom_filter)

    # Gather info about POI type (combines the tag info from "amenity" and "shop")
    pois["poi_type"] = pois["amenity"]
    pois["poi_type"] = pois["poi_type"].fillna(pois["shop"])

    pois.to_file(os.path.join(out_path, f"pois_{city}.geojson"), driver="GeoJSON")
