import osmnx
import pandas as pd
import os

RADIUS = 20
out_path = os.path.join("data", "osm_poi_single")
os.makedirs(out_path, exist_ok=True)

data = pd.read_csv(os.path.join("data", "tist_500_label.csv"))

# chunksize = 2

# start, end = i * chunksize, (i + 1) * chunksize
# gdf_list = []
# for i in range(len(data)):
for i, row in data.iterrows():
    gdf = osmnx.geometries.geometries_from_point(
        (row["latitude"], row["longitude"]), dist=RADIUS, tags={"amenity": True}
    )
    if "node" in gdf.index:
        gdf["lon_base"] = row["longitude"]
        gdf["lat_base"] = row["latitude"]

        gdf.loc["node"].to_file(os.path.join(out_path, f"pois_{i}.geojson"), driver="GeoJSON")
        print(f"Finished processing record {i} and wrote to file new pois:", len(gdf))
    else:
        print("No records for ", i)

#     if len(gdf) > 0:
#         gdf_list.append(gdf.loc["node"])
# if len(gdf_list) == 0:
#     print("No records found for any of the points in the chunk")
#     continue
# final_gdf = pd.concat(gdf_list)
