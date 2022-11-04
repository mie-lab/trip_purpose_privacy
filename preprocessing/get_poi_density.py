import argparse
import os
import matplotlib.pyplot as plt
from foursquare_privacy.add_poi import POI_processor
from foursquare_privacy.utils.io import read_gdf_csv, read_poi_geojson


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="data", type=str)
    parser.add_argument("-c", "--city", default="newyorkcity", type=str)
    parser.add_argument("-p", "--poi_data", default="foursquare", type=str)
    parser.add_argument("-b", "--buffer", default=500, type=int)
    args = parser.parse_args()
    city = args.city
    data_raw = read_gdf_csv(os.path.join(args.data_path, f"checkin_{city}_features.csv"))
    pois = read_poi_geojson(os.path.join(args.data_path, f"pois_{city}_{args.poi_data}.geojson"))

    poi_process = POI_processor(data_raw, pois)
    poi_process(buffer=args.buffer)
    count_per_lon_lat = (
        poi_process.geom_with_pois.groupby(["latitude", "longitude"])
        .agg({"poi_type": "count"})
        .rename(columns={"poi_type": "poi_density"})
    )

    # plt.hist(count_per_lon_lat["poi_density"], bins=100)
    # plt.xlabel("Number of surrounding POIs (within 500m)", fontsize=15)
    # plt.tight_layout()
    # plt.savefig(os.path.join("figures", f"poi_density_histogram_{args.poi_data}.png"))
    # plt.show()

    poi_density_per_venue = data_raw.merge(
        count_per_lon_lat, how="left", left_on=["latitude", "longitude"], right_on=["latitude", "longitude"],
    )[["venue_id", "poi_density"]]

    poi_density_per_venue.to_csv(
        os.path.join(args.data_path, f"poi_density_{args.city}_{args.poi_data}_{args.buffer}.csv"), index=False
    )
