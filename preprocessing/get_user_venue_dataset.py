import os
import pandas as pd

from foursquare_privacy.user_features import *
from foursquare_privacy.utils.io import read_gdf_csv

if __name__ == "__main__":
    for city in ["tokyo", "newyorkcity"]:
        data = pd.read_csv(os.path.join("data", f"foursquare_{city}.csv"))
        data["local_time"] = pd.to_datetime(data["local_time"])

        # merging
        data = merge_repeated_checkins(data)

        # group by longitude and latitude to clean up the venue ID
        actual_venue = data.groupby(["user_id", "latitude", "longitude"]).agg(
            {"label": "first", "category": "first", "geometry": "first"}
        )
        actual_venue["venue_id"] = np.arange(len(actual_venue))
        # merge with data again to have everything as input for the user features
        data = data.drop(["venue_id", "label", "category"], axis=1).merge(
            actual_venue, left_on=["user_id", "latitude", "longitude"], right_index=True, how="left"
        )
        actual_venue = actual_venue.reset_index().set_index(["user_id", "venue_id"])

        # get features and merge them
        nr_visits = get_visit_count_features(data)
        user_venue_data = actual_venue.merge(nr_visits, left_index=True, right_index=True, how="inner")
        time_feats = time_features(data)
        user_venue_data = user_venue_data.merge(time_feats, left_index=True, right_index=True, how="inner")
        print("merged time and visit count features", len(nr_visits), len(time_feats), len(user_venue_data))
        duration_features = get_duration_feature(data)
        user_venue_data = user_venue_data.merge(duration_features, left_index=True, right_index=True, how="inner")
        print("merged time and visit count features with duration features", len(user_venue_data))

        user_venue_data.reset_index(inplace=True)
        user_venue_data.index.name = "id"

        assert all(pd.isna(user_venue_data).sum() == 0), "NaNs in user venue dataframe"

        user_venue_data.to_csv(os.path.join("data", f"foursquare_{city}_features.csv"))

        print(f"Saved user-features dataframe for {city}, length {len(user_venue_data)}")
