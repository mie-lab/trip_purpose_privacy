import os
import pandas as pd

from foursquare_privacy.user_features import *
from foursquare_privacy.utils.io import read_gdf_csv

if __name__ == "__main__":
    for city in ["yumuv", "tokyo", "newyorkcity"]:
        # for yumuv we don't need to change the venue ID
        correct_venue_id = False if city == "yumuv" else True

        data = pd.read_csv(os.path.join("data", f"checkin_{city}.csv"))

        # to datetime
        data["local_time"] = pd.to_datetime(data["local_time"])
        if "finished_at" in data.columns:
            data["finished_at"] = pd.to_datetime(data["finished_at"])

        # merging
        data = merge_repeated_checkins(data)

        # group by longitude and latitude to clean up the venue ID
        if correct_venue_id:
            print("Replacing venue ID by long-lat grouping")
            actual_venue = data.groupby(["latitude", "longitude"]).agg({"venue_id": "first"})
            # merge with data again to have everything as input for the user features
            data = (
                data.drop(["venue_id"], axis=1, errors="ignore")
                .merge(actual_venue, left_on=["latitude", "longitude"], right_index=True, how="left")
                .reset_index()
            )
        grouped_visits_user_venue = data.groupby(["user_id", "venue_id"]).agg(
            {"label": "first", "category": "first", "geometry": "first"}
        )

        # get features and merge them
        nr_visits = get_visit_count_features(data)
        user_venue_data = grouped_visits_user_venue.merge(nr_visits, left_index=True, right_index=True, how="inner")
        time_feats = time_features(data)
        user_venue_data = user_venue_data.merge(time_feats, left_index=True, right_index=True, how="inner")
        print("merged time and visit count features", len(nr_visits), len(time_feats), len(user_venue_data))
        duration_features = get_duration_feature(data)
        user_venue_data = user_venue_data.merge(duration_features, left_index=True, right_index=True, how="inner")
        print("merged time and visit count features with duration features", len(user_venue_data))

        user_venue_data.reset_index(inplace=True)
        user_venue_data.index.name = "id"

        assert all(pd.isna(user_venue_data).sum() == 0), "NaNs in user venue dataframe"

        user_venue_data.to_csv(os.path.join("data", f"checkin_{city}_features.csv"))

        print(f"Saved user-features dataframe for {city}, length {len(user_venue_data)}")
