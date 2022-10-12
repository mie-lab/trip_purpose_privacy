import datetime
import pandas as pd
import os

from foursquare_privacy.utils.io import txt_to_df


def gt(dt_str):
    dt, _, us = dt_str[:-1].partition(".")
    dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    return dt + datetime.timedelta(microseconds=0)


df = txt_to_df("/Users/ninawiedemann/MIE/general_data/loc-gowalla_totalCheckins.txt")

# make datetime
df["dt"] = df[1].apply(gt)

# give column names
df = df.rename(
    columns={0: "user_id", 1: "started_at_raw", 2: "latitude", 3: "longitude", 4: "venue_id", "dt": "started_at_utc"}
)
df.index.name = "id"

df.drop(["started_at_raw"], axis=1, inplace=True)

# write to database
gdf.to_postgis(
    "check_ins", engine_graphrep, schema="gowalla", if_exists="fail", index=True, index_label="id", chunksize=10000
)

