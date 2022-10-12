from shapely import wkt
import os
import pandas as pd
import geopandas as gpd
from datetime import timedelta

from foursquare_privacy.utils.io import txt_to_df
from foursquare_privacy.utils.purpose_categories import get_purpose_category


def month_to_number(x):
    month_dict = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }
    month = month_dict[x[:3]]
    return f"2012/{month}/" + x[4:]


# #  column description by website:
# 1. User ID (anonymized)
# 2. Venue ID (Foursquare)
# 3. Venue category ID (Foursquare)
# 4. Venue category name (Fousquare)
# 5. Latitude
# 6. Longitude
# 7. Timezone offset in minutes (The offset in minutes between when this check-in occurred and the same time in UTC)
# 8. UTC time
column_mapping = {
    0: "user_id",
    1: "venue_id",
    2: "category_id",
    3: "category",
    4: "latitude",
    5: "longitude",
    6: "timezone_offset",
    7: "utc_time",
}

for city in ["NYC", "TKY"]:
    df = txt_to_df(os.path.join("data", "foursquare_ny_tokio_raw", f"dataset_TSMC2014_{city}.txt"), encoding="latin-1")
    # rename categories
    df.rename(
        columns=column_mapping, inplace=True,
    )

    # get datetime object
    df["time"] = df["utc_time"].str[4:-10]
    df["time"] = df["time"].apply(month_to_number)
    df["time"] = pd.to_datetime(df["time"], format="%Y/%m/%d %H:%M:%S")

    # get local time
    df["timezone_offset"] = df["timezone_offset"].astype(int)
    df["local_time"] = df.apply(lambda row: row["time"] + timedelta(minutes=row["timezone_offset"]), axis=1)

    # get label
    df["label"] = df["category"].apply(get_purpose_category)

    # to gdf
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="epsg:4326")
    gdf["geometry"] = gdf["geometry"].apply(wkt.dumps)

    # save
    gdf.index.name = "id"
    gdf.to_csv(os.path.join("data", f"foursquare_{city.lower()}.csv"))
    print("Saved to file", city, "number records", len(gdf))
