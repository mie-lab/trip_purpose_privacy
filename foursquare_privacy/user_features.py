from gettext import npgettext


import numpy as np
import pandas as pd


def get_visit_count_features(data):
    """Compute features about how often a person has visited a venue"""
    grouped_by_user = data.groupby(["user_id", "venue_id"]).agg({"venue_id": "count"})
    total_places_by_user = (data.groupby("user_id").count()[["venue_id"]]).rename(
        columns={"venue_id": "places_by_user"}
    )
    nr_visits = (
        grouped_by_user.rename(columns={"venue_id": "count"})
        .reset_index()
        .merge(total_places_by_user, left_on="user_id", right_index=True, how="left")
    )
    nr_visits["feat_visit_count_log"] = np.log(nr_visits["count"])
    nr_visits["feat_visit_ratio"] = nr_visits["count"] / nr_visits["places_by_user"]
    return nr_visits.set_index(["user_id", "venue_id"])[["feat_visit_count_log", "feat_visit_ratio"]]


def daytime(x):
    # one hot vector for time
    res = [0, 0, 0, 0]
    if x > 7 and x < 12:
        res[0] = 1
    elif x >= 12 and x < 17:
        res[1] = 1
    elif x >= 17 and x < 22:
        res[2] = 1
    else:
        res[3] = 1
    return pd.Series(res)


def time_features(inp_data, col_name="local_time"):
    """Get hour and day features, grouped by user and venue"""
    data = inp_data.copy()
    data["feat_hour"] = data[col_name].dt.hour
    data["feat_day"] = data[col_name].dt.dayofweek
    data["feat_is_weekend"] = (data["feat_day"] > 4).astype(int)
    daytime_df = data["feat_hour"].apply(daytime)
    daytime_df.columns = ["feat_morning", "feat_afternoon", "feat_evening", "feat_night"]
    data = pd.concat((data, daytime_df), axis=1)
    for time_range, period in zip(["hour", "day"], [24, 7]):
        data[f"feat_{time_range}_sin"] = np.sin(data[f"feat_{time_range}"] / period * 2 * np.pi)
        data[f"feat_{time_range}_cos"] = np.cos(data[f"feat_{time_range}"] / period * 2 * np.pi)
    include_mean_features = ["feat_is_weekend", "feat_hour_sin", "feat_hour_cos"] + list(daytime_df.columns)
    visit_times = data.groupby(["user_id", "venue_id"]).agg({feat: "mean" for feat in include_mean_features})
    return visit_times


def merge_repeated_checkins(data, max_hours=1):
    # check duration at stay
    data = data.sort_values(["user_id", "local_time"])
    data["prev_check_in_time"] = data["local_time"].shift(1)
    data["prev_venue_id"] = data["venue_id"].shift(1)
    data["time_diff"] = (data["local_time"] - data["prev_check_in_time"]).dt.total_seconds() / 3600
    len_data_prev = len(data)
    data = data[(data["venue_id"] != data["prev_venue_id"]) | (data["time_diff"] > max_hours)]
    print(f"Ratio of repeated entries of venues within {max_hours} hours (deleted):", 1 - len(data) / len_data_prev)
    return data.drop(["prev_check_in_time", "prev_venue_id", "time_diff"], axis=1)


def get_duration_feature(inp_data):
    data = inp_data.copy()
    # problem: now we have the time diff to the previous one. We need that for merging two entries (delete second check in)
    # however, for computing the duration we want to have the time diff to the next one. so we roll again
    if "finished_at" not in data.columns:
        print("Approximate finished at by next check in.")
        data["finished_at"] = data["local_time"].shift(-1)
    data["duration"] = (data["finished_at"] - data["local_time"]).dt.total_seconds() / 3600
    nanmean = np.nanmean(data.loc[data["duration"] > 0, "duration"])
    data.loc[data["duration"] < 0, "duration"] = nanmean
    data.loc[pd.isna(data["duration"]), "duration"] = nanmean
    data["feat_log_duration"] = np.log(data["duration"] + 1)
    grouped_duration_data = data.groupby(["user_id", "venue_id"]).agg({"feat_log_duration": "mean"})
    # # Code to include std:
    # "log_duration": ["mean", "std"]
    # grouped_duration_data.columns = grouped_duration_data.columns.to_flat_index()
    # return grouped_duration_data.rename(
    #     columns={("log_duration", "mean"): "feat_log_duration", ("log_duration", "std"): "feat_log_duration_std"}
    # )
    return grouped_duration_data

