import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18})
import pandas as pd
import psycopg2
import json
import seaborn as sns
from collections import defaultdict
from shapely.geometry import Point
import trackintel as ti
from sqlalchemy import create_engine
import geopandas as gpd
from shapely.geometry import Point


def auswertung(sp, gt_column="purpose_gt", printout=True):
    # auswertung
    res_dict = {}
    for method in ["FREQ", "OSNA"]:
        if printout:
            print(f"------------ {method} -------------")
        for lab in ["home", "work"]:
            gt_lab = sp[sp[gt_column] == lab]
            correct_lab = gt_lab[gt_lab[f"purpose_{method}"] == lab]
            if printout:
                print(lab, len(correct_lab) / len(gt_lab))
            res_dict[lab, method] = len(correct_lab) / len(gt_lab)
    return res_dict


def add_fake_geom(sp):
    fake_geom = Point(1, 2)
    sp["geom"] = fake_geom
    sp = gpd.GeoDataFrame(sp, geometry="geom")
    return sp


# Remove errand, unknown etc from purpose column
def set_na_purposes(sp):
    sp.loc[sp["purpose_gt"] == "unknown", "purpose_gt"] = pd.NA
    sp.loc[sp["purpose_gt"] == "errand", "purpose_gt"] = pd.NA
    sp.loc[sp["purpose_gt"] == "<NA>", "purpose_gt"] = pd.NA
    sp.loc[sp["purpose_gt"] == "wait", "purpose_gt"] = pd.NA


def replace_wrong_locs(sp):
    # find the rows where one location has more than one purpose
    uni_locs = sp.groupby("location_id").agg({"purpose_gt": "nunique"})
    # 1) for the ones with <= 1 purposes, we just have to get the first element
    locs_with_one = uni_locs[uni_locs["purpose_gt"] <= 1].index
    sp_okay_locs = sp[sp["location_id"].isin(locs_with_one)]

    def unique_non_null(s):
        return s.dropna().unique()

    okay_locs_grouped = sp_okay_locs.groupby("location_id").agg({"purpose_gt": unique_non_null})
    okay_locs_grouped["purpose_loc"] = okay_locs_grouped["purpose_gt"].str[0]
    # 2) deal with the one that have more than one loc assigned
    def most_often(elem_list):
        # rm nans
        elem_list = [elem for elem in elem_list if not pd.isna(elem)]
        uni, counts = np.unique(elem_list, return_counts=True)
        return uni[np.argmax(counts)]

    locs_more_than_one = uni_locs[uni_locs["purpose_gt"] > 1].index
    sp_bad_locs = sp[sp["location_id"].isin(locs_more_than_one)]
    bad_locs_grouped = sp_bad_locs.groupby("location_id").agg({"purpose_gt": list})
    # replace with the most occuring value
    bad_locs_grouped["purpose_loc"] = bad_locs_grouped["purpose_gt"].apply(most_often)
    # 3) concat both
    new_loc_purposes = pd.concat((okay_locs_grouped[["purpose_loc"]], bad_locs_grouped[["purpose_loc"]]))
    # uni_locs[["location_purpose"]].droplevel(1, axis=1)
    # merge with original table
    sp = sp.merge(new_loc_purposes, left_on="location_id", right_index=True, how="left")
    # fill nans
    #     sp.loc[pd.isna(sp["location_purpose"]), "location_purpose"] = sp.loc[pd.isna(sp["location_purpose"]), "purpose_gt"]
    return sp


# set only the gt purpose of the most active home / work location

# change label: use the location as home location which was labelled most often as home bzw work
def reduce_home_work_to_one(sp):
    get_home = lambda x: sum(x == "home")
    get_work = lambda x: sum(x == "work")
    homework_by_loc = (
        sp.groupby(["user_id", "location_id"])
        .agg({"purpose_loc": [get_home, get_work]})
        .rename(columns={"<lambda_0>": "home", "<lambda_1>": "work"})["purpose_loc"]
    )
    loc_home = homework_by_loc.groupby("user_id").apply(
        lambda x: x["home"].idxmax()[1]
    )  # ["location_id"] # .reset_index().groupby("user_id").agg()
    loc_home = pd.DataFrame(loc_home).rename(columns={0: "location_id"}).reset_index()
    loc_home["purpose_clean"] = "home"
    print(len(loc_home))
    loc_work = homework_by_loc.groupby("user_id").apply(
        lambda x: x["work"].idxmax()[1]
    )  # ["location_id"] # .reset_index().groupby("user_id").agg()
    loc_work = pd.DataFrame(loc_work).rename(columns={0: "location_id"}).reset_index()
    loc_work["purpose_clean"] = "work"
    print(len(loc_work))
    out = pd.concat([loc_home, loc_work])
    sp = sp.merge(out, left_on=["user_id", "location_id"], right_on=["user_id", "location_id"], how="left")
    return sp

