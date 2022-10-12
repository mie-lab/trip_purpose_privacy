import pandas as pd
import numpy as np

from foursquare_privacy.utils.clean_home_work import *


if __name__ == "__main__":

    DBLOGIN_FILE = "../dblogin_mielab.json"

    with open(DBLOGIN_FILE) as json_file:
        LOGIN_DATA = json.load(json_file)

    engine = create_engine(
        "postgresql://{user}:{password}@{host}:{port}/case_study_cache".format(**LOGIN_DATA)
    )  # load green class data
    sql = "SELECT study, id, user_id, purpose, started_at, finished_at, location_id FROM staypoints WHERE\
    study!='Yumuv' and study != 'Geolife'"
    sp = pd.read_sql(sql, engine)
    print("Data read successfully", len(sp))

    sp["purpose_gt"] = sp["purpose"].copy()

    # add geom and datetimes
    sp = add_fake_geom(sp)
    sp["started_at"] = pd.to_datetime(sp["started_at"], utc=True)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"], utc=True)

    print(len(sp), np.unique(sp["study"], return_counts=True), np.unique(sp["purpose_gt"], return_counts=True))

    # save the original purpose column
    sp["purpose_original"] = sp["purpose_gt"].copy()

    # fix location IDs (ensure that they are unique per study):
    sp["orig_location_id"] = sp["location_id"].copy()
    current_max = 0
    for study_name in sp["study"].unique():
        sp.loc[sp["study"] == study_name, "location_id"] += current_max
        current_max = sp.loc[sp["study"] == study_name, "location_id"].max() + 1

    print("Setting errand purposes to NaN...")
    set_na_purposes(sp)
    print(np.unique(sp["purpose_gt"].dropna(), return_counts=True))

    # multiple labels per location id
    print("Fix problem 1: Some locations have multiple different labels")
    sp = replace_wrong_locs(sp)

    # home / work missing for some users
    print("Removing users which have no work or home location")
    sp["is_home"] = sp["purpose_loc"] == "home"
    sp["is_work"] = sp["purpose_loc"] == "work"
    check_missing = sp.groupby("user_id").agg({"is_home": "sum", "is_work": "sum"})
    users_with_missing = check_missing[(check_missing["is_home"] == 0) | (check_missing["is_work"] == 0)].index
    print("Ratio of users with missing home / work", len(users_with_missing) / sp["user_id"].nunique())
    print(len(sp))
    sp = sp[~sp["user_id"].isin(users_with_missing)]
    sp.drop(["is_home", "is_work"], axis=1, inplace=True)
    print(len(sp))

    # plot: how many per person
    # how many home locations per person
    test = sp[sp["purpose_loc"] == "home"]
    test = test.groupby(["user_id", "location_id"]).agg({"purpose_loc": "first", "location_id": "count"})
    nr_home_per_person = test.groupby("user_id").agg({"purpose_loc": "count"})
    print("Average number of home locations per person: ", nr_home_per_person.mean())
    plt.hist(nr_home_per_person, bins=100)
    plt.savefig(os.path.join("figures", "n_home_locs_per_user.png"))

    # multiple home / work locations per person
    print("Fix problem 2: multiple home and work locations per person")
    sp = reduce_home_work_to_one(sp)

    # TESTS
    # each location should just have one purpose
    test_whether_correct = sp.groupby("location_id").agg({"purpose_loc": "nunique"})
    assert not np.any(test_whether_correct["purpose_loc"] > 1)
    # each user should have only one home and one work location
    for purp in ["home", "work"]:
        test_test = sp[sp["purpose_clean"] == purp]
        test_test = test_test.groupby(["user_id", "location_id"]).agg(
            {"purpose_clean": "first", "location_id": "count"}
        )
        nr_home_per_person = test_test.groupby("user_id").agg({"purpose_clean": "count"})
        print(f"Average number {purp} per person: ", nr_home_per_person.mean(), nr_home_per_person.std())

    # EVALUATE ON ALL DATA
    # add purpose with freq method
    sp.drop("purpose", axis=1, inplace=True)
    sp = ti.analysis.location_identifier(sp, method="FREQ", pre_filter=False)
    sp["purpose_FREQ"] = sp["purpose"]

    # add purpose with osna method:
    sp.drop("purpose", axis=1, inplace=True)
    sp = ti.analysis.location_identifier(sp, method="OSNA", pre_filter=False)
    sp["purpose_OSNA"] = sp["purpose"]

    auswertung(sp, "purpose_clean")

    # plot by time period:
    sp = sp.sort_values(["study", "user_id", "started_at"])
    sp["sp_id"] = sp.groupby(["study", "user_id"])["started_at"].rank()

    print("Running location identification for varying thresholds...")
    main_dict = {}
    for cutoff in [2, 4, 6, 8] + list(np.arange(10, 160, 10)):
        # FREQ
        sp_period = sp[sp["sp_id"] < cutoff]
        #     sp_period.drop("purpose", axis=1, inplace=True, errors="ignore")
        sp_period["purpose"] = pd.NA
        sp_period = ti.analysis.location_identifier(sp_period, method="FREQ", pre_filter=False)
        sp_period["purpose_FREQ"] = sp_period["purpose"]

        # OSNA
        sp_period.drop("purpose", axis=1, inplace=True)
        sp_period = ti.analysis.location_identifier(sp_period, method="OSNA", pre_filter=False)
        sp_period["purpose_OSNA"] = sp_period["purpose"]

        # EVAL
        res_dict = auswertung(sp_period, "purpose_clean", printout=False)
        print("cutoff", cutoff, "Results:", res_dict)
        main_dict[cutoff] = res_dict

    main_dict = (
        pd.DataFrame(main_dict)
        .reset_index()
        .rename(columns={"level_0": "variable", "level_1": "method"})
        .drop("index", axis=1, errors="ignore")
    )
    main_dict = main_dict.groupby(["variable", "method"]).agg(
        {x: "first" for x in main_dict.columns if x not in ["variable", "method"]}
    )
    main_dict.to_csv(os.path.join("figures", "time_analysis_homework.csv"))

    # plot
    col = ["green", "blue"]
    plt.figure(figsize=(15, 6))
    for var, c in zip(["home", "work"], col):
        for meth, ls in zip(["FREQ", "OSNA"], ["-", "--"]):
            value_series = main_dict.loc[var].loc[meth]
            plt.plot(value_series.index, value_series.values, c=c, linestyle=ls, label=f"{var} - {meth}")
    plt.legend(loc="lower right", title="Predicted variable - Method")
    plt.xlabel("Number of staypoints (aka tracking duration)")
    plt.ylabel("Prediction accuracy")
    plt.savefig(os.path.join("figures", "freq_osna_tracking_period.png"))
