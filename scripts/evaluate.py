import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"font.size": 20})
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from foursquare_privacy.plotting import plot_confusion_matrix, user_mae_plot, main_plot
from foursquare_privacy.utils.user_distribution import get_user_dist_mae, user_identification_accuracy


def results_to_dataframe(result_dict):
    result_df = pd.DataFrame(result_dict).swapaxes(1, 0)
    result_df.index.name = "method"
    result_df.reset_index(inplace=True)
    result_df["obfuscation"] = result_df["method"].apply(
        lambda x: int(x.split("_")[-1]) if x not in ["temporal_features", "random_results"] else pd.NA
    )
    result_df["method"] = result_df["method"].apply(lambda x: " ".join(x.split("_")[:-1]))
    return result_df.sort_values(["obfuscation", "method"])


def load_results(base_path):
    files_for_eval = [f for f in os.listdir(base_path) if f.startswith("predictions")]
    # make random file at first
    any_results_file = pd.read_csv(os.path.join(base_path, files_for_eval[0]))
    rand_results = baseline_random(any_results_file)
    rand_results.to_csv(os.path.join(base_path, "predictions_random_results.csv"))

    files_for_eval = [f for f in os.listdir(base_path) if f.startswith("predictions")]
    result_dict = {}
    for pred in files_for_eval:
        result_df = pd.read_csv(os.path.join(base_path, pred))
        name = pred[12:-4]
        acc = accuracy_score(result_df["ground_truth"], result_df["prediction"])
        bal_acc = balanced_accuracy_score(result_df["ground_truth"], result_df["prediction"])
        user_mae = np.mean(get_user_dist_mae(result_df))
        try:
            user_mae_probs = np.mean(get_user_dist_mae(result_df, True))
            user_identify = user_identification_accuracy(result_df)
        except AssertionError:
            user_mae_probs, user_identify = pd.NA, pd.NA
        result_dict[name] = {
            "Accuracy": acc,
            "Balanced accuracy": bal_acc,
            "User-wise MAE": user_mae,
            "User-wise MAE probs": user_mae_probs,
            "User profile identification": user_identify,
        }
    return result_dict


def baseline_random(results):
    """Make random results based on distribution of ground truth labels"""
    # add random as baseline
    uni, count = np.unique(results["label"], return_counts=True)
    count = count / np.sum(count)
    results.loc[:, ["proba_" + u for u in uni]] = count
    results["prediction"] = np.random.choice(np.unique(results["ground_truth"]), p=count, size=len(results))
    return results


def plot_results_for_one(base_path="outputs/xgb_foursquare_newyorkcity_spatial_1"):
    # INPUT: single configuration, all csv files for this configuration
    out_path = base_path
    # combine results
    result_dict = load_results(base_path)
    result_df = results_to_dataframe(result_dict)
    result_df.to_csv(os.path.join(out_path, "results_table.csv"), index=False)

    # main plot
    main_plot(result_df, out_path)

    # user MAE plot
    user_mae_plot(result_df, out_path)


def poi_density_analysis(result_csv_path, data_path="data", out_path="figures"):
    # Input: SINGLE CSV
    results = pd.read_csv(result_csv_path)
    city = "newyorkcity" if "newyorkcity" in result_csv_path else "tokyo"
    poi_density = pd.read_csv(os.path.join(data_path, f"poi_density_{city}.csv"))

    results = results.merge(poi_density, left_on="venue_id", right_on="venue_id", how="left")
    results["Sample"] = (
        (results["prediction"] == results["ground_truth"]).astype(int).map({0: "erronous", 1: "correct"})
    )
    sns.kdeplot(data=results, x="poi_density", hue="Sample", common_norm=False)
    plt.xlabel("POI density (in 500m surrounding)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "kde_poi_density.png"))


def load_save_all_results(base_path="outputs/cluster_runs_all"):
    results = []
    info_columns = ["model", "poi_data", "city", "split"]
    for subdir in os.listdir(base_path):
        if subdir[0] == ".":
            continue
        infos = subdir.split("_")
        result_dict = load_results(os.path.join(base_path, subdir))
        result_df = results_to_dataframe(result_dict)
        for i, col in enumerate(info_columns):
            result_df[col] = infos[i]
        results.append(result_df)
    all_results = pd.concat(results)
    all_results.to_csv("outputs/pooled_results.csv")


if __name__ == "__main__":
    # # 1) Multiple runs
    load_save_all_results()

    # # 2) One run (but all files of that run)
    # plot_results_for_one(base_path="outputs/cluster_runs_all/xgb_osm_newyorkcity_spatial_1")

    # # 3) Single files
    # poi_density_analysis(
    #     "outputs/cluster_runs_all/xgb_foursquare_newyorkcity_spatial_1/predictions_all_features_100.csv"
    # )

    # # Confusion matrix for one
    # base_path = "outputs/cluster_runs_all/xgb_osm_newyorkcity_spatial_1"
    # results_one = pd.read_csv(os.path.join(base_path, "predictions_all_features_0.csv"))
    # plot_confusion_matrix(
    #     results_one["ground_truth"],
    #     results_one["prediction"],
    #     col_names=np.unique(results_one["label"]),
    #     out_path=os.path.join(base_path, "confusion_matrix.png"),
    # )
