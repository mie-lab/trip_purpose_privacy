import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

plt.rcParams.update({"font.size": 20})
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from foursquare_privacy.plotting import plot_confusion_matrix, user_mae_plot, main_plot
from foursquare_privacy.utils.user_distribution import (
    get_user_dist_mae,
    user_identification_accuracy,
    privacy_loss,
    get_user_dist_euclidean,
)


def results_to_dataframe(result_dict):
    result_df = pd.DataFrame(result_dict).swapaxes(1, 0)
    result_df.index.name = "method"
    result_df.reset_index(inplace=True)
    result_df["obfuscation"] = result_df["method"].apply(
        lambda x: int(x.split("_")[-1]) if x not in ["temporal_features", "random_results"] else pd.NA
    )
    result_df["method"] = result_df["method"].apply(lambda x: " ".join(x.split("_")[:-1]))
    return result_df.sort_values(["method", "obfuscation"])


def load_results(base_path, top_k=5):
    files_for_eval = [f for f in os.listdir(base_path) if f.startswith("predictions")]
    # make random file at first
    if not any(["random" in f for f in files_for_eval]):
        any_results_file = pd.read_csv(os.path.join(base_path, files_for_eval[0]))
        rand_results = baseline_random(any_results_file)
        rand_results.to_csv(os.path.join(base_path, "predictions_random_results.csv"))

    files_for_eval = [f for f in os.listdir(base_path) if f.startswith("predictions")]
    result_dict = {}
    for pred in files_for_eval:
        print(base_path, pred)
        result_df = pd.read_csv(os.path.join(base_path, pred))
        name = pred[12:-4]

        # # Privacy loss tests
        # if "proba_Dining" not in result_df.columns:
        #     continue
        # result_dict[name] = {
        #     "privacyloss_rank_1": privacy_loss(result_df, p=1, mode="rank"),
        #     "privacyloss_rank_2": privacy_loss(result_df, p=2, mode="rank"),
        #     "privacyloss_rank_3": privacy_loss(result_df, p=3, mode="rank"),
        #     "privacyloss_dist_1": privacy_loss(result_df, p=1, mode="distance"),
        #     "privacyloss_dist_2": privacy_loss(result_df, p=2, mode="distance"),
        #     "privacyloss_dist_3": privacy_loss(result_df, p=3, mode="distance"),
        #     "privacyloss_softmax": privacy_loss(result_df, p=3, mode="softmax"),
        # }
        acc = accuracy_score(result_df["ground_truth"], result_df["prediction"])
        bal_acc = balanced_accuracy_score(result_df["ground_truth"], result_df["prediction"])
        user_mae = np.mean(get_user_dist_mae(result_df))
        user_identify = user_identification_accuracy(result_df, top_k)
        euclid = np.mean(get_user_dist_euclidean(result_df, False))

        if "proba_Dining" in result_df.columns:
            user_mae_probs = np.mean(get_user_dist_mae(result_df, True))
            user_identify_probs = user_identification_accuracy(result_df, top_k, True)
            priv_loss = privacy_loss(result_df, p=1, mode="softmax")
            euclid_probs = np.mean(get_user_dist_euclidean(result_df, True))
        else:
            user_mae_probs, user_identify_probs = pd.NA, pd.NA
            euclid_probs = euclid
            priv_loss = privacy_loss(result_df, p=1, mode="softmax", use_probabilities=False)
        loss_mean = np.mean(priv_loss)
        loss_median = np.median(priv_loss)
        result_dict[name] = {
            "Accuracy": acc,
            "Balanced accuracy": bal_acc,
            "User-wise MAE": user_mae,
            "User-wise MAE probs": user_mae_probs,
            "Profile identification": user_identify,
            "Profile identification probs": user_identify_probs,
            "Privacy loss (mean)": loss_mean,
            "Privacy loss (median)": loss_median,
            "User profiling error": euclid,
            "User profiling error probs": euclid_probs,
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


def plot_results_for_one(base_path, out_path):
    # INPUT: single configuration, all csv files for this configuration

    # combine results
    result_dict = load_results(base_path)
    result_df = results_to_dataframe(result_dict)
    result_df.to_csv(os.path.join(out_path, "results_table.csv"), index=False)

    # main plot
    main_plot(result_df, out_path)

    # user identification plot
    user_mae_plot(result_df, out_path, metric="Profile identification")

    # user MAE plot
    user_mae_plot(result_df, out_path, metric="User-wise MAE")


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


def load_save_all_results(base_path="outputs/cluster_runs_all", out_path="outputs"):
    results = []
    info_columns = [
        "model",
        "poi_data",
        "city",
        "split",
        "embed",
        "lda",
        "inbuffer",
        "closestk",
        "xgbdepth",
        "kfold",
        "poi_keep_ratio",
    ]
    for subdir in os.listdir(base_path):
        if (
            # subdir != "xgb_foursquare_newyorkcity_user_True_False_True_True_6_10_1" # for testing
            subdir[0] == "."
            or (not os.path.isdir(os.path.join(base_path, subdir)))
            or len(os.listdir(os.path.join(base_path, subdir))) < 1
        ):
            continue
        infos = subdir.split("_")
        result_dict = load_results(os.path.join(base_path, subdir))
        result_df = results_to_dataframe(result_dict)
        for i, col in enumerate(info_columns):
            if i >= len(infos):
                result_df[col] = pd.NA
                # older files don't have so many entries
                break
            result_df[col] = infos[i]
        results.append(result_df)
        # save intermediate results
        all_results = pd.concat(results)
        all_results.to_csv(os.path.join(out_path, "pooled_results_new.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inp_path", type=str, required=True)
    parser.add_argument("-o", "--out_path", type=str, default="outputs")
    parser.add_argument("-m", "--mode", default="result_table", type=str)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    if args.mode == "result_table":
        # 1) Summarize multiple runs in different configurations in one table
        load_save_all_results(args.inp_path, args.out_path)
    elif args.mode == "main_plot":
        # 2) One run (but all files of that run)
        plot_results_for_one(args.inp_path, args.out_path)
    elif args.mode == "single_file":
        # 3) Single files
        # density analysis for one
        poi_density_analysis(args.inp_path, out_path=args.out_path)
        # Confusion matrix for one
        results_one = pd.read_csv(args.inp_path)
        plot_confusion_matrix(
            results_one["ground_truth"],
            results_one["prediction"],
            col_names=np.unique(results_one["label"]),
            out_path=os.path.join(args.out_path, "confusion_matrix.png"),
        )
