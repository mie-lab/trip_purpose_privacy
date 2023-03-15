import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix


def poi_dist_plot(label_list, out_path=None, size=(8, 6), title="Foursquare POI distribution"):
    uni, counts = np.unique(label_list, return_counts=True)
    plt.figure(figsize=size)
    plt.bar(uni, counts)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()


def plot_confusion_matrix(labels, pred, col_names=None, normalize="sensitivity", out_path=None):
    """
    """
    if col_names is None:
        col_names = np.unique(labels)

    # make confusion matrix
    data_confusion = confusion_matrix(labels, pred)
    if normalize == "sensitivity":
        data_confusion = data_confusion / np.expand_dims(np.sum(data_confusion, axis=1), 1)
    else:
        data_confusion = data_confusion / np.sum(data_confusion, axis=0)

    # TODO: possibility for several matrices to have std
    data_confusion = np.array([data_confusion])

    def data_to_label(data, text):
        out_shape = data.shape
        if np.all(text == 0):
            return (np.asarray(["{0:.2f}".format(data) for data in data.flatten()])).reshape(*out_shape)
        return (
            np.asarray(
                [
                    "{0:.2f}\n".format(data) + "\u00B1" + "{0:.2f}".format(text)
                    for data, text in zip(data.flatten(), text.flatten())
                ]
            )
        ).reshape(*out_shape)

    sens_stds = np.std(data_confusion, axis=0)
    data_confusion = np.mean(data_confusion, axis=0)

    labels = data_to_label(data_confusion, sens_stds)

    # ACTUAL PLOT
    plt.figure(figsize=(20, 10))
    df_cm = pd.DataFrame(data_confusion, index=[i for i in col_names], columns=[i for i in col_names])

    sn.set(font_scale=1.8)

    short_col_names = [col[0].upper() for col in col_names]

    plt.xticks(np.arange(len(col_names)) + 0.5, col_names, fontsize="18", va="center")
    plt.yticks(np.arange(len(col_names)) + 0.5, short_col_names, rotation=0, fontsize="18", va="center")
    # sn.heatmap(df_cm, annot=True, fmt="g", cmap="YlGnBu")
    sn.heatmap(df_cm, annot=labels, fmt="", cmap="YlGnBu")
    # ax.xaxis.tick_bottom()
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True,
        rotation=90,
    )
    plt.xlabel("$\\bf{Predictions}$", fontsize=20)
    plt.ylabel("$\\bf{Ground\ truth}$", fontsize=20)
    plt.tight_layout()
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)


def main_plot(result_df, out_path=None):
    result_df.sort_values(["obfuscation", "method"], inplace=True)
    data_feats = result_df[result_df["method"] == "all features"]
    data_nearest = result_df[result_df["method"] == "spatial join"]
    user_acc = result_df.set_index("method").loc["temporal"]["Accuracy"]
    rand_acc = result_df.set_index("method").loc["random"]["Accuracy"]

    plt.figure(figsize=(10, 6))
    plt.plot(data_feats["obfuscation"], data_feats["Accuracy"], label="All features")
    if "spatial features" in result_df["method"].unique():
        data_only_spatial = result_df[result_df["method"] == "spatial features"]
        plt.plot(data_only_spatial["obfuscation"], data_only_spatial["Accuracy"], label="Spatial features")
    plt.plot(data_nearest["obfuscation"], data_nearest["Accuracy"], label="Spatial join")
    plt.plot(
        data_nearest["obfuscation"],
        [user_acc for _ in range(len(data_nearest))],
        label="Only temporal features",
        linestyle="--",
    )
    plt.plot(
        data_nearest["obfuscation"],
        [rand_acc for _ in range(len(data_nearest))],
        label="Random",
        linestyle="--",
        c="grey",
    )
    plt.xlabel("Obfuscation radius (in meters)")
    plt.ylabel("Accuracy")
    # plt.xticks(np.arange(len(data_nearest)), data_nearest["obfuscation"])
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(os.path.join(out_path, "main_result_plot.png"))
    else:
        plt.show()


def user_mae_plot(result_df, out_path=None, metric="User-wise MAE"):
    metric_probs = metric + " probs"
    user_results = result_df.dropna(subset=[metric_probs])
    user_results = user_results.sort_values(["obfuscation", "method"])

    timefeats_mae = {}
    timefeats_mae[metric] = user_results.set_index("method").loc["temporal"][metric]
    timefeats_mae[metric_probs] = user_results.set_index("method").loc["temporal"][metric_probs]

    random_mae = {}
    random_mae[metric] = user_results.set_index("method").loc["random"][metric]
    random_mae[metric_probs] = user_results.set_index("method").loc["random"][metric_probs]

    if "spatial features" in user_results["method"].unique():
        user_results = user_results[user_results["method"] == "spatial features"]
        feature_label = "Spatial features"
    else:
        user_results = user_results[user_results["method"] == "all features"]
        feature_label = "All features"

    x_obfuscation = user_results["obfuscation"].unique()

    plt.figure(figsize=(10, 6))
    plot_lines = []
    styles = ["-", "--", "-."]
    cols = ["blue", "green", "grey"]
    feat_labels = [feature_label, "Temporal features", "Random"]
    mode_labels = ["Hard labels", "Soft labels"]
    for i, feat_type in enumerate([user_results, timefeats_mae, random_mae]):
        plot_lines_inner = []
        for j, mode in enumerate([metric, metric_probs]):
            data_to_plot = feat_type[mode]
            if isinstance(data_to_plot, float):
                (l1,) = plt.plot(
                    x_obfuscation,
                    [data_to_plot for _ in range(len(user_results))],
                    label=f"{mode_labels[j]} ({feat_labels[i]})",
                    c=cols[i],
                    linestyle=styles[j],
                )
            else:
                (l1,) = plt.plot(
                    x_obfuscation,
                    data_to_plot,
                    label=f"{mode_labels[j]} ({feat_labels[i]})",
                    c=cols[i],
                    linestyle=styles[j],
                )
            plot_lines_inner.append(l1)
        plot_lines.append(plot_lines_inner)
    plt.xlabel("Obfuscation radius (in meters)")
    label_mapping = {
        "User-wise MAE": "User-wise distribution MAE",
        "Profile identification": "User identification (top-5 accuracy)",
    }
    plt.ylabel(label_mapping[metric])
    if metric == "Profile identification":
        legend1 = plt.legend(plot_lines[0], mode_labels, loc="upper right")
        plt.legend([l[0] for l in plot_lines], feat_labels, loc="center right")
    else:
        legend1 = plt.legend(plot_lines[0], mode_labels, loc="upper left")
        plt.legend([l[0] for l in plot_lines], feat_labels, loc="lower right")
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(os.path.join(out_path, f"user_performance_({metric}).png"))
    else:
        plt.show()


def plot_configurations(
    all_results,
    compare_col="split",
    compare_second_level="poi_data",
    eval_col="Accuracy",
    agg_method="mean",
    out_path="figures",
):
    cols_avail = ["method", "model", "poi_data", "city", "split"]
    filter_columns = {"split": "spatial", "model": "xgb"}  # "poi_data": "foursquare",
    assert compare_col not in filter_columns.keys() and compare_second_level not in filter_columns.keys()
    print(
        "Aggregating over",
        [
            col
            for col in cols_avail
            if col not in filter_columns.keys() and col != compare_col and col != compare_second_level
        ],
    )

    # filter by all settings in filter_columns
    results_filtered = all_results.copy()
    for key, val in filter_columns.items():
        results_filtered = results_filtered[results_filtered[key] == val]

    results_filtered["obfuscation"] = results_filtered["obfuscation"].fillna(-1)
    # group by the relevant columns
    results_grouped = (
        results_filtered.groupby(["obfuscation", compare_col, compare_second_level])
        .agg({eval_col: agg_method})
        .reset_index()
    )

    # print(results_grouped)
    plot_lines = []
    styles = ["-", "--", "-."]
    cols = ["blue", "green", "red"]
    plt.figure(figsize=(8, 7))
    for i, uni_val in enumerate(results_filtered[compare_col].unique()):
        plot_lines_innter = []
        for j, uni_val_2 in enumerate(results_filtered[compare_second_level].unique()):
            cond1 = results_grouped[compare_col] == uni_val
            cond2 = results_grouped[compare_second_level] == uni_val_2
            results_sub = results_grouped[cond1 & cond2]
            if len(results_sub) == 1:
                obs = results_grouped["obfuscation"].unique()
                one_val = results_sub.iloc[0][eval_col]
                (l1,) = plt.plot(
                    obs,
                    [one_val for _ in range(len(obs))],
                    c=cols[i],
                    linestyle=styles[j],
                    label=f"{uni_val}_{uni_val_2}",
                )
            else:
                (l1,) = plt.plot(
                    results_sub["obfuscation"],
                    results_sub[eval_col],
                    c=cols[i],
                    linestyle=styles[j],
                    label=f"{uni_val}_{uni_val_2}",
                )
            plot_lines_innter.append(l1)
        plot_lines.append(plot_lines_innter)

        #
    legend1 = plt.legend(
        plot_lines[0],
        results_filtered[compare_second_level].unique(),
        loc="upper left",
        title=compare_second_level.replace("_", " "),
    )
    plt.ylabel(eval_col)
    plt.xlabel("Obfuscation radius (in m)")
    plt.legend(
        [l[0] for l in plot_lines],
        results_filtered[compare_col].unique(),
        loc="lower right",
        title=compare_col.replace("_", " "),
    )
    plt.gca().add_artist(legend1)
    filtered_keys = "_".join(list(filter_columns.values()))
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(os.path.join(out_path, f"{compare_col}_{compare_second_level}_{eval_col}_({filtered_keys}).png"))
    else:
        plt.show()


if __name__ == "__main__":
    # Testing
    data_confusion = np.random.rand(2, 5, 5)
    plot_confusion_matrix(data_confusion, np.random.randint(0, 100, 5).astype(str))
