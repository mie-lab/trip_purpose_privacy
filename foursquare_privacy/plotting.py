import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_label_distribution(labels, out_path=os.path.join("data", "label_distribution.png")):
    uni, counts = np.unique(labels, return_counts=True)
    plt.bar(uni, counts)
    plt.xticks(rotation=90)
    plt.savefig(out_path)
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
                    "{0:.2f}\n".format(data) + u"\u00B1" + "{0:.2f}".format(text)
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
        rotation=0,
    )
    plt.xlabel("$\\bf{Predictions}$", fontsize=20)
    plt.ylabel("$\\bf{Ground\ truth}$", fontsize=20)
    plt.tight_layout()
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)


def main_plot(result_df, out_path):
    result_df.sort_values(["obfuscation", "method"], inplace=True)
    data_feats = result_df[result_df["method"] == "all features"]
    data_nearest = result_df[result_df["method"] == "spatial join"]
    user_acc = result_df.set_index("method").loc["temporal"]["Accuracy"]

    plt.figure(figsize=(10, 6))
    plt.plot(data_feats["obfuscation"], data_feats["Accuracy"], label="All features")
    plt.plot(data_nearest["obfuscation"], data_nearest["Accuracy"], label="Spatial join")
    plt.plot(
        data_nearest["obfuscation"],
        [user_acc for _ in range(len(data_nearest))],
        label="Only temporal features",
        linestyle="--",
    )
    plt.plot(
        data_nearest["obfuscation"], [0.08 for _ in range(len(data_nearest))], label="Random", linestyle="--", c="grey"
    )
    plt.xlabel("Obfuscation (in meters)")
    plt.ylabel("Accuracy")
    # plt.xticks(np.arange(len(data_nearest)), data_nearest["obfuscation"])
    plt.legend()
    plt.savefig(os.path.join(out_path, "main_result_plot.png"))


def user_mae_plot(result_df, out_path):
    user_results = result_df.dropna(subset=["User-wise MAE probs"])
    user_results = user_results.sort_values(["obfuscation", "method"])
    timefeats_mae = user_results.set_index("method").loc["temporal"]["User-wise MAE"]
    timefeats_mae_probs = user_results.set_index("method").loc["temporal"]["User-wise MAE probs"]
    user_results = user_results[user_results["method"] != "temporal"]

    plt.figure(figsize=(10, 6))
    plt.plot(user_results["obfuscation"], user_results["User-wise MAE"], label="Hard labels", c="blue")
    plt.plot(user_results["obfuscation"], user_results["User-wise MAE probs"], label="Soft labels", c="red")
    plt.plot(
        user_results["obfuscation"],
        [timefeats_mae for _ in range(len(user_results))],
        label="Hard labels (temporal features)",
        linestyle="--",
        c="blue",
    )
    plt.plot(
        user_results["obfuscation"],
        [timefeats_mae_probs for _ in range(len(user_results))],
        label="Soft labels (temporal features)",
        linestyle="--",
        c="red",
    )
    plt.xlabel("Obfuscation (in meters)")
    plt.ylabel("User-wise distribution MAE")
    # plt.xticks(np.arange(len(data_nearest)), data_nearest["obfuscation"])
    plt.legend()
    plt.savefig(os.path.join(out_path, "user_mae_probs_by_obfuscation.png"))


if __name__ == "__main__":
    # Testing
    data_confusion = np.random.rand(2, 5, 5)
    plot_confusion_matrix(data_confusion, np.random.randint(0, 100, 5).astype(str))
