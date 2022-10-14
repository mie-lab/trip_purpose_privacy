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
    plt.figure(figsize=(20, 20))
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
    )
    plt.xlabel("$\\bf{Predictions}$", fontsize=20)
    plt.ylabel("$\\bf{Ground\ truth}$", fontsize=20)
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)


if __name__ == "__main__":
    # Testing
    data_confusion = np.random.rand(2, 5, 5)
    plot_confusion_matrix(data_confusion, np.random.randint(0, 100, 5).astype(str))
