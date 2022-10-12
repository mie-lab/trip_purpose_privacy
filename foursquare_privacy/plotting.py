import os
import numpy as np
import matplotlib.pyplot as plt


def plot_label_distribution(labels, out_path=os.path.join("data", "label_distribution.png")):
    uni, counts = np.unique(labels, return_counts=True)
    plt.bar(uni, counts)
    plt.xticks(rotation=90)
    plt.savefig(out_path)
    plt.show()
