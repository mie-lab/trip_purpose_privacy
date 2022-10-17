import os
import pandas as pd
import numpy as np
from foursquare_privacy.mlp import train_model

if __name__ == "__main__":
    dataset = pd.read_csv("data/poionly_dataset.csv")

    features = dataset[[col for col in dataset.columns if col.startswith("feat")]]
    # TODO: these features are always 0
    features.drop(["feat_dist_poi_residential", "feat_count_poi_residential"], axis=1, inplace=True)
    labels = dataset["label"]
    labels_onehot = pd.get_dummies(labels, prefix="label_")

    # train test split
    train_len = int(len(features) * 0.9)
    train_x, test_x = features[:train_len], features[train_len:]
    train_y, test_y = labels_onehot[:train_len], labels_onehot[train_len:]

    # normalize
    mean_x, std_x = np.mean(train_x, axis=0), np.std(train_x, axis=0)
    train_normed_x = (train_x - mean_x) / std_x
    test_normed_x = (test_x - mean_x) / std_x

    config = {"batch_size": 8, "epochs": 50, "learning_rate": 1e-5}

    model = train_model(
        np.array(train_normed_x),
        np.array(train_y),
        np.array(test_normed_x),
        np.array(test_y),
        save_path=os.path.join("trained_models", "test2"),
        **config
    )
