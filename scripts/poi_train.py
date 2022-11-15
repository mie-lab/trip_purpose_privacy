import pickle
import json
import pandas as pd
import numpy as np
import os
import geopandas as gpd
import xgboost
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch

warnings.filterwarnings("ignore")

from foursquare_privacy.models.xgb import XGBWrapper
from spacegraph_codebase.test import embed_points


class SimplePoint:
    def __init__(self, coord, label, label_finegrained=None):
        self.coord = coord
        self.label = label
        self.label_finegrained = label_finegrained


def dataset_from_neighborgraph(
    neighborgraphs, pointset, nr_types, count_mode="add_and_first", use_embedding=False, model_dir=None
):
    if use_embedding:
        assert model_dir is not None, "If embedding should be used, specify an embedding model"
    # get count vector of surrounding poi types
    poi_set_dict = {p_id: SimplePoint(p_coords, label1, label2) for p_id, p_coords, (label1, label2), _ in pointset}
    nr_uni_labels = len(np.unique([p.label for p in poi_set_dict.values()]))

    if use_embedding:
        embedded_coordinates = embed_points(model_dir, neighborgraphs, pointset, nr_types).detach().numpy()

    x_data, y_data = [], []
    for ind, (p_id, p_surrounding, _, _, dists_surrounding) in enumerate(neighborgraphs):
        # get y
        label = poi_set_dict[p_id].label
        y_data.append(label)

        # get x
        coords = poi_set_dict[p_id].coord
        if "add" in count_mode:
            one_hot = np.zeros(nr_uni_labels)
            for p in p_surrounding:
                one_hot[poi_set_dict[p].label] += 1
        elif count_mode == "stack":
            nr_surrounding = len(p_surrounding)
            one_hot = np.zeros(nr_uni_labels * nr_surrounding)
            for i, p in enumerate(p_surrounding):
                one_hot[nr_surrounding * i + poi_set_dict[p].label] = 1
        else:
            raise NotImplementedError("count_mode must be add_and_first, add or stack")
        if count_mode == "add_and_first":
            one_hot2 = np.zeros(nr_uni_labels)
            one_hot2[poi_set_dict[p_surrounding[0]].label] += 1
            one_hot = np.hstack((one_hot, one_hot2))
        # if count_mode == "add_and_distance":
        # TODO
        if use_embedding:
            x_data.append(np.hstack((one_hot, embedded_coordinates[ind])))
        else:
            x_data.append(np.hstack((coords, one_hot)))
    return np.array(x_data), np.array(y_data)


def concat_all_data(data_dict):
    all_x, all_y = data_dict["training"]
    for key in ["validation", "test"]:
        x_data, y_data = data_dict[key]
        all_x = np.vstack((all_x, x_data))
        all_y = np.concatenate((all_y, y_data))
    print("all data shape", all_x.shape, all_y.shape)
    return all_x, all_y


if __name__ == "__main__":
    MODEL_DIR = "../space2vec/spacegraph/model_dir/join_mydata_32/"
    DATA_DIR = "data/space2vec_foursquare_newyorkcity"
    SAVE_PATH = "poi_pred_test"
    xgb_params = {"max_depth": 15}

    with open(os.path.join(DATA_DIR, "poi_type.json"), "r") as infile:
        poi_type = json.load(infile)

    with open(os.path.join(DATA_DIR, "pointset.pkl"), "rb") as infile:
        nr_types, pointset = pickle.load(infile, encoding="latin-1")

    # get train, test and val data
    data_dict = {}
    for mode in ["training", "validation", "test"]:
        with open(os.path.join(DATA_DIR, f"neighborgraphs_{mode}.pkl"), "rb") as infile:
            neighborgraphs = pickle.load(infile, encoding="latin-1")
        x_data, y_data = dataset_from_neighborgraph(
            neighborgraphs, pointset, nr_types, count_mode="add", use_embedding=False, model_dir=MODEL_DIR
        )
        data_dict[mode] = (x_data, y_data)

    x_train, y_train = data_dict["training"]
    # init model
    model = XGBWrapper(xgb_params)
    model.fit(x_train, y_train)
    # validate
    x_val, y_val = data_dict["validation"]
    pred_val_proba = model.predict(x_val)
    pred_val = np.argmax(pred_val_proba, axis=1)
    print("Accuracy on val data", accuracy_score(pred_val, y_val))

    # print("Fitting on all data...")
    all_x, all_y = concat_all_data(data_dict)
    model = XGBWrapper(xgb_params)
    model.fit(all_x, all_y)

    # save model
    model.save(os.path.join(SAVE_PATH))
