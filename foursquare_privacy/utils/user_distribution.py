import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances
from scipy.stats import rankdata


def get_dist_per_user(results, use_probabilities=False):
    # Either using the hard labels or there must be probabilities in the results dataframe
    assert use_probabilities == False or any([col.startswith("proba") for col in results.columns])

    if use_probabilities:
        gt_dummies = pd.concat((results[["user_id"]], pd.get_dummies(results["label"], prefix="gt")), axis=1)
        pred_columns = [col for col in results if col.startswith("proba")]
        pred_dummies = results[["user_id"] + pred_columns]
    else:
        gt_dummies = pd.concat((results[["user_id"]], pd.get_dummies(results["ground_truth"], prefix="gt")), axis=1)
        pred_dummies = pd.concat((results[["user_id"]], pd.get_dummies(results["prediction"], prefix="pred")), axis=1)
        # Fix: some columns might be missing because not all classes are predicted. Insert those with zeros
        pred_columns = [f"pred_{col[3:]}" for col in gt_dummies.columns if col != "user_id"]
        pred_columns_existing = [col for col in pred_columns if col in pred_dummies.columns]
        pred_dummies = pred_dummies[["user_id"] + pred_columns_existing]

        if len(pred_dummies.columns) < len(gt_dummies.columns):
            missing_cols = [col for col in pred_columns if col not in pred_dummies.columns]
            for col_name in missing_cols:
                pred_dummies[col_name] = 0

    agg_dict_gt = {col: "mean" for col in gt_dummies.columns if col.startswith("gt_")}
    agg_dict_pred = {col: "mean" for col in pred_columns}

    # group by user
    gt_user = gt_dummies.groupby(["user_id"]).agg(agg_dict_gt)
    pred_user = pred_dummies.groupby("user_id").agg(agg_dict_pred)
    return gt_user, pred_user


def get_user_dist_mae(results, use_probabilities=False):
    gt_user, pred_user = get_dist_per_user(results, use_probabilities)

    assert all(gt_user.index == pred_user.index)

    mae = np.mean(np.absolute(np.array(gt_user) - np.array(pred_user)), axis=1)
    return mae


def user_identification_accuracy(results, top_k=5, use_probabilities=False):
    gt_user, pred_user = get_dist_per_user(results, use_probabilities)

    # use np array
    gt_user = np.array(gt_user)
    pred_user = np.array(pred_user)

    # compute distances between users
    distance_matrix = pairwise_distances(gt_user, pred_user)
    argsort_matrix = np.argsort(distance_matrix, axis=1)

    bool_matrix_gt_in_top_k = argsort_matrix[:, :top_k] == np.expand_dims(np.arange(len(argsort_matrix)), 1)
    top_k_acc = np.sum(np.any(bool_matrix_gt_in_top_k, axis=1)) / len(distance_matrix)
    # print(f"Top {top_k} accuracy: {round(top_k_acc, 4)} (Random: {round(top_k / len(distance_matrix), 3)})")
    return top_k_acc


def privacy_loss(results, p=2, mode="rank", use_probabilities=True):
    gt_user, pred_user = get_dist_per_user(results, use_probabilities)
    gt_user = np.array(gt_user)
    pred_user = np.array(pred_user)
    # compute distances between users
    distance_matrix = pairwise_distances(gt_user, pred_user)
    if mode == "distance":
        # OPTION 1: based on actual distance
        if np.any(distance_matrix == 0):
            distance_matrix = distance_matrix + 1e-7
        prob_matrix = (1 / distance_matrix) ** p
    elif mode == "rank":
        # # OPTION 2: based on rank
        argsort_matrix = rankdata(distance_matrix, axis=1)
        prob_matrix = (1 / argsort_matrix) ** p
    elif mode == "softmax":
        prob_matrix = np.exp(1 / distance_matrix) # Softmax --> not really better
    else:
        raise RuntimeError("Wrong mode")
    # prob_matrix[prob_matrix < np.mean(prob_matrix)] = 0 # doesn't help
    # # OPTION 3: just use softmax
    # normalize
    prob_matrix = prob_matrix / np.expand_dims(np.sum(prob_matrix, axis=1), 1)
    matched = prob_matrix.diagonal()

    return matched * len(matched) # matched has the prob for the informed, * len(matched) = / (1 / len(matched))
