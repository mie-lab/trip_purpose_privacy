import pandas as pd
import numpy as np


def get_user_dist_mae(results):
    gt_dummies = pd.concat((results[["user_id"]], pd.get_dummies(results["ground_truth"], prefix="gt")), axis=1)
    pred_dummies = pd.concat((results[["user_id"]], pd.get_dummies(results["prediction"], prefix="pred")), axis=1)

    # Fix: some columns might be missing because not all classes are predicted. Insert those with zeros
    pred_columns = [f"pred_{col.split('_')[1]}" for col in gt_dummies.columns if col != "user_id"]
    if len(pred_dummies.columns) < len(gt_dummies.columns):
        missing_cols = [col for col in pred_columns if col not in pred_dummies.columns]
        for col_name in missing_cols:
            pred_dummies[col_name] = 0
    # reorder
    pred_dummies = pred_dummies[["user_id"] + pred_columns]

    agg_dict_gt = {col: "mean" for col in gt_dummies.columns if col.startswith("gt_")}
    agg_dict_pred = {col: "mean" for col in pred_columns}

    # group by user
    gt_user = gt_dummies.groupby(["user_id"]).agg(agg_dict_gt)
    pred_user = pred_dummies.groupby("user_id").agg(agg_dict_pred)

    assert all(gt_user.index == pred_user.index)

    mae = np.mean(np.absolute(np.array(gt_user) - np.array(pred_user)), axis=1)
    return mae

