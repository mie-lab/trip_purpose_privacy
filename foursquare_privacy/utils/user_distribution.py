import pandas as pd
import numpy as np


def get_user_dist_mae(results):
    gt_dummies = pd.concat((results[["user_id"]], pd.get_dummies(results["ground_truth"], prefix="gt")), axis=1)
    pred_dummies = pd.concat((results[["user_id"]], pd.get_dummies(results["prediction"], prefix="pred")), axis=1)
    agg_dict_gt = {col: "mean" for col in gt_dummies.columns if col.startswith("gt_")}
    agg_dict_pred = {col: "mean" for col in pred_dummies.columns if col.startswith("pred_")}

    # group by user
    gt_user = gt_dummies.groupby(["user_id"]).agg(agg_dict_gt)
    pred_user = pred_dummies.groupby("user_id").agg(agg_dict_pred)
    assert all(gt_user.index == pred_user.index)

    mae = np.mean(np.absolute(np.array(gt_user) - np.array(pred_user)), axis=1)
    return mae
