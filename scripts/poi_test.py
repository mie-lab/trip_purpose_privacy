import json
import os
import pandas as pd
import argparse
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from foursquare_privacy.utils.io import read_gdf_csv, read_poi_geojson
from foursquare_privacy.models.xgb import XGBWrapper
from foursquare_privacy.models.mlp import MLPWrapper
from foursquare_privacy.utils.user_distribution import get_user_dist_mae
from foursquare_privacy.utils.spatial_folds import spatial_split, venue_split
from foursquare_privacy.add_poi import POI_processor, get_nearest
from foursquare_privacy.location_masking import LocationMasker

from run import print_results
from train import dataset_from_neighborgraph

model_dict = {"xgb": {"model_class": XGBWrapper, "config": {}}, "mlp": {"model_class": MLPWrapper, "config": {}}}
# xgb config: tree_method="gpu_hist", gpu_id=0 if gpu available

results_dict = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="data", type=str)
    parser.add_argument("-c", "--city", default="newyorkcity", type=str)
    parser.add_argument("-o", "--out_dir", default=os.path.join("outputs", "test"), type=str)
    parser.add_argument("-p", "--poi_data", default="foursquare", type=str)
    parser.add_argument("-m", "--model", default="xgb", type=str)
    parser.add_argument("-f", "--fold_mode", default="spatial", type=str)
    parser.add_argument("-k", "--kfold", default=4, type=int)
    parser.add_argument("-n", "--neighbors", default=10, type=int)
    parser.add_argument("-l", "--load_model", default="poi_pred_test", type=str)

    args = parser.parse_args()

    city = args.city

    out_dir_base = args.out_dir
    os.makedirs(out_dir_base, exist_ok=True)
    out_name = f"{args.model}_{args.poi_data}_{args.city}_{args.fold_mode}"
    out_dir = os.path.join(out_dir_base, out_name)
    print(out_dir_base, out_dir)
    if os.path.exists(out_dir):
        print("Warning: Output directory already exists, may be overwriting files")
    os.makedirs(out_dir, exist_ok=True)

    # get model
    ModelClass = model_dict[args.model]["model_class"]
    model_config = model_dict[args.model]["config"]

    # load data
    data_raw = read_gdf_csv(os.path.join(args.data_path, f"checkin_{city}_features.csv"))
    checkin_id_arr = data_raw.index.astype(int) + 1000000
    uni_labels = np.unique(data_raw["label"])
    label_mapping = {elem: i for i, elem in enumerate(uni_labels)}
    data_raw["ground_truth"] = data_raw["label"].map(label_mapping)

    with open(os.path.join(args.data_path, f"space2vec_{args.poi_data}_{args.city}", "pointset.pkl"), "rb") as infile:
        nr_poi_types, poi_pointset = pickle.load(infile, encoding="latin-1")
    poi_coord_arr = np.array([p[1] for p in poi_pointset])
    poi_id_array = np.array([p[0] for p in poi_pointset])
    print("POI len, uni and max", len(poi_id_array), len(np.unique(poi_id_array)), np.max(poi_id_array))

    model = XGBWrapper({})
    model.load(args.load_model)
    print("Loaded model")

    # if args.poi_data == "both":
    #     pois_1 = read_poi_geojson(os.path.join(args.data_path, f"pois_{city}_foursquare.geojson"))
    #     pois_2 = read_poi_geojson(os.path.join(args.data_path, f"pois_{city}_osm.geojson"))
    #     pois = pd.concat((pois_2[["id", "poi_my_label", "poi_type", "geometry"]], pois_1))
    #     pois_1, pois_2 = None, None  # free space of variable
    # else:
    #     pois = read_poi_geojson(os.path.join(args.data_path, f"pois_{city}_{args.poi_data}.geojson"))
    # print(f"Using {len(pois)} pois")

    # Split data
    # np.random.seed(42)
    # if args.fold_mode == "spatial":
    #     folds = spatial_split(data_raw, args.kfold)
    # elif args.fold_mode == "venue":
    #     folds = venue_split(data_raw, args.kfold)
    # else:
    #     raise ValueError("fold_mode argument must be one of spatial, venue")
    # print("Fold lengths", [len(f) for f in folds])

    # 1) USER-FEATURES: check the performance with solely the user features
    # _, results_only_user = cross_validation(data_raw, folds)
    # print_results(results_only_user, "temporal_features")

    temporal_feats = [col for col in data_raw.columns if col.startswith("feat")]

    # obfuscate coordinates
    for masking in [0, 25, 50, 100, 200, 400, 800, 1200, 1600]:
        print(f"-------- Masking {masking} ---------")
        if masking == 0:
            data = data_raw.copy()
        else:
            masker = LocationMasker(data_raw)
            data = masker(masking)
            # # double check the latitude difference
            # print(
            #     f"Average latitude difference (masking {masking})",
            #     (data_raw["latitude"] - data["latitude"]).abs().mean(),
            # )

        data_coord_arr = np.swapaxes(np.vstack([data.geometry.x.values, data.geometry.y.values]), 1, 0)
        print("Coordinate array shape", data_coord_arr.shape)

        # spatial join: neares k points
        closest_pois, distance_of_closest = get_nearest(
            data_coord_arr, poi_coord_arr, k_neighbors=args.neighbors, remove_first=False
        )

        # build neighbor list for data
        data_neighbor_list = []
        for counter, positive_sampled_index in enumerate(closest_pois):
            positive_sampled = poi_id_array[positive_sampled_index]
            neighbor_tuple = (checkin_id_arr[counter], positive_sampled, 0, 0, distance_of_closest[counter])
            data_neighbor_list.append(neighbor_tuple)

        # build pointset for data
        pointset_checkins = []
        for i, row in data.iterrows():
            transformed_id = int(i) + 1000000
            pointset_checkin_tuple = (
                transformed_id,
                (row["longitude"], row["latitude"]),
                (row["ground_truth"], 0),
                "test",
            )
            pointset_checkins.append(pointset_checkin_tuple)

        pointset_all = poi_pointset + pointset_checkins
        x_inference, y_inference = dataset_from_neighborgraph(
            data_neighbor_list, pointset_all, nr_poi_types, count_mode="add", use_embedding=False
        )
        pred_inference_proba = model.predict(x_inference)
        pred_inference = np.argmax(pred_inference_proba, axis=1)

        result_df = data[["user_id", "venue_id", "label", "ground_truth"]].copy()
        proba_columns = [f"proba_{u}" for u in uni_labels]
        result_df.loc[:, proba_columns] = pred_inference_proba
        result_df.loc[:, "prediction"] = pred_inference
        print_results(result_df, f"poi_trained_{masking}", out_dir)

        # 2) CLOSEST_POI - Use simply the nearest poi label
        # spatial_joined = data.sjoin_nearest(pois, how="left")  # , distance_col="distance")
        # spatial_joined["ground_truth"] = spatial_joined["label"]
        # spatial_joined["prediction"] = spatial_joined["poi_my_label"]
        # print_results(spatial_joined, f"spatial_join_{masking}")

        # # get poi features
        # poi_process = POI_processor(data, pois)
        # poi_process(buffer=buffering)
        # poi_features = poi_process.distance_count_features()
        # if args.lda:
        #     lda_features = poi_process.lda_features()
        #     assert len(poi_features) == len(lda_features)
        #     poi_features = poi_features.merge(lda_features, left_index=True, right_index=True)
        # del poi_process

        # # version 2: together with user features
        # dataset = data.merge(poi_features, left_on=["latitude", "longitude"], right_index=True, how="left")
        # print("Percentage of rows with at least one NaN", dataset.isna().any(axis=1).sum() / len(dataset))
        # dataset = dataset.fillna(0)
        # print("Merge user featuers and POI features", len(poi_features), len(data), len(dataset))
        # if any(pd.isna(dataset)):
        #     print("Attention: NaNs in data", sum(pd.isna(dataset)))

        # _, result_df = cross_validation(dataset, folds, models=[], save_feature_importance=True)
        # print_results(result_df, f"all_features_{masking}")

        # dataset_spatial = dataset.drop(temporal_feats, axis=1)
        # _, result_df = cross_validation(dataset_spatial, folds, models=[])
        # print_results(result_df, f"spatial_features_{masking}")

        # # CODE to train only on non-obfuscated and test on others
        # if masking == 0:
        #     models, result_df = cross_validation(dataset, folds, models)
        # else:
        #     _, result_df = cross_validation(dataset, folds, models)

    with open(os.path.join(out_dir, "results.json"), "w") as outfile:
        json.dump(results_dict, outfile)
