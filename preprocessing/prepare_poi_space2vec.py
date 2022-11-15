import argparse
import os
import json
import pickle
import numpy as np
from sklearn.metrics import pairwise_distances
from collections import defaultdict

from foursquare_privacy.add_poi import get_nearest
from foursquare_privacy.utils.io import read_poi_geojson

# sliding window approach:
def overlapping_spatial_split(coord_arr, buffer=500, kfold=9):
    ind_arr = np.arange(len(coord_arr))
    assert np.sqrt(kfold) == int(np.sqrt(kfold))
    kfold_sqrt = int(np.sqrt(kfold))
    # lower and upper bounds of each fold by quantiles
    bounds_lat = (
        [-np.inf] + [np.quantile(coord_arr[:, 0], (i + 1) / kfold_sqrt) for i in range(kfold_sqrt - 1)] + [np.inf]
    )
    bounds_lon = [-np.inf] + [np.quantile(coord_arr[:, 1], (i + 1) / kfold_sqrt) for i in range(kfold_sqrt)] + [np.inf]

    folds = []
    for i in range(kfold_sqrt):
        for j in range(kfold_sqrt):
            below_bounds = (coord_arr[:, 0] <= bounds_lat[i + 1] + buffer) & (
                coord_arr[:, 1] <= bounds_lon[j + 1] + buffer
            )
            above_bounds = (coord_arr[:, 0] > bounds_lat[i] - buffer) & (coord_arr[:, 1] > bounds_lon[j] - buffer)
            indices_spatial = ind_arr[below_bounds & above_bounds]
            folds.append(indices_spatial)
    return folds


def old_version_get_nearest():
    coord_arr = np.swapaxes(np.vstack([poi["geometry"].x.values, poi["geometry"].y.values]), 1, 0)
    id_arr = np.arange(len(coord_arr))
    folds = overlapping_spatial_split(coord_arr, 500)  # 500 meter buffer
    print("Fold length sum", sum([len(f) for f in folds]), "(nr pois before:", len(coord_arr))

    # compute closest distances
    dict_distances = defaultdict(list)
    dict_closest = defaultdict(list)
    for fold in folds:
        fold_arr = np.array(fold)
        # fold contains all the indices of coord_arr. Now I want to get the closest neighbors of each point
        dist_matrix = pairwise_distances(coord_arr[fold], coord_arr[fold])
        closest_neighbors = np.argsort(dist_matrix, axis=1)[:, 1 : nr_neighbors + 1]
        closest_distances = np.sort(dist_matrix, axis=1)[:, 1 : nr_neighbors + 1]
        corresponding_ids = id_arr[fold]
        for i, poi_id in enumerate(corresponding_ids):
            dict_distances[poi_id].extend(list(closest_distances[i].flatten()))
            dict_closest[poi_id].extend(list(fold_arr[closest_neighbors[i].flatten()]))
        # print("done with one fold")

    # Post-processing:
    counter = 0
    for poi_id in id_arr:
        # handle the border case
        if len(dict_distances[poi_id]) > nr_neighbors:
            new_closest_neighbor_inds = np.argsort(dict_distances[poi_id])
            new_closest_neighbors = np.array(dict_closest[poi_id])[new_closest_neighbor_inds]
            # Remove duplicate entries
            dict_closest[poi_id] = get_ordered_unique(new_closest_neighbors)[:nr_neighbors]
            counter += 1


def get_ordered_unique(arr):
    set_new, elems_new = set(), []
    for elem in arr:
        if elem not in set_new:
            set_new.add(elem)
            elems_new.append(elem)
    return elems_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="data", type=str)
    parser.add_argument("-c", "--city", default="newyorkcity", type=str)
    parser.add_argument("-p", "--poi_data", default="foursquare", type=str)
    parser.add_argument("-n", "--neighbors", default=10, type=int)
    parser.add_argument("-o", "--out_path", default="data", type=str)
    args = parser.parse_args()

    out_path = os.path.join(args.out_path, f"space2vec_{args.poi_data}_{args.city}")
    os.makedirs(out_path, exist_ok=True)
    nr_neighbors = args.neighbors

    # LOAD data
    poi = read_poi_geojson(os.path.join(args.data_path, f"pois_{args.city}_{args.poi_data}.geojson"))
    mapping_prev_ids = {i: int(old_id) for i, old_id in enumerate(poi["id"].values)}
    with open(os.path.join(out_path, "poi_id_mapping.json"), "w") as outfile:
        json.dump(mapping_prev_ids, outfile)
    print("Saved mapping from old IDs to new IDs")
    poi["id"] = np.arange(len(poi))
    poi.set_index("id", inplace=True)

    # PART 1: POI types
    # add the main categories:
    main_types = np.unique(poi["poi_my_label"])
    start_mapping = len(main_types)
    main_type_mapping = {elem: i for i, elem in enumerate(main_types)}
    sub_types = [t for t in poi["poi_type"].unique() if t not in main_types]
    poi_id_mapping = {elem: i + start_mapping for i, elem in enumerate(sub_types)}
    poi_id_mapping.update(main_type_mapping)
    # reversed
    id_poi_mapping = {str(i): elem for elem, i in poi_id_mapping.items()}

    # SAVE the poi types
    with open(os.path.join(out_path, "poi_type.json"), "w") as outfile:
        json.dump(id_poi_mapping, outfile)
    print("Saved POI types")

    # PART 2: POI list with categories
    # update table
    poi["poi_type_id"] = poi["poi_type"].map(poi_id_mapping)
    poi["poi_my_label_id"] = poi["poi_my_label"].map(poi_id_mapping)
    # train test splot
    rand_perm = np.random.permutation(len(poi))
    train_cutoff = int(len(poi) * 0.8)
    val_cutoff = int(len(poi) * 0.9)
    split_label_arr = np.array(["training" for _ in range(len(poi))]).astype(str)
    split_label_arr[rand_perm[train_cutoff:val_cutoff]] = "validation"
    split_label_arr[rand_perm[val_cutoff:]] = "test"
    poi["split"] = split_label_arr
    poi.loc[poi["split"] == "validati", "split"] = "validation"
    # convert table into tuple
    my_poi_data = []
    for elem_id, row in poi.iterrows():
        this_tuple = (
            elem_id,
            (row["geometry"].x, row["geometry"].y),
            (row["poi_my_label_id"], row["poi_type_id"]),
            row["split"],
        )
        my_poi_data.append(this_tuple)
    number_of_pois = len(id_poi_mapping)

    # Save the poi data with the categories
    with open(os.path.join(out_path, "pointset.pkl"), "wb") as outfile:
        pickle.dump((number_of_pois, my_poi_data), outfile)
    print("Saved POI-label data")

    # PART 3: sample the spatially closest
    coord_arr = np.swapaxes(np.vstack([poi["geometry"].x.values, poi["geometry"].y.values]), 1, 0)
    closest, distance_of_closest = get_nearest(coord_arr, coord_arr, k_neighbors=nr_neighbors)
    print("Finished positive sampling")

    # convert index
    poi_id_list = list(poi.index)
    poi_id_array = np.array(poi_id_list)
    poi_id_set = set(poi_id_list)

    # Negative sampling:
    all_tuples = []
    for counter, positive_sampled_index in enumerate(closest):
        elem_id = poi_id_list[counter]
        positive_sampled = poi_id_array[positive_sampled_index]
        leftover = list(poi_id_set - set([elem_id] + list(positive_sampled)))
        negative_sampled = list(np.random.choice(leftover, nr_neighbors))

        mode = poi.loc[elem_id, "split"]
        all_tuples.append((elem_id, tuple(positive_sampled), mode, negative_sampled, distance_of_closest[counter]))
    print("Finisher negative sampling")

    for mode in ["training", "validation", "test"]:
        out_tuple = [the_tuple for the_tuple in all_tuples if the_tuple[2] == mode]
        with open(os.path.join(out_path, f"neighborgraphs_{mode}.pkl"), "wb") as outfile:
            pickle.dump(out_tuple, outfile)
        print("Saved graph data", mode)
