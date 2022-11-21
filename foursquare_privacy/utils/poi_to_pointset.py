import numpy as np


def get_poi_id_mapping(poi):
    main_types = np.unique(poi["poi_my_label"])
    start_mapping = len(main_types)
    main_type_mapping = {elem: i for i, elem in enumerate(main_types)}
    sub_types = [t for t in poi["poi_type"].unique() if t not in main_types]
    poi_id_mapping = {elem: i + start_mapping for i, elem in enumerate(sub_types)}
    poi_id_mapping.update(main_type_mapping)
    return poi_id_mapping

def table_to_pointset(poi):
    my_poi_data = []
    for elem_id, row in poi.iterrows():
        this_tuple = (
            elem_id,
            (row["geometry"].x, row["geometry"].y),
            (row["poi_my_label_id"], row["poi_type_id"]),
            row["split"],
        )
        my_poi_data.append(this_tuple)

    return my_poi_data