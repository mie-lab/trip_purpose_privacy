import numpy as np


def spatial_split(data, kfold=9):
    assert np.sqrt(kfold) == int(np.sqrt(kfold))
    kfold_sqrt = int(np.sqrt(kfold))
    # lower and upper bounds of each fold by quantiles
    bounds_lat = (
        [-np.inf] + [np.quantile(data["latitude"], (i + 1) / kfold_sqrt) for i in range(kfold_sqrt - 1)] + [np.inf]
    )
    bounds_lon = (
        [-np.inf] + [np.quantile(data["longitude"], (i + 1) / kfold_sqrt) for i in range(kfold_sqrt)] + [np.inf]
    )

    folds = []
    for i in range(kfold_sqrt):
        for j in range(kfold_sqrt):
            below_bounds = (data["latitude"] <= bounds_lat[i + 1]) & (data["longitude"] <= bounds_lon[j + 1])
            above_bounds = (data["latitude"] > bounds_lat[i]) & (data["longitude"] > bounds_lon[j])
            indices_spatial = data[below_bounds & above_bounds].index
            # print(i, j, len(indices_spatial))
            folds.append(indices_spatial)
    return folds