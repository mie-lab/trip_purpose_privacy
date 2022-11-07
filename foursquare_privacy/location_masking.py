import geopandas as gpd
import pandas as pd
import numpy as np


class LocationMasker:
    def __init__(self, data) -> None:
        self.data = data
        assert all(self.data.geometry.type == "Point"), "Geometry must be Point data"

    def __call__(self, masking_factor):
        """
        Perturb the data randomly by masking_factor meter

        Parameters
        ----------
        masking_factor : int, optional
            Perturbation in meter, by default 20
        """
        data_masked = self.data.copy()
        # make random unit vectors
        translation = np.random.rand(len(data_masked), 2) - 0.5
        # scale by random factor --> maximally the masking factor
        translation_size = np.expand_dims(np.random.uniform(low=0, high=masking_factor, size=len(translation)), 1)
        # translata by unit vector multiplied by factor
        translation = translation / np.expand_dims(np.linalg.norm(translation, axis=1), 1) * translation_size

        data_masked["latitude"] = data_masked["latitude"].values + translation[:, 0]
        data_masked["longitude"] = data_masked["longitude"].values + translation[:, 1]

        # update geometry
        orig_crs = data_masked.crs
        data_masked.geometry = gpd.points_from_xy(x=data_masked["longitude"], y=data_masked["latitude"])
        data_masked.crs = orig_crs
        return data_masked
