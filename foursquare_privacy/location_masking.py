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
        # make random unit vectors and scale by masking factor
        translation = np.random.rand(len(self.data), 2) - 0.5
        translation = translation / np.expand_dims(np.linalg.norm(translation, axis=1), 1)

        self.data["lat_masked"] = self.data["latitude"].values + translation[:, 0]
        self.data["lon_masked"] = self.data["longitude"].values + translation[:, 1]

        # update geometry
        orig_crs = self.data.crs
        self.data.geometry = gpd.points_from_xy(x=self.data["lon_masked"], y=self.data["lat_masked"])
        self.data.crs = orig_crs
