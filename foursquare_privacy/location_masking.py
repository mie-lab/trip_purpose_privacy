from dataclasses import dataclass


class LocationMasker:
    def __init__(self, data) -> None:
        self.data = data
        assert all(self.data.geometry.type == "Point"), "Geometry must be Point data"

    def __call__(self, masking_factor=20):
        """
        Perturb the data randomly by masking_factor meter

        Parameters
        ----------
        masking_factor : int, optional
            Perturbation in meter, by default 20
        """
        pass
