import os
import geopandas as gpd
import pandas as pd

import gensim
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel

from foursquare_privacy.utils.io import read_gdf_csv, read_poi_geojson


class POI_processor:
    def __init__(self, data, poi_path=os.path.join("data", "pois_newyorkcity_labelled.geojson")):
        self.data = data

        self.poi = read_poi_geojson(poi_path)

        assert self.poi.crs == self.data.crs, f"must be in the same CRS, poi: {self.poi.crs}, data: {self.data.crs}"

    def __call__(self, buffer=200):
        self.buffer = buffer

        # we only need to add POIs by unique longitude or latitude --> group
        grouped_by_geom = self.data.groupby(["latitude", "longitude"]).agg({"geometry": "first"})

        # buffer the points
        grouped_by_geom.set_geometry("geometry", inplace=True)
        grouped_by_geom["buffered"] = grouped_by_geom.geometry.buffer(buffer)
        grouped_by_geom.set_geometry("buffered", inplace=True)
        grouped_by_geom.crs = self.data.crs

        # reduce poi data to point data TODO: should we do that
        pois_only_points = self.poi[self.poi.geometry.type == "Point"]

        # join buffered points with POIs and add geometry column of POIs
        # TODO: take care of the ones without any nearby POIs
        joined_in_buffer = grouped_by_geom.sjoin(pois_only_points)
        joined_in_buffer.crs = self.data.crs
        joined_in_buffer = joined_in_buffer.merge(
            pois_only_points[["geometry"]], how="left", left_on="index_right", right_index=True, suffixes=("", "_poi")
        )
        joined_in_buffer.set_geometry("geometry", inplace=True)
        joined_in_buffer.crs = self.data.crs
        # compute distance
        joined_in_buffer["distance"] = joined_in_buffer["geometry"].distance(joined_in_buffer["geometry_poi"])

        # save the result in the object
        self.geom_with_pois = joined_in_buffer.reset_index()

    def distance_count_features(self):
        """
        Compute features that indicate the distance of the closest POI and the POI count for each POI type

        Returns
        -------
        Dataframe with index (latitude, longitude) and columns feat_dist_<poi_type> and feat_count_<poi_type>
        """
        assert hasattr(self, "geom_with_pois"), "must first call the __call__ method to merge geoms with POIs"
        # get one hot vectors of POIs
        poi_columns_one_hot = pd.get_dummies(self.geom_with_pois["poi_my_label"], prefix="poi")
        poi_type_list = poi_columns_one_hot.columns
        geom_with_poi_feats = pd.concat((self.geom_with_pois, poi_columns_one_hot), axis=1)

        for poi_col in poi_type_list:
            # replace POI indicator by distance
            geom_with_poi_feats[poi_col] = geom_with_poi_feats[poi_col] * geom_with_poi_feats["distance"]
            # the ones where no POI is nearby are filled with the buffer distance
            geom_with_poi_feats.loc[geom_with_poi_feats[poi_col] == 0, poi_col] = self.buffer

        # aggregate the POIs at one location (group by the location and aggregate pois with distance and count)
        def agg_x(x):
            return sum(x < self.buffer)

        agg_dict = {poi_col: ["min", agg_x] for poi_col in poi_type_list}
        geom_with_poi_feats = geom_with_poi_feats.groupby(["latitude", "longitude"]).agg(agg_dict)

        # flatten index
        geom_with_poi_feats.columns = geom_with_poi_feats.columns.to_flat_index()

        # convert to features
        for poi_col in poi_type_list:
            # distance: now invert --> the closer the POI, the higher the feature value
            geom_with_poi_feats["feat_dist_" + poi_col] = 200 - geom_with_poi_feats[(poi_col, "min")]
            # counts: simply copy the column
            geom_with_poi_feats["feat_count_" + poi_col] = geom_with_poi_feats[(poi_col, "agg_x")]

        # return only the final feature columns
        feat_cols = [col for col in geom_with_poi_feats.columns if type(col) != tuple and col.startswith("feat")]
        return geom_with_poi_feats[feat_cols]

    def lda_features(self, categories=8):
        assert hasattr(self, "geom_with_pois"), "must first call the __call__ method to merge geoms with POIs"

        geom_with_pois_grouped = self.geom_with_pois.groupby(["latitude", "longitude"])
        texts = geom_with_pois_grouped["poi_type"].apply(list).to_list()

        dct = Dictionary(texts)
        corpus = [dct.doc2bow(line) for line in texts]

        lda = LdaModel(corpus, num_topics=categories)
        vector = lda[corpus]

        # the lda array
        dense_ldavector = gensim.matutils.corpus2dense(vector, num_terms=categories).T
        # the index arr
        index_arr = geom_with_pois_grouped.count().reset_index()[["latitude", "longitude"]].values

        lda_vec_df = pd.DataFrame(dense_ldavector, columns=[f"feat_lda_{num}" for num in range(categories)])
        lda_vec_df["latitude"] = index_arr[:, 0]
        lda_vec_df["longitude"] = index_arr[:, 1]
        return lda_vec_df.set_index(["latitude", "longitude"])
