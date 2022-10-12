import os
import pandas as pd
import geopandas as gpd
import json
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from foursquare_privacy.utils.purpose_categories import purpose_categories

DBLOGIN_FILE = "../dblogin_mielab.json"

with open(DBLOGIN_FILE) as json_file:
    LOGIN_DATA = json.load(json_file)

engine_graphrep = create_engine("postgresql://{user}:{password}@{host}:{port}/graph_data".format(**LOGIN_DATA))

sql = "SELECT user_id, venue_id, started_at_local, purpose, country_code, geom FROM tist.staypoints"
sp_tist = gpd.read_postgis(sql, engine_graphrep, geom_col="geom")

users_greater_500 = pd.read_csv(os.path.join("data", "users_greater_500.csv"))

tist_500 = sp_tist[sp_tist["user_id"].isin(users_greater_500["user_id"].values)]
print(len(tist_500))

tist_500.to_csv("../homework_privacy/data/tist_500.csv")

tist_500["label"] = tist_500["purpose"].apply(purpose_categories)

# clean other labels
tist_500 = tist_500[tist_500["label"] != "other"]
print(len(tist_500))

tist_500["longitude"] = tist_500["geom"].x
tist_500["latitude"] = tist_500["geom"].y

tist_500.drop(["geom"], axis=1).to_csv("../homework_privacy/data/tist_500_label.csv", index=False)


uni, counts = np.unique(tist_500["label"], return_counts=True)

plt.bar(uni, counts)
plt.xticks(rotation=90)
plt.savefig(os.path.join("data", "foursquare_500_label_distribution.png"))
plt.show()
