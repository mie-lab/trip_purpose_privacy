# Where you go is who you are - A study on machine learning based semantic privacy attacks

This code bases accompanies our paper on semantic privacy attacks (under review), where we aim to quantify the risks for potential attackers to profile users based on their raw location data. To reproduce our results, follow the instructions below.

### Installation:

Install the code in a virtual environment by executing the following lines:
```
cd trip_purpose_privacy
python -m venv priv_env
source priv_env/bin/activate
pip install -e .
```

## Preprocessing steps:


### 1) Download the Foursquare NYC and Tokyo data from section 2 [this website](https://sites.google.com/site/yangdingqi/home/foursquare-dataset). Extract the zip file into the [data](data) folder and rename the folder to `foursquare_ny_tokio_raw`.

Execute the following steps to preprocess the data, to add the POI labels according to our taxonomy mentioned above:

```
# Preprocess the raw (txt) data into a GeoDataFrame with longitude and latitude
python preprocessing/preprocess_ny_tokyo.py
```

### 2) Preprocess the Foursquare POIs

```
python preprocessing/preprocess_foursquare_pois.py
```

### 3) Preprocess the yumuv data (propriety - skip this step):

#### 3.1) Download data from database and align with check-in dataset format
```
python preprocessing/preprocess_yumuv.py
```
#### 3.2) Get Swiss POIs: For this step, download the global Foursquare POI data in section 3 on [this website](https://sites.google.com/site/yangdingqi/home/foursquare-dataset). Extract the zip into the [data](data) folder. The folder should be named "dataset_TIST2015". Then run:
```
python preprocessing/get_swiss_pois.py
```

### 4) Add temporal information about user-venue visitation patterns to all three datasets (NY, Tokyo, yumuv)

```
# Group by user and venue ID and aggregate user features (visit times, count and duration)
python preprocessing/get_user_venue_dataset.py
```

### 5) Get OSM POIs

Download OSM data with pyrosm package (install via `pip install pyrosm`) and select and label the relevant ones:
```
python preprocessing/preprocess_osm_pois.py
```

## Prediction

```
# Cross validation for different masking and user split
python scripts/run.py     
```
