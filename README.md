# Trip purpose protection

Understanding the predictability of activity purposes and the effect of location protection measures

Preprocessing steps:

### 1) Make POI taxonomy (NOTE: This first step can be omitted because our taxonomy is on GitHub)
Download Foursquare POI Taxonomy from [here](https://github.com/Factual/places/blob/master/categories/integrated_places_files/integrated_category_taxonomy.json). Download the json file to the folder [data](data) and rename the json file to `foursquare_taxonomy_raw.json`. We then divide some categories and form a slightly different taxonomy with the following script:

```
python preprocessing/convert_foursquare_taxonomy.py 
```

### 2) Download the Foursquare NYC and Tokyo data [here](https://www-public.it-sudparis.eu/~zhang_da/pub/dataset_tsmc2014.zip) (or simply download the data of section 2 on [this website](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)). Extract the zip file into the [data](data) folder and rename the folder to `foursquare_ny_tokio_raw`.

Execute the following steps to preprocess the data, to add the POI labels according to our taxonomy mentioned above:

```
# Preprocess the raw (txt) data into a GeoDataFrame with longitude and latitude
python preprocessing/preprocess_ny_tokyo.py

# Preprocess Foursquare POIs
python preprocessing/preprocess_foursquare_pois.py
```

### 3) Preprocess the yumuv data (propriety):

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

## Prediction

```
# Cross validation for different masking and user split
python scripts/run.py     
```
