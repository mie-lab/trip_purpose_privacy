# Where you go is who you are - A study on machine learning based semantic privacy attacks

This code accompanies our [paper](https://doi.org/10.1186/s40537-024-00888-8) on semantic privacy attacks, titled **Where you go is who you are - A study on machine learning based semantic privacy attacks**, published in the *Journal of Big Data*.

In this work, we aim to quantify the risks for potential attackers to profile users based on their raw location data, i.e. to find out their interest in different types of places. To reproduce our results, follow the instructions below.

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
python scripts/run.py -h


usage: run.py [-h] [-d DATA_PATH] [-c CITY] [-o OUT_DIR]
              [-p POI_DATA] [-m MODEL] [-x EMBED_MODEL_PATH]
              [-f FOLD_MODE] [-k KFOLD] [-b BUFFER_FACTOR]
              [--min_buffer MIN_BUFFER] [--lda] [--embed]
              [--closestk] [--inbuffer]
              [--poi_keep_ratio POI_KEEP_RATIO]
              [--xgbdepth XGBDEPTH]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
  -c CITY, --city CITY
  -o OUT_DIR, --out_dir OUT_DIR
  -p POI_DATA, --poi_data POI_DATA
  -m MODEL, --model MODEL
  -x EMBED_MODEL_PATH, --embed_model_path EMBED_MODEL_PATH
  -f FOLD_MODE, --fold_mode FOLD_MODE
  -k KFOLD, --kfold KFOLD
  -b BUFFER_FACTOR, --buffer_factor BUFFER_FACTOR
  --min_buffer MIN_BUFFER
  --lda
  --embed
  --closestk
  --inbuffer
  --poi_keep_ratio POI_KEEP_RATIO
  --xgbdepth XGBDEPTH
  ```

  Examples of the commands that we ran for analysis are given in `sh_commands.sh`. However, the --embed flag can not easily be used, since it requires to clone our version of the [space-to-vec code base](https://github.com/gengchenmai/space2vec) that you can get [here](https://github.com/NinaWie/space2vec), and then to train embedding models on the foursquare POI data.

## Evaluation

```
python scripts/evaluate.py -i outputs/test
```

## Citation

If you build up on this work, please consider citing our paper:

Wiedemann, N., Janowicz, K., Raubal, M. et al. Where you go is who you are: a study on machine learning based semantic privacy attacks. J Big Data 11, 39 (2024).

```bib
@article{wiedemann2024you,
  title={Where you go is who you are: a study on machine learning based semantic privacy attacks},
  author={Wiedemann, Nina and Janowicz, Krzysztof and Raubal, Martin and Kounadi, Ourania},
  journal={Journal of Big Data},
  volume={11},
  number={1},
  pages={39},
  year={2024},
  publisher={Springer}
}
```