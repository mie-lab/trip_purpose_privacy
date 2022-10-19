# Trip purpose protection

Understanding the predictability of activity purposes and the effect of location protection measures

Preprocessing steps:

```
# Preprocess the raw (txt) data into a GeoDataFrame with longitude and latitude
python preprocessing/preprocess_ny_tokyo.py

# Group by user and venue ID and aggregate user features (visit times, count and duration)
python preprocessing/get_user_venue_dataset.py
```

Prediction
```
# Cross validation for different masking and user split
python scripts/baseline_xgb.py     
```