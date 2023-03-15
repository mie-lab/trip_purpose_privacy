# main command to train with 10 fold cross validation
python scripts/run.py --embed --inbuffer --closestk -m xgb -c newyorkcity -p foursquare -k 10 -d data --embed_model_path=privacy_models
# for spatial cross validation
python scripts/run.py --embed --inbuffer --closestk -m xgb -c newyorkcity -p foursquare -k 9 -d data --embed_model_path=privacy_models --fold_mode spatial
# keep only 75% of the POIs
python scripts/run.py --embed --inbuffer --closestk -m xgb -c newyorkcity -p foursquare -k 10 -d data --embed_model_path=privacy_models --poi_keep_ratio 0.75