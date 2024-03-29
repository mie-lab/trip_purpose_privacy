import os
import pickle
import xgboost


class XGBWrapper:
    def __init__(self, xgb_params):
        self.model = xgboost.XGBClassifier(**xgb_params)

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)
        self.feature_importance = self.model.get_booster().get_score(importance_type="gain")

    def predict(self, test_x):
        return self.model.predict_proba(test_x)

    def save(self, save_path):
        os.makedirs(os.path.join("trained_models", save_path), exist_ok=True)
        with open(os.path.join("trained_models", save_path, "xgb_model.p"), "wb") as outfile:
            pickle.dump(self.model, outfile)

    def load(self, load_path):
        with open(os.path.join("trained_models", load_path, "xgb_model.p"), "rb") as infile:
            self.model = pickle.load(infile)
