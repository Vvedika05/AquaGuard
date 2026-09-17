
from pathlib import Path
import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"

def load_model(path=DEFAULT_MODEL_PATH):
    return joblib.load(path)

def predict_sample(model, sample):
    X = pd.DataFrame([sample]) if isinstance(sample, dict) else sample.copy()
    pred = int(model.predict(X)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X)[0, 1])
    return pred, probability
