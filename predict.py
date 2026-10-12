"""Load the saved model and predict final marks for one student."""
from pathlib import Path

import joblib
import pandas as pd

from preprocess import FEATURES

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_marks(model, inputs):
    """inputs: dict with one numeric value for each feature in FEATURES.
    Returns the predicted Final Marks clipped to the 0-100 range."""
    row = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    return min(max(prediction, 0.0), 100.0)
