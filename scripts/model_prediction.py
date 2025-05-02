# scripts/model_prediction.py

import joblib
import pandas as pd

def predict(model_path, input_data):
    """
    Load the model from disk and make predictions on new input data.
    """
    model = joblib.load(model_path)
    prediction = model.predict(input_data)
    return prediction
