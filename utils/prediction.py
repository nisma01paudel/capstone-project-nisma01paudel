import joblib
import pandas as pd

# Load the saved model
def load_model():
    try:
        model = joblib.load('model.pkl')  # Ensure the path matches where the model is saved
        return model
    except FileNotFoundError:
        raise FileNotFoundError("Model file not found. Please ensure the model is trained and saved correctly.")

# Predict function
def predict(features):
    model = load_model()
    prediction = model.predict([features])
    return prediction
