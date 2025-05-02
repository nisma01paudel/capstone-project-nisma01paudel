# pages/flood_analysis/model_training.py

import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def display():
    st.subheader("⚙️ Model Training for Flood Prediction")

    st.markdown("""
        Train a machine learning model to predict flood risks based on the available data.
    """)

    # Load flood data
    flood_data = pd.read_csv("data/filled_precipitation_data_Very_Final.csv")

    # Feature selection
    features = flood_data[['Precipitation', 'Temperature', 'Humidity']]
    target = flood_data['Flood_Risk']  # Assuming 'Flood_Risk' is the target variable

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Save model for future use
    import joblib
    joblib.dump(model, 'models/flood_random_forest_model.pkl')
    st.success("Model has been trained and saved!")
