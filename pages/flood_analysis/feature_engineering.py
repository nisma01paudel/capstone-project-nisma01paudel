# pages/flood_analysis/feature_engineering.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

def display():
    st.subheader("🛠️ Feature Engineering for Flood Prediction")

    st.markdown("""
        This section helps transform raw flood data into model-ready features for training machine learning models.
    """)

    # Load data (make sure to adjust the path if needed)
    flood_data = pd.read_csv("data/filled_precipitation_data_Very_Final.csv")
    
    # Example Feature Engineering
    st.markdown("#### Feature Engineering Example")
    scaler = StandardScaler()

    # Example feature to scale
    scaled_features = scaler.fit_transform(flood_data[['Precipitation', 'Temperature', 'Humidity']])

    # Display the scaled features
    scaled_data = pd.DataFrame(scaled_features, columns=['Precipitation', 'Temperature', 'Humidity'])
    st.write(scaled_data.head())
