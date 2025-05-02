# tabs/feat_engineering_page.py

import streamlit as st
from sklearn.preprocessing import StandardScaler

def display():
    st.subheader("🛠️ Feature Engineering")

    # Provide options for feature engineering (e.g., scaling, selection)
    st.markdown("""
        Here, you can transform raw data into model-ready features.
    """)
    
    # Example feature scaling
    scaler = StandardScaler()
    climate_data = pd.read_csv("data/dailyclimate-2.csv")
    scaled_data = scaler.fit_transform(climate_data[['Temperature', 'Humidity']])

    st.write(scaled_data)
