# pages/flood_analysis/prediction.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def display():
    st.subheader("🔮 Flood Risk Prediction")

    st.markdown("""
        Use the trained model to predict flood risks based on new input data.
    """)

    # Load trained model and scaler
    model = joblib.load('models/flood_random_forest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    # User input for prediction
    precipitation = st.number_input("Precipitation", min_value=0.0)
    temperature = st.number_input("Temperature", min_value=-50.0)
    humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0)

    if st.button("Predict Flood Risk"):
        input_data = pd.DataFrame([[precipitation, temperature, humidity]], columns=['Precipitation', 'Temperature', 'Humidity'])
        input_data_scaled = scaler.transform(input_data)

        # Predict flood risk
        prediction = model.predict(input_data_scaled)

        if prediction[0] == 1:
            st.success("Flood risk detected!")
        else:
            st.success("No flood risk predicted.")
