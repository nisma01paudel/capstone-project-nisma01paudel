# tabs/prediction_page.py

import streamlit as st
from scripts.model_prediction import predict
import pandas as pd

def display():
    st.subheader("🔮 Prediction")
    st.markdown("""
        Input new data and predict future flood or climate risks using the trained models.
    """)

    # Take input data from user
    temperature = st.number_input("Temperature")
    precipitation = st.number_input("Precipitation")
    
    # Predict based on input data
    input_data = pd.DataFrame([[temperature, precipitation]], columns=['Temperature', 'Precipitation'])
    prediction = predict('models/climate_model.pkl', input_data)
    
    st.write(f"Predicted Climate Risk: {prediction}")
