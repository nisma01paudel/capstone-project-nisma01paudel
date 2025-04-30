import streamlit as st
from utils.prediction import predict

def prediction_page():
    st.title("Prediction")
    
    # Collect feature inputs from the user
    t2m = st.number_input("Enter Temperature (T2M)", min_value=-50.0, max_value=50.0)
    rh2m = st.number_input("Enter Humidity (RH2M)", min_value=0.0, max_value=100.0)
    ws10m = st.number_input("Enter Wind Speed (WS10M)", min_value=0.0, max_value=100.0)
    ws50m = st.number_input("Enter Wind Speed (WS50M)", min_value=0.0, max_value=100.0)
    
    features = [t2m, rh2m, ws10m, ws50m]
    
    # Button to trigger prediction
    if st.button("Predict"):
        prediction = predict(features)
        st.write(f"Predicted Precipitation: {prediction[0]} mm")
