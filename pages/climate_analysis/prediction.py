# pages/climate_analysis/prediction.py

import streamlit as st
import pandas as pd
import joblib

def display():
    st.subheader("🔮 Climate Risk Prediction")

    st.markdown("""
        Use the trained model to predict climate risks based on new input data.
    """)

    try:
        model = joblib.load('models/climate_risk_classifier.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except FileNotFoundError:
        st.error("Trained model or scaler not found. Please run model training first.")
        return

    # User inputs
    st.write("### Enter Climate Parameters")
    temperature = st.number_input("Temperature (°C) [T2M]", min_value=-50.0, format="%.2f")
    humidity = st.number_input("Relative Humidity (%) [RH2M]", min_value=0.0, max_value=100.0, format="%.2f")
    precipitation = st.number_input("Precipitation (mm) [PRECTOT]", min_value=0.0, format="%.2f")

    if st.button("📈 Predict Climate Risk"):
        input_df = pd.DataFrame(
            [[temperature, humidity, precipitation]],
            columns=['T2M', 'RH2M', 'PRECTOT']
        )

        # Scale input
        try:
            input_scaled = scaler.transform(input_df)
        except ValueError as ve:
            st.error(f"Scaler input error: {ve}")
            return

        # Make prediction
        prediction = model.predict(input_scaled)

        # Show result
        if prediction[0] == 1:
            st.success("⚠️ Climate risk **detected**.")
        else:
            st.info("✅ No significant climate risk predicted.")

        st.write("### Input Summary")
        st.dataframe(input_df)

    st.divider()
