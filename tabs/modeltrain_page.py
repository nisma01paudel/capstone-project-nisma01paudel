# tabs/modeltrain_page.py

import streamlit as st
from scripts.model_train import train_model

def display():
    st.subheader("⚙️ Model Training")
    st.markdown("""
        Train machine learning models on the available climate and flood data to predict future risks.
    """)

    # Train the model (you can provide more options for tuning)
    model, mae = train_model(climate_data, target_column="Climate_Risk", model_save_path="models/climate_model.pkl")
    
    st.write(f"Trained Model - MAE: {mae:.2f}")
