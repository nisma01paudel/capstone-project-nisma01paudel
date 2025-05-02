# tabs/overview_page.py

import streamlit as st

def display():
    st.subheader("🌐 Dashboard Overview")
    st.markdown("""
        **Flood & Climate Dashboard Overview**

        - **Climate Analysis**: Analyze climate data and predict risks.
        - **Flood Analysis**: Understand flood forecasts and risk factors.
        - **Model Predictions**: Use trained models to predict climate and flood-related risks.
    """)
