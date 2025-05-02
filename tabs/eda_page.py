# tabs/eda_page.py

import streamlit as st
import pandas as pd

def display():
    st.subheader("📊 Data Exploration (EDA)")
    st.markdown("""
        This section allows you to explore the climate and flood data to gain insights into patterns, trends, and correlations.
    """)

    # Load the datasets
    climate_data = pd.read_csv("data/dailyclimate-2.csv")
    flood_data = pd.read_csv("data/filled_precipitation_data_Very_Final.csv")
    
    st.write("Climate Data:", climate_data.head())
    st.write("Flood Data:", flood_data.head())
