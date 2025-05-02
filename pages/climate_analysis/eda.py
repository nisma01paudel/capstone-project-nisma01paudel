# pages/climate_analysis/eda.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display():
    st.subheader("📊 Climate Data - EDA")
    st.markdown("""
        Explore the climate-related datasets to understand the data distribution and correlations.
    """)

    # Load the climate data
    try:
        climate_data = pd.read_csv("data/dailyclimate-2.csv")
    except Exception as e:
        st.error(f"Failed to load climate data: {e}")
        return

    # Preview the dataset
    st.write("### Data Preview")
    st.dataframe(climate_data.head())

    # Show column types
    st.write("### Column Types")
    st.write(climate_data.dtypes)

    # Drop non-numeric columns for correlation analysis
    numeric_df = climate_data.select_dtypes(include=['number'])

    # Check if there's anything to correlate
    if numeric_df.empty:
        st.warning("No numeric columns available for correlation.")
        return

    # Correlation heatmap
    st.markdown("#### Correlation Heatmap")
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)
