# eda.py
import streamlit as st
import pandas as pd
from scripts.visualization import (
    plot_time_series,
    plot_time_series_double,
    other_visualize,
)

def display():
    st.subheader("📊 Flood Data - EDA")
    st.markdown("""
        Explore the flood-related datasets to understand the data distribution and trends.
    """)

    # Load the data
    try:
        flood_data = pd.read_csv("data/filled_precipitation_data_Very_Final.csv")
        flood_data.columns = flood_data.columns.map(str)  # ✅ Convert column names to strings
    except Exception as e:
        st.error(f"Failed to load flood data: {e}")
        return

    # Display data
    st.subheader("Data Preview")
    st.dataframe(flood_data.head(), height=200)

    st.divider()

    st.subheader("Data Overview")
    st.write(f"Data shape: {flood_data.shape[0]} Rows, {flood_data.shape[1]} Columns")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Types")
        st.write(flood_data.dtypes)
    with col2:
        st.write("Missing Values")
        st.write(flood_data.isnull().sum())

    st.divider()

    st.subheader("Statistical Description")
    st.dataframe(flood_data.describe())

    st.divider()

    # Time series if 'year' column is present
    if 'year' in flood_data.columns:
        st.subheader("📈 Time Series Visualization")

        numeric_cols = flood_data.select_dtypes(include=['float64', 'int64']).columns
        value_col = st.selectbox("Select a numeric column to plot:", numeric_cols)

        fig = plot_time_series(
            flood_data,
            x_axis="year",
            y_axis=value_col,
            figsize=(10, 5)
        )
        st.pyplot(fig)

        st.divider()

        st.subheader("📊 Time Series Comparison")

        col1, col2 = st.columns(2)
        with col1:
            col_1 = st.selectbox("Column 1", numeric_cols)
        with col2:
            col_2 = st.selectbox("Column 2", numeric_cols, index=1)

        fig2 = plot_time_series_double(
            flood_data,
            col1=col_1,
            col2=col_2,
            x_axis="year",
            figsize=(10, 5)
        )
        st.pyplot(fig2)

    st.divider()

    st.subheader("📌 Advanced Visualizations")

    col1, col2, col3 = st.columns(3)
    with col1:
        plot_type = st.selectbox("Plot Type", ["box", "dist", "scatter", "bar", "pairplot"])
        group_by = st.selectbox("Group By (for bar)", list(flood_data.columns))
    with col2:
        x_col = st.selectbox("X-Axis", list(flood_data.columns))
        pair_cols = st.multiselect("Pairplot Columns", list(flood_data.columns))
    with col3:
        y_col = st.selectbox("Y-Axis", list(flood_data.columns))

    # Only proceed if valid strings are selected
    if all(isinstance(col, str) for col in [x_col, y_col, group_by]):
        fig = other_visualize(
            flood_data,
            plot_type=plot_type,
            x=x_col,
            y=y_col,
            group_by=group_by,
            cols=pair_cols,
            title=f"{plot_type.capitalize()} of {y_col} vs {x_col}",
        )

        st.plotly_chart(fig) if 'plotly' in str(type(fig)).lower() else st.pyplot(fig)
    else:
        st.error("Please ensure all selected columns are valid.")
