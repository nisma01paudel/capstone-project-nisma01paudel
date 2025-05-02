import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

def display():
    st.subheader("🛠️ Feature Engineering")

    df = pd.read_csv("data/climate_data_nepal_district_wise_monthly.csv")

    st.write("### Raw Data")
    st.dataframe(df.head())

    st.markdown("Creating scaled features for temperature, humidity, and precipitation.")

    selected_cols = ['T2M', 'RH2M', 'PRECTOT']
    if not all(col in df.columns for col in selected_cols):
        st.error("One or more required columns are missing.")
        return

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[selected_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=[f"{col}_scaled" for col in selected_cols])
    
    st.write("### Scaled Features")
    st.dataframe(scaled_df.head())

    # Combine with original for potential saving
    df_scaled_combined = pd.concat([df, scaled_df], axis=1)
    df_scaled_combined.to_csv("data/engineered_climate_data.csv", index=False)

    st.success("Feature engineering completed and saved to `data/engineered_climate_data.csv`")
