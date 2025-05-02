import streamlit as st
from pages.climate_analysis import overview, eda, feature_engineering, model_training, prediction
from pages.flood_analysis import overview as flood_overview, eda as flood_eda, feature_engineering as flood_feat_eng, model_training as flood_model_train, prediction as flood_prediction
import pages.climate_analysis.overview as climate_overview
import pages.climate_analysis.eda as climate_eda
import pages.climate_analysis.feature_engineering as climate_feateng
import pages.climate_analysis.model_training as climate_modeltrain
import pages.climate_analysis.prediction as climate_pred

def main():
    st.title("Flood and Climate Change Dashboard - Nepal")

    menu = ["Overview", "Climate Analysis", "Flood Analysis", "About"]
    choice = st.sidebar.selectbox("Select Section", menu)

    if choice == "Overview":
        st.subheader("Welcome to the Flood and Climate Change Dashboard")
        st.write("Explore the various aspects of climate change and flood risks in Nepal.")
    
    elif choice == "Climate Analysis":
        st.subheader("Climate Analysis")
        sub_menu = ["Overview", "EDA", "Feature Engineering", "Model Training", "Prediction"]
        climate_choice = st.sidebar.selectbox("Select Action", sub_menu)

        if climate_choice == "Overview":
            overview.display()
        elif climate_choice == "EDA":
            eda.display()
        elif climate_choice == "Feature Engineering":
            feature_engineering.display()
        elif climate_choice == "Model Training":
            model_training.display()
        elif climate_choice == "Prediction":
            prediction.display()

    elif choice == "Flood Analysis":
        st.subheader("Flood Analysis")
        sub_menu = ["Overview", "EDA", "Feature Engineering", "Model Training", "Prediction"]
        flood_choice = st.sidebar.selectbox("Select Action", sub_menu)

        if flood_choice == "Overview":
            flood_overview.display()
        elif flood_choice == "EDA":
            flood_eda.display()
        elif flood_choice == "Feature Engineering":
            flood_feat_eng.display()
        elif flood_choice == "Model Training":
            flood_model_train.display()
        elif flood_choice == "Prediction":
            flood_prediction.display()

    elif choice == "About":
        st.subheader("About")
        st.write("This project analyzes the impact of climate change and flood risks in Nepal.")

if __name__ == "__main__":
    main()
