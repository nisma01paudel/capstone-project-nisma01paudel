import streamlit as st

def display():
    # Display a warning message for first-time users
    st.warning("If this is your first time using the app, please visit the **About the project** tab to understand the background, data, and usage instructions.")

    st.header("🌐 Dashboard Overview")

    # Enhanced project overview for Climate and Forest Fire
    st.markdown("""
    Welcome to the **Climate & Forest Fire Impact Dashboard for Nepal**! 🇳🇵  
    This dashboard is part of the **Omdena Batch II Capstone Project** and is designed to help you explore, analyze, and predict environmental hazards in Nepal using real-world datasets.

    ### 📌 What You Can Do Here:
    - **🌡️ Climate Analysis**:  
      Dive deep into monthly climate data from 1981 onwards. Analyze temperature, precipitation, humidity, and other atmospheric features across Nepalese districts.
      
    - **🔥 Forest Fire Analysis**:  
      Understand the occurrence and risk of forest fires using both CSV-based fire records and shapefile-based fire event mapping.

    - **🤖 Predictive Modeling**:  
      Use machine learning models trained on climate and forest fire data to forecast future environmental risks.

    ### 🔍 How to Use:
    - Select the type of data you wish to explore from the dropdown below.
    - Navigate through tabs to perform EDA, feature engineering, model training, and prediction.
    """)

    # Session-managed selection for data type
    if "selected_type" not in st.session_state:
        st.session_state["selected_type"] = "Climate"  # Default

    option = st.selectbox(
        "Select a data category to explore:",
        ["Climate", "Agriculture"],
        index=["Climate", "Agriculture" ].index(st.session_state["selected_type"])
    )

    if st.session_state["selected_type"] != option:
        st.session_state["selected_type"] = option

    # Load appropriate modules
    if option == "Climate":
        from pages.climate_analysis import climate_overview
        climate_overview.display()

    if option == "Agriculture":
        from pages.agriculture_analysis import agri_overview
        agri_overview.display()
