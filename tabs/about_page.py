# tabs/about_page.py

import streamlit as st

def display():
    st.subheader("📖 About This Project")
    st.markdown("""
        **Flood & Climate Dashboard - Nepal**

        This dashboard allows users to explore and predict the impacts of climate change and flooding in Nepal.
        It integrates climate data, flood forecasts, and machine learning models to provide insights into future risks.

        **Developed by:** Pramod Aryal and the Omdena team.
    """)
