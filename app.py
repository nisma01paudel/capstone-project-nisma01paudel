import streamlit as st
from utils.st_exploratory import exploratory_page
from utils.st_model_training import model_training_page
from utils.st_prediction import prediction_page

def main():
    st.title("Climate and Air Quality Data Analysis")

    menu = ["Exploratory Data Analysis", "Model Training", "Prediction"]
    choice = st.sidebar.selectbox("Select Page", menu)

    if choice == "Exploratory Data Analysis":
        exploratory_page()
    elif choice == "Model Training":
        model_training_page()
    elif choice == "Prediction":
        prediction_page()

if __name__ == "__main__":
    main()
