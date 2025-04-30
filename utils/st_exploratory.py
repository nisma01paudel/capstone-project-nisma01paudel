import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocess import merge_data

def exploratory_page():
    st.title("Exploratory Data Analysis")
    
    # Load data
    data = merge_data()
    
    st.write("Dataset Overview:")
    st.dataframe(data.head())

    # Show basic statistics
    st.write("Basic Statistics:")
    st.write(data.describe())
    
    # Visualize temperature over time
    st.write("Temperature Over Time")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='DATE', y='T2M')
    plt.title('Temperature Over Time')
    plt.xticks(rotation=45)
    st.pyplot()
    
    # Visualize correlation matrix
    st.write("Correlation Matrix:")
    
    # Only include numeric columns for correlation matrix
    numeric_data = data.select_dtypes(include='number')  
    corr = numeric_data.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot()
