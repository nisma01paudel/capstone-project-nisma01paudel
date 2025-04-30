import streamlit as st
from utils.preprocess import merge_data
from utils.model_training import train_model, save_model

def model_training_page():
    st.title("Model Training")
    
    # Load and prepare data
    st.write("Loading and preparing data...")
    data = merge_data()

    # Reduce size to speed up experimentation (optional)
    data = data.sample(frac=0.2, random_state=42)

    # --- Define target and features ---
    target_column = 'Precip'  # Or change to another numeric column like 'T2M', 'Humidity_2m', etc.

    if target_column not in data.columns:
        raise ValueError(f"❌ '{target_column}' not found in dataset columns.")

    # Drop rows with missing target values
    data = data.dropna(subset=[target_column])

    # Define features: remove target and non-numeric/non-relevant columns
    non_features = ['DATE', 'DISTRICT', 'District', 'ADM2_PCODE', 'version', 'Unnamed: 0']
    feature_columns = [col for col in data.columns if col not in non_features + [target_column]]

    # Select only numeric columns from the remaining features
    X = data[feature_columns].select_dtypes(include='number')
    y = data[target_column]

    # Button to trigger model training
    if st.button("Train Model"):
        # Train the model
        model, mse, y_test, predictions = train_model(X, y)
        
        # Save the trained model
        model_filename = 'trained_model.joblib'  # You can change this to any desired filename
        save_model(model, model_filename)
        
        st.write("Model Training Complete!")
        st.write(f"Mean Squared Error: {mse:.4f}")
        
        # Optionally display the predicted vs actual plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)
        
        st.success(f"Model saved successfully as {model_filename}")
