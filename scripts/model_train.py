# scripts/model_train.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import joblib

def train_model(data, target_column, model_save_path):
    """
    Train the model using the provided data and target column, then save the model.
    """
    features = data.drop(columns=[target_column])
    target = data[target_column]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")
    
    # Save the model to disk
    joblib.dump(model, model_save_path)
    return model, mae
