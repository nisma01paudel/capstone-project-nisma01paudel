from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib  # For saving the model

# Train model function
def train_model(X, y):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions and calculate MSE
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Return the model, MSE, and predictions
    return model, mse, y_test, predictions

# Save model function
def save_model(model, filename='model.pkl'):
    # Save the trained model to the specified filename using joblib
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")
