import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
def display():
    st.subheader("⚙️ Model Training for Climate Risk Classification")

    data_path = "data/climate_data_nepal_district_wise_monthly.csv"
    df = pd.read_csv(data_path)

    if 'DATE' in df.columns:
        df = df.drop(columns=['DATE'])

    required_columns = ['T2M', 'RH2M', 'PRECTOT']
    if not all(col in df.columns for col in required_columns):
        st.error("Missing required columns.")
        return

    # Add noisy target
    import numpy as np
    np.random.seed(42)
    risk = ((df['PRECTOT'] > 100) & (df['T2M'] > 30)).astype(int)
    noise = np.random.binomial(1, 0.1, size=len(risk))
    df['Climate_Risk'] = np.abs(risk - noise)

    X = df[required_columns]
    y = df['Climate_Risk']

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.pkl')

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model Accuracy: {acc*100:.2f}%")
    st.code(classification_report(y_test, y_pred))

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/climate_random_forest_model.pkl')
