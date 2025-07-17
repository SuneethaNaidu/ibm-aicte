# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Page title
st.title("👔 Employee Salary Prediction using Random Forest")

# Upload CSV file
uploaded_file = st.file_uploader("C:/Chinnu work/adult 3.csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Dataset")
    st.write(data.head())

    # Drop rows with missing values (if any)
    data.replace(' ?', np.nan, inplace=True)
    data.dropna(inplace=True)

    # Encode categorical features
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Features and Target
    X = data.drop('salary', axis=1)
    y = data['salary']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Display Results
    st.subheader("✅ Model Accuracy")
    st.write(f"Accuracy: **{accuracy * 100:.2f}%**")

    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # User Input for Prediction
    st.subheader("🧠 Predict Salary Category")

    input_data = {}
    for col in X.columns:
        dtype = data[col].dtype
        if dtype == 'int64' or dtype == 'float64':
            input_data[col] = st.number_input(f"Enter {col}", value=float(data[col].mean()))
        else:
            input_data[col] = st.selectbox(f"Select {col}", data[col].unique())

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_df = scaler.transform(input_df)
        prediction = model.predict(input_df)[0]
        result = label_encoders['salary'].inverse_transform([prediction])[0]
        st.success(f"Predicted Salary Category: **{result}**")
