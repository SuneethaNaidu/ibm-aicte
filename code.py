import logging
logging.getLogger('joblib').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.caching').setLevel(logging.ERROR)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Title
st.title("Employee Salary Prediction using Random Forest")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Chinnu work/adult 3.csv")
    return df

df = load_data()

# Encode categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split data
X = df.drop('income', axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# User input for prediction
st.sidebar.header("Enter Employee Details")
def user_input_features():
    age = st.sidebar.slider('Age', 17, 90, 30)
    workclass = st.sidebar.selectbox('Workclass', label_encoders['workclass'].classes_)
    education = st.sidebar.selectbox('Education', label_encoders['education'].classes_)
    marital_status = st.sidebar.selectbox('Marital Status', label_encoders['marital-status'].classes_)
    occupation = st.sidebar.selectbox('Occupation', label_encoders['occupation'].classes_)
    relationship = st.sidebar.selectbox('Relationship', label_encoders['relationship'].classes_)
    race = st.sidebar.selectbox('Race', label_encoders['race'].classes_)
    gender = st.sidebar.selectbox('Gender', label_encoders['gender'].classes_)
    hours = st.sidebar.slider('Hours per week', 1, 99, 40)

    # Map user input to encoded values
    data = {
        'age': age,
        'workclass': label_encoders['workclass'].transform([workclass])[0],
        'fnlwgt': 0,  # Placeholder
        'education': label_encoders['education'].transform([education])[0],
        'educational-num': 10,  # Placeholder
        'marital-status': label_encoders['marital-status'].transform([marital_status])[0],
        'occupation': label_encoders['occupation'].transform([occupation])[0],
        'relationship': label_encoders['relationship'].transform([relationship])[0],
        'race': label_encoders['race'].transform([race])[0],
        'gender': label_encoders['gender'].transform([gender])[0],
        'capital-gain': 0,  # Placeholder
        'capital-loss': 0,  # Placeholder
        'hours-per-week': hours,
        'native-country': 0  # Placeholder
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Predict
prediction = model.predict(input_df)
prediction_label = label_encoders['income'].inverse_transform(prediction)

# Output
st.subheader("Prediction")
st.write(f"Predicted Salary Category: **{prediction_label[0]}**")

# Optional: Accuracy score
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: **{acc * 100:.2f}%**")
