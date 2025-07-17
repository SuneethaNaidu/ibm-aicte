import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Chinnu work/adult 3.csv")

    return df

@st.cache_data
def preprocess_data(df):
    df = df.copy()
    df = df.replace('?', np.nan).dropna()

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop('income', axis=1)
    y = df['income']
    return X, y, label_encoders

def main():
    st.title("💼 Employee Salary Prediction")
    st.write("Predict whether a person earns more than 50K or not using employee attributes.")

    df = load_data()
    X, y, label_encoders = preprocess_data(df)

    model = RandomForestClassifier()
    model.fit(X, y)

    st.sidebar.header("Enter Employee Details")
    input_data = {}

    for col in X.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            val = st.sidebar.selectbox(col, sorted(df[col].unique()))
        else:
            val = st.sidebar.slider(col, int(df[col].min()), int(df[col].max()))
        input_data[col] = val

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode inputs if needed
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]
    result = label_encoders['income'].inverse_transform([prediction])[0]

    st.subheader("📊 Prediction Result")
    st.success(f"The model predicts this person earns: **{result}**")

if __name__ == '__main__':
    main()
