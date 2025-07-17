import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Title
st.title("💼 Employee Salary Prediction using Random Forest")

# Upload CSV file
uploaded_file = st.file_uploader("C:/Chinnu work/adult 3.csv")

@st.cache_data
def preprocess_data(df):
    df.columns = df.columns.str.strip()  # Clean column names
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'salary' not in df.columns:
            st.error("❌ 'salary' column not found in CSV.")
        else:
            df, encoders = preprocess_data(df)

            # Split dataset
            X = df.drop("salary", axis=1)
            y = df["salary"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"✅ Model Trained Successfully with Accuracy: {acc:.2%}")

            # Prediction interface
            st.subheader("🎯 Predict Salary Category")
            user_data = {}
            for col in X.columns:
                if col in encoders:
                    options = encoders[col].classes_
                    val = st.selectbox(f"{col}", options)
                    user_data[col] = encoders[col].transform([val])[0]
                else:
                    user_data[col] = st.number_input(f"{col}", value=0)

            if st.button("Predict Salary"):
                input_df = pd.DataFrame([user_data])
                pred = model.predict(input_df)[0]
                salary_label = encoders['salary'].inverse_transform([pred])[0]
                st.success(f"💰 Predicted Salary Category: {salary_label}")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
else:
    st.info("👆 Please upload a CSV file to begin.")
