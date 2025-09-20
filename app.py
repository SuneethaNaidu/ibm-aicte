import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

st.title("💼 Employee Salary Prediction App")
st.write("An interactive machine learning app to analyze employee data and predict monthly salaries.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload Employee Salary CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Original Dataset")
    st.dataframe(df.head())

    # Dataset info
    st.write("### Dataset Info")
    st.write(df.describe())
    st.write(df.info())

    # Missing values
    st.write("### Missing Values")
    st.write(df.isna().sum())

    # Drop unnecessary columns
    if "Hire_Date" in df.columns and "Employee_ID" in df.columns:
        df.drop(["Hire_Date", "Employee_ID"], inplace=True, axis=1)

    # Boxplots
    st.subheader("📈 Boxplots")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].boxplot(df['Monthly_Salary'])
    ax[0].set_title("Monthly Salary")
    ax[1].boxplot(df['Age'])
    ax[1].set_title("Age")
    st.pyplot(fig)

    # Encode categorical features
    label_encoder = LabelEncoder()
    if "Gender" in df.columns:
        df['Gender'] = label_encoder.fit_transform(df['Gender'])
    if "Education_Level" in df.columns:
        df['Education_Level'] = label_encoder.fit_transform(df['Education_Level'])

    if "Job_Title" in df.columns:
        dummies_job = pd.get_dummies(df['Job_Title'], prefix='JobTitle').astype(int)
        df = pd.concat([df.drop('Job_Title', axis=1), dummies_job], axis=1)

    if "Department" in df.columns:
        dummies_dept = pd.get_dummies(df['Department'], prefix='Department').astype(int)
        df = pd.concat([df.drop('Department', axis=1), dummies_dept], axis=1)

    st.subheader("🔑 Encoded Dataset")
    st.dataframe(df.head())

    # MinMax Scaling
    minmax_scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove("Monthly_Salary")
    df[numeric_cols] = minmax_scaler.fit_transform(df[numeric_cols])

    st.subheader("⚖️ Scaled Dataset")
    st.dataframe(df.head())

    # Split features & target
    X = df.drop(columns=["Monthly_Salary"])
    y = df["Monthly_Salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluation function
    def evaluate_model(y_true, y_pred, model_name=""):
        st.write(f"### {model_name} Evaluation")
        st.write("MAE :", mean_absolute_error(y_true, y_pred))
        st.write("MSE :", mean_squared_error(y_true, y_pred))
        st.write("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
        st.write("R2  :", r2_score(y_true, y_pred))

    # Models
    st.subheader("🤖 Model Training & Evaluation")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    evaluate_model(y_test, lr.predict(X_test), "Linear Regression")

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    evaluate_model(y_test, rf.predict(X_test), "Random Forest")

    # Prediction
    st.subheader("📝 Predict Salary for a New Employee")
    user_input = {}
    for col in X.columns:
        val = st.number_input(f"Enter {col}", min_value=0.0, value=float(X[col].mean()))
        user_input[col] = val

    if st.button("Predict Salary"):
        input_df = pd.DataFrame([user_input])
        salary_pred = rf.predict(input_df)[0]
        st.success(f"💰 Predicted Monthly Salary: {salary_pred:.2f}")
else:
    st.info("👆 Please upload a CSV file to get started.")
