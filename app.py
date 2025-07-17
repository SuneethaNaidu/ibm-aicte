import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.title("💼 Employee Salary Prediction")

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Chinnu work/adult 3.csv")
    df = df.dropna()
    return df

data = load_data()

# Encode categorical columns
le_dict = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

X = data.drop("income", axis=1)
y = data["income"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sidebar input
st.sidebar.header("Enter employee info:")

def user_input():
    input_data = {}
    for col in X.columns:
        if col in le_dict:
            options = list(le_dict[col].classes_)
            val = st.sidebar.selectbox(col, options)
            input_data[col] = le_dict[col].transform([val])[0]
        else:
            val = st.sidebar.slider(col, int(X[col].min()), int(X[col].max()), int(X[col].mean()))
            input_data[col] = val
    return pd.DataFrame([input_data])

input_df = user_input()

if st.button("Predict Salary Category"):
    prediction = model.predict(input_df)
    output = le_dict["income"].inverse_transform(prediction)[0]
    st.success(f"🧠 Predicted Salary: **{output}**")
