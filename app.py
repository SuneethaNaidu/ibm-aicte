import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load and display data
data = pd.read_csv("C:/Certiport/Employee_Salary.csv")
df = pd.DataFrame(data)
print('Original Dataset:')
display(df)
print(df.info())

# Check for missing values and value counts
print("\nMissing Values:\n", df.isna().sum())
print("\nJob Title Distribution:\n", df['Job_Title'].value_counts())
print("\nDepartment Distribution:\n", df['Department'].value_counts())
print("\nEducation Level Distribution:\n", df['Education_Level'].value_counts())

# Drop unnecessary columns
df.drop(['Hire_Date', 'Employee_ID'], inplace=True, axis=1)

# Boxplots
plt.boxplot(df['Monthly_Salary'])
plt.title("Boxplot - Monthly Salary")
plt.show()

plt.boxplot(df['Age'])
plt.title("Boxplot - Age")
plt.show()

# Encode categorical features
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Education_Level'] = label_encoder.fit_transform(df['Education_Level'])

# One-hot encode 'Job_Title' and 'Department'
dummies_job = pd.get_dummies(df['Job_Title'], prefix='JobTitle').astype(int)
dummies_dept = pd.get_dummies(df['Department'], prefix='Department').astype(int)

# Drop original categorical columns and concatenate dummies
df.drop(['Job_Title', 'Department'], inplace=True, axis=1)
df_encoded = pd.concat([df, dummies_job, dummies_dept], axis=1)

print("\nEncoded Dataset:")
display(df_encoded)

# MinMax Scaling
minmax_scaler = MinMaxScaler()
scaled_cols = ['Age', 'Work_Hours_Per_Week', 'Years_At_Company', 'Education_Level',
               'Performance_Score', 'Projects_Handled', 'Overtime_Hours', 'Sick_Days',
               'Remote_Work_Frequency', 'Team_Size', 'Training_Hours', 'Promotions',
               'Employee_Satisfaction_Score', 'Resigned']

df_scaled = df_encoded.copy()
df_scaled[scaled_cols] = minmax_scaler.fit_transform(df_scaled[scaled_cols])

print("\nAfter MinMax Scaling:")
display(df_scaled)

# Split features and target
X = df_scaled.drop(columns=['Monthly_Salary'])
y = df_scaled['Monthly_Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Data Shape:", X_train.shape)

# Evaluation functionfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def evaluate_model(y_true, y_pred, model_name=""):
    print(f"\n {model_name} Evaluation:")
    print("MAE :", mean_absolute_error(y_true, y_pred))
    print("MSE :", mean_squared_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2  :", r2_score(y_true, y_pred))
# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate_model(y_test, lr.predict(X_test), "Linear Regression")


