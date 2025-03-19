import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load the data
file_path = "data/salary_data.csv"
df = pd.read_csv(file_path)

# Convert categorical variables into numeric using Label Encoding
label_encoders = {}
categorical_columns = ["Gender", "Education Level", "Job Title"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Transform the column
    label_encoders[col] = le  # Store encoder for future transformations

# Ensure numeric columns are valid
df["Years of Experience"] = pd.to_numeric(df["Years of Experience"], errors="coerce")
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

# Drop rows with missing values
df = df.dropna()

# Define features (X) and target variable (y)
X = df.drop(columns=["Salary", "Age"])  # Remove Salary (target) and Age (optional)
y = df["Salary"]

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print model coefficients
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Function to predict salary for new inputs
def predict_salary(new_data):
    """Predict salary based on new input data."""
    encoded_data = []
    
    # Encode categorical values using stored label encoders
    for i, col in enumerate(categorical_columns):
        if new_data[i] in label_encoders[col].classes_:
            encoded_value = label_encoders[col].transform([new_data[i]])[0]
        else:
            raise ValueError(f"Invalid value '{new_data[i]}' for column '{col}'. Expected one of {list(label_encoders[col].classes_)}")
        encoded_data.append(encoded_value)
    
    # Append numerical features
    encoded_data.append(new_data[-1])  # Years of Experience
    
    # Convert to numpy array and reshape for prediction
    encoded_data = np.array(encoded_data).reshape(1, -1)
    
    # Predict salary
    predicted_salary = model.predict(encoded_data)
    return predicted_salary[0]

# Example inputs for prediction (Modify these values)
new_inputs = [
    ["Male", "Bachelor's", "Software Engineer", 5],  # Gender, Education, Job Title, Years of Experience
    ["Female", "Master's", "Data Analyst", 3]  # Another test case
]

# Predict salaries
try:
    predicted_salaries = [predict_salary(row) for row in new_inputs]
    print("Predicted Salaries:", predicted_salaries)
except ValueError as e:
    print("Error:", e)