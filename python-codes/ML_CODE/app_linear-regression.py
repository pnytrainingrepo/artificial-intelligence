import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import streamlit as st

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Select numerical variables and drop rows with missing values
df = df[['age', 'sibsp', 'parch', 'fare', 'pclass']].dropna()

# Split the data into features and target variable
X = df[['age', 'sibsp', 'parch', 'pclass']]
y = df['fare']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Define the prediction function
def predict_fare(age, sibsp, parch, pclass):
    input_data = np.array([[age, sibsp, parch, pclass]])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
st.title("Titanic Fare Predictor")

st.write("""
This app predicts the fare based on age, siblings/spouses aboard, parents/children aboard, and passenger class.
""")

age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
pclass = st.selectbox("Passenger Class", options=[1, 2, 3])

if st.button("Predict Fare"):
    prediction = predict_fare(age, sibsp, parch, pclass)
    st.write(f"Predicted Fare: ${prediction:.2f}")
