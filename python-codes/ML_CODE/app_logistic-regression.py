import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import streamlit as st

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Select relevant features and drop rows with missing values
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].dropna()

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# Split the data into features and target variable
X = df.drop('survived', axis=1)
y = df['survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Define the prediction function
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    input_data = pd.DataFrame({
        'pclass': [pclass],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'sex_male': [1 if sex == 'male' else 0],
        'embarked_Q': [1 if embarked == 'Q' else 0],
        'embarked_S': [1 if embarked == 'S' else 0]
    })
    prediction = model.predict(input_data)
    return 'Survived' if prediction[0] == 1 else 'Did not survive'

# Streamlit app
st.title("Titanic Survival Predictor")

st.write("""
This app predicts whether a passenger survived the Titanic disaster based on their personal information.
""")

pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=0)
sex = st.selectbox("Sex", options=['male', 'female'], index=0)
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", options=['C', 'Q', 'S'], index=2)

if st.button("Predict Survival"):
    prediction = predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)
    st.write(f"Prediction: {prediction}")
