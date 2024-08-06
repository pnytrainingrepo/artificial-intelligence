# 1. Load the Dataset

import pandas as pd

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic_data = pd.read_csv(url)

# Display the first few rows of the dataset
titanic_data.head()

# 2. Preprocess the Data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Select features and target variable
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'

X = titanic_data[features]
y = titanic_data[target]

# Handle missing values and encode categorical variables
numerical_features = ['Age', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# 3. Define the Hypothesis (Model)

import numpy as np

# Initialize parameters
n_features = X_train.shape[1]
theta = np.zeros(n_features + 1)  # +1 for the intercept term

# Add intercept term to X
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]


# 4. Define the Cost Function [logistic loss function]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5  # Small constant to prevent division by zero
    cost = -1/m * (y.dot(np.log(h + epsilon)) + (1 - y).dot(np.log(1 - h + epsilon)))
    return cost


# 5. Implement the Optimizer [Gradient Descent to minimize the cost function]

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)
    
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradients = X.T.dot(h - y) / m
        theta = theta - learning_rate * gradients
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

# 6. Train the Model

# Define hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Train the model
theta_optimal, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations)

print("Optimal parameters:", theta_optimal)
print("Cost history over iterations:", cost_history)


# 7. Evaluate the Model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predict function
def predict(X, theta):
    probabilities = sigmoid(X.dot(theta))
    return probabilities >= 0.5

# Make predictions on the test set
y_pred = predict(X_test, theta_optimal)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
