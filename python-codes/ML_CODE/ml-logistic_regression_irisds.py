# 1. Load the Dataset

import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for better visualization
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = y
iris_df.head()


# 2. Preprocess the Data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# 3. Define the Hypothesis (Model)

import numpy as np

# Add intercept term to X
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Initialize parameters
n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))
theta = np.zeros((n_classes, n_features))



# 4. Define the Cost Function [logistic loss function]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def compute_cost(X, y, theta):
    m = len(y)
    h = softmax(X.dot(theta.T))
    epsilon = 1e-5  # Small constant to prevent division by zero
    y_one_hot = np.eye(n_classes)[y]
    cost = -1/m * np.sum(y_one_hot * np.log(h + epsilon))
    return cost



# 5. Implement the Optimizer [Gradient Descent to minimize the cost function]

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)
    y_one_hot = np.eye(n_classes)[y]
    
    for i in range(num_iterations):
        h = softmax(X.dot(theta.T))
        gradients = -1/m * (y_one_hot - h).T.dot(X)
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
    probabilities = softmax(X.dot(theta.T))
    return np.argmax(probabilities, axis=1)

# Make predictions on the test set
y_pred = predict(X_test, theta_optimal)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

