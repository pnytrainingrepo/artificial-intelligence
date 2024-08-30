import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the labels to one-hot encoding
lb = LabelBinarizer()
y_onehot = lb.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

# Function to build and compile an ANN model with a specified activation function
def build_model(activation_function):
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))  # Input layer
    model.add(layers.Dense(10, activation=activation_function))  # Hidden layer with custom activation function
    model.add(layers.Dense(10, activation=activation_function))  # Second hidden layer with custom activation function
    model.add(layers.Dense(3, activation='softmax'))  # Output layer for 3 classes (Iris dataset)
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# List of activation functions to evaluate
activations = ['relu', 'sigmoid', 'tanh']
results = {}

# Train and evaluate the model with each activation function
for activation in activations:
    print(f"\nTraining model with {activation} activation function...")
    model = build_model(activation)
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)  # Train the model silently
    
    # Evaluate on test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy with {activation}: {accuracy:.4f}")
    results[activation] = accuracy

# Report the accuracy for each activation function
print("\nSummary of results:")
for activation, accuracy in results.items():
    print(f"Activation: {activation}, Test Accuracy: {accuracy:.4f}")
