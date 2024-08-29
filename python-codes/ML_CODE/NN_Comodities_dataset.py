# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset (assuming historical commodity prices in CSV format)
data = pd.read_csv('commodity_prices.csv')  # Replace with your dataset
print(data.head())

# Data Preprocessing
# Assume 'Date' is the index and 'Price' is the target variable
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Extract price as a numpy array
prices = data['Price'].values.reshape(-1, 1)

# Normalize the prices (scale between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Define a function to create time steps for NN input
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Set the time step (lookback period)
time_step = 60  # Example: 60 days

# Create the dataset
X, y = create_dataset(scaled_prices, time_step)

# Split data into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the MLPRegressor (a simple feed-forward NN)
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

# Train the model
mlp_model.fit(X_train, y_train)

# Predict on the test set
y_pred_scaled = mlp_model.predict(X_test)

# Inverse transform the scaled predictions and test values
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Visualize the actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test_actual)), y_test_actual, label='Actual Prices', color='blue')
plt.plot(range(len(y_pred)), y_pred, label='Predicted Prices', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Commodity Price Prediction (MLPRegressor)')
plt.legend()
plt.show()
