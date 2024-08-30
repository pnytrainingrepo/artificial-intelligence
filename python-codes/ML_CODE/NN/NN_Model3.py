import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 1: Load historical weather data from CSV
data = pd.read_csv('weather_data.csv')  # Replace with your CSV file path

# Step 2: Preprocess the data

# Handle missing values (for simplicity, filling with the mean)
data.fillna(data.mean(), inplace=True)

# Convert categorical variables to numerical (if any)
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Split features and target
X = data.drop(columns=['target_column'])  # Replace 'target_column' with your target variable
y = data['target_column']  # Replace with your target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Create the Neural Network Model

def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  # Output layer for regression task
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Initialize model
input_shape = (X_train.shape[1],)
model = create_model(input_shape)

# Step 4: Train the model with validation
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Step 5: Evaluate the model on test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.4f}")

# Step 6: Print Mean Squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Step 7: Predict weather conditions on new or unseen data
new_data = np.array([[value1, value2, ..., valueN]])  # Replace with actual new data
new_data_scaled = scaler.transform(new_data)
predicted_weather = model.predict(new_data_scaled)
print(f"Predicted weather condition: {predicted_weather}")
