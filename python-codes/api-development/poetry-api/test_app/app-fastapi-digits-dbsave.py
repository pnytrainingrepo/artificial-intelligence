# Step 1: Train and Save the Digit Classifier Model

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pickle

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a StandardScaler and a LogisticRegression model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=10000))
])

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

# Save the model to a file
with open('digit_classifier_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# Step 2: Create a FastAPI Application

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional
import pickle
import numpy as np
from PIL import Image, ImageOps
import io
import os

# Load the saved model
with open('digit_classifier_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the database model
class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    predicted_digit: int
    filename: str

# Create a PostgreSQL database connection string
DATABASE_URL = "postgresql://username:password@localhost:5432/yourdatabase"
engine = create_engine(DATABASE_URL)
SQLModel.metadata.create_all(engine)

# Initialize the FastAPI app
app = FastAPI()

# Dependency to get a new database session
def get_session():
    with Session(engine) as session:
        yield session

# Helper function to process the image
def read_imagefile(file: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file))

    # Ensure the image is in grayscale
    image = image.convert("L")  # Convert to grayscale

    # Resize to 8x8 pixels (matching the digits dataset)
    image = image.resize((8, 8), Image.ANTIALIAS)

    # Apply inversion to get black digits on white background
    image = ImageOps.invert(image)

    # Normalize the pixel values to match the digits dataset
    image_np = np.array(image) / 16.0

    # Flatten the image into a vector
    return image_np.flatten().reshape(1, -1)

# Create an endpoint for digit prediction
@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...), session: Session = Depends(get_session)):
    # Check file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only PNG and JPG are supported.")

    # Read the image file
    image = await file.read()
    input_data = read_imagefile(image)
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_digit = int(prediction[0])
    
    # Save the prediction to the database
    prediction_entry = Prediction(predicted_digit=predicted_digit, filename=file.filename)
    session.add(prediction_entry)
    session.commit()
    session.refresh(prediction_entry)
    
    return {"filename": file.filename, "predicted_digit": predicted_digit, "id": prediction_entry.id}

# Run the app with: uvicorn script_name:app --reload

# curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: multipart/form-data" -F "file=@path_to_your_digit_image.png"