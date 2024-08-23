# Step 1: Install Required Packages

# pip install fastapi uvicorn sqlmodel psycopg2-binary

# Step 2: Define the Database Model with SQLModel

from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional

# Define the Prediction table
class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    sentiment: str

# Create a PostgreSQL database connection string
DATABASE_URL = "postgresql://username:password@localhost:5432/yourdatabase"

# Create the database engine
engine = create_engine(DATABASE_URL)

# Create the tables in the database
SQLModel.metadata.create_all(engine)


# Step 3: Update the FastAPI Application to Save Predictions

from fastapi import FastAPI, Depends
from pydantic import BaseModel
import pickle
from sqlmodel import Session

# Load the saved model
with open('text_classification_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = FastAPI()

# Dependency to get a new database session
def get_session():
    with Session(engine) as session:
        yield session

# Define the input data schema
class TextData(BaseModel):
    text: str

@app.post("/predict/")
async def predict_sentiment(data: TextData, session: Session = Depends(get_session)):
    # Make prediction
    prediction = model.predict([data.text])
    sentiment = "positive" if prediction[0] == 1 else "negative"
    
    # Save the prediction to the database
    prediction_entry = Prediction(text=data.text, sentiment=sentiment)
    session.add(prediction_entry)
    session.commit()
    session.refresh(prediction_entry)
    
    return {"text": data.text, "sentiment": sentiment, "id": prediction_entry.id}

# Run the app with: uvicorn script_name:app --reload
# curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"text": "This is a test text"}'


