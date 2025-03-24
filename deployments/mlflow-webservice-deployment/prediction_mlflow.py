from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import numpy as np
from typing import Dict, Any

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Initialize FastAPI app
app = FastAPI()

# Pydantic model for input validation
class TripPredictionInput(BaseModel):
    trip_distance: float
    pickup_hour: int
    pickup_day: int
    PULocationID: int
    DOLocationID: int
    total_amount: float


class ModelService:

    def __init__(self, model_uri, vectorizer_uri):
        self.model = None
        self.vectorizer = None
        self.model_uri = model_uri
        self.vectorizer_uri = vectorizer_uri
        
    def load_model(self):
        """Load the XGBoost model from MLflow"""
        try:
            self.model = mlflow.pyfunc.load_model(self.model_uri) 
            
        except Exception as e:
            raise RuntimeError("Failed to load model from MLflow")
    
    def load_vectorizer(self):
        """Load the dictionary vectorizer from MLflow artifacts"""
        try:
            # Load vectorizer from MLflow artifacts

            artifact_path = mlflow.artifacts.download_artifacts(self.vectorizer_uri)
            import pickle
            with open(artifact_path, 'rb') as f:
                self.dv = pickle.load(f)
            
        except Exception as e:
            raise RuntimeError("Failed to load vectorizer from MLflow")
    

# Initialize model service
# Replace with your actual MLflow URIs
MODEL_URI =  'runs:/c36592810dac4653bc0dfd44374a3712/models_mlflow'
VECTORIZER_URI = 'runs:/c36592810dac4653bc0dfd44374a3712/preprocessor/preprocessor.b'
model_service = ModelService(MODEL_URI, VECTORIZER_URI)

@app.on_event("startup")
async def startup_event():
    """Load model and vectorizer on startup"""
    model_service.load_model()
    model_service.load_vectorizer()


@app.post("/predict")
async def predict_trip_duration(rides:TripPredictionInput):
    """ 
    Args:
        rides (TripPredictionInput): Input features for the trip
    
    Returns:
        dict: Predicted trip duration in minutes
    """
    try:
        # Convert input to dictionary
        ride_dict = rides.model_dump()

        # Vectorize the input
        X_input = model_service.dv.fit_transform([ride_dict])
        
        # Make prediction
        predicted_duration = model_service.model.predict(X_input)[0]
        
        return {
            "predicted_duration": float(predicted_duration)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

