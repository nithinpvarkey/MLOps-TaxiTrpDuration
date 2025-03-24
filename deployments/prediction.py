from fastapi import FastAPI,HTTPException
import pickle
from pydantic import BaseModel



# Define the input model for API
class TripPredictionInput(BaseModel):
    trip_distance: float
    pickup_hour: int
    pickup_day: int
    PULocationID: int
    DOLocationID: int
    total_amount: float

# Create FastAPI app
app = FastAPI(
    title="Taxi Trip Duration Prediction API",
)


# Load the saved model and vectorizer
def load_model(model_path='lin_reg.bin'):
    try:
        with open(model_path, 'rb') as f_in:
            dv, model = pickle.load(f_in)
        return dv, model
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, 
            detail="Model file not found."
        )
# Load model at startup
dv, lr_model = load_model()



@app.post("/predict")
def predict_trip_duration(rides:TripPredictionInput):
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
        X_input = dv.fit_transform([ride_dict])
        
        # Make prediction
        predicted_duration = lr_model.predict(X_input)[0]
        
        return {
            "predicted_duration": float(predicted_duration)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    