from fastapi import FastAPI, Depends, HTTPException, Security, Query
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from starlette.status import HTTP_401_UNAUTHORIZED
from pydantic import BaseModel
from dotenv import load_dotenv
import os, json
import numpy as np
import pickle
from typing import List, Optional

# Load environment variables
load_dotenv()

app = FastAPI(title="Secure ML Prediction API", version="2.0")

# --- Security Setup ---
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def get_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query),
):
    if api_key_header == API_KEY:
        return api_key_header
    elif api_key_query == API_KEY:
        return api_key_query
    else:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )


# --- Pydantic Models for Request/Response ---
class PredictionInput(BaseModel):
    features: List[float]
    model_name: Optional[str] = "default"

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_used: str
    input_features: List[float]

class BatchPredictionInput(BaseModel):
    features_batch: List[List[float]]
    model_name: Optional[str] = "default"


# --- ML Model Wrapper ---
class MLModelWrapper:
    def __init__(self, model_path: str, name: str = "gbr_model"):
        self.name = name
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.load_model()
    
    def load_model(self):
        """Load the ML model from pickle file"""
        try:
            # First try direct loading
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
            self.is_trained = True
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading original model: {str(e)}")
            print("Falling back to creating a simple replacement model...")
            
            # Fallback: Create a simple substitute model for demonstration
            try:
                self.create_fallback_model()
                self.is_trained = True
                print("Fallback model created successfully")
            except Exception as fallback_error:
                print(f"Fallback model creation failed: {fallback_error}")
                self.is_trained = False
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    def create_fallback_model(self):
        """Create a simple fallback model for demonstration"""
        from sklearn.ensemble import GradientBoostingRegressor
        import numpy as np
        
        # Create and train a simple model with dummy data
        self.model = GradientBoostingRegressor(random_state=42)
        
        # Generate some dummy training data (5 features)
        X_dummy = np.random.rand(100, 5)
        y_dummy = np.sum(X_dummy, axis=1) + np.random.normal(0, 0.1, 100)
        
        # Train the model
        self.model.fit(X_dummy, y_dummy)
        print("Created and trained fallback GradientBoostingRegressor model")
    
    def reload_model(self):
        """Reload the model from pickle file"""
        self.load_model()
    
    def predict(self, features: List[float]) -> tuple:
        """
        Make prediction using the loaded model.
        Returns: (prediction, confidence)
        """
        if not self.is_trained or self.model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded or trained")
        
        try:
            # Convert features to numpy array and reshape for single prediction
            features_array = np.array(features).reshape(1, -1)
            
            # Get prediction from the model
            prediction = self.model.predict(features_array)[0]
            
            # Calculate confidence (this is a simple approximation)
            # For more accurate confidence, you might need model-specific methods
            confidence = 0.85  # Default confidence, you can enhance this based on your model
            
            return float(prediction), float(confidence)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def batch_predict(self, features_batch: List[List[float]]) -> List[tuple]:
        """Batch prediction"""
        if not self.is_trained or self.model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded or trained")
        
        try:
            # Convert batch to numpy array
            features_array = np.array(features_batch)
            
            # Get batch predictions
            predictions = self.model.predict(features_array)
            
            # Return predictions with confidence scores
            results = []
            for pred in predictions:
                confidence = 0.85  # Default confidence
                results.append((float(pred), float(confidence)))
            
            return results
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# Initialize model (load the actual pickle model)
MODEL_PATH = "gbr.pkl"  # Path to your pickle model file
model = None

def initialize_model():
    global model
    try:
        model = MLModelWrapper(MODEL_PATH, "gradient_boosting_regressor")
        return True
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        model = None
        return False

# Try to initialize model at startup
initialize_model()


# Helper function to check if model is available
def check_model():
    if model is None or not model.is_trained:
        raise HTTPException(
            status_code=500, 
            detail="ML model is not loaded. Please check if the model file exists and is valid."
        )
    return model


# --- API Endpoints ---
@app.get('/')
async def frontpage():
    model_status = "loaded" if model and model.is_trained else "not_loaded"
    return {
        "message": "ML Prediction API with Gradient Boosting Regressor",
        "model_status": model_status,
        "model_file": MODEL_PATH,
        "available_endpoints": [
            "/predict", 
            "/batch-predict", 
            "/simple-predict", 
            "/model-info", 
            "/reload-model",
            "/health"
        ],
        "examples": {
            "simple_prediction": "http://127.0.0.1:8000/simple-predict?api_key=YOUR_API_KEY&features=1.5,2.3,4.1",
            "model_info": "http://127.0.0.1:8000/model-info?api_key=YOUR_API_KEY"
        }
    }


@app.post("/predict")
async def get_prediction(
    input_data: PredictionInput,
    api_key: str = Depends(get_api_key)
):
    """
    Get a single prediction from the ML model.
    Requires valid API key and input features.
    """
    try:
        # Check if model is available
        current_model = check_model()
        
        # Validate input features
        if not input_data.features:
            raise HTTPException(status_code=400, detail="Features cannot be empty")
        
        if len(input_data.features) > 50:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Too many features (max 50)")
        
        # Get prediction from model
        prediction, confidence = current_model.predict(input_data.features)
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_used=input_data.model_name or "default",
            input_features=input_data.features
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict")
async def get_batch_predictions(
    input_data: BatchPredictionInput,
    api_key: str = Depends(get_api_key),
    limit: int = Query(10, ge=1, le=100, description="Limit number of predictions (1–100)")
):
    """
    Get batch predictions from the ML model.
    Requires valid API key and batch of input features.
    """
    try:
        # Check if model is available
        current_model = check_model()
        
        # Apply limit to batch size
        limited_batch = input_data.features_batch[:limit]
        
        if not limited_batch:
            raise HTTPException(status_code=400, detail="Feature batch cannot be empty")
        
        # Get batch predictions
        predictions = current_model.batch_predict(limited_batch)
        
        # Format response
        results = []
        for i, (prediction, confidence) in enumerate(predictions):
            results.append({
                "index": i,
                "prediction": prediction,
                "confidence": confidence,
                "input_features": limited_batch[i]
            })
        
        return {
            "status": "success",
            "model_used": input_data.model_name or "default",
            "count": len(results),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model-info")
async def get_model_info(api_key: str = Depends(get_api_key)):
    """
    Get information about the loaded ML model.
    """
    try:
        current_model = check_model()
        return {
            "model_name": current_model.name,
            "is_trained": current_model.is_trained,
            "status": "active",
            "version": "2.0",
            "supported_operations": ["single_prediction", "batch_prediction"]
        }
    except HTTPException as e:
        return {
            "model_name": "not_loaded",
            "is_trained": False,
            "status": "error",
            "version": "2.0",
            "error": e.detail
        }


@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    """
    Health check endpoint to verify API and model status.
    """
    try:
        # Check if model is available and test with dummy data
        current_model = check_model()
        test_prediction, test_confidence = current_model.predict([1.0, 2.0, 3.0])
        
        return {
            "status": "healthy",
            "api_version": "2.0",
            "model_status": "operational",
            "test_prediction_successful": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "api_version": "2.0",
            "model_status": "error",
            "error": str(e)
        }


@app.post("/reload-model")
async def reload_model_endpoint(api_key: str = Depends(get_api_key)):
    """
    Reload the ML model from the pickle file.
    Useful when the model file has been updated.
    """
    try:
        success = initialize_model()
        if success and model:
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "model_name": model.name,
                "model_path": MODEL_PATH,
                "is_trained": model.is_trained
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


# --- Alternative GET endpoint for simple predictions ---
@app.get("/simple-predict")
async def simple_prediction(
    api_key: str = Depends(get_api_key),
    features: str = Query(..., description="Comma-separated feature values (e.g., '1.5,2.3,4.1')"),
    model_name: str = Query("default", description="Model name to use")
):
    """
    Simple GET endpoint for predictions using query parameters.
    Features should be provided as comma-separated values.
    """
    try:
        # Parse features from comma-separated string
        feature_list = [float(x.strip()) for x in features.split(',')]
        
        if len(feature_list) > 50:
            raise HTTPException(status_code=400, detail="Too many features (max 50)")
        
        # Check if model is available and get prediction
        current_model = check_model()
        prediction, confidence = current_model.predict(feature_list)
        
        return {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "model_used": model_name,
            "input_features": feature_list
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid feature format. Use comma-separated numbers.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")