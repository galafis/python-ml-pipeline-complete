"""
FastAPI Application for ML Model Serving
RESTful API for model predictions and information
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List, Any
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Pipeline API",
    description="RESTful API for machine learning model predictions",
    version="1.0.0"
)

# Global variables for model components
model_components = None


class PredictionRequest(BaseModel):
    """Schema for prediction input data"""
    data: List[List[float]]  # Assume tabular matrix input


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    prediction: List[Any]
    status: str = "success"


class ModelInfo(BaseModel):
    """Schema for model information"""
    model_loaded: bool
    model_type: str = "unknown"
    version: str


# Model loading configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'modelo.joblib')


def load_model():
    """Load the trained model from disk"""
    global model_components
    try:
        if os.path.exists(MODEL_PATH):
            model_components = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
            model_components = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_components = None


# Load model on startup
load_model()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting ML Pipeline API")
    if model_components is None:
        logger.warning("Model not loaded - predictions will fail")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "ML Pipeline API is running", "status": "healthy"}


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    return ModelInfo(
        model_loaded=model_components is not None,
        model_type=type(model_components).__name__ if model_components else "none",
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Receive data and return prediction from trained model.

    Args:
        request: PredictionRequest containing input data matrix

    Returns:
        PredictionResponse with predictions

    Raises:
        HTTPException: If model is not available or prediction fails
    """
    # Check if model is loaded
    if model_components is None:
        raise HTTPException(
            status_code=500,
            detail="Model not available - check server logs for loading errors"
        )

    try:
        # Convert input data to numpy array
        X = np.array(request.data)

        # Validate input shape
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array (matrix)")

        # Log prediction request
        logger.info(f"Processing prediction request with shape: {X.shape}")

        # Make prediction
        predictions = model_components.predict(X)

        # Convert predictions to list for JSON serialization
        pred_list = predictions.tolist()

        logger.info(f"Prediction completed successfully: {len(pred_list)} predictions")

        return PredictionResponse(
            prediction=pred_list,
            status="success"
        )

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(ve)}"
        )
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(exc)}"
        )


@app.post("/predict/batch")
async def predict_batch(request: PredictionRequest):
    """
    Batch prediction endpoint for larger datasets.
    Same as /predict but optimized for batch processing.
    """
    return await predict(request)


@app.post("/reload-model")
async def reload_model():
    """
    Reload the model from disk (useful for model updates).
    """
    try:
        load_model()
        if model_components is not None:
            return {"message": "Model reloaded successfully", "status": "success"}
        else:
            return {"message": "Model reload failed", "status": "error"}
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
