"""
FastAPI Application for ML Model Serving
RESTful API for model predictions and information
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import os
from typing import List
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
    features: List[float]
    
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: float

class ModelInfo(BaseModel):
    model_type: str
    feature_count: int
    classes: List[int]
    version: str

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model_components
    
    model_path = os.getenv("MODEL_PATH", "models/trained/best_model.pkl")
    
    try:
        if os.path.exists(model_path):
            model_components = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
            # Create a dummy model for demonstration
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            model_components = {
                'model': RandomForestClassifier(n_estimators=10, random_state=42),
                'scaler': StandardScaler(),
                'feature_engineer': None
            }
            
            # Train on dummy data
            X_dummy = np.random.randn(100, 5)
            y_dummy = np.random.randint(0, 2, 100)
            
            model_components['scaler'].fit(X_dummy)
            model_components['model'].fit(model_components['scaler'].transform(X_dummy), y_dummy)
            
            logger.info("Dummy model created for demonstration")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_components = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "ML Pipeline API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model_components is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction"""
    if model_components is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        
        # Apply scaling
        if model_components['scaler']:
            features_scaled = model_components['scaler'].transform(features)
        else:
            features_scaled = features
        
        # Make prediction
        prediction = model_components['model'].predict(features_scaled)[0]
        probabilities = model_components['model'].predict_proba(features_scaled)[0]
        
        # Get confidence (max probability)
        confidence = float(probabilities.max())
        probability = float(probabilities[prediction])
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get model information"""
    if model_components is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model = model_components['model']
        
        return ModelInfo(
            model_type=type(model).__name__,
            feature_count=model.n_features_in_ if hasattr(model, 'n_features_in_') else 5,
            classes=model.classes_.tolist() if hasattr(model, 'classes_') else [0, 1],
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(features_list: List[List[float]]):
    """Make batch predictions"""
    if model_components is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = np.array(features_list)
        
        # Apply scaling
        if model_components['scaler']:
            features_scaled = model_components['scaler'].transform(features)
        else:
            features_scaled = features
        
        # Make predictions
        predictions = model_components['model'].predict(features_scaled)
        probabilities = model_components['model'].predict_proba(features_scaled)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            results.append({
                "index": i,
                "prediction": int(pred),
                "probability": float(probs[pred]),
                "confidence": float(probs.max())
            })
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

