"""
Complete ML Pipeline
Orchestrates the entire machine learning workflow
"""

import joblib
import os
from pathlib import Path
import logging

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class MLPipeline:
    def __init__(self, config=None):
        self.config = config or {}
        self.data_loader = DataLoader(config)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        
        self.model = None
        self.scaler = None
        self.feature_selector = None
        
    def train(self, X_train, y_train):
        """Train the complete pipeline"""
        logger.info("Starting pipeline training...")
        
        # Feature engineering
        X_train_engineered = self.feature_engineer.create_features(X_train)
        
        # Feature scaling
        X_train_scaled, _, self.scaler = self.feature_engineer.scale_features(
            X_train_engineered, X_train_engineered
        )
        
        # Train models
        models = self.model_trainer.train_with_hyperparameter_tuning(
            X_train_scaled, y_train
        )
        
        # Select best model (highest accuracy)
        best_model_name = None
        best_score = 0
        
        for name, model in models.items():
            # Quick evaluation on training data
            predictions = model.predict(X_train_scaled)
            metrics = self.evaluator.evaluate_classification(y_train, predictions)
            
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_model_name = name
                self.model = model
        
        logger.info(f"Best model selected: {best_model_name} with accuracy: {best_score:.4f}")
        
        return self.model
    
    def predict(self, X_test):
        """Make predictions using the trained pipeline"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Apply same transformations as training
        X_test_engineered = self.feature_engineer.create_features(X_test)
        X_test_scaled = self.scaler.transform(X_test_engineered)
        
        return self.model.predict(X_test_scaled)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Apply same transformations as training
        X_test_engineered = self.feature_engineer.create_features(X_test)
        X_test_scaled = self.scaler.transform(X_test_engineered)
        
        return self.model.predict_proba(X_test_scaled)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the pipeline on test data"""
        predictions = self.predict(X_test)
        
        try:
            probabilities = self.predict_proba(X_test)
        except:
            probabilities = None
        
        return self.evaluator.evaluate_classification(y_test, predictions, probabilities)
    
    def save_model(self, model_path):
        """Save the trained pipeline"""
        if self.model is None:
            raise ValueError("No model to save. Train the pipeline first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and preprocessing components
        pipeline_components = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_engineer': self.feature_engineer
        }
        
        joblib.dump(pipeline_components, model_path)
        logger.info(f"Pipeline saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained pipeline"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        pipeline_components = joblib.load(model_path)
        
        self.model = pipeline_components['model']
        self.scaler = pipeline_components['scaler']
        self.feature_engineer = pipeline_components['feature_engineer']
        
        logger.info(f"Pipeline loaded from {model_path}")
    
    def run_complete_pipeline(self, data_path=None):
        """Run the complete ML pipeline from data loading to evaluation"""
        logger.info("Running complete ML pipeline...")
        
        # Load and split data
        X_train, X_test, y_train, y_test = self.data_loader.load_and_split(data_path)
        
        # Train pipeline
        self.train(X_train, y_train)
        
        # Evaluate pipeline
        metrics = self.evaluate(X_test, y_test)
        
        # Generate evaluation report
        predictions = self.predict(X_test)
        try:
            probabilities = self.predict_proba(X_test)
        except:
            probabilities = None
        
        report = self.evaluator.generate_evaluation_report(
            y_test, predictions, probabilities, "Complete Pipeline"
        )
        
        logger.info("Pipeline execution completed successfully!")
        
        return {
            'model': self.model,
            'metrics': metrics,
            'report': report,
            'test_predictions': predictions
        }

