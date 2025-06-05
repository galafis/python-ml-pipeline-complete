"""
Complete Machine Learning Pipeline
Main execution script for the ML pipeline
"""

import logging
import yaml
from pathlib import Path

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from pipeline import MLPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    """Main pipeline execution"""
    logger.info("Starting ML Pipeline execution...")
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    # Initialize pipeline
    pipeline = MLPipeline(config)
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        X_train, X_test, y_train, y_test = data_loader.load_and_split()
        
        # Feature engineering
        logger.info("Performing feature engineering...")
        X_train_engineered = feature_engineer.create_features(X_train)
        X_test_engineered = feature_engineer.create_features(X_test)
        
        # Scale features
        X_train_scaled, X_test_scaled, scaler = feature_engineer.scale_features(
            X_train_engineered, X_test_engineered
        )
        
        # Train models
        logger.info("Training models with hyperparameter tuning...")
        best_models = model_trainer.train_with_hyperparameter_tuning(
            X_train_scaled, y_train
        )
        
        # Evaluate models
        logger.info("Evaluating models...")
        best_model_name = None
        best_score = 0
        
        for name, model in best_models.items():
            predictions = model.predict(X_test_scaled)
            metrics = evaluator.evaluate_classification(y_test, predictions)
            
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}")
            
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_model_name = name
        
        # Save best model
        best_model = best_models[best_model_name]
        pipeline.save_model(best_model, f"models/trained/best_model.pkl")
        
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Best model: {best_model_name} with accuracy: {best_score:.4f}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

