"""
Model Training Module
Handles model training and hyperparameter optimization
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
import mlflow
import mlflow.sklearn
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
    def get_param_grid(self, model_name):
        """Get parameter grid for hyperparameter tuning"""
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear']
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        }
        return param_grids.get(model_name, {})
    
    def train_with_hyperparameter_tuning(self, X_train, y_train):
        """Train models with hyperparameter optimization"""
        logger.info("Starting model training with hyperparameter tuning...")
        
        best_models = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Get parameter grid
            param_grid = self.get_param_grid(name)
            
            if param_grid:
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    model, param_grid, cv=3, 
                    scoring='accuracy', n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_score = grid_search.best_score_
                best_params = grid_search.best_params_
            else:
                # Train with default parameters
                model.fit(X_train, y_train)
                best_model = model
                best_score = cross_val_score(model, X_train, y_train, cv=3).mean()
                best_params = {}
            
            best_models[name] = best_model
            
            # Log to MLflow
            try:
                with mlflow.start_run(run_name=f"{name}_training"):
                    mlflow.log_params(best_params)
                    mlflow.log_metric("cv_score", best_score)
                    mlflow.sklearn.log_model(best_model, name)
            except Exception as e:
                logger.warning(f"MLflow logging failed for {name}: {str(e)}")
            
            logger.info(f"{name} - CV Score: {best_score:.4f}")
        
        return best_models
    
    def train_single_model(self, model_name, X_train, y_train, params=None):
        """Train a single model with given parameters"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        if params:
            model.set_params(**params)
        
        model.fit(X_train, y_train)
        return model

