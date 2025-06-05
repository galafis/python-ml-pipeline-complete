"""
Unit tests for the ML Pipeline
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from pipeline import MLPipeline

class TestDataLoader:
    def setup_method(self):
        self.data_loader = DataLoader()
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        df = self.data_loader.generate_sample_data(n_samples=100, n_features=5)
        
        assert len(df) == 100
        assert len(df.columns) == 6  # 5 features + 1 target
        assert 'target' in df.columns
        assert not df.isnull().any().any()
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        # Create data with missing values
        df = pd.DataFrame({
            'num_col': [1, 2, np.nan, 4],
            'cat_col': ['A', 'B', np.nan, 'C'],
            'target': [0, 1, 0, 1]
        })
        
        df_clean = self.data_loader.handle_missing_values(df)
        assert not df_clean.isnull().any().any()
    
    def test_encode_categorical(self):
        """Test categorical encoding"""
        df = pd.DataFrame({
            'cat_col': ['A', 'B', 'A', 'C'],
            'num_col': [1, 2, 3, 4]
        })
        
        df_encoded = self.data_loader.encode_categorical(df)
        assert df_encoded['cat_col'].dtype in ['int64', 'int32']

class TestFeatureEngineer:
    def setup_method(self):
        self.feature_engineer = FeatureEngineer()
    
    def test_create_features(self):
        """Test feature creation"""
        df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4],
            'feature_1': [2, 3, 4, 5],
            'feature_2': [3, 4, 5, 6]
        })
        
        df_new = self.feature_engineer.create_features(df)
        
        # Should have original features plus new ones
        assert len(df_new.columns) > len(df.columns)
        assert 'feature_0_x_feature_1' in df_new.columns
    
    def test_scale_features(self):
        """Test feature scaling"""
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(20, 5)
        
        X_train_scaled, X_test_scaled, scaler = self.feature_engineer.scale_features(
            X_train, X_test
        )
        
        # Check that scaling was applied
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        assert abs(X_train_scaled.mean()) < 0.1  # Should be close to 0

class TestModelTrainer:
    def setup_method(self):
        self.model_trainer = ModelTrainer()
        
        # Create sample data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5)
        self.y_train = np.random.randint(0, 2, 100)
    
    def test_train_single_model(self):
        """Test single model training"""
        model = self.model_trainer.train_single_model(
            'random_forest', self.X_train, self.y_train
        )
        
        assert model is not None
        assert hasattr(model, 'predict')
        
        # Test prediction
        predictions = model.predict(self.X_train[:10])
        assert len(predictions) == 10

class TestModelEvaluator:
    def setup_method(self):
        self.evaluator = ModelEvaluator()
    
    def test_evaluate_classification(self):
        """Test classification evaluation"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        metrics = self.evaluator.evaluate_classification(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1

class TestMLPipeline:
    def setup_method(self):
        self.pipeline = MLPipeline()
    
    def test_pipeline_training(self):
        """Test complete pipeline training"""
        # Generate sample data
        data_loader = DataLoader()
        X_train, X_test, y_train, y_test = data_loader.load_and_split()
        
        # Train pipeline
        model = self.pipeline.train(X_train, y_train)
        
        assert model is not None
        assert self.pipeline.model is not None
        assert self.pipeline.scaler is not None
    
    def test_pipeline_prediction(self):
        """Test pipeline prediction"""
        # Generate and train on sample data
        data_loader = DataLoader()
        X_train, X_test, y_train, y_test = data_loader.load_and_split()
        
        self.pipeline.train(X_train, y_train)
        
        # Make predictions
        predictions = self.pipeline.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

