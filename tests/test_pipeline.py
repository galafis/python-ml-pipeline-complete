"""
Unit tests for the ML Pipeline
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch
import tempfile

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
            'feature1': [1, 2, None, 4],
            'feature2': [None, 2, 3, 4],
            'target': [0, 1, 0, 1]
        })
        
        # Test handling missing values
        df_clean = self.data_loader.handle_missing_values(df)
        assert not df_clean.isnull().any().any()
    
    def test_load_data(self):
        """Test data loading from file"""
        # Create temporary CSV file
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            
            try:
                df_loaded = self.data_loader.load_data(f.name)
                assert df_loaded.shape == (3, 3)
                assert list(df_loaded.columns) == ['feature1', 'feature2', 'target']
            finally:
                os.unlink(f.name)


class TestFeatureEngineer:
    def setup_method(self):
        self.feature_engineer = FeatureEngineer()
    
    def test_fit_transform(self):
        """Test feature transformation"""
        X = pd.DataFrame({
            'categorical': ['A', 'B', 'A', 'C'],
            'numerical': [1, 2, 3, 4]
        })
        
        X_transformed = self.feature_engineer.fit_transform(X)
        
        # Check that transformation occurred
        assert X_transformed.shape[0] == 4
        assert X_transformed.shape[1] >= X.shape[1]  # Should have at least same or more features
        assert not pd.isna(X_transformed).any().any()
    
    def test_transform(self):
        """Test transform method after fitting"""
        X_train = pd.DataFrame({
            'categorical': ['A', 'B', 'A'],
            'numerical': [1, 2, 3]
        })
        X_test = pd.DataFrame({
            'categorical': ['A', 'B'],
            'numerical': [4, 5]
        })
        
        # Fit on training data
        self.feature_engineer.fit_transform(X_train)
        
        # Transform test data
        X_test_transformed = self.feature_engineer.transform(X_test)
        
        assert X_test_transformed.shape[0] == 2
        assert not pd.isna(X_test_transformed).any().any()


class TestModelTrainer:
    def setup_method(self):
        from sklearn.linear_model import LogisticRegression
        self.model_trainer = ModelTrainer(LogisticRegression(random_state=42))
    
    def test_initialization(self):
        """Test model trainer initialization"""
        assert hasattr(self.model_trainer, 'estimator')
        assert not hasattr(self.model_trainer, 'trained') or not self.model_trainer.trained
    
    def test_fit(self):
        """Test model fitting"""
        X = np.random.rand(50, 3)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        
        self.model_trainer.fit(X, y)
        
        assert hasattr(self.model_trainer, 'trained')
        assert self.model_trainer.trained
    
    def test_predict(self):
        """Test model prediction"""
        X = np.random.rand(50, 3)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        
        # Fit the model first
        self.model_trainer.fit(X, y)
        
        # Make predictions
        X_test = np.random.rand(10, 3)
        predictions = self.model_trainer.predict(X_test)
        
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)


class TestModelEvaluator:
    def setup_method(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(random_state=42)
        self.model_evaluator = ModelEvaluator(self.model)
    
    def test_evaluate(self):
        """Test model evaluation"""
        X = np.random.rand(100, 3)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        
        # Train the model first
        self.model.fit(X, y)
        
        # Evaluate the model
        metrics = self.model_evaluator.evaluate(X, y)
        
        # Check that required metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_evaluate_with_cross_validation(self):
        """Test model evaluation with cross-validation"""
        X = np.random.rand(100, 3)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        
        # Evaluate with cross-validation
        cv_scores = self.model_evaluator.cross_validate(X, y, cv=3)
        
        assert 'test_accuracy' in cv_scores
        assert len(cv_scores['test_accuracy']) == 3


class TestMLPipeline:
    def setup_method(self):
        from sklearn.linear_model import LogisticRegression
        self.pipeline = MLPipeline(LogisticRegression(random_state=42))
    
    def test_initialization(self):
        """Test pipeline initialization"""
        assert hasattr(self.pipeline, 'data_loader')
        assert hasattr(self.pipeline, 'feature_engineer')
        assert hasattr(self.pipeline, 'model_trainer')
        assert hasattr(self.pipeline, 'model_evaluator')
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        X = pd.DataFrame({
            'feature1': np.random.rand(20),
            'feature2': ['A', 'B'] * 10
        })
        
        X_processed = self.pipeline.preprocess_data(X)
        
        assert X_processed.shape[0] == 20
        assert not pd.isna(X_processed).any().any()
    
    def test_train_model(self):
        """Test model training through pipeline"""
        X = pd.DataFrame(np.random.rand(50, 3))
        y = (X.iloc[:, 0] + X.iloc[:, 1] > 1).astype(int)
        
        # Preprocess data
        X_processed = self.pipeline.preprocess_data(X)
        
        # Train model
        self.pipeline.train_model(X_processed, y)
        
        assert hasattr(self.pipeline.model_trainer, 'trained')
        assert self.pipeline.model_trainer.trained
    
    def test_evaluate_model(self):
        """Test model evaluation through pipeline"""
        X = pd.DataFrame(np.random.rand(50, 3))
        y = (X.iloc[:, 0] + X.iloc[:, 1] > 1).astype(int)
        
        # Preprocess and train
        X_processed = self.pipeline.preprocess_data(X)
        self.pipeline.train_model(X_processed, y)
        
        # Evaluate
        metrics = self.pipeline.evaluate_model(X_processed, y)
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        # Generate sample data
        df = self.pipeline.data_loader.generate_sample_data(n_samples=100, n_features=4)
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Run complete pipeline
        X_processed = self.pipeline.preprocess_data(X)
        self.pipeline.train_model(X_processed, y)
        metrics = self.pipeline.evaluate_model(X_processed, y)
        
        # Make predictions
        predictions = self.pipeline.model_trainer.predict(X_processed[:5])
        
        assert len(predictions) == 5
        assert 'accuracy' in metrics
        assert self.pipeline.model_trainer.trained


# Example tests for API integration (commented out for now)
# Uncomment and adapt these when implementing API functionality

# @pytest.mark.integration
# def test_api_predict():
#     """Test API prediction endpoint"""
#     import requests
#     
#     payload = {
#         'data': [[0.1, 0.2, 0.3], [0.8, 0.9, 0.1]]
#     }
#     
#     response = requests.post('http://localhost:8000/predict', json=payload)
#     
#     assert response.status_code == 200
#     result = response.json()
#     assert 'predictions' in result
#     assert len(result['predictions']) == 2

# @pytest.mark.integration  
# def test_api_train():
#     """Test API training endpoint"""
#     import requests
#     
#     payload = {
#         'data': [[0.1, 0.2, 0.3], [0.8, 0.9, 0.1]],
#         'target': [0, 1]
#     }
#     
#     response = requests.post('http://localhost:8000/train', json=payload)
#     
#     assert response.status_code == 200
#     result = response.json()
#     assert 'status' in result
#     assert result['status'] == 'success'

# @pytest.fixture
# def sample_data():
#     """Fixture to provide sample data for tests"""
#     np.random.seed(42)
#     X = pd.DataFrame({
#         'feature1': np.random.rand(100),
#         'feature2': np.random.choice(['A', 'B', 'C'], 100),
#         'feature3': np.random.randn(100)
#     })
#     y = (X['feature1'] + (X['feature2'] == 'A').astype(int) > 0.8).astype(int)
#     return X, y

# @pytest.mark.parametrize("n_samples,n_features", [
#     (50, 3),
#     (100, 5), 
#     (200, 10)
# ])
# def test_pipeline_with_different_data_sizes(n_samples, n_features):
#     """Test pipeline with different data sizes"""
#     from sklearn.linear_model import LogisticRegression
#     
#     pipeline = MLPipeline(LogisticRegression(random_state=42))
#     df = pipeline.data_loader.generate_sample_data(n_samples=n_samples, n_features=n_features)
#     
#     X = df.drop('target', axis=1)
#     y = df['target']
#     
#     X_processed = pipeline.preprocess_data(X)
#     pipeline.train_model(X_processed, y)
#     metrics = pipeline.evaluate_model(X_processed, y)
#     
#     assert X_processed.shape[0] == n_samples
#     assert 'accuracy' in metrics
#     assert pipeline.model_trainer.trained
