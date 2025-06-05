"""
Data Loading and Preprocessing Module
Handles data ingestion, cleaning, and preparation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config=None):
        self.config = config or {}
        self.label_encoders = {}
    
    def load_data(self, file_path):
        """Load data from various formats"""
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                return pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Numerical columns - fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns - fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def remove_outliers(self, df, method='iqr'):
        """Remove outliers using IQR method"""
        if method == 'iqr':
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def preprocess_data(self, df):
        """Complete data preprocessing pipeline"""
        logger.info("Starting data preprocessing...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        logger.info("Missing values handled")
        
        # Remove outliers
        df = self.remove_outliers(df)
        logger.info("Outliers removed")
        
        # Encode categorical variables
        df = self.encode_categorical(df)
        logger.info("Categorical variables encoded")
        
        return df
    
    def load_and_split(self, data_path=None, target_column=None):
        """Load data and split into train/test sets"""
        # Use config values if not provided
        data_path = data_path or self.config.get('data', {}).get('train_path', 'data/sample_data.csv')
        target_column = target_column or self.config.get('data', {}).get('target_column', 'target')
        
        # Generate sample data if file doesn't exist
        if not pd.io.common.file_exists(data_path):
            logger.warning(f"Data file {data_path} not found. Generating sample data...")
            df = self.generate_sample_data()
        else:
            df = self.load_data(data_path)
        
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Separate features and target
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            # If no target column, create a sample target
            X = df
            y = np.random.randint(0, 2, size=len(df))
        
        # Split data
        test_size = self.config.get('training', {}).get('test_size', 0.2)
        random_state = self.config.get('training', {}).get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split completed. Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def generate_sample_data(self, n_samples=1000, n_features=10):
        """Generate sample dataset for demonstration"""
        np.random.seed(42)
        
        # Generate numerical features
        X_numerical = np.random.randn(n_samples, n_features-2)
        
        # Generate categorical features
        categories = ['A', 'B', 'C', 'D']
        cat_feature1 = np.random.choice(categories, n_samples)
        cat_feature2 = np.random.choice(['Type1', 'Type2', 'Type3'], n_samples)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features-2)]
        df = pd.DataFrame(X_numerical, columns=feature_names)
        df['category1'] = cat_feature1
        df['category2'] = cat_feature2
        
        # Generate target variable
        # Create some correlation with features
        target_prob = (
            0.3 * df['feature_0'] + 
            0.2 * df['feature_1'] + 
            0.1 * (df['category1'] == 'A').astype(int) +
            np.random.randn(n_samples) * 0.1
        )
        df['target'] = (target_prob > target_prob.median()).astype(int)
        
        logger.info(f"Generated sample dataset with {n_samples} samples and {n_features} features")
        
        return df

