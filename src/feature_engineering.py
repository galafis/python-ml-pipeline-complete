"""
Feature Engineering Module
Advanced feature creation and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        self.polynomial_features = None
        
    def create_features(self, df):
        """Create new features from existing ones"""
        logger.info("Creating new features...")
        
        df_new = df.copy()
        
        # Polynomial features for numerical columns
        numerical_cols = df_new.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) >= 2:
            # Create interaction features for first few numerical columns
            for i, col1 in enumerate(numerical_cols[:3]):
                for col2 in numerical_cols[i+1:4]:
                    df_new[f'{col1}_x_{col2}'] = df_new[col1] * df_new[col2]
        
        # Statistical features
        if len(numerical_cols) > 1:
            df_new['numerical_mean'] = df_new[numerical_cols].mean(axis=1)
            df_new['numerical_std'] = df_new[numerical_cols].std(axis=1)
            df_new['numerical_sum'] = df_new[numerical_cols].sum(axis=1)
        
        # Binning features
        for col in numerical_cols[:3]:  # Limit to first 3 columns
            df_new[f'{col}_binned'] = pd.cut(df_new[col], bins=5, labels=False)
        
        logger.info(f"Created {len(df_new.columns) - len(df.columns)} new features")
        return df_new
    
    def scale_features(self, X_train, X_test, method='standard'):
        """Scale numerical features"""
        logger.info(f"Scaling features using {method} method...")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, scaler
    
    def select_features(self, X_train, y_train, X_test, k=10):
        """Select top k features using statistical tests"""
        logger.info(f"Selecting top {k} features...")
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        selected_features = selector.get_support(indices=True)
        logger.info(f"Selected features: {selected_features}")
        
        return X_train_selected, X_test_selected, selector

