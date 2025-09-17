"""
model_evaluator.py
------------------
Módulo de avaliação automática de modelos ML.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Optional, Union, Tuple
import warnings


class ModelEvaluator:
    """Classe responsável pela avaliação automática de modelos de classificação e regressão."""
    
    def __init__(self, estimator: BaseEstimator):
        """
        Inicializa o ModelEvaluator com um estimator scikit-learn.
        
        Args:
            estimator (BaseEstimator): Estimador (modelo) do scikit-learn treinado.
        """
        self.estimator = estimator
        self.model_type: Optional[str] = None
        self.last_metrics: Optional[Dict[str, float]] = None
    
    def _detect_model_type(self, y_true: np.ndarray) -> str:
        """
        Detecta automaticamente se o modelo é de classificação ou regressão.
        
        Args:
            y_true (np.ndarray): Valores verdadeiros do target.
            
        Returns:
            str: 'classification' ou 'regression'.
        """
        # Verifica se o estimador tem método predict_proba (classificação)
        if hasattr(self.estimator, 'predict_proba'):
            return 'classification'
        
        # Verifica se y_true contém apenas valores inteiros e poucos valores únicos
        unique_values = np.unique(y_true)
        if (len(unique_values) <= 20 and 
            np.all(np.equal(np.mod(y_true, 1), 0)) and 
            np.all(unique_values >= 0)):
            return 'classification'
        
        # Caso contrário, assume regressão
        return 'regression'
    
    def _evaluate_classification(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Avalia modelo de classificação.
        
        Args:
            X (np.ndarray): Features de entrada.
            y_true (np.ndarray): Target verdadeiro.
            
        Returns:
            Dict[str, float]: Dicionário com métricas de classificação.
        """
        y_pred = self.estimator.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC AUC apenas para classificação binária
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2 and hasattr(self.estimator, 'predict_proba'):
            try:
                y_proba = self.estimator.predict_proba(X)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except (ValueError, AttributeError):
                # Em caso de erro, skip ROC AUC
                pass
        
        return metrics
    
    def _evaluate_regression(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Avalia modelo de regressão.
        
        Args:
            X (np.ndarray): Features de entrada.
            y_true (np.ndarray): Target verdadeiro.
            
        Returns:
            Dict[str, float]: Dicionário com métricas de regressão.
        """
        y_pred = self.estimator.predict(X)
        
        mse = mean_squared_error(y_true, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Avalia o modelo automaticamente detectando o tipo (classificação/regressão).
        
        Args:
            X (np.ndarray): Features de entrada.
            y_true (np.ndarray): Target verdadeiro.
            
        Returns:
            Dict[str, float]: Dicionário com métricas apropriadas.
        """
        # Detecta o tipo do modelo
        self.model_type = self._detect_model_type(y_true)
        
        # Avalia de acordo com o tipo
        if self.model_type == 'classification':
            metrics = self._evaluate_classification(X, y_true)
        else:
            metrics = self._evaluate_regression(X, y_true)
        
        # Armazena as métricas
        self.last_metrics = metrics
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5, 
                      scoring: Optional[str] = None) -> Dict[str, Any]:
        """
        Realiza validação cruzada do modelo.
        
        Args:
            X (np.ndarray): Features de entrada.
            y (np.ndarray): Target.
            cv (int): Número de folds para validação cruzada.
            scoring (Optional[str]): Métrica para scoring. Se None, usa padrão do tipo.
            
        Returns:
            Dict[str, Any]: Resultados da validação cruzada.
        """
        # Detecta o tipo se ainda não foi detectado
        if self.model_type is None:
            self.model_type = self._detect_model_type(y)
        
        # Define scoring padrão se não fornecido
        if scoring is None:
            scoring = 'accuracy' if self.model_type == 'classification' else 'r2'
        
        # Realiza validação cruzada
        scores = cross_val_score(self.estimator, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'scoring': scoring
        }
    
    def generate_report(self, X: np.ndarray, y_true: np.ndarray) -> str:
        """
        Gera relatório completo de avaliação do modelo.
        
        Args:
            X (np.ndarray): Features de entrada.
            y_true (np.ndarray): Target verdadeiro.
            
        Returns:
            str: Relatório formatado.
        """
        metrics = self.evaluate(X, y_true)
        
        # Cabeçalho do relatório
        report = f"""
=================================================
RELATÓRIO DE AVALIAÇÃO DO MODELO
=================================================
Tipo do Modelo: {self.model_type.upper()}
Tamanho do Dataset: {len(y_true)} amostras
"""
        
        # Métricas específicas por tipo
        if self.model_type == 'classification':
            report += f"""
MÉTRICAS DE CLASSIFICAÇÃO:
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1']:.4f}"""
            
            if 'roc_auc' in metrics:
                report += f"\n- ROC AUC: {metrics['roc_auc']:.4f}"
        
        else:  # regressão
            report += f"""
MÉTRICAS DE REGRESSÃO:
- MSE: {metrics['mse']:.4f}
- RMSE: {metrics['rmse']:.4f}
- MAE: {metrics['mae']:.4f}
- R²: {metrics['r2']:.4f}"""
        
        report += "\n=================================================\n"
        
        return report
    
    def __str__(self) -> str:
        """
        Representação string customizada da classe.
        
        Returns:
            str: Representação formatada.
        """
        if self.last_metrics is None:
            return f"ModelEvaluator(estimator={type(self.estimator).__name__}, não avaliado)"
        
        # String base
        result = f"ModelEvaluator(estimator={type(self.estimator).__name__}, tipo={self.model_type})"
        
        # Adiciona métricas principais
        if self.model_type == 'classification':
            accuracy = self.last_metrics.get('accuracy', 0)
            f1 = self.last_metrics.get('f1', 0)
            result += f"\n  Accuracy: {accuracy:.3f}, F1: {f1:.3f}"
        else:
            r2 = self.last_metrics.get('r2', 0)
            rmse = self.last_metrics.get('rmse', 0)
            result += f"\n  R²: {r2:.3f}, RMSE: {rmse:.3f}"
        
        return result


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    
    print("=== EXEMPLO DE USO - CLASSIFICAÇÃO ===")
    # Dados de classificação
    X_class, y_class = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.3, random_state=42)
    
    # Modelo de classificação
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_c, y_train_c)
    
    # Avaliação
    evaluator_clf = ModelEvaluator(clf)
    metrics_clf = evaluator_clf.evaluate(X_test_c, y_test_c)
    print("Métricas:", metrics_clf)
    print("\nRelatório:")
    print(evaluator_clf.generate_report(X_test_c, y_test_c))
    print("Representação:", evaluator_clf)
    
    print("\n=== EXEMPLO DE USO - REGRESSÃO ===")
    # Dados de regressão
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    
    # Modelo de regressão
    reg = LinearRegression()
    reg.fit(X_train_r, y_train_r)
    
    # Avaliação
    evaluator_reg = ModelEvaluator(reg)
    metrics_reg = evaluator_reg.evaluate(X_test_r, y_test_r)
    print("Métricas:", metrics_reg)
    print("\nRelatório:")
    print(evaluator_reg.generate_report(X_test_r, y_test_r))
    print("Representação:", evaluator_reg)
