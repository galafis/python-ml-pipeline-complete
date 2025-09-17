"""pipeline.py
-----------
Pipeline completo integrando ingestão, processamento, treino, avaliação e serialização.
"""
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from typing import Any, Optional


class MLPipeline:
    """
    Pipeline padrão para projetos de ML, integrando todos os componentes do fluxo.
    """

    def __init__(self, estimator: Any):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.trainer = ModelTrainer(estimator)
        self.evaluator: Optional[ModelEvaluator] = None
        self.trained = False

    def load_data(self, path: str):
        """Carrega os dados de entrada."""
        return self.data_loader.load_data(path)

    def preprocess_data(self, X):
        """Processa os dados com feature engineering."""
        return self.feature_engineer.fit_transform(X)

    def train_model(self, X, y):
        """Treina o modelo."""
        model = self.trainer.fit(X, y)
        self.trained = True
        return model

    def evaluate_model(self, X, y):
        """Avalia o modelo treinado."""
        self.evaluator = ModelEvaluator(self.trainer.estimator)
        return self.evaluator.evaluate(X, y)

    def save_artifacts(self, model_path: str):
        """Salva o modelo treinado e os artefatos."""
        self.trainer.save_model(model_path)


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    # Simula dados
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Inicializa pipeline
    pipeline = MLPipeline(LogisticRegression())

    # Pré-processa
    data_proc = pipeline.preprocess_data(X)

    # Treina
    pipeline.train_model(data_proc, y)

    # Avalia
    metrics = pipeline.evaluate_model(data_proc, y)
    print('Avaliação:', metrics)

    # Salva artefatos
    pipeline.save_artifacts('modelo.joblib')
