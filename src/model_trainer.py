"""model_trainer.py
----------------
Módulo de treinamento e serialização de modelos ML.
"""
import joblib
from sklearn.base import BaseEstimator


class ModelTrainer:
    """Classe responsável pelo treino, avaliação e serialização do modelo."""

    def __init__(self, estimator: BaseEstimator):
        """
        Inicializa o ModelTrainer com um estimator scikit-learn.

        Args:
            estimator (BaseEstimator): Estimador (modelo) do scikit-learn.
        """
        self.estimator = estimator
        self.trained = False

    def fit(self, X, y) -> BaseEstimator:
        """
        Ajusta o modelo aos dados de treino.

        Args:
            X (array-like): Features de entrada.
            y (array-like): Target.

        Returns:
            BaseEstimator: Modelo ajustado.
        """
        self.estimator.fit(X, y)
        self.trained = True
        return self.estimator

    def score(self, X, y) -> float:
        """
        Calcula o score do modelo.

        Args:
            X (array-like): Features de entrada.
            y (array-like): Target.

        Returns:
            float: Score do modelo.
        """
        if not self.trained:
            raise ValueError("Modelo não treinado.")
        return self.estimator.score(X, y)

    def save_model(self, path: str) -> None:
        """
        Salva o modelo treinado em disco (formato joblib).

        Args:
            path (str): Caminho para salvar o modelo.
        """
        if not self.trained:
            raise ValueError("Modelo não treinado.")
        joblib.dump(self.estimator, path)


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X = np.random.rand(10, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    trainer = ModelTrainer(LogisticRegression())
    trainer.fit(X, y)
    print("Score:", trainer.score(X, y))
    trainer.save_model("model.joblib")
