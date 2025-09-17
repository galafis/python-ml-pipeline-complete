"""
data_loader.py
--------------
Componente de carregamento de dados do pipeline ML.
"""
import pandas as pd


class DataLoader:
    """Classe para gerenciamento de ingestão dos dados."""

    def __init__(self):
        pass

    def load_data(self, path):
        """
        Carrega um dataset a partir de um caminho CSV.

        Args:
            path (str): Caminho para o arquivo CSV.

        Returns:
            pd.DataFrame: DataFrame de dados carregados.
        """
        return pd.read_csv(path)
