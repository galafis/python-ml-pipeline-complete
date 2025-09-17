"""
feature_engineering.py
----------------------
Módulo de engenharia de features do pipeline ML.
Contém classe FeatureEngineer para transformação padronizada dos dados.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Classe para engenharia de features e transformação dos dados.
    
    Implementa métodos padronizados para detecção automática de tipos de colunas,
    aplicação de transformações apropriadas (one-hot encoding para categóricas,
    standardização para numéricas) seguindo boas práticas do scikit-learn.
    
    Attributes:
        categorical_features (list): Lista de features categóricas detectadas.
        numerical_features (list): Lista de features numéricas detectadas.
        preprocessor (ColumnTransformer): Transformador composto para aplicar as transformações.
        fitted_ (bool): Indica se o transformador foi ajustado aos dados.
    
    Example:
        >>> import pandas as pd
        >>> from feature_engineering import FeatureEngineer
        >>> 
        >>> # Dados de exemplo
        >>> data = pd.DataFrame({
        ...     'age': [25, 30, 35, 40],
        ...     'salary': [50000, 60000, 70000, 80000],
        ...     'department': ['IT', 'HR', 'IT', 'Finance'],
        ...     'experience': ['Junior', 'Senior', 'Mid', 'Senior']
        ... })
        >>> 
        >>> # Inicializar e aplicar transformações
        >>> fe = FeatureEngineer()
        >>> X_transformed = fe.fit_transform(data)
        >>> print(X_transformed.shape)
    """
    
    def __init__(self):
        """
        Inicializa o FeatureEngineer.
        
        Define os atributos internos para armazenar informações sobre
        as features e o estado do transformador.
        """
        self.categorical_features = []
        self.numerical_features = []
        self.preprocessor = None
        self.fitted_ = False
    
    def _detect_categorical_columns(self, X):
        """
        Detecta automaticamente colunas categóricas no dataset.
        
        Identifica colunas com tipos object, category, bool ou string,
        que são consideradas categóricas para transformação.
        
        Args:
            X (pd.DataFrame): Dataset de entrada.
            
        Returns:
            list: Lista de nomes das colunas categóricas.
        """
        categorical_cols = []
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category', 'bool', 'string']:
                categorical_cols.append(col)
            elif X[col].dtype in ['int64', 'float64'] and X[col].nunique() <= 10:
                # Considera colunas numéricas com poucos valores únicos como categóricas
                categorical_cols.append(col)
        
        return categorical_cols
    
    def _detect_numerical_columns(self, X):
        """
        Detecta automaticamente colunas numéricas no dataset.
        
        Identifica colunas com tipos numéricos (int, float) que não foram
        classificadas como categóricas.
        
        Args:
            X (pd.DataFrame): Dataset de entrada.
            
        Returns:
            list: Lista de nomes das colunas numéricas.
        """
        numerical_cols = []
        categorical_cols = self._detect_categorical_columns(X)
        
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64', 'int32', 'float32'] and col not in categorical_cols:
                numerical_cols.append(col)
        
        return numerical_cols
    
    def fit(self, X, y=None):
        """
        Ajusta o transformador aos dados de treino.
        
        Detecta automaticamente os tipos de colunas e configura o pipeline
        de transformação apropriado para cada tipo de feature.
        
        Args:
            X (pd.DataFrame): Dataset de entrada para treino.
            y (array-like, optional): Variável target (não utilizada).
            
        Returns:
            self: Instância ajustada do FeatureEngineer.
        """
        # Detectar tipos de colunas
        self.categorical_features = self._detect_categorical_columns(X)
        self.numerical_features = self._detect_numerical_columns(X)
        
        # Criar transformadores
        transformers = []
        
        if self.numerical_features:
            transformers.append((
                'num',
                StandardScaler(),
                self.numerical_features
            ))
        
        if self.categorical_features:
            transformers.append((
                'cat',
                OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                self.categorical_features
            ))
        
        # Configurar o preprocessador
        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )
            
            # Ajustar o preprocessador
            self.preprocessor.fit(X)
        
        self.fitted_ = True
        return self
    
    def transform(self, X):
        """
        Aplica as transformações aos dados.
        
        Utiliza o preprocessador ajustado para transformar os dados de entrada,
        aplicando standardização às features numéricas e one-hot encoding
        às features categóricas.
        
        Args:
            X (pd.DataFrame): Dataset a ser transformado.
            
        Returns:
            np.ndarray: Array numpy com os dados transformados.
            
        Raises:
            ValueError: Se o transformador não foi ajustado previamente.
        """
        if not self.fitted_:
            raise ValueError("FeatureEngineer deve ser ajustado antes da transformação. Use fit() ou fit_transform().")
        
        if self.preprocessor is None:
            # Caso não haja transformadores (dataset vazio ou sem features válidas)
            return X.values if hasattr(X, 'values') else X
        
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Ajusta o transformador e aplica as transformações em uma única operação.
        
        Método conveniente que combina fit() e transform() para eficiência
        em pipelines de machine learning.
        
        Args:
            X (pd.DataFrame): Dataset de entrada.
            y (array-like, optional): Variável target (não utilizada).
            
        Returns:
            np.ndarray: Array numpy com os dados transformados.
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """
        Retorna os nomes das features após a transformação.
        
        Útil para entender quais features foram criadas após as transformações,
        especialmente após one-hot encoding.
        
        Args:
            input_features (array-like, optional): Nomes das features de entrada.
            
        Returns:
            np.ndarray: Array com os nomes das features transformadas.
        """
        if not self.fitted_:
            raise ValueError("FeatureEngineer deve ser ajustado antes de obter nomes das features.")
        
        if self.preprocessor is None:
            return input_features if input_features is not None else []
        
        return self.preprocessor.get_feature_names_out(input_features)


# Exemplo de uso
if __name__ == "__main__":
    # Criar dados de exemplo
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 40, 28],
        'salary': [50000, 60000, 70000, 80000, 55000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT'],
        'experience': ['Junior', 'Senior', 'Mid', 'Senior', 'Junior'],
        'years_employed': [2, 5, 8, 12, 3]
    })
    
    print("Dados originais:")
    print(sample_data)
    print(f"Shape: {sample_data.shape}")
    
    # Aplicar transformações
    fe = FeatureEngineer()
    X_transformed = fe.fit_transform(sample_data)
    
    print("\nDados transformados:")
    print(f"Shape: {X_transformed.shape}")
    print(f"Features categóricas detectadas: {fe.categorical_features}")
    print(f"Features numéricas detectadas: {fe.numerical_features}")
    
    # Mostrar nomes das features transformadas
    feature_names = fe.get_feature_names_out()
    print(f"\nNomes das features após transformação: {feature_names}")
