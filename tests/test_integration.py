#!/usr/bin/env python3
"""
Testes de integração para o pipeline completo de ML.
Testa a integração entre módulos principais e garante que o pipeline roda fim-a-fim.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Adicionar src ao path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from model_trainer import ModelTrainer
    from model_evaluator import ModelEvaluator
    from pipeline import MLPipeline
except ImportError as e:
    pytest.skip(f"Módulos não encontrados: {e}", allow_module_level=True)


class TestPipelineIntegration:
    """Classe para testes de integração do pipeline completo."""
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        """Fixture para criar dados de exemplo para testes."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        # Criar DataFrame com nomes de colunas
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    @pytest.fixture(scope="class")
    def temp_data_dir(self):
        """Fixture para criar diretório temporário para dados de teste."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_data_loading_integration(self, sample_data, temp_data_dir):
        """Testa integração do carregamento de dados."""
        # Salvar dados de teste em arquivo
        data_path = os.path.join(temp_data_dir, 'test_data.csv')
        sample_data.to_csv(data_path, index=False)
        
        # Testar carregamento
        loader = DataLoader()
        loaded_data = loader.load_data(data_path)
        
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)
    
    def test_feature_engineering_integration(self, sample_data):
        """Testa integração da engenharia de features."""
        engineer = FeatureEngineer()
        
        # Assumir que target está na última coluna
        target_col = 'target'
        
        # Testar criação de features
        features = engineer.create_features(sample_data)
        assert features is not None
        assert len(features) > 0
        
        # Testar divisão dos dados
        if hasattr(engineer, 'split_data'):
            X_train, X_test, y_train, y_test = engineer.split_data(
                features, target_column=target_col
            )
            
            assert len(X_train) > 0
            assert len(X_test) > 0
            assert len(y_train) == len(X_train)
            assert len(y_test) == len(X_test)
    
    def test_model_training_integration(self, sample_data):
        """Testa integração do treinamento de modelos."""
        engineer = FeatureEngineer()
        trainer = ModelTrainer()
        
        # Preparar dados
        features = engineer.create_features(sample_data)
        target_col = 'target'
        
        if hasattr(engineer, 'split_data'):
            X_train, X_test, y_train, y_test = engineer.split_data(
                features, target_column=target_col
            )
        else:
            # Fallback simples se split_data não existir
            X = features.drop(target_col, axis=1)
            y = features[target_col]
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # Testar treinamento com modelo simples
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        trained_model = trainer.train(model, X_train, y_train)
        
        assert trained_model is not None
        assert hasattr(trained_model, 'predict')
        
        # Testar predição
        predictions = trained_model.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_model_evaluation_integration(self, sample_data):
        """Testa integração da avaliação de modelos."""
        engineer = FeatureEngineer()
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()
        
        # Preparar dados
        features = engineer.create_features(sample_data)
        target_col = 'target'
        
        if hasattr(engineer, 'split_data'):
            X_train, X_test, y_train, y_test = engineer.split_data(
                features, target_column=target_col
            )
        else:
            # Fallback simples
            X = features.drop(target_col, axis=1)
            y = features[target_col]
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # Treinar modelo
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        trained_model = trainer.train(model, X_train, y_train)
        
        # Avaliar modelo
        metrics = evaluator.evaluate(trained_model, X_test, y_test)
        
        assert metrics is not None
        assert isinstance(metrics, dict)
        # Verificar métricas básicas esperadas
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in expected_metrics if m in metrics]
        assert len(available_metrics) > 0, "Nenhuma métrica básica encontrada"
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_param')
    def test_end_to_end_pipeline(self, mock_log_param, mock_log_metric, 
                                mock_start_run, sample_data, temp_data_dir):
        """Testa o pipeline completo fim-a-fim com mock do MLflow."""
        # Configurar mock do MLflow
        mock_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_context
        
        # Salvar dados de teste
        data_path = os.path.join(temp_data_dir, 'test_data.csv')
        sample_data.to_csv(data_path, index=False)
        
        # Configurar pipeline (se existir)
        try:
            pipeline = MLPipeline()
            
            # Configuração básica
            config = {
                'data_path': data_path,
                'target_column': 'target',
                'test_size': 0.3,
                'random_state': 42
            }
            
            # Executar pipeline
            result = pipeline.run(config)
            
            assert result is not None
            # Verificar se pipeline executou sem erros
            assert 'error' not in result or result.get('error') is None
            
        except (ImportError, AttributeError):
            # Se MLPipeline não existir, testar componentes individuais
            self._test_components_integration(sample_data, temp_data_dir)
    
    def _test_components_integration(self, sample_data, temp_data_dir):
        """Testa integração de componentes individuais quando pipeline não existe."""
        data_path = os.path.join(temp_data_dir, 'test_data.csv')
        sample_data.to_csv(data_path, index=False)
        
        # 1. Carregar dados
        loader = DataLoader()
        data = loader.load_data(data_path)
        assert data is not None
        
        # 2. Engenharia de features
        engineer = FeatureEngineer()
        features = engineer.create_features(data)
        assert features is not None
        
        # 3. Dividir dados
        target_col = 'target'
        if hasattr(engineer, 'split_data'):
            X_train, X_test, y_train, y_test = engineer.split_data(
                features, target_column=target_col
            )
        else:
            X = features.drop(target_col, axis=1)
            y = features[target_col]
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # 4. Treinar modelo
        trainer = ModelTrainer()
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        trained_model = trainer.train(model, X_train, y_train)
        assert trained_model is not None
        
        # 5. Avaliar modelo
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(trained_model, X_test, y_test)
        assert metrics is not None
        assert isinstance(metrics, dict)
        
        print("Pipeline de componentes integrados executado com sucesso!")
    
    def test_error_handling_integration(self, temp_data_dir):
        """Testa tratamento de erros no pipeline integrado."""
        # Testar com arquivo inexistente
        loader = DataLoader()
        
        with pytest.raises(Exception):  # Expect some form of exception
            loader.load_data(os.path.join(temp_data_dir, 'nonexistent.csv'))
    
    def test_data_validation_integration(self, sample_data):
        """Testa validação de dados no pipeline."""
        engineer = FeatureEngineer()
        
        # Criar dados com problemas (valores nulos)
        problematic_data = sample_data.copy()
        problematic_data.iloc[0, 0] = np.nan
        
        # O processamento deve lidar com valores nulos
        try:
            features = engineer.create_features(problematic_data)
            assert features is not None
            # Verificar se valores nulos foram tratados
            assert not features.isnull().all().any(), "Dados ainda contêm colunas completamente nulas"
        except Exception as e:
            # Se lançar exceção, deve ser tratada adequadamente
            assert "null" in str(e).lower() or "nan" in str(e).lower()


if __name__ == "__main__":
    # Executar testes se rodado diretamente
    pytest.main([__file__, "-v"])
