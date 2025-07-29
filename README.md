# Python ML Pipeline Complete

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Pipeline completo end-to-end de Machine Learning com MLOps, desde ingestão de dados até deployment em produção, incluindo monitoramento, versionamento e automação.

## 🎯 Visão Geral

Sistema integrado de Machine Learning que implementa as melhores práticas de MLOps para desenvolvimento, treinamento, avaliação e deployment de modelos em ambiente de produção.

### ✨ Características Principais

- **🔄 Pipeline End-to-End**: Ingestão → Processamento → Treinamento → Deploy
- **🧠 Múltiplos Algoritmos**: Classificação, regressão, clustering, ensemble
- **📊 MLOps Completo**: MLflow, DVC, versionamento de modelos
- **🐳 Containerização**: Docker e Kubernetes para deployment
- **🌐 API REST**: FastAPI para serving de modelos
- **📈 Monitoramento**: Drift detection e performance tracking

## 🛠️ Stack Tecnológico

### Machine Learning Core
- **Scikit-learn**: Algoritmos de ML clássicos
- **XGBoost/LightGBM**: Gradient boosting avançado
- **TensorFlow/PyTorch**: Deep learning
- **Optuna**: Otimização de hiperparâmetros

### MLOps e Deployment
- **MLflow**: Experiment tracking e model registry
- **DVC**: Versionamento de dados e modelos
- **Docker**: Containerização
- **Kubernetes**: Orquestração
- **FastAPI**: API REST para serving

### Data Processing
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica
- **Polars**: Processamento rápido
- **Dask**: Computação distribuída

### Visualização e Interface
- **Streamlit**: Dashboards interativos
- **Plotly**: Visualizações interativas
- **Evidently**: Monitoramento de drift

## 📁 Estrutura do Projeto

```
python-ml-pipeline-complete/
├── src/                            # Código fonte principal
│   ├── data_loader.py              # Carregamento de dados
│   ├── feature_engineering.py     # Engenharia de features
│   ├── model_trainer.py           # Treinamento de modelos
│   ├── model_evaluator.py         # Avaliação de modelos
│   ├── pipeline.py                # Pipeline principal
│   ├── main.py                    # Script principal
│   └── api/                       # API FastAPI
│       └── main.py                # Servidor API
├── config/                        # Configurações
│   ├── config.yaml               # Configuração principal
│   └── model_config.yaml         # Configuração de modelos
├── data/                          # Datasets
│   ├── raw/                      # Dados brutos
│   ├── processed/                # Dados processados
│   └── features/                 # Features engineered
├── models/                        # Modelos treinados
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Testes automatizados
├── docker/                        # Dockerfiles
├── requirements.txt               # Dependências
└── README.md                      # Documentação
```

## 🚀 Quick Start

### Pré-requisitos

- Python 3.9+
- Docker (opcional)
- MLflow server (opcional)

### Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/galafis/python-ml-pipeline-complete.git
cd python-ml-pipeline-complete
```

2. **Configure o ambiente:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

3. **Configure MLflow:**
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
```

4. **Execute o pipeline:**
```bash
python src/main.py
```

## 🔄 Pipeline de Machine Learning

### 1. Data Loading e Preprocessing
```python
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer

# Carregar dados
loader = DataLoader()
data = loader.load_data('data/raw/dataset.csv')

# Engenharia de features
engineer = FeatureEngineer()
features = engineer.create_features(data)
X_train, X_test, y_train, y_test = engineer.split_data(features)
```

### 2. Model Training
```python
from src.model_trainer import ModelTrainer

# Configurar treinamento
trainer = ModelTrainer()

# Treinar múltiplos modelos
models = {
    'random_forest': RandomForestClassifier(),
    'xgboost': XGBClassifier(),
    'lightgbm': LGBMClassifier()
}

# Treinamento com MLflow tracking
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        trained_model = trainer.train(model, X_train, y_train)
        trainer.log_model(trained_model, name)
```

### 3. Model Evaluation
```python
from src.model_evaluator import ModelEvaluator

# Avaliar modelos
evaluator = ModelEvaluator()

for name, model in trained_models.items():
    metrics = evaluator.evaluate(model, X_test, y_test)
    
    print(f"{name} Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

### 4. Hyperparameter Tuning
```python
import optuna
from optuna.integration import MLflowCallback

def objective(trial):
    # Definir espaço de busca
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    
    # Treinar modelo
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Avaliar
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy

# Otimização com MLflow integration
mlflc = MLflowCallback(tracking_uri="http://localhost:5000")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, callbacks=[mlflc])
```

## 🌐 API de Serving

### FastAPI Server
```python
from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="ML Model API")

# Carregar modelo do MLflow
model = mlflow.pyfunc.load_model("models:/best_model/Production")

@app.post("/predict")
async def predict(data: dict):
    # Converter para DataFrame
    df = pd.DataFrame([data])
    
    # Fazer predição
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0].max()
    
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability),
        "model_version": model.metadata.run_id
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
```

### Executar API
```bash
# Desenvolvimento
uvicorn src.api.main:app --reload --port 8000

# Produção
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📊 Dashboard Streamlit

### Aplicação Interativa
```python
import streamlit as st
import plotly.express as px
import mlflow

st.title("ML Pipeline Dashboard")

# Sidebar para seleção de experimento
experiments = mlflow.search_experiments()
selected_exp = st.sidebar.selectbox("Experimento", experiments)

# Métricas dos runs
runs = mlflow.search_runs(experiment_ids=[selected_exp.experiment_id])

# Visualizar métricas
fig = px.scatter(runs, x='metrics.accuracy', y='metrics.f1_score', 
                 color='tags.model_type', title='Model Performance')
st.plotly_chart(fig)

# Comparação de modelos
best_runs = runs.nlargest(5, 'metrics.accuracy')
st.dataframe(best_runs[['run_id', 'metrics.accuracy', 'metrics.f1_score']])
```

## 🐳 Containerização e Deploy

### Dockerfile para API
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
  
  mlflow:
    image: python:3.9-slim
    ports:
      - "5000:5000"
    command: >
      bash -c "pip install mlflow &&
               mlflow server --host 0.0.0.0 --port 5000"
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: ml-pipeline:latest
        ports:
        - containerPort: 8000
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
```

## 📈 Monitoramento e Observabilidade

### Data Drift Detection
```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Configurar monitoramento
column_mapping = ColumnMapping()
column_mapping.target = 'target'
column_mapping.prediction = 'prediction'

# Criar relatório de drift
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df, 
           column_mapping=column_mapping)

# Salvar relatório
report.save_html("reports/data_drift_report.html")
```

### Model Performance Monitoring
```python
import mlflow
from datetime import datetime

def log_prediction_metrics(y_true, y_pred, model_version):
    with mlflow.start_run():
        # Log métricas de performance
        mlflow.log_metric("accuracy", accuracy_score(y_true, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_true, y_pred))
        mlflow.log_metric("timestamp", datetime.now().timestamp())
        
        # Log versão do modelo
        mlflow.set_tag("model_version", model_version)
        mlflow.set_tag("environment", "production")
```

## 🧪 Testes e Validação

### Executar Testes
```bash
# Testes unitários
pytest tests/test_pipeline.py -v

# Testes de integração
pytest tests/test_integration.py -v

# Testes de API
pytest tests/test_api.py -v

# Coverage report
pytest --cov=src tests/
```

### Testes de Modelo
```python
import pytest
from src.model_trainer import ModelTrainer

def test_model_training():
    trainer = ModelTrainer()
    model = trainer.train(RandomForestClassifier(), X_train, y_train)
    
    assert model is not None
    assert hasattr(model, 'predict')
    
    # Testar predições
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert all(pred in [0, 1] for pred in predictions)
```

## 🔧 Configuração Avançada

### Configuração YAML
```yaml
# config/config.yaml
data:
  train_path: "data/processed/train.csv"
  test_path: "data/processed/test.csv"
  target_column: "target"

models:
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  
  xgboost:
    n_estimators: 200
    learning_rate: 0.1
    max_depth: 6

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "ml_pipeline_experiment"

deployment:
  model_name: "production_model"
  stage: "Production"
  api_port: 8000
```

## 📊 Casos de Uso Práticos

### 1. Classificação de Clientes
- Segmentação automática de clientes
- Predição de churn
- Scoring de crédito

### 2. Previsão de Demanda
- Forecasting de vendas
- Otimização de estoque
- Planejamento de produção

### 3. Detecção de Anomalias
- Fraud detection
- Monitoramento de qualidade
- Manutenção preditiva

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- Email: gabrieldemetrios@gmail.com

---

⭐ Se este projeto foi útil, considere deixar uma estrela!

