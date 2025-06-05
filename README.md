# 🇧🇷 Pipeline Completo de Machine Learning | 🇺🇸 Complete Machine Learning Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

**Pipeline end-to-end de Machine Learning com MLOps e deployment em produção**

[🚀 Features](#-funcionalidades) • [📊 Modelos](#-modelos-implementados) • [⚡ Quick Start](#-quick-start) • [🔧 MLOps](#-mlops)

</div>

---

## 🇧🇷 Português

### 🚀 Visão Geral

Pipeline **completo de Machine Learning** desenvolvido em Python, implementando as melhores práticas de MLOps:

- 🔄 **Pipeline End-to-End**: Desde ingestão de dados até deployment
- 🧠 **Múltiplos Algoritmos**: Classificação, regressão, clustering
- 📊 **Monitoramento**: Tracking de experimentos com MLflow
- 🐳 **Containerização**: Docker para deployment consistente
- 🌐 **API REST**: FastAPI para servir modelos em produção
- 📈 **Visualizações**: Dashboards interativos com Streamlit

### 🎯 Objetivos do Pipeline

- **Automatizar** todo o ciclo de vida de ML
- **Padronizar** processos de desenvolvimento
- **Facilitar** deployment e monitoramento
- **Garantir** reprodutibilidade de experimentos
- **Acelerar** time-to-market de modelos

### 🛠️ Stack Tecnológico

#### Machine Learning Core
- **scikit-learn**: Algoritmos de ML clássicos
- **xgboost**: Gradient boosting avançado
- **lightgbm**: Gradient boosting eficiente
- **catboost**: Gradient boosting para dados categóricos
- **optuna**: Otimização de hiperparâmetros

#### Deep Learning
- **tensorflow**: Framework de deep learning
- **keras**: API de alto nível para TensorFlow
- **pytorch**: Framework alternativo de DL
- **transformers**: Modelos de linguagem pré-treinados

#### Data Processing
- **pandas**: Manipulação de dados estruturados
- **numpy**: Computação numérica
- **polars**: Processamento rápido de dados
- **dask**: Computação paralela e distribuída

#### MLOps e Deployment
- **mlflow**: Tracking e gerenciamento de experimentos
- **dvc**: Versionamento de dados e modelos
- **docker**: Containerização
- **kubernetes**: Orquestração de containers
- **fastapi**: API REST para servir modelos

#### Visualização e Monitoramento
- **streamlit**: Dashboards interativos
- **plotly**: Gráficos interativos
- **wandb**: Monitoramento de experimentos
- **evidently**: Monitoramento de drift de dados

### 📋 Estrutura do Pipeline

```
python-ml-pipeline-complete/
├── 📁 src/                        # Código fonte principal
│   ├── 📁 data/                   # Módulos de dados
│   │   ├── 📄 ingestion.py        # Ingestão de dados
│   │   ├── 📄 validation.py       # Validação de dados
│   │   ├── 📄 preprocessing.py    # Pré-processamento
│   │   └── 📄 feature_engineering.py # Engenharia de features
│   ├── 📁 models/                 # Módulos de modelos
│   │   ├── 📄 base_model.py       # Classe base para modelos
│   │   ├── 📄 classification.py   # Modelos de classificação
│   │   ├── 📄 regression.py       # Modelos de regressão
│   │   ├── 📄 clustering.py       # Modelos de clustering
│   │   └── 📄 ensemble.py         # Modelos ensemble
│   ├── 📁 training/               # Módulos de treinamento
│   │   ├── 📄 trainer.py          # Classe principal de treinamento
│   │   ├── 📄 hyperparameter_tuning.py # Otimização de hiperparâmetros
│   │   ├── 📄 cross_validation.py # Validação cruzada
│   │   └── 📄 model_selection.py  # Seleção de modelos
│   ├── 📁 evaluation/             # Módulos de avaliação
│   │   ├── 📄 metrics.py          # Métricas de avaliação
│   │   ├── 📄 visualization.py    # Visualizações
│   │   ├── 📄 reports.py          # Relatórios automáticos
│   │   └── 📄 model_interpretation.py # Interpretabilidade
│   ├── 📁 deployment/             # Módulos de deployment
│   │   ├── 📄 api.py              # API FastAPI
│   │   ├── 📄 batch_prediction.py # Predições em lote
│   │   ├── 📄 model_serving.py    # Servir modelos
│   │   └── 📄 monitoring.py       # Monitoramento em produção
│   └── 📁 utils/                  # Utilitários
│       ├── 📄 config.py           # Configurações
│       ├── 📄 logging.py          # Sistema de logs
│       ├── 📄 database.py         # Conexões de banco
│       └── 📄 helpers.py          # Funções auxiliares
├── 📁 data/                       # Dados do projeto
│   ├── 📁 raw/                    # Dados brutos
│   ├── 📁 processed/              # Dados processados
│   ├── 📁 features/               # Features engineered
│   └── 📁 external/               # Dados externos
├── 📁 models/                     # Modelos treinados
│   ├── 📁 experiments/            # Experimentos MLflow
│   ├── 📁 production/             # Modelos em produção
│   └── 📁 artifacts/              # Artefatos de modelo
├── 📁 notebooks/                  # Jupyter notebooks
│   ├── 📄 01_data_exploration.ipynb # Exploração de dados
│   ├── 📄 02_feature_engineering.ipynb # Engenharia de features
│   ├── 📄 03_model_development.ipynb # Desenvolvimento de modelos
│   ├── 📄 04_model_evaluation.ipynb # Avaliação de modelos
│   └── 📄 05_model_interpretation.ipynb # Interpretação de modelos
├── 📁 tests/                      # Testes automatizados
│   ├── 📄 test_data_processing.py # Testes processamento
│   ├── 📄 test_models.py          # Testes modelos
│   ├── 📄 test_api.py             # Testes API
│   └── 📄 test_integration.py     # Testes integração
├── 📁 configs/                    # Arquivos de configuração
│   ├── 📄 model_config.yaml       # Configuração de modelos
│   ├── 📄 data_config.yaml        # Configuração de dados
│   └── 📄 deployment_config.yaml  # Configuração deployment
├── 📁 docker/                     # Arquivos Docker
│   ├── 📄 Dockerfile.training     # Container para treinamento
│   ├── 📄 Dockerfile.api          # Container para API
│   └── 📄 docker-compose.yml      # Orquestração local
├── 📁 kubernetes/                 # Manifests Kubernetes
│   ├── 📄 deployment.yaml         # Deployment
│   ├── 📄 service.yaml            # Service
│   └── 📄 ingress.yaml            # Ingress
├── 📁 scripts/                    # Scripts de automação
│   ├── 📄 train_model.py          # Script de treinamento
│   ├── 📄 evaluate_model.py       # Script de avaliação
│   ├── 📄 deploy_model.py         # Script de deployment
│   └── 📄 batch_predict.py        # Script predição em lote
├── 📁 streamlit_app/              # Dashboard Streamlit
│   ├── 📄 app.py                  # Aplicação principal
│   ├── 📄 pages/                  # Páginas do dashboard
│   └── 📄 components/             # Componentes reutilizáveis
├── 📄 requirements.txt            # Dependências Python
├── 📄 requirements-dev.txt        # Dependências desenvolvimento
├── 📄 setup.py                    # Setup do pacote
├── 📄 pyproject.toml             # Configuração do projeto
├── 📄 Makefile                   # Comandos automatizados
├── 📄 .github/workflows/         # CI/CD GitHub Actions
├── 📄 README.md                  # Este arquivo
├── 📄 LICENSE                    # Licença MIT
└── 📄 .gitignore                # Arquivos ignorados
```

### 🚀 Funcionalidades Principais

#### 1. 📊 Ingestão e Processamento de Dados

**Ingestão Flexível**
```python
from src.data.ingestion import DataIngestion

# Múltiplas fontes de dados
ingestion = DataIngestion()

# Banco de dados
data_db = ingestion.from_database(
    connection_string="postgresql://user:pass@host:port/db",
    query="SELECT * FROM sales_data WHERE date >= '2024-01-01'"
)

# APIs REST
data_api = ingestion.from_api(
    url="https://api.example.com/data",
    headers={"Authorization": "Bearer token"},
    params={"limit": 10000}
)

# Arquivos (CSV, Parquet, JSON)
data_file = ingestion.from_file(
    file_path="data/raw/sales_data.csv",
    file_type="csv",
    parse_dates=["date"]
)
```

**Validação de Dados**
```python
from src.data.validation import DataValidator

validator = DataValidator()

# Validação de schema
schema_validation = validator.validate_schema(
    data=df,
    expected_columns=["id", "feature1", "feature2", "target"],
    column_types={"id": "int64", "feature1": "float64"}
)

# Detecção de anomalias
anomalies = validator.detect_anomalies(
    data=df,
    methods=["isolation_forest", "local_outlier_factor"],
    contamination=0.1
)

# Verificação de qualidade
quality_report = validator.data_quality_report(df)
```

#### 2. 🔧 Engenharia de Features

**Feature Engineering Automatizada**
```python
from src.data.feature_engineering import FeatureEngineer

fe = FeatureEngineer()

# Features temporais
temporal_features = fe.create_temporal_features(
    data=df,
    date_column="date",
    features=["year", "month", "day_of_week", "is_weekend"]
)

# Features de agregação
agg_features = fe.create_aggregation_features(
    data=df,
    group_by=["customer_id"],
    agg_columns=["amount"],
    agg_functions=["mean", "sum", "std", "count"]
)

# Features de interação
interaction_features = fe.create_interaction_features(
    data=df,
    feature_pairs=[("feature1", "feature2"), ("feature3", "feature4")]
)

# Seleção automática de features
selected_features = fe.select_features(
    X=X_train,
    y=y_train,
    method="mutual_info",
    k_best=20
)
```

#### 3. 🧠 Modelos de Machine Learning

**Classificação**
```python
from src.models.classification import ClassificationPipeline

# Pipeline de classificação
clf_pipeline = ClassificationPipeline()

# Múltiplos algoritmos
models = {
    "random_forest": {"n_estimators": 100, "max_depth": 10},
    "xgboost": {"n_estimators": 100, "learning_rate": 0.1},
    "lightgbm": {"n_estimators": 100, "num_leaves": 31},
    "logistic_regression": {"C": 1.0, "max_iter": 1000}
}

# Treinamento e avaliação
results = clf_pipeline.train_and_evaluate(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    models=models,
    cv_folds=5
)

# Modelo ensemble
ensemble_model = clf_pipeline.create_ensemble(
    models=results["trained_models"],
    method="voting",  # ou "stacking"
    weights=[0.3, 0.3, 0.2, 0.2]
)
```

**Regressão**
```python
from src.models.regression import RegressionPipeline

reg_pipeline = RegressionPipeline()

# Modelos de regressão
reg_models = {
    "linear_regression": {},
    "random_forest": {"n_estimators": 100},
    "xgboost": {"n_estimators": 100, "learning_rate": 0.1},
    "neural_network": {"hidden_layer_sizes": (100, 50)}
}

# Treinamento com validação cruzada
reg_results = reg_pipeline.train_with_cv(
    X=X_train,
    y=y_train,
    models=reg_models,
    cv_folds=5,
    scoring=["r2", "mse", "mae"]
)
```

#### 4. 🎯 Otimização de Hiperparâmetros

**Optuna para Otimização Bayesiana**
```python
from src.training.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner()

# Definir espaço de busca
search_space = {
    "n_estimators": ("int", 50, 500),
    "max_depth": ("int", 3, 20),
    "learning_rate": ("float", 0.01, 0.3),
    "subsample": ("float", 0.6, 1.0)
}

# Otimização
best_params = tuner.optimize(
    model_class="xgboost",
    X_train=X_train,
    y_train=y_train,
    search_space=search_space,
    n_trials=100,
    cv_folds=5,
    scoring="roc_auc"
)

# Treinamento com melhores parâmetros
best_model = tuner.train_best_model(
    best_params=best_params,
    X_train=X_train,
    y_train=y_train
)
```

#### 5. 📈 Avaliação e Interpretabilidade

**Métricas Abrangentes**
```python
from src.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator()

# Avaliação completa
evaluation_report = evaluator.comprehensive_evaluation(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    task_type="classification"
)

# Métricas incluem:
# - Accuracy, Precision, Recall, F1-score
# - ROC-AUC, PR-AUC
# - Confusion Matrix
# - Classification Report
# - Feature Importance
```

**Interpretabilidade com SHAP**
```python
from src.evaluation.model_interpretation import ModelInterpreter

interpreter = ModelInterpreter()

# Valores SHAP
shap_values = interpreter.calculate_shap_values(
    model=trained_model,
    X_test=X_test,
    model_type="tree"  # ou "linear", "deep"
)

# Visualizações SHAP
interpreter.plot_shap_summary(shap_values, X_test)
interpreter.plot_shap_waterfall(shap_values, X_test, instance_idx=0)
interpreter.plot_shap_dependence(shap_values, X_test, feature="feature1")
```

#### 6. 🌐 API REST com FastAPI

**Servir Modelos em Produção**
```python
from fastapi import FastAPI, HTTPException
from src.deployment.api import ModelAPI

app = FastAPI(title="ML Model API", version="1.0.0")
model_api = ModelAPI()

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Carregar modelo
        model = model_api.load_model("production_model_v1.pkl")
        
        # Fazer predição
        prediction = model_api.predict(
            model=model,
            features=request.features
        )
        
        return {
            "prediction": prediction,
            "model_version": "v1.0",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    # Predições em lote
    predictions = model_api.batch_predict(
        model_name=request.model_name,
        data=request.data
    )
    return {"predictions": predictions}
```

#### 7. 📊 MLflow para Tracking

**Tracking de Experimentos**
```python
import mlflow
import mlflow.sklearn
from src.utils.mlflow_utils import MLflowTracker

tracker = MLflowTracker()

# Iniciar experimento
with mlflow.start_run():
    # Log parâmetros
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1
    })
    
    # Treinar modelo
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Avaliar modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log métricas
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted")
    })
    
    # Log modelo
    mlflow.sklearn.log_model(model, "model")
    
    # Log artefatos
    mlflow.log_artifact("feature_importance.png")
```

### 🐳 Containerização e Deployment

#### Docker para Desenvolvimento
```dockerfile
# Dockerfile.training
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY configs/ ./configs/

CMD ["python", "scripts/train_model.py"]
```

#### Docker para Produção
```dockerfile
# Dockerfile.api
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/production/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Kubernetes Deployment
```yaml
# kubernetes/deployment.yaml
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
        - name: MODEL_PATH
          value: "/app/models/production_model.pkl"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### 📊 Dashboard Streamlit

**Interface Interativa**
```python
import streamlit as st
from src.streamlit_app.components import ModelDashboard

st.set_page_config(
    page_title="ML Pipeline Dashboard",
    page_icon="🤖",
    layout="wide"
)

dashboard = ModelDashboard()

# Sidebar para seleção
st.sidebar.title("ML Pipeline Dashboard")
page = st.sidebar.selectbox(
    "Selecione uma página",
    ["Visão Geral", "Experimentos", "Modelos", "Predições", "Monitoramento"]
)

if page == "Visão Geral":
    dashboard.show_overview()
elif page == "Experimentos":
    dashboard.show_experiments()
elif page == "Modelos":
    dashboard.show_models()
elif page == "Predições":
    dashboard.show_predictions()
elif page == "Monitoramento":
    dashboard.show_monitoring()
```

### 🎯 Competências Demonstradas

#### Machine Learning
- ✅ **Algoritmos Supervisionados**: Classificação e regressão
- ✅ **Algoritmos Não-Supervisionados**: Clustering e redução de dimensionalidade
- ✅ **Ensemble Methods**: Voting, bagging, boosting, stacking
- ✅ **Deep Learning**: Redes neurais com TensorFlow/PyTorch

#### MLOps e DevOps
- ✅ **Versionamento**: Git, DVC para dados e modelos
- ✅ **Containerização**: Docker, Kubernetes
- ✅ **CI/CD**: GitHub Actions, automated testing
- ✅ **Monitoramento**: MLflow, Wandb, Evidently

#### Engenharia de Software
- ✅ **Arquitetura**: Design patterns, SOLID principles
- ✅ **Testes**: Unit tests, integration tests, pytest
- ✅ **Documentação**: Docstrings, README, API docs
- ✅ **Performance**: Profiling, optimization, caching

### 🚀 Quick Start

#### Instalação Local
```bash
# Clonar repositório
git clone https://github.com/galafis/python-ml-pipeline-complete.git
cd python-ml-pipeline-complete

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Treinar modelo
python scripts/train_model.py --config configs/model_config.yaml

# Iniciar API
uvicorn src.deployment.api:app --reload

# Iniciar dashboard
streamlit run streamlit_app/app.py
```

#### Docker Compose
```bash
# Iniciar todos os serviços
docker-compose up -d

# Serviços disponíveis:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Streamlit: http://localhost:8501
```

### 📈 Casos de Uso Práticos

#### 1. E-commerce: Recomendação de Produtos
- Algoritmos colaborativos e baseados em conteúdo
- Features de comportamento do usuário
- A/B testing para otimização

#### 2. Finanças: Detecção de Fraude
- Modelos de anomalia em tempo real
- Features de transação e comportamento
- Alertas automáticos

#### 3. Saúde: Diagnóstico Assistido
- Classificação de imagens médicas
- Análise de dados clínicos
- Interpretabilidade para médicos

#### 4. Marketing: Segmentação de Clientes
- Clustering de comportamento
- Predição de churn
- Otimização de campanhas

---

## 🇺🇸 English

### 🚀 Overview

**Complete Machine Learning pipeline** developed in Python, implementing MLOps best practices:

- 🔄 **End-to-End Pipeline**: From data ingestion to deployment
- 🧠 **Multiple Algorithms**: Classification, regression, clustering
- 📊 **Monitoring**: Experiment tracking with MLflow
- 🐳 **Containerization**: Docker for consistent deployment
- 🌐 **REST API**: FastAPI for serving models in production
- 📈 **Visualizations**: Interactive dashboards with Streamlit

### 🎯 Pipeline Objectives

- **Automate** the entire ML lifecycle
- **Standardize** development processes
- **Facilitate** deployment and monitoring
- **Ensure** experiment reproducibility
- **Accelerate** model time-to-market

### 🚀 Main Features

#### 1. 📊 Data Ingestion and Processing
- Flexible data ingestion from multiple sources
- Automated data validation and quality checks
- Feature engineering and selection
- Data preprocessing pipelines

#### 2. 🧠 Machine Learning Models
- Classification and regression algorithms
- Ensemble methods and model stacking
- Hyperparameter optimization with Optuna
- Cross-validation and model selection

#### 3. 📈 Evaluation and Interpretability
- Comprehensive evaluation metrics
- SHAP values for model interpretation
- Feature importance analysis
- Model performance visualization

#### 4. 🌐 Production Deployment
- FastAPI REST API for model serving
- Docker containerization
- Kubernetes orchestration
- Batch prediction capabilities

#### 5. 📊 MLOps and Monitoring
- MLflow for experiment tracking
- Model versioning and registry
- Performance monitoring
- Data drift detection

### 🎯 Skills Demonstrated

#### Machine Learning
- ✅ **Supervised Algorithms**: Classification and regression
- ✅ **Unsupervised Algorithms**: Clustering and dimensionality reduction
- ✅ **Ensemble Methods**: Voting, bagging, boosting, stacking
- ✅ **Deep Learning**: Neural networks with TensorFlow/PyTorch

#### MLOps and DevOps
- ✅ **Versioning**: Git, DVC for data and models
- ✅ **Containerization**: Docker, Kubernetes
- ✅ **CI/CD**: GitHub Actions, automated testing
- ✅ **Monitoring**: MLflow, Wandb, Evidently

#### Software Engineering
- ✅ **Architecture**: Design patterns, SOLID principles
- ✅ **Testing**: Unit tests, integration tests, pytest
- ✅ **Documentation**: Docstrings, README, API docs
- ✅ **Performance**: Profiling, optimization, caching

---

## 📄 Licença | License

MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes | see [LICENSE](LICENSE) file for details

## 📞 Contato | Contact

**GitHub**: [@galafis](https://github.com/galafis)  
**LinkedIn**: [Gabriel Demetrios Lafis](https://linkedin.com/in/galafis)  
**Email**: gabriel.lafis@example.com

---

<div align="center">

**Desenvolvido com ❤️ para Machine Learning em Produção | Developed with ❤️ for Production Machine Learning**

[![GitHub](https://img.shields.io/badge/GitHub-galafis-blue?style=flat-square&logo=github)](https://github.com/galafis)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

</div>

