# 🤖 Python Ml Pipeline Complete

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-1.4-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#english) | [Português](#português)

---

## English

### 🎯 Overview

**Python Ml Pipeline Complete** — Data Science project - python-ml-pipeline-complete

Total source lines: **1,812** across **15** files in **2** languages.

### ✨ Key Features

- **Production-Ready Architecture**: Modular, well-documented, and following best practices
- **Comprehensive Implementation**: Complete solution with all core functionality
- **Clean Code**: Type-safe, well-tested, and maintainable codebase
- **Easy Deployment**: Docker support for quick setup and deployment

### 🚀 Quick Start

#### Prerequisites
- Python 3.12+
- Docker and Docker Compose (optional)

#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/galafis/python-ml-pipeline-complete.git
cd python-ml-pipeline-complete
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### Running

```bash
python src/main.py
```

## 🐳 Docker

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Project Structure

```
python-ml-pipeline-complete/
├── config/
│   ├── config.yaml
│   └── model_config.yaml
├── data/
│   ├── features/
│   ├── processed/
│   └── raw/
├── docker/
│   └── docker-compose.yml
├── docs/
├── k8s/
│   └── deployment.yaml
├── models/
├── notebooks/
│   └── README.md
├── reports/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── __init__.py
│   ├── app_dashboard.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── main.py
│   ├── model_evaluator.py
│   ├── model_trainer.py
│   ├── monitoring.py
│   └── pipeline.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_integration.py
│   └── test_pipeline.py
├── CONTRIBUTING.md
├── README.md
└── requirements.txt
```

### 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | 14 files |
| HTML | 1 files |

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)

---

## Português

### 🎯 Visão Geral

**Python Ml Pipeline Complete** — Data Science project - python-ml-pipeline-complete

Total de linhas de código: **1,812** em **15** arquivos em **2** linguagens.

### ✨ Funcionalidades Principais

- **Arquitetura Pronta para Produção**: Modular, bem documentada e seguindo boas práticas
- **Implementação Completa**: Solução completa com todas as funcionalidades principais
- **Código Limpo**: Type-safe, bem testado e manutenível
- **Fácil Implantação**: Suporte Docker para configuração e implantação rápidas

### 🚀 Início Rápido

#### Pré-requisitos
- Python 3.12+
- Docker e Docker Compose (opcional)

#### Instalação

1. **Clone the repository**
```bash
git clone https://github.com/galafis/python-ml-pipeline-complete.git
cd python-ml-pipeline-complete
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### Execução

```bash
python src/main.py
```

### 🧪 Testes

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Estrutura do Projeto

```
python-ml-pipeline-complete/
├── config/
│   ├── config.yaml
│   └── model_config.yaml
├── data/
│   ├── features/
│   ├── processed/
│   └── raw/
├── docker/
│   └── docker-compose.yml
├── docs/
├── k8s/
│   └── deployment.yaml
├── models/
├── notebooks/
│   └── README.md
├── reports/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── __init__.py
│   ├── app_dashboard.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── main.py
│   ├── model_evaluator.py
│   ├── model_trainer.py
│   ├── monitoring.py
│   └── pipeline.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_integration.py
│   └── test_pipeline.py
├── CONTRIBUTING.md
├── README.md
└── requirements.txt
```

### 🛠️ Stack Tecnológica

| Tecnologia | Uso |
|------------|-----|
| Python | 14 files |
| HTML | 1 files |

### 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)
