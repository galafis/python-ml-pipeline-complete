# Notebooks - Análise Exploratória e Exemplos

Este diretório contém Jupyter notebooks com análises exploratórias de dados (EDA) e exemplos práticos do pipeline de Machine Learning.

## 📊 Conteúdo

### Análise Exploratória de Dados (EDA)
- **Data profiling**: Estatísticas descritivas e qualidade dos dados
- **Visualizações**: Distribuições, correlações e padrões
- **Feature analysis**: Análise de importância e engenharia de features
- **Data validation**: Verificação de consistência e outliers

### Exemplos Práticos
- **Model training**: Treinamento de diferentes algoritmos
- **Hyperparameter tuning**: Otimização com Optuna
- **Model evaluation**: Métricas e validação cruzada
- **MLflow integration**: Tracking de experimentos

## 🚀 Como Usar

1. **Instalação das dependências**:
   ```bash
   pip install jupyter
   pip install -r requirements.txt
   ```

2. **Iniciar Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Executar notebooks em ordem**:
   - `01_data_exploration.ipynb` - EDA inicial
   - `02_feature_engineering.ipynb` - Engenharia de features
   - `03_model_training.ipynb` - Treinamento de modelos
   - `04_model_evaluation.ipynb` - Avaliação e comparação

## 📚 Referências e Links Úteis

### Documentação Oficial
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)

### Tutoriais e Exemplos
- [Kaggle Learn](https://www.kaggle.com/learn) - Cursos práticos de ML
- [Google AI Education](https://ai.google/education/) - Recursos educacionais
- [Fast.ai](https://www.fast.ai/) - Curso prático de deep learning

### Datasets de Exemplo
- [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [OpenML](https://www.openml.org/)

### Boas Práticas
- [ML Engineering Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Data Science Project Structure](https://drivendata.github.io/cookiecutter-data-science/)
- [MLOps Principles](https://ml-ops.org/)

## 🔧 Configuração do Ambiente

### Variáveis de Ambiente
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Jupyter Extensions Recomendadas
```bash
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

---

**Nota**: Certifique-se de que o servidor MLflow esteja rodando antes de executar os notebooks que fazem tracking de experimentos.
