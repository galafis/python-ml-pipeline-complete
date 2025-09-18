"""
main.py
-------
Ponto de entrada principal do pipeline de Machine Learning.
Utiliza argumentos de linha de comando para configurar e executar o pipeline completo.

Exemplo de uso:
    python main.py --data data.csv --model logistic --output modelo.joblib
    python main.py --data data.csv --model random_forest --test-size 0.3 --random-state 42
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Adiciona o diretório src ao path para importações
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import MLPipeline  # noqa: E402


def get_model(model_name: str, **kwargs) -> Any:
    """
    Retorna o modelo especificado com os parâmetros fornecidos.

    Args:
        model_name (str): Nome do modelo ('logistic', 'random_forest', 'svm')
        **kwargs: Parâmetros adicionais para o modelo

    Returns:
        Any: Instância do modelo scikit-learn

    Raises:
        ValueError: Se o modelo especificado não for suportado
    """
    models = {
        'logistic': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'svm': SVC
    }

    if model_name not in models:
        raise ValueError(f"Modelo '{model_name}' não suportado. Opções: {list(models.keys())}")

    # Filtra apenas os parâmetros relevantes para cada modelo
    if model_name == 'logistic':
        valid_params = ['random_state', 'max_iter', 'C']
    elif model_name == 'random_forest':
        valid_params = ['n_estimators', 'random_state', 'max_depth']
    elif model_name == 'svm':
        valid_params = ['C', 'kernel', 'random_state']

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return models[model_name](**filtered_kwargs)


def create_parser() -> argparse.ArgumentParser:
    """
    Cria e configura o parser de argumentos de linha de comando.

    Returns:
        argparse.ArgumentParser: Parser configurado
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de Machine Learning - Treinamento e Avaliação",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exemplos de uso:
  %(prog)s --data dataset.csv --model logistic --output modelo.joblib
  %(prog)s --data dataset.csv --model random_forest --test-size 0.2 --n-estimators 100
  %(prog)s --data dataset.csv --model svm --C 1.0 --kernel rbf --random-state 42
        """
    )

    # Argumentos obrigatórios
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help="Caminho para o arquivo CSV de dados"
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['logistic', 'random_forest', 'svm'],
        required=True,
        help="Tipo de modelo a ser usado"
    )

    # Argumentos opcionais
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='model.joblib',
        help="Caminho para salvar o modelo treinado (padrão: model.joblib)"
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help="Proporção dos dados para teste (padrão: 0.2)"
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help="Seed para reprodutibilidade (padrão: 42)"
    )

    # Parâmetros específicos dos modelos
    parser.add_argument(
        '--max-iter',
        type=int,
        default=1000,
        help="Número máximo de iterações (LogisticRegression)"
    )

    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help="Parâmetro de regularização (LogisticRegression/SVM)"
    )

    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help="Número de árvores (RandomForestClassifier)"
    )

    parser.add_argument(
        '--max-depth',
        type=int,
        help="Profundidade máxima das árvores (RandomForestClassifier)"
    )

    parser.add_argument(
        '--kernel',
        type=str,
        default='rbf',
        choices=['linear', 'poly', 'rbf', 'sigmoid'],
        help="Kernel do SVM (padrão: rbf)"
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Modo verboso - exibe informações detalhadas"
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Valida os argumentos fornecidos.

    Args:
        args (argparse.Namespace): Argumentos parseados

    Raises:
        FileNotFoundError: Se o arquivo de dados não existir
        ValueError: Se os argumentos forem inválidos
    """
    # Verifica se o arquivo de dados existe
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {args.data}")

    # Verifica se test_size está no intervalo válido
    if not 0 < args.test_size < 1:
        raise ValueError("test-size deve estar entre 0 e 1")

    # Cria o diretório de saída se não existir
    output_dir = Path(args.output).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)


def main():
    """
    Função principal que executa o pipeline completo de ML.

    Carrega os dados, treina o modelo especificado, avalia o desempenho
    e salva o modelo treinado conforme os argumentos fornecidos.
    """
    # Parse dos argumentos
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Validação dos argumentos
        validate_arguments(args)

        if args.verbose:
            print("Iniciando pipeline de ML...")
            print(f"Dados: {args.data}")
            print(f"Modelo: {args.model}")
            print(f"Saída: {args.output}")
            print(f"Test size: {args.test_size}")
            print(f"Random state: {args.random_state}")
            print("-" * 50)

        # Preparação dos parâmetros do modelo
        model_params = {
            'random_state': args.random_state,
            'max_iter': args.max_iter,
            'C': args.C,
            'n_estimators': args.n_estimators,
            'kernel': args.kernel
        }

        if args.max_depth is not None:
            model_params['max_depth'] = args.max_depth

        # Criação do modelo
        estimator = get_model(args.model, **model_params)

        if args.verbose:
            print(f"Modelo criado: {estimator.__class__.__name__}")
            print(f"Parâmetros: {estimator.get_params()}")

        # Inicialização do pipeline
        pipeline = MLPipeline(estimator)

        # Carregamento dos dados
        if args.verbose:
            print("Carregando dados...")

        data = pipeline.load_data(args.data)

        # Assumindo que a última coluna é o target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        if args.verbose:
            print(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")
            print(f"Classes únicas: {sorted(y.unique())}")

        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y if len(y.unique()) > 1 else None
        )

        if args.verbose:
            print(f"Divisão treino/teste: {len(X_train)}/{len(X_test)} amostras")

        # Pré-processamento
        if args.verbose:
            print("Aplicando feature engineering...")

        X_train_processed = pipeline.preprocess_data(X_train)
        X_test_processed = pipeline.feature_engineer.transform(X_test)

        # Treinamento
        if args.verbose:
            print("Treinando modelo...")

        model = pipeline.train_model(X_train_processed, y_train)

        # Avaliação
        if args.verbose:
            print("Avaliando modelo...")

        train_metrics = pipeline.evaluate_model(X_train_processed, y_train)
        test_metrics = pipeline.evaluate_model(X_test_processed, y_test)

        # Exibição dos resultados
        print("\n=== RESULTADOS ===")
        print(f"Modelo: {model.__class__.__name__}")
        print(f"Dados de treino: {len(X_train)} amostras")
        print(f"Dados de teste: {len(X_test)} amostras")

        print("\nMétricas de Treino:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print("\nMétricas de Teste:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Salvamento do modelo
        if args.verbose:
            print(f"\nSalvando modelo em: {args.output}")

        pipeline.save_artifacts(args.output)

        print("\n✓ Pipeline executado com sucesso!")
        print(f"✓ Modelo salvo em: {args.output}")

    except Exception as e:
        print(f"❌ Erro durante a execução: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
