# Multi-stage build para produção do ML Pipeline
# Baseado em python:3.11-slim para otimização de tamanho e segurança

# Stage 1: Base com dependências do sistema
FROM python:3.11-slim as base

# Metadados do container
LABEL maintainer="Gabriel Demetrios Lafis <gabrieldemetrios@gmail.com>" \
      description="ML Pipeline FastAPI production container" \
      version="1.0"

# Configurar variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Criar usuário não-root para segurança
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Instalar dependências do sistema essenciais
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
       && rm -rf /var/lib/apt/lists/* \
       && apt-get clean

# Stage 2: Dependências Python
FROM base as dependencies

# Definir diretório de trabalho
WORKDIR /app

# Copiar e instalar requirements primeiro (para cache de layers)
COPY requirements.txt .

# Upgrade pip e instalar dependências
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Stage 3: Aplicação final
FROM dependencies as production

# Copiar código fonte
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

# Criar diretórios necessários com permissões corretas
RUN mkdir -p /app/models /app/data /app/logs /app/reports \
    && chown -R appuser:appuser /app

# Configurar health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Mudar para usuário não-root
USER appuser

# Expor porta
EXPOSE 8000

# Configurar comando padrão com boas práticas para produção
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--access-log", \
     "--log-level", "info", \
     "--no-server-header"]
