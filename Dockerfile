# syntax=docker/dockerfile:1
# Imagem multi-stage:
#   builder  -> resolve dependências de produção com Poetry a partir do poetry.lock
#   runtime  -> base enxuta com o venv pronto, sem Poetry nem compiladores
#   pipeline -> roda `dvc repro` (treino) sobre o repositório montado em /app
#   api      -> serve o modelo campeão via FastAPI

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.5.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN pip install "poetry==${POETRY_VERSION}" && python -m venv /opt/venv

WORKDIR /app

# Camada de dependências: só é refeita quando pyproject/lock mudam
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root --no-ansi

COPY src ./src
RUN poetry install --only main --no-ansi


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN useradd --create-home --uid 1000 app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src ./src
COPY params.yaml ./


FROM runtime AS pipeline

# git é necessário para o DVC ler o repositório montado
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app

USER app
CMD ["dvc", "repro"]


FROM runtime AS api

COPY models/champion ./models/champion
USER app

ENV PORT=8000 \
    MODEL_PATH=models/champion
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"

CMD ["sh", "-c", "uvicorn purchase_propensity.api.main:app --host 0.0.0.0 --port ${PORT}"]
