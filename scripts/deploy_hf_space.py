"""Publica a API de inferência no Hugging Face Spaces (SDK Docker, plano gratuito).

Uso:
    poetry run python scripts/deploy_hf_space.py

Requer HF_USER e HF_TOKEN (permissão Write) no .env e o modelo em models/champion
(gerado por `dvc repro` ou baixado com `dvc pull`). O Space usa o mesmo Dockerfile do
projeto: o alvo padrão é o último estágio (`api`).
"""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from pydantic_settings import BaseSettings, SettingsConfigDict

SPACE_NAME = "propensao-compra-api"
PROJECT_FILES = ("Dockerfile", "pyproject.toml", "poetry.lock", "params.yaml")
PROJECT_DIRS = ("src", "models/champion")
SPACE_CARD = """---
title: Propensão de Compra
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
short_description: Probabilidade de uma sessão de e-commerce terminar em compra
---

# Propensão de Compra — API

API FastAPI do Tech Challenge Fase 2 (FIAP 11MLET). Serve o modelo `champion` do MLflow
Model Registry. Documentação interativa em `/docs`.

Código, pipeline DVC e experimentos: https://github.com/joao-victor-galvao/propensao-compra-mlops
"""


class DeploySettings(BaseSettings):
    """Credenciais do Hugging Face lidas do .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hf_user: str
    hf_token: str


def stage_space_files(destination: Path) -> None:
    """Copia para ``destination`` só o necessário para construir a imagem da API."""
    for file in PROJECT_FILES:
        shutil.copy2(file, destination / file)
    for directory in PROJECT_DIRS:
        shutil.copytree(
            directory, destination / directory, ignore=shutil.ignore_patterns("__pycache__")
        )
    (destination / "README.md").write_text(SPACE_CARD, encoding="utf-8")


def main() -> None:
    """Cria o Space (se não existir) e envia os arquivos; o HF faz o build da imagem."""
    if not Path("models/champion/MLmodel").exists():
        raise SystemExit("models/champion não encontrado: rode `dvc repro` ou `dvc pull` antes.")
    settings = DeploySettings()  # type: ignore[call-arg]
    api = HfApi(token=settings.hf_token)
    repo_id = f"{settings.hf_user}/{SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        stage_space_files(Path(staging))
        api.upload_folder(
            folder_path=staging,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy da API de propensão de compra",
        )
    print(f"Space publicado: https://huggingface.co/spaces/{repo_id}")
    print(f"API: https://{settings.hf_user}-{SPACE_NAME}.hf.space/docs")


if __name__ == "__main__":
    main()
