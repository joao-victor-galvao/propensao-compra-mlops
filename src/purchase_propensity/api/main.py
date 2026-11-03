"""API de inferência: expõe o modelo campeão do Model Registry via HTTP."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from purchase_propensity import __version__
from purchase_propensity.api.predictor import PurchasePredictor
from purchase_propensity.api.schemas import (
    BatchPrediction,
    BatchRequest,
    Health,
    Prediction,
    Session,
)
from purchase_propensity.config import Settings


def create_app(model_path: Path | None = None, threshold: float | None = None) -> FastAPI:
    """Cria a aplicação; o modelo é carregado uma única vez na inicialização.

    Args:
        model_path: Pasta do modelo exportado. Padrão: ``MODEL_PATH`` do ambiente.
        threshold: Limiar de decisão. Padrão: ``DECISION_THRESHOLD`` do ambiente.

    Returns:
        Aplicação FastAPI configurada.
    """
    settings = Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = PurchasePredictor.from_path(
            model_path or settings.model_path,
            threshold if threshold is not None else settings.decision_threshold,
        )
        yield

    app = FastAPI(
        title="Propensão de Compra",
        description="Probabilidade de uma sessão de navegação terminar em compra.",
        version=__version__,
        lifespan=lifespan,
    )
    _register_routes(app)
    return app


def _predictor(request: Request) -> PurchasePredictor:
    predictor: PurchasePredictor = request.app.state.predictor
    return predictor


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", tags=["infra"])
    def health(request: Request) -> Health:
        """Confirma que a API está no ar e informa qual versão do modelo está carregada."""
        predictor = _predictor(request)
        return Health(
            status="ok", model_name=predictor.model_name, model_version=predictor.model_version
        )

    @app.post("/predict", tags=["predição"])
    def predict(session: Session, request: Request) -> Prediction:
        """Retorna a probabilidade de a sessão terminar em compra e a decisão no limiar."""
        return _predictor(request).predict([session])[0]

    @app.post("/predict/batch", tags=["predição"])
    def predict_batch(batch: BatchRequest, request: Request) -> BatchPrediction:
        """Prevê várias sessões de uma vez (até 1.000), na mesma ordem do envio."""
        return BatchPrediction(predictions=_predictor(request).predict(batch.sessions))


app = create_app()
