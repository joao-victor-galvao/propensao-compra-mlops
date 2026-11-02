"""Contratos de entrada e saída da API (validados pelo Pydantic)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.config import JsonDict

# Grafia igual à do dataset original ("June" é o único mês por extenso)
Month = Literal["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VisitorType = Literal["Returning_Visitor", "New_Visitor", "Other"]

SESSION_EXAMPLE: JsonDict = {
    "Administrative": 3,
    "Administrative_Duration": 80.5,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 42,
    "ProductRelated_Duration": 1650.3,
    "BounceRates": 0.004,
    "ExitRates": 0.018,
    "PageValues": 25.7,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False,
}


class Session(BaseModel):
    """Comportamento de navegação de uma sessão no e-commerce."""

    model_config = ConfigDict(json_schema_extra={"example": SESSION_EXAMPLE})

    Administrative: int = Field(ge=0, description="Páginas administrativas visitadas")
    Administrative_Duration: float = Field(ge=0, description="Segundos em páginas administrativas")
    Informational: int = Field(ge=0, description="Páginas informativas visitadas")
    Informational_Duration: float = Field(ge=0, description="Segundos em páginas informativas")
    ProductRelated: int = Field(ge=0, description="Páginas de produto visitadas")
    ProductRelated_Duration: float = Field(ge=0, description="Segundos em páginas de produto")
    BounceRates: float = Field(ge=0, le=1, description="Taxa de rejeição média das páginas")
    ExitRates: float = Field(ge=0, le=1, description="Taxa de saída média das páginas")
    PageValues: float = Field(ge=0, description="Valor médio das páginas antes da transação")
    SpecialDay: float = Field(ge=0, le=1, description="Proximidade de uma data comemorativa")
    Month: Month
    OperatingSystems: int = Field(ge=1)
    Browser: int = Field(ge=1)
    Region: int = Field(ge=1)
    TrafficType: int = Field(ge=1)
    VisitorType: VisitorType
    Weekend: bool


class BatchRequest(BaseModel):
    """Lote de sessões para predição."""

    sessions: list[Session] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    """Propensão de compra de uma sessão."""

    purchase_probability: float = Field(ge=0, le=1)
    will_purchase: bool
    threshold: float


class BatchPrediction(BaseModel):
    """Predições na mesma ordem das sessões enviadas."""

    predictions: list[Prediction]


class Health(BaseModel):
    """Estado da API e do modelo carregado."""

    status: Literal["ok"]
    model_name: str
    model_version: str
