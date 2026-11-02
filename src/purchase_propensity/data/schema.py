"""Contrato de colunas do dataset Online Shoppers Purchasing Intention."""

from typing import Final

TARGET: Final = "Revenue"

NUMERIC_FEATURES: Final = (
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
)

# Códigos inteiros (sistema operacional, navegador...) não têm ordem: tratados como categorias.
CATEGORICAL_FEATURES: Final = (
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
)

BOOLEAN_FEATURES: Final = ("Weekend",)

FEATURES: Final = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
ALL_COLUMNS: Final = (*FEATURES, TARGET)
