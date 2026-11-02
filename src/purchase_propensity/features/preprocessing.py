"""Transformações de features aplicadas dentro do Pipeline do Scikit-Learn."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from purchase_propensity.data.schema import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """Cria o ColumnTransformer: escala numéricas e aplica one-hot nas categóricas.

    Ficar dentro do Pipeline garante que o mesmo pré-processamento seja salvo
    junto do modelo e reaplicado na inferência, sem vazamento de dados do teste.

    Returns:
        ColumnTransformer ainda não ajustado.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            ("categorical", _build_one_hot_encoder(), list(CATEGORICAL_FEATURES)),
            ("boolean", "passthrough", list(BOOLEAN_FEATURES)),
        ],
        verbose_feature_names_out=False,
    )


def _build_one_hot_encoder() -> OneHotEncoder:
    # Categorias novas na inferência (ex.: navegador nunca visto) viram vetor zerado.
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
