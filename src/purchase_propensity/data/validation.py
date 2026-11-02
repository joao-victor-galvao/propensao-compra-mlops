"""Validação do dataset bruto: falha cedo se o contrato de colunas for quebrado."""

import pandas as pd

from purchase_propensity.data.schema import ALL_COLUMNS, NUMERIC_FEATURES, TARGET


class DataValidationError(ValueError):
    """O dataset não respeita o schema esperado."""


def validate_raw(df: pd.DataFrame) -> None:
    """Valida colunas, tipos numéricos e ausência de nulos no alvo.

    Args:
        df: Dataset bruto.

    Raises:
        DataValidationError: Se alguma regra do schema for violada.
    """
    _check_required_columns(df)
    _check_numeric_columns(df)
    _check_target(df)


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(ALL_COLUMNS) - set(df.columns))
    if missing:
        raise DataValidationError(f"Colunas ausentes: {missing}")


def _check_numeric_columns(df: pd.DataFrame) -> None:
    non_numeric = [col for col in NUMERIC_FEATURES if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise DataValidationError(f"Colunas que deveriam ser numéricas: {non_numeric}")


def _check_target(df: pd.DataFrame) -> None:
    if df[TARGET].isna().any():
        raise DataValidationError(f"A coluna alvo '{TARGET}' possui valores nulos")
    if df[TARGET].nunique() != 2:  # noqa: PLR2004 - classificação binária
        raise DataValidationError(f"A coluna alvo '{TARGET}' deve ser binária")
