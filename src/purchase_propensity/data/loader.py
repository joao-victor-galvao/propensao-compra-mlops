"""Leitura e escrita dos dados em disco."""

from pathlib import Path

import pandas as pd

from purchase_propensity.data.schema import TARGET


def load_raw(path: Path) -> pd.DataFrame:
    """Lê o CSV bruto do dataset.

    Args:
        path: Caminho do arquivo CSV.

    Returns:
        DataFrame com as colunas originais.
    """
    return pd.read_csv(path)


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Salva um DataFrame em parquet, criando a pasta de destino se preciso."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_features_and_target(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Lê um split processado e separa as features do alvo.

    Args:
        path: Caminho do parquet gerado pelo estágio de pré-processamento.

    Returns:
        Tupla ``(X, y)``.
    """
    df = pd.read_parquet(path)
    return df.drop(columns=[TARGET]), df[TARGET]
