"""Limpeza e divisão treino/teste do dataset bruto."""

import pandas as pd
from sklearn.model_selection import train_test_split

from purchase_propensity.data.schema import ALL_COLUMNS, BOOLEAN_FEATURES, TARGET


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas, mantém só as colunas do schema e converte booleanos para 0/1.

    Args:
        df: Dataset bruto já validado.

    Returns:
        Novo DataFrame limpo.
    """
    cleaned = df.loc[:, list(ALL_COLUMNS)].drop_duplicates().reset_index(drop=True)
    for column in (*BOOLEAN_FEATURES, TARGET):
        cleaned[column] = cleaned[column].astype(bool).astype(int)
    return cleaned


def split_dataset(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide em treino e teste preservando a proporção do alvo.

    Args:
        df: Dataset limpo.
        test_size: Fração destinada ao teste.
        seed: Semente para reprodutibilidade.

    Returns:
        Tupla ``(train, test)``.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])
    return train.reset_index(drop=True), test.reset_index(drop=True)
