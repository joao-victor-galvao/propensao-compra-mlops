import pandas as pd
import pytest

from purchase_propensity.data.loader import load_features_and_target, save_dataset
from purchase_propensity.data.preparation import clean_dataset, split_dataset
from purchase_propensity.data.schema import FEATURES, TARGET
from purchase_propensity.data.validation import DataValidationError, validate_raw


def test_validate_raw_accepts_valid_dataset(raw_df: pd.DataFrame) -> None:
    validate_raw(raw_df)


def test_validate_raw_rejects_missing_column(raw_df: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="PageValues"):
        validate_raw(raw_df.drop(columns=["PageValues"]))


def test_validate_raw_rejects_null_target(raw_df: pd.DataFrame) -> None:
    raw_df[TARGET] = raw_df[TARGET].astype(object)
    raw_df.loc[0, TARGET] = None
    with pytest.raises(DataValidationError, match="nulos"):
        validate_raw(raw_df)


def test_clean_dataset_converts_booleans_and_drops_duplicates(raw_df: pd.DataFrame) -> None:
    duplicated = pd.concat([raw_df, raw_df.head(5)])
    cleaned = clean_dataset(duplicated)
    assert len(cleaned) == len(raw_df)
    assert set(cleaned[TARGET].unique()) <= {0, 1}
    assert set(cleaned["Weekend"].unique()) <= {0, 1}


def test_split_dataset_is_stratified_and_reproducible(raw_df: pd.DataFrame) -> None:
    cleaned = clean_dataset(raw_df)
    train_a, test_a = split_dataset(cleaned, test_size=0.25, seed=42)
    train_b, _ = split_dataset(cleaned, test_size=0.25, seed=42)
    assert len(test_a) == len(cleaned) // 4
    assert train_a.equals(train_b)
    assert abs(train_a[TARGET].mean() - test_a[TARGET].mean()) < 0.05


def test_save_and_load_roundtrip(raw_df: pd.DataFrame, tmp_path) -> None:
    path = tmp_path / "nested" / "train.parquet"
    save_dataset(clean_dataset(raw_df), path)
    features, target = load_features_and_target(path)
    assert list(features.columns) == list(FEATURES)
    assert target.name == TARGET
