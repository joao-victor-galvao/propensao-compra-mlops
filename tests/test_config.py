from pathlib import Path

import pytest
from pydantic import ValidationError

from purchase_propensity.config import Params, load_params


def test_load_params_reads_project_file() -> None:
    params = load_params(Path("params.yaml"))
    assert params.seed == 42
    assert params.registry.selection_metric.startswith("test_")
    assert params.train.models


def test_params_rejects_invalid_test_size() -> None:
    raw = load_params(Path("params.yaml")).model_dump()
    raw["data"]["test_size"] = 1.5
    with pytest.raises(ValidationError):
        Params.model_validate(raw)
