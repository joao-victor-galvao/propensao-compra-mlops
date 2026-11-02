import numpy as np
import pandas as pd
import pytest

from purchase_propensity.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

N_ROWS = 200


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Amostra sintética com o mesmo schema do dataset real (formato do CSV bruto)."""
    rng = np.random.default_rng(0)
    data: dict[str, object] = {col: rng.random(N_ROWS) * 10 for col in NUMERIC_FEATURES}
    categories = {
        "Month": ["Feb", "Mar", "May", "Nov"],
        "OperatingSystems": [1, 2, 3],
        "Browser": [1, 2],
        "Region": [1, 3, 9],
        "TrafficType": [1, 2, 13],
        "VisitorType": ["Returning_Visitor", "New_Visitor", "Other"],
    }
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice(categories[col], N_ROWS)
    data["Weekend"] = rng.choice([True, False], N_ROWS)
    data["Revenue"] = rng.choice([True, False], N_ROWS, p=[0.3, 0.7])
    return pd.DataFrame(data)
