"""
tests/conftest.py
-----------------
Shared pytest fixtures available to every test module in this suite.

ENGINEERING NOTE
----------------
All fixtures generate SYNTHETIC data clearly labelled as such.
They must never be used as proof of real-world model performance.
"""

import pytest
import numpy as np
import lasio
import sys
import pandas as pd
from pathlib import Path

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_TRAINING = PROJECT_ROOT / "data" / "raw" / "training"


@pytest.fixture(scope="session")
def synthetic_las_path():
    """Return the path to WELL_A.las if it exists, else skip the test."""
    p = DATA_TRAINING / "WELL_A.las"
    if not p.exists():
        pytest.skip(
            "WELL_A.las not found. Run `python create_synthetic_las.py` first."
        )
    return p


@pytest.fixture(scope="session")
def small_depth():
    """A tiny 100-sample depth array at 0.5 ft spacing."""
    return np.arange(8000.0, 8050.0, 0.5)


@pytest.fixture(scope="session")
def clean_df(small_depth):
    """
    A minimal clean DataFrame with GR, RHOB, NPHI indexed by depth.
    Represents a simple clean-sand interval — no defects.
    """
    n = len(small_depth)
    rng = np.random.default_rng(seed=0)
    return pd.DataFrame({
        "DEPT": small_depth,
        "GR":   np.clip(rng.normal(25, 4, n), 0, 300),
        "RHOB": np.clip(rng.normal(2.30, 0.02, n), 1.0, 3.5),
        "NPHI": np.clip(rng.normal(0.18, 0.01, n), -0.1, 0.6),
    }).set_index("DEPT")
