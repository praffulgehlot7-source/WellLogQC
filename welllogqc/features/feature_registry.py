"""
feature_registry.py
-------------------
Central registry of well log features and feature matrix builder.
"""

from typing import Any
import pandas as pd

from welllogqc.config.defaults import SPIKE_WINDOW
from welllogqc.features.derived_logs import add_derived_features
from welllogqc.features.rolling_stats import add_rolling_stats
from welllogqc.features.gradient_features import add_gradients

# Base canonical curves with standard units and descriptions
_CANONICAL_LOGS: dict[str, tuple[str, str]] = {
    "GR": ("GAPI", "Gamma Ray log"),
    "CALI": ("in", "Caliper borehole diameter"),
    "BS": ("in", "Nominal drill bit size"),
    "RHOB": ("g/cm3", "Bulk density log"),
    "NPHI": ("v/v", "Neutron porosity log"),
    "RT": ("ohm.m", "True formation resistivity (deep)"),
    "RS": ("ohm.m", "Shallow formation resistivity"),
    "DT": ("us/ft", "Sonic acoustic slowness"),
    "PEF": ("b/e", "Photoelectric absorption factor"),
}

# Construct the central feature registry
FEATURE_REGISTRY: dict[str, dict[str, str]] = {}

# 1. Original canonical curves
for curve_name, (unit, desc) in _CANONICAL_LOGS.items():
    FEATURE_REGISTRY[curve_name] = {
        "source": "original",
        "unit": unit,
        "description": desc,
    }

# 2. Rolling statistics for each canonical curve
for curve_name, (unit, desc) in _CANONICAL_LOGS.items():
    FEATURE_REGISTRY[f"{curve_name}_ROLL_MEAN"] = {
        "source": "rolling",
        "unit": unit,
        "description": f"Rolling mean of {curve_name}",
    }
    FEATURE_REGISTRY[f"{curve_name}_ROLL_STD"] = {
        "source": "rolling",
        "unit": unit,
        "description": f"Rolling standard deviation of {curve_name}",
    }
    FEATURE_REGISTRY[f"{curve_name}_ROLL_VAR"] = {
        "source": "rolling",
        "unit": f"({unit})^2",
        "description": f"Rolling sample variance of {curve_name}",
    }

# 3. Depth gradients for each canonical curve
for curve_name, (unit, desc) in _CANONICAL_LOGS.items():
    FEATURE_REGISTRY[f"{curve_name}_GRAD"] = {
        "source": "gradient",
        "unit": f"{unit}/ft",
        "description": f"Depth gradient (first difference / depth step) of {curve_name}",
    }
    FEATURE_REGISTRY[f"{curve_name}_ABS_GRAD"] = {
        "source": "gradient",
        "unit": f"{unit}/ft",
        "description": f"Absolute depth gradient of {curve_name}",
    }

# 4. Derived petrophysical logs and missing indicators
FEATURE_REGISTRY["CALI_DEV"] = {
    "source": "derived",
    "unit": "in",
    "description": "Caliper deviation from bit size (CALI - BS)",
}
FEATURE_REGISTRY["DN_SEP"] = {
    "source": "derived",
    "unit": "v/v",
    "description": "Apparent density-neutron porosity separation",
}

for curve_name in ["GR", "CALI", "BS", "RHOB", "NPHI", "RT", "DT", "PEF"]:
    FEATURE_REGISTRY[f"MISSING_{curve_name}"] = {
        "source": "derived",
        "unit": "binary",
        "description": f"Missing value indicator flag for {curve_name} (1.0=missing, 0.0=present)",
    }

FEATURE_REGISTRY["VSH_LINEAR"] = {
    "source": "derived",
    "unit": "v/v",
    "description": "Linear shale volume calculated from gamma ray",
}
FEATURE_REGISTRY["PHID"] = {
    "source": "derived",
    "unit": "v/v",
    "description": "Density porosity calculated from bulk density",
}
FEATURE_REGISTRY["SW_ARCHIE"] = {
    "source": "derived",
    "unit": "v/v",
    "description": "Water saturation calculated via Archie equation",
}


def build_feature_matrix(
    df: pd.DataFrame,
    window: int = SPIKE_WINDOW
) -> pd.DataFrame:
    """Build the full feature engineering matrix from input well log curves.

    Applies the sequential transformation pipeline:
        1. add_derived_features: adds CALI_DEV, DN_SEP, and MISSING_* flags
        2. add_rolling_stats: adds rolling mean, std, and var for curves present in df
        3. add_gradients: adds depth gradient and absolute gradient for curves present in df

    Only calculates rolling and gradient features for numeric curves actually
    present in the input DataFrame.

    Args:
        df: pandas DataFrame containing raw or preprocessed well log data,
            indexed by depth ('DEPT').
        window: Window size (in samples) for rolling window operations. Defaults to 11.

    Returns:
        pd.DataFrame: Complete feature matrix containing original columns and
            all engineered features as float64.
    """
    # Track curves present in the original input DataFrame
    original_curves = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]

    # Step 1: Add derived petrophysical logs and missing flags
    feat_df = add_derived_features(df)

    # Step 2: Add rolling statistics only for curves originally present
    feat_df = add_rolling_stats(feat_df, curves=original_curves, window=window)

    # Step 3: Add depth gradients only for curves originally present
    feat_df = add_gradients(feat_df, curves=original_curves)

    return feat_df
