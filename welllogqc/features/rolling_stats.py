"""
rolling_stats.py
----------------
Rolling window statistics calculation for well log features.
"""

import numpy as np
import pandas as pd

from welllogqc.config.defaults import SPIKE_WINDOW


def add_rolling_stats(
    df: pd.DataFrame,
    curves: list[str] | None = None,
    window: int = SPIKE_WINDOW,
    center: bool = True
) -> pd.DataFrame:
    """Add rolling mean, standard deviation, and variance for specified curves.

    For each target curve, adds:
      - {CURVE}_ROLL_MEAN: Rolling average over the specified window
      - {CURVE}_ROLL_STD: Rolling standard deviation
      - {CURVE}_ROLL_VAR: Rolling sample variance

    Uses min_periods=1 to maintain coverage across window boundaries and around gaps.
    All engineered columns are typed as float64.

    Args:
        df: pandas DataFrame containing well log data, indexed by depth ('DEPT').
        curves: Optional list of curve column names to process. If None, defaults
            to all numeric columns present in df.
        window: Number of samples in the rolling window. Defaults to 11 (odd integer).
        center: Whether to set the window label at the center of the window.
            Defaults to True.

    Returns:
        pd.DataFrame: A copy of df containing original columns plus the new rolling
            statistics columns.

    Raises:
        ValueError: If window is not a positive integer.
        ValueError: If any curve in the curves list is not present in df.columns.
    """
    if window <= 0:
        raise ValueError(f"window ({window}) must be a positive integer.")

    if curves is None:
        target_curves = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        for c in curves:
            if c not in df.columns:
                raise ValueError(f"Required curve '{c}' not found in DataFrame columns.")
        target_curves = list(curves)

    out = df.copy()

    for c in target_curves:
        s = df[c].astype(np.float64)
        roll = s.rolling(window=window, min_periods=1, center=center)
        out[f"{c}_ROLL_MEAN"] = roll.mean().astype(np.float64)
        out[f"{c}_ROLL_STD"] = roll.std().astype(np.float64)
        out[f"{c}_ROLL_VAR"] = roll.var().astype(np.float64)

    return out
