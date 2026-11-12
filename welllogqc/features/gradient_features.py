"""
gradient_features.py
--------------------
Depth gradient feature extraction for well logs.
"""

import numpy as np
import pandas as pd


def add_gradients(
    df: pd.DataFrame,
    curves: list[str] | None = None
) -> pd.DataFrame:
    """Add first derivative (depth gradient) and absolute gradient for curves.

    For each target curve, adds:
      - {CURVE}_GRAD: (first difference) / depth_step
      - {CURVE}_ABS_GRAD: absolute value of {CURVE}_GRAD

    Depth step is estimated as the median of absolute adjacent depth differences:
    median(abs(diff(index))). If the DataFrame has fewer than 2 rows or the step
    is non-positive, a default depth step of 1.0 is used.
    All engineered columns are float64.

    Args:
        df: pandas DataFrame containing well log data, indexed by depth ('DEPT').
        curves: Optional list of curve names to compute gradients for. If None,
            defaults to all numeric columns present in df.

    Returns:
        pd.DataFrame: A copy of df containing original columns plus {CURVE}_GRAD
            and {CURVE}_ABS_GRAD for each target curve.

    Raises:
        ValueError: If any specified curve is not present in df.columns.
    """
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

    if len(df) > 1:
        depth_series = pd.Series(df.index, index=df.index).astype(np.float64)
        diffs = depth_series.diff().abs().dropna()
        if len(diffs) > 0 and float(diffs.median()) > 0.0:
            depth_step = float(diffs.median())
        else:
            depth_step = 1.0
    else:
        depth_step = 1.0

    for c in target_curves:
        s = df[c].astype(np.float64)
        grad = (s.diff() / depth_step).astype(np.float64)
        abs_grad = grad.abs().astype(np.float64)
        out[f"{c}_GRAD"] = grad
        out[f"{c}_ABS_GRAD"] = abs_grad

    return out
