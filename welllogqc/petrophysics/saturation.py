"""
saturation.py
-------------
Water saturation calculation routines using the Archie equation.
"""

import numpy as np
import pandas as pd


def calc_archie_sw(
    df: pd.DataFrame,
    rw: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    phi_col: str = "PHID"
) -> pd.Series:
    """Calculate water saturation (Sw) using the classic Archie equation.

    The Archie formula is:
        Sw = ((a * Rw) / (RT * phi^m))^(1/n)

    Values are clipped to the physical interval [0.0, 1.0]. Where RT or phi
    is non-positive or missing (NaN), the output is np.nan.

    Args:
        df: pandas DataFrame containing well log data, indexed by depth ('DEPT').
            Must contain the 'RT' column and the column specified by phi_col.
        rw: Formation water resistivity at formation temperature in ohm.m.
            Must be strictly positive.
        a: Tortuosity / lithology factor. Defaults to 1.0. Must be strictly positive.
        m: Cementation exponent. Defaults to 2.0. Must be strictly positive.
        n: Saturation exponent. Defaults to 2.0. Must be strictly positive.
        phi_col: Name of the porosity column to use. Defaults to 'PHID'.

    Returns:
        pd.Series: Water saturation named 'SW_ARCHIE' with values clipped to [0.0, 1.0],
            dtype float64, sharing the index of df.

    Raises:
        ValueError: If 'RT' is not present in df.columns.
        ValueError: If phi_col is not present in df.columns.
        ValueError: If rw <= 0.
        ValueError: If a <= 0, m <= 0, or n <= 0.
    """
    if "RT" not in df.columns:
        raise ValueError("Required curve 'RT' not found in DataFrame columns.")

    if phi_col not in df.columns:
        raise ValueError(
            f"Required porosity column '{phi_col}' not found in DataFrame columns."
        )

    if rw <= 0.0:
        raise ValueError(f"Formation water resistivity 'rw' ({rw}) must be strictly positive (> 0).")

    if a <= 0.0:
        raise ValueError(f"Tortuosity factor 'a' ({a}) must be strictly positive (> 0).")

    if m <= 0.0:
        raise ValueError(f"Cementation exponent 'm' ({m}) must be strictly positive (> 0).")

    if n <= 0.0:
        raise ValueError(f"Saturation exponent 'n' ({n}) must be strictly positive (> 0).")

    rt = df["RT"].astype(np.float64)
    phi = df[phi_col].astype(np.float64)

    sw = pd.Series(np.nan, index=df.index, dtype=np.float64, name="SW_ARCHIE")

    # Valid points require RT > 0 and phi > 0, non-null
    valid_mask = rt.notna() & phi.notna() & (rt > 0.0) & (phi > 0.0)
    if valid_mask.any():
        rt_valid = rt[valid_mask]
        phi_valid = phi[valid_mask]
        sw_val = ((a * rw) / (rt_valid * (phi_valid ** m))) ** (1.0 / n)
        sw.loc[valid_mask] = sw_val.clip(lower=0.0, upper=1.0).astype(np.float64)

    return sw
