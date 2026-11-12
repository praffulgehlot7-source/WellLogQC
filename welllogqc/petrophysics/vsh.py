"""
vsh.py
------
Shale volume (Vsh) calculation routines.
"""

import numpy as np
import pandas as pd

from welllogqc.config.defaults import GR_CLEAN, GR_SHALE


def calc_vsh_linear(
    df: pd.DataFrame,
    gr_clean: float = GR_CLEAN,
    gr_shale: float = GR_SHALE
) -> pd.Series:
    """Calculate the shale volume using the linear Index of Gamma Ray (IGR) method.

    IGR is computed as:
        IGR = (GR - gr_clean) / (gr_shale - gr_clean)
        Vsh = clip(IGR, 0.0, 1.0)

    Args:
        df: pandas DataFrame containing well log data, indexed by depth ('DEPT').
            Must contain the 'GR' (Gamma Ray) column.
        gr_clean: Clean sand/carbonate gamma-ray baseline value in GAPI. Defaults to 15.0.
        gr_shale: Shale gamma-ray baseline value in GAPI. Defaults to 120.0.

    Returns:
        pd.Series: Shale volume index named 'VSH_LINEAR' with values in [0.0, 1.0],
            dtype float64, sharing the index of df. Contains np.nan where GR is np.nan.

    Raises:
        ValueError: If 'GR' is not present in df.columns.
        ValueError: If gr_clean >= gr_shale.
    """
    if "GR" not in df.columns:
        raise ValueError("Required curve 'GR' not found in DataFrame columns.")

    if gr_clean >= gr_shale:
        raise ValueError(
            f"gr_clean ({gr_clean}) must be strictly less than gr_shale ({gr_shale})."
        )

    gr = df["GR"].astype(np.float64)
    igr = (gr - float(gr_clean)) / float(gr_shale - gr_clean)
    vsh = igr.clip(lower=0.0, upper=1.0).astype(np.float64)
    vsh.name = "VSH_LINEAR"
    return vsh
