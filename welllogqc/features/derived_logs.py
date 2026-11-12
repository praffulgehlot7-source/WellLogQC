"""
derived_logs.py
---------------
Derived petrophysical indicators and missing-curve flag engineering.
"""

import numpy as np
import pandas as pd

from welllogqc.config.defaults import RHOMAT, RHOFLUID

TARGET_MISSING_CURVES: list[str] = [
    "GR", "CALI", "BS", "RHOB", "NPHI", "RT", "DT", "PEF"
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived domain features and missing-curve binary indicators.

    Calculates:
      - CALI_DEV: CALI - BS (borehole enlargement / washout). If BS is absent,
        column is created with np.nan. Added if CALI is present in df.
      - DN_SEP: NPHI - (2.65 - RHOB) / (2.65 - 1.0) (apparent density-neutron
        porosity crossover / separation). Set to np.nan if either NPHI or RHOB
        is absent from df. Added if either source curve is present.
      - MISSING_{CURVE}: Binary float64 indicator (1.0 = missing/NaN, 0.0 = present)
        for each of the 8 canonical curves: GR, CALI, BS, RHOB, NPHI, RT, DT, PEF.

    Args:
        df: pandas DataFrame containing well log data, indexed by depth ('DEPT').

    Returns:
        pd.DataFrame: A copy of df containing all original columns plus engineered
            derived features and missing indicators as float64.
    """
    out = df.copy()

    # 1. Caliper deviation from bit size
    if "CALI" in df.columns:
        if "BS" in df.columns:
            out["CALI_DEV"] = (
                df["CALI"].astype(np.float64) - df["BS"].astype(np.float64)
            ).astype(np.float64)
        else:
            out["CALI_DEV"] = pd.Series(np.nan, index=df.index, dtype=np.float64)

    # 2. Density-Neutron porosity separation
    rho_mat = float(RHOMAT)
    rho_fl = float(RHOFLUID)
    rho_diff = rho_mat - rho_fl

    has_nphi = "NPHI" in df.columns
    has_rhob = "RHOB" in df.columns

    if has_nphi and has_rhob:
        phi_d = (rho_mat - df["RHOB"].astype(np.float64)) / rho_diff
        out["DN_SEP"] = (df["NPHI"].astype(np.float64) - phi_d).astype(np.float64)
    elif has_nphi or has_rhob:
        out["DN_SEP"] = pd.Series(np.nan, index=df.index, dtype=np.float64)

    # 3. Missing curve indicators for canonical curves
    for curve in TARGET_MISSING_CURVES:
        col_name = f"MISSING_{curve}"
        if curve in df.columns:
            out[col_name] = df[curve].isna().astype(np.float64)
        else:
            out[col_name] = pd.Series(1.0, index=df.index, dtype=np.float64)

    return out
