"""
ntg.py
------
Net-to-Gross (NTG) pay estimation and depth-sample classification.
"""

from typing import Any
import numpy as np
import pandas as pd

from welllogqc.config.defaults import VSH_CUTOFF, PHID_CUTOFF


def calc_ntg(
    df: pd.DataFrame,
    gross_top: float,
    gross_base: float,
    vsh_col: str = "VSH_LINEAR",
    phid_col: str = "PHID",
    vsh_cutoff: float = VSH_CUTOFF,
    phid_cutoff: float = PHID_CUTOFF,
    qc_flag_depths: list[tuple[float, float]] | None = None
) -> dict[str, Any]:
    """Calculate Net-to-Gross (NTG) and classify depth samples within an interval.

    Classifies each depth sample in the interval [gross_top, gross_base] as:
      - 'RESERVOIR': vsh <= vsh_cutoff AND phid >= phid_cutoff AND not QC-flagged
      - 'NON_RESERVOIR': fails vsh or phid cutoff AND not QC-flagged
      - 'QC_REVIEW': depth falls within a QC-flagged interval
      - 'INSUFFICIENT': required petrophysical curves are NaN and not QC-flagged

    Args:
        df: pandas DataFrame containing well log data, indexed by depth ('DEPT').
        gross_top: Top depth of the gross evaluation interval in ft.
        gross_base: Base depth of the gross evaluation interval in ft.
        vsh_col: Name of the shale volume column. Defaults to 'VSH_LINEAR'.
        phid_col: Name of the porosity column. Defaults to 'PHID'.
        vsh_cutoff: Upper cutoff threshold for shale volume. Defaults to 0.5.
        phid_cutoff: Lower cutoff threshold for porosity. Defaults to 0.05.
        qc_flag_depths: Optional list of (depth_top, depth_base) tuples marking intervals
            flagged for QC review.

    Returns:
        dict: Calculation results containing:
            - 'ntg' (float): Net-to-gross ratio (net_ft / gross_ft).
            - 'net_ft' (float): Net reservoir thickness in ft.
            - 'gross_ft' (float): Total gross interval thickness in ft.
            - 'classification' (pd.Series): Categorical label per sample.
            - 'summary' (dict): Counts for 'RESERVOIR', 'NON_RESERVOIR',
              'QC_REVIEW', and 'INSUFFICIENT'.

    Raises:
        ValueError: If vsh_col is not present in df.columns.
        ValueError: If phid_col is not present in df.columns.
        ValueError: If gross_top > gross_base.
    """
    if vsh_col not in df.columns:
        raise ValueError(
            f"Required shale volume column '{vsh_col}' not found in DataFrame columns."
        )

    if phid_col not in df.columns:
        raise ValueError(
            f"Required porosity column '{phid_col}' not found in DataFrame columns."
        )

    if gross_top > gross_base:
        raise ValueError(
            f"gross_top ({gross_top}) must be less than or equal to gross_base ({gross_base})."
        )

    gross_ft = float(gross_base - gross_top)
    mask = (df.index >= gross_top) & (df.index <= gross_base)
    sub_df = df.loc[mask]

    classification = pd.Series(index=sub_df.index, dtype="object", name="CLASSIFICATION")

    if len(sub_df) == 0:
        summary = {
            "RESERVOIR": 0,
            "NON_RESERVOIR": 0,
            "QC_REVIEW": 0,
            "INSUFFICIENT": 0,
        }
        return {
            "ntg": 0.0,
            "net_ft": 0.0,
            "gross_ft": gross_ft,
            "classification": classification,
            "summary": summary,
        }

    is_qc = pd.Series(False, index=sub_df.index)
    if qc_flag_depths:
        for d_top, d_base in qc_flag_depths:
            t, b = min(d_top, d_base), max(d_top, d_base)
            is_qc |= (sub_df.index >= t) & (sub_df.index <= b)

    vsh = sub_df[vsh_col].astype(np.float64)
    phid = sub_df[phid_col].astype(np.float64)
    is_nan = vsh.isna() | phid.isna()

    # Rule 1: QC_REVIEW takes precedence for flagged intervals
    classification[is_qc] = "QC_REVIEW"

    # Rule 2: INSUFFICIENT data where curves are NaN (and not flagged for QC)
    classification[~is_qc & is_nan] = "INSUFFICIENT"

    # Rule 3: RESERVOIR where cutoffs are met, not flagged, and valid data
    is_res = ~is_qc & ~is_nan & (vsh <= vsh_cutoff) & (phid >= phid_cutoff)
    classification[is_res] = "RESERVOIR"

    # Rule 4: NON_RESERVOIR where cutoffs fail, not flagged, and valid data
    is_non_res = ~is_qc & ~is_nan & ~is_res
    classification[is_non_res] = "NON_RESERVOIR"

    summary = {
        "RESERVOIR": int((classification == "RESERVOIR").sum()),
        "NON_RESERVOIR": int((classification == "NON_RESERVOIR").sum()),
        "QC_REVIEW": int((classification == "QC_REVIEW").sum()),
        "INSUFFICIENT": int((classification == "INSUFFICIENT").sum()),
    }

    ntg_val = float(summary["RESERVOIR"] / len(sub_df))
    net_ft = float(ntg_val * gross_ft)

    return {
        "ntg": ntg_val,
        "net_ft": net_ft,
        "gross_ft": gross_ft,
        "classification": classification,
        "summary": summary,
    }
