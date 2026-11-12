"""
caliper_qc.py
-------------
Borehole mechanical quality control evaluating washouts and tight hole restrictions.
"""

from typing import Any
import numpy as np
import pandas as pd

from welllogqc.config.defaults import CALI_BS_THRESHOLD, CALI_BS_TIGHT
from welllogqc.qc.interval_utils import QC_COLS, empty_qc_dataframe, find_contiguous_intervals


def check_caliper(
    df: pd.DataFrame,
    well_name: str,
    washout_threshold: float = CALI_BS_THRESHOLD,
    tight_threshold: float = CALI_BS_TIGHT,
) -> pd.DataFrame:
    """Evaluates borehole condition by comparing caliper (CALI) against bit size (BS).

    Flags severe washouts (hole enlargement) and tight hole restrictions (hole constriction).
    If bit size is unavailable, uses an absolute caliper threshold (CALI > 14 in) to flag
    potential washouts with INFO severity.

    Args:
        df: Canonical well-log DataFrame indexed by DEPT (float ft).
        well_name: Identifier of the well being inspected.
        washout_threshold: Caliper enlargement above bit size (inches) to flag washout.
        tight_threshold: Caliper constriction below bit size (inches) to flag tight hole.

    Returns:
        pd.DataFrame containing QC flags matching the standard QC_COLS schema.

    Raises:
        ValueError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty or "CALI" not in df.columns:
        return empty_qc_dataframe()

    depths = df.index.to_numpy(dtype=np.float64)
    cali = df["CALI"].to_numpy(dtype=np.float64)
    flags: list[dict[str, Any]] = []

    if "BS" in df.columns:
        bs = df["BS"].to_numpy(dtype=np.float64)
        diff = cali - bs

        # 1. Borehole Washout ((CALI - BS) > washout_threshold)
        mask_washout = (diff > washout_threshold) & (~np.isnan(diff))
        if np.any(mask_washout):
            washout_intervals = find_contiguous_intervals(mask_washout, depths)
            for start, end in washout_intervals:
                d_top = float(min(depths[start], depths[end]))
                d_base = float(max(depths[start], depths[end]))
                sub_diff = diff[start : end + 1]
                sub_cali = cali[start : end + 1]
                max_washout = float(np.max(sub_diff))
                max_cali = float(np.max(sub_cali))
                n_samples = end - start + 1
                flags.append({
                    "well_name": well_name,
                    "depth_top": d_top,
                    "depth_base": d_base,
                    "curve": "CALI",
                    "issue": "WASHOUT",
                    "severity": "WARNING",
                    "method": "caliper_bitsize_diff",
                    "value": max_washout,
                    "threshold": f"CALI - BS > {washout_threshold:.1f} in",
                    "description": (
                        f"Borehole washout detected: CALI exceeds bit size by up to {max_washout:.2f} in "
                        f"(peak CALI: {max_cali:.2f} in) across {d_top:.2f}-{d_base:.2f} ft ({n_samples} samples)."
                    ),
                })

        # 2. Tight Hole ((CALI - BS) < tight_threshold)
        mask_tight = (diff < tight_threshold) & (~np.isnan(diff))
        if np.any(mask_tight):
            tight_intervals = find_contiguous_intervals(mask_tight, depths)
            for start, end in tight_intervals:
                d_top = float(min(depths[start], depths[end]))
                d_base = float(max(depths[start], depths[end]))
                sub_diff = diff[start : end + 1]
                min_diff = float(np.min(sub_diff))
                n_samples = end - start + 1
                flags.append({
                    "well_name": well_name,
                    "depth_top": d_top,
                    "depth_base": d_base,
                    "curve": "CALI",
                    "issue": "TIGHT_HOLE",
                    "severity": "WARNING",
                    "method": "caliper_bitsize_diff",
                    "value": min_diff,
                    "threshold": f"CALI - BS < {tight_threshold:.1f} in",
                    "description": (
                        f"Tight hole / undergauge borehole detected: CALI is {abs(min_diff):.2f} in "
                        f"below bit size across {d_top:.2f}-{d_base:.2f} ft ({n_samples} samples)."
                    ),
                })
    else:
        # Fallback: Absolute caliper check when bit size curve is absent
        mask_large = (cali > 14.0) & (~np.isnan(cali))
        if np.any(mask_large):
            large_intervals = find_contiguous_intervals(mask_large, depths)
            for start, end in large_intervals:
                d_top = float(min(depths[start], depths[end]))
                d_base = float(max(depths[start], depths[end]))
                sub_cali = cali[start : end + 1]
                max_cali = float(np.max(sub_cali))
                n_samples = end - start + 1
                flags.append({
                    "well_name": well_name,
                    "depth_top": d_top,
                    "depth_base": d_base,
                    "curve": "CALI",
                    "issue": "WASHOUT",
                    "severity": "INFO",
                    "method": "absolute_caliper_threshold",
                    "value": max_cali,
                    "threshold": "CALI > 14.0 in (bit size unavailable)",
                    "description": (
                        f"Potential borehole washout detected: CALI reaches {max_cali:.2f} in "
                        f"across {d_top:.2f}-{d_base:.2f} ft (bit size curve absent, {n_samples} samples)."
                    ),
                })

    if not flags:
        return empty_qc_dataframe()

    res_df = pd.DataFrame(flags)[QC_COLS]
    res_df["depth_top"] = res_df["depth_top"].astype(np.float64)
    res_df["depth_base"] = res_df["depth_base"].astype(np.float64)
    res_df["value"] = res_df["value"].astype(np.float64)
    return res_df.sort_values(by=["depth_top", "depth_base", "issue"], ignore_index=True)
