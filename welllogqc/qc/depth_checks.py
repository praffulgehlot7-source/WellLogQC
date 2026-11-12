"""
depth_checks.py
---------------
Quality control checks for depth monotonicity, duplicate depths, and step regularity.
"""

from typing import Any
import numpy as np
import pandas as pd

from welllogqc.qc.interval_utils import QC_COLS, empty_qc_dataframe, find_contiguous_intervals


def check_depth(
    df: pd.DataFrame,
    well_name: str,
    step_tolerance: float = 0.001,
) -> pd.DataFrame:
    """Checks depth index for duplicates, non-monotonic reversals, and variable step sizes.

    Identifies anomalies in depth indexing and aggregates contiguous flagged samples into
    single intervals according to QC standards.

    Args:
        df: Canonical well-log DataFrame indexed by DEPT (float ft).
        well_name: Identifier of the well being inspected.
        step_tolerance: Permissible tolerance for depth step variation (ft).

    Returns:
        pd.DataFrame containing QC flags matching the standard QC_COLS schema.

    Raises:
        ValueError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty or len(df) < 2:
        return empty_qc_dataframe()

    depths = df.index.to_numpy(dtype=np.float64)
    flags: list[dict[str, Any]] = []

    # 1. Duplicate Depths
    is_dup = df.index.duplicated(keep=False)
    if np.any(is_dup):
        dup_intervals = find_contiguous_intervals(is_dup, depths)
        for start, end in dup_intervals:
            d_top = float(min(depths[start], depths[end]))
            d_base = float(max(depths[start], depths[end]))
            n_dups = end - start + 1
            flags.append({
                "well_name": well_name,
                "depth_top": d_top,
                "depth_base": d_base,
                "curve": "DEPT",
                "issue": "DUPLICATE_DEPTH",
                "severity": "CRITICAL",
                "method": "duplicate_detection",
                "value": d_top,
                "threshold": "Depths must be strictly unique",
                "description": (
                    f"Duplicate depth values detected at {d_top:.2f} ft "
                    f"({n_dups} samples share this depth value)."
                ),
            })

    # 2. Monotonicity and Step Regularity
    steps = np.diff(depths)
    step_depths = depths[:-1]  # reference depth for each step interval

    # Non-monotonic reversals (step < -step_tolerance)
    mask_reversal = steps < -step_tolerance
    if np.any(mask_reversal):
        rev_intervals = find_contiguous_intervals(mask_reversal, step_depths)
        for start, end in rev_intervals:
            d_top = float(min(depths[start], depths[end + 1]))
            d_base = float(max(depths[start], depths[end + 1]))
            sub_steps = steps[start : end + 1]
            min_step = float(np.min(sub_steps))
            flags.append({
                "well_name": well_name,
                "depth_top": d_top,
                "depth_base": d_base,
                "curve": "DEPT",
                "issue": "NON_MONOTONIC_DEPTH",
                "severity": "CRITICAL",
                "method": "monotonicity_check",
                "value": min_step,
                "threshold": "Depth step must be positive (depth strictly increasing)",
                "description": (
                    f"Depth reversal detected across {d_top:.2f}-{d_base:.2f} ft: "
                    f"depth decreases by {abs(min_step):.4f} ft."
                ),
            })

    # Variable step size (step deviates from median step)
    pos_steps = steps[steps > step_tolerance]
    if len(pos_steps) > 0:
        median_step = float(np.median(pos_steps))
        # Exclude reversals or duplicates from step consistency check
        mask_var_step = (np.abs(steps - median_step) > step_tolerance) & (steps > step_tolerance)
        if np.any(mask_var_step):
            var_intervals = find_contiguous_intervals(mask_var_step, step_depths)
            for start, end in var_intervals:
                d_top = float(min(depths[start], depths[end + 1]))
                d_base = float(max(depths[start], depths[end + 1]))
                sub_steps = steps[start : end + 1]
                mean_step = float(np.mean(sub_steps))
                flags.append({
                    "well_name": well_name,
                    "depth_top": d_top,
                    "depth_base": d_base,
                    "curve": "DEPT",
                    "issue": "VARIABLE_STEP",
                    "severity": "WARNING",
                    "method": "step_consistency",
                    "value": mean_step,
                    "threshold": f"Depth step must be {median_step:.4f} ft +/- {step_tolerance:.4f} ft",
                    "description": (
                        f"Variable depth step detected across {d_top:.2f}-{d_base:.2f} ft: "
                        f"observed step is {mean_step:.4f} ft (expected {median_step:.4f} ft)."
                    ),
                })

    if not flags:
        return empty_qc_dataframe()

    res_df = pd.DataFrame(flags)[QC_COLS]
    res_df["depth_top"] = res_df["depth_top"].astype(np.float64)
    res_df["depth_base"] = res_df["depth_base"].astype(np.float64)
    res_df["value"] = res_df["value"].astype(np.float64)
    return res_df.sort_values(by=["depth_top", "depth_base"], ignore_index=True)
