"""
range_checks.py
---------------
Evaluates log curve values against physically plausible boundary thresholds.
"""

from typing import Any
import numpy as np
import pandas as pd

from welllogqc.config.defaults import CURVE_RANGES
from welllogqc.qc.interval_utils import QC_COLS, empty_qc_dataframe, find_contiguous_intervals


def check_ranges(
    df: pd.DataFrame,
    well_name: str,
    curve_ranges: dict[str, tuple[float, float, str]] | None = None,
) -> pd.DataFrame:
    """Verifies that curve values fall within conservative engineering and physical limits.

    Samples exceeding minimum or maximum thresholds are aggregated into contiguous
    interval flags. Curves not present in df are safely bypassed.

    Args:
        df: Canonical well-log DataFrame indexed by DEPT (float ft).
        well_name: Identifier of the well being inspected.
        curve_ranges: Optional dictionary mapping canonical curve names to
            (min_valid, max_valid, unit) tuples. Defaults to CURVE_RANGES from config.

    Returns:
        pd.DataFrame containing QC flags matching the standard QC_COLS schema.

    Raises:
        ValueError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        return empty_qc_dataframe()

    if curve_ranges is None:
        curve_ranges = CURVE_RANGES

    depths = df.index.to_numpy(dtype=np.float64)
    flags: list[dict[str, Any]] = []

    for curve, bounds in curve_ranges.items():
        if curve not in df.columns:
            continue

        min_val, max_val, unit = bounds
        vals = df[curve].to_numpy(dtype=np.float64)

        # 1. Below Minimum Physical Bound
        mask_low = (~np.isnan(vals)) & (vals < min_val)
        if np.any(mask_low):
            low_intervals = find_contiguous_intervals(mask_low, depths)
            for start, end in low_intervals:
                d_top = float(min(depths[start], depths[end]))
                d_base = float(max(depths[start], depths[end]))
                sub_vals = vals[start : end + 1]
                min_observed = float(np.min(sub_vals))
                n_samples = end - start + 1
                flags.append({
                    "well_name": well_name,
                    "depth_top": d_top,
                    "depth_base": d_base,
                    "curve": curve,
                    "issue": "OUT_OF_RANGE_LOW",
                    "severity": "RANGE_VIOLATION",
                    "method": "physical_range_bounds",
                    "value": min_observed,
                    "threshold": f"{curve} must be between {min_val} and {max_val} {unit}",
                    "description": (
                        f"{curve} value ({min_observed:.2f} {unit}) falls below physical minimum of {min_val} {unit} "
                        f"across {d_top:.2f}-{d_base:.2f} ft ({n_samples} samples)."
                    ),
                })

        # 2. Above Maximum Physical Bound
        mask_high = (~np.isnan(vals)) & (vals > max_val)
        if np.any(mask_high):
            high_intervals = find_contiguous_intervals(mask_high, depths)
            for start, end in high_intervals:
                d_top = float(min(depths[start], depths[end]))
                d_base = float(max(depths[start], depths[end]))
                sub_vals = vals[start : end + 1]
                max_observed = float(np.max(sub_vals))
                n_samples = end - start + 1
                flags.append({
                    "well_name": well_name,
                    "depth_top": d_top,
                    "depth_base": d_base,
                    "curve": curve,
                    "issue": "OUT_OF_RANGE_HIGH",
                    "severity": "RANGE_VIOLATION",
                    "method": "physical_range_bounds",
                    "value": max_observed,
                    "threshold": f"{curve} must be between {min_val} and {max_val} {unit}",
                    "description": (
                        f"{curve} value ({max_observed:.2f} {unit}) exceeds physical maximum of {max_val} {unit} "
                        f"across {d_top:.2f}-{d_base:.2f} ft ({n_samples} samples)."
                    ),
                })

    if not flags:
        return empty_qc_dataframe()

    res_df = pd.DataFrame(flags)[QC_COLS]
    res_df["depth_top"] = res_df["depth_top"].astype(np.float64)
    res_df["depth_base"] = res_df["depth_base"].astype(np.float64)
    res_df["value"] = res_df["value"].astype(np.float64)
    return res_df.sort_values(by=["depth_top", "depth_base", "curve"], ignore_index=True)
