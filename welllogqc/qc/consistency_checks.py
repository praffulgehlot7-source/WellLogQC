"""
consistency_checks.py
---------------------
Consistency and data integrity checks: density-neutron crossover, stuck curve detection,
and missing value (null) intervals.
"""

from typing import Any
import numpy as np
import pandas as pd

from welllogqc.config.defaults import RHOFLUID, RHOMAT, STUCK_TOLERANCE, STUCK_WINDOW
from welllogqc.qc.interval_utils import QC_COLS, empty_qc_dataframe, find_contiguous_intervals


def check_density_neutron(
    df: pd.DataFrame,
    well_name: str,
    matrix_density: float = RHOMAT,
    fluid_density: float = RHOFLUID,
    crossover_threshold: float = 0.25,
) -> pd.DataFrame:
    """Detects extreme density-neutron crossover intervals.

    Calculates density porosity (phi_D) using user-defined or default lithology matrix
    and fluid densities. Identifies zones where phi_D - NPHI exceeds crossover_threshold,
    indicative of gas presence or loss of tool contact due to washouts. Only executes
    if both RHOB and NPHI curves are present.

    Args:
        df: Canonical well-log DataFrame indexed by DEPT (float ft).
        well_name: Identifier of the well being inspected.
        matrix_density: Lithology matrix grain density (g/cm3), default 2.65.
        fluid_density: Borehole fluid density (g/cm3), default 1.00.
        crossover_threshold: Porosity difference (v/v) defining extreme crossover.

    Returns:
        pd.DataFrame containing QC flags matching the standard QC_COLS schema.

    Raises:
        ValueError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty or "RHOB" not in df.columns or "NPHI" not in df.columns:
        return empty_qc_dataframe()

    depths = df.index.to_numpy(dtype=np.float64)
    rhob = df["RHOB"].to_numpy(dtype=np.float64)
    nphi = df["NPHI"].to_numpy(dtype=np.float64)

    denom = matrix_density - fluid_density
    if abs(denom) < 1e-4:
        return empty_qc_dataframe()

    phi_d = (matrix_density - rhob) / denom
    crossover = phi_d - nphi
    valid_mask = (~np.isnan(rhob)) & (~np.isnan(nphi))
    mask_crossover = (crossover > crossover_threshold) & valid_mask

    if not np.any(mask_crossover):
        return empty_qc_dataframe()

    flags: list[dict[str, Any]] = []
    intervals = find_contiguous_intervals(mask_crossover, depths)

    for start, end in intervals:
        d_top = float(min(depths[start], depths[end]))
        d_base = float(max(depths[start], depths[end]))
        sub_crossover = crossover[start : end + 1]
        max_crossover = float(np.max(sub_crossover))
        n_samples = end - start + 1
        flags.append({
            "well_name": well_name,
            "depth_top": d_top,
            "depth_base": d_base,
            "curve": "RHOB/NPHI",
            "issue": "DENSITY_NEUTRON_CROSSOVER",
            "severity": "INFO",
            "method": "density_neutron_crossover",
            "value": max_crossover,
            "threshold": f"phi_D - NPHI > {crossover_threshold:.2f} v/v",
            "description": (
                f"Extreme density-neutron crossover (phi_D - NPHI reaches {max_crossover:.2f} v/v) "
                f"across {d_top:.2f}-{d_base:.2f} ft (potential gas effect or pad standoff, {n_samples} samples)."
            ),
        })

    res_df = pd.DataFrame(flags)[QC_COLS]
    res_df["depth_top"] = res_df["depth_top"].astype(np.float64)
    res_df["depth_base"] = res_df["depth_base"].astype(np.float64)
    res_df["value"] = res_df["value"].astype(np.float64)
    return res_df.sort_values(by=["depth_top", "depth_base"], ignore_index=True)


def check_stuck_curve(
    df: pd.DataFrame,
    well_name: str,
    window: int = STUCK_WINDOW,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Detects continuous runs of identical values indicating stuck tool sensors.

    Scans each curve for consecutive non-null values that remain invariant within
    the specified tolerance over at least window samples.

    Args:
        df: Canonical well-log DataFrame indexed by DEPT (float ft).
        well_name: Identifier of the well being inspected.
        window: Minimum number of consecutive identical samples to flag.
        tolerance: Floating point difference tolerance for identity comparison.

    Returns:
        pd.DataFrame containing QC flags matching the standard QC_COLS schema.

    Raises:
        ValueError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty or len(df) < window:
        return empty_qc_dataframe()

    depths = df.index.to_numpy(dtype=np.float64)
    flags: list[dict[str, Any]] = []

    for curve in df.columns:
        vals = df[curve].to_numpy(dtype=np.float64)
        n = len(vals)
        start_run = 0
        current_len = 1

        for i in range(1, n):
            v_curr = vals[i]
            v_prev = vals[i - 1]

            if not np.isnan(v_curr) and not np.isnan(v_prev) and abs(v_curr - v_prev) <= tolerance:
                current_len += 1
            else:
                if current_len >= window:
                    end_run = i - 1
                    d_top = float(min(depths[start_run], depths[end_run]))
                    d_base = float(max(depths[start_run], depths[end_run]))
                    stuck_val = float(vals[start_run])
                    flags.append({
                        "well_name": well_name,
                        "depth_top": d_top,
                        "depth_base": d_base,
                        "curve": curve,
                        "issue": "STUCK_TOOL",
                        "severity": "WARNING",
                        "method": "consecutive_identical",
                        "value": stuck_val,
                        "threshold": f"Run length >= {window} identical samples",
                        "description": (
                            f"Stuck curve / invariant sensor reading detected on {curve} across "
                            f"{d_top:.2f}-{d_base:.2f} ft ({current_len} consecutive samples constant at {stuck_val:.4f})."
                        ),
                    })
                start_run = i
                current_len = 1

        if current_len >= window:
            end_run = n - 1
            d_top = float(min(depths[start_run], depths[end_run]))
            d_base = float(max(depths[start_run], depths[end_run]))
            stuck_val = float(vals[start_run])
            flags.append({
                "well_name": well_name,
                "depth_top": d_top,
                "depth_base": d_base,
                "curve": curve,
                "issue": "STUCK_TOOL",
                "severity": "WARNING",
                "method": "consecutive_identical",
                "value": stuck_val,
                "threshold": f"Run length >= {window} identical samples",
                "description": (
                    f"Stuck curve / invariant sensor reading detected on {curve} across "
                    f"{d_top:.2f}-{d_base:.2f} ft ({current_len} consecutive samples constant at {stuck_val:.4f})."
                ),
            })

    if not flags:
        return empty_qc_dataframe()

    res_df = pd.DataFrame(flags)[QC_COLS]
    res_df["depth_top"] = res_df["depth_top"].astype(np.float64)
    res_df["depth_base"] = res_df["depth_base"].astype(np.float64)
    res_df["value"] = res_df["value"].astype(np.float64)
    return res_df.sort_values(by=["depth_top", "depth_base", "curve"], ignore_index=True)


def check_null_intervals(
    df: pd.DataFrame,
    well_name: str,
) -> pd.DataFrame:
    """Scans for contiguous missing data (NaN) intervals across all log curves.

    Aggregates consecutive NaN samples into distinct interval flags per curve.

    Args:
        df: Canonical well-log DataFrame indexed by DEPT (float ft).
        well_name: Identifier of the well being inspected.

    Returns:
        pd.DataFrame containing QC flags matching the standard QC_COLS schema.

    Raises:
        ValueError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        return empty_qc_dataframe()

    depths = df.index.to_numpy(dtype=np.float64)
    flags: list[dict[str, Any]] = []

    for curve in df.columns:
        vals = df[curve].to_numpy(dtype=np.float64)
        is_null = np.isnan(vals)

        if np.any(is_null):
            null_intervals = find_contiguous_intervals(is_null, depths)
            for start, end in null_intervals:
                d_top = float(min(depths[start], depths[end]))
                d_base = float(max(depths[start], depths[end]))
                n_samples = end - start + 1
                flags.append({
                    "well_name": well_name,
                    "depth_top": d_top,
                    "depth_base": d_base,
                    "curve": curve,
                    "issue": "NULL_INTERVAL",
                    "severity": "WARNING",
                    "method": "null_interval_scan",
                    "value": np.nan,
                    "threshold": "No null (NaN) values permitted",
                    "description": (
                        f"Missing data on {curve}: {n_samples} contiguous null (NaN) samples "
                        f"across {d_top:.2f}-{d_base:.2f} ft."
                    ),
                })

    if not flags:
        return empty_qc_dataframe()

    res_df = pd.DataFrame(flags)[QC_COLS]
    res_df["depth_top"] = res_df["depth_top"].astype(np.float64)
    res_df["depth_base"] = res_df["depth_base"].astype(np.float64)
    res_df["value"] = res_df["value"].astype(np.float64)
    return res_df.sort_values(by=["depth_top", "depth_base", "curve"], ignore_index=True)
