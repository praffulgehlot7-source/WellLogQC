"""
spike_detection.py
------------------
Detects single-point and narrow-interval anomalous spikes using a rolling z-score algorithm.
"""

from typing import Any
import numpy as np
import pandas as pd

from welllogqc.qc.interval_utils import QC_COLS, empty_qc_dataframe, find_contiguous_intervals


def detect_spikes(
    df: pd.DataFrame,
    well_name: str,
    curves: list[str] | None = None,
    zscore_threshold: float = 4.0,
    window: int = 11,
) -> pd.DataFrame:
    """Identifies anomalous spikes on log curves via a centred rolling z-score.

    Calculates rolling local mean and standard deviation. Samples where
    |value - rolling_mean| > zscore_threshold * rolling_std are flagged and grouped
    into contiguous spike intervals. Missing values and zero-variance windows are handled safely.

    Args:
        df: Canonical well-log DataFrame indexed by DEPT (float ft).
        well_name: Identifier of the well being inspected.
        curves: Optional list of curve names to inspect. If None, inspects all
            available curves except bit size ('BS').
        zscore_threshold: Multiplier of rolling standard deviation defining a spike.
        window: Number of samples in the rolling window (odd integer recommended).

    Returns:
        pd.DataFrame containing QC flags matching the standard QC_COLS schema.

    Raises:
        ValueError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty or len(df) < max(3, window // 2):
        return empty_qc_dataframe()

    if curves is None:
        target_curves = [c for c in df.columns if c != "BS"]
    else:
        target_curves = [c for c in curves if c in df.columns]

    if not target_curves:
        return empty_qc_dataframe()

    depths = df.index.to_numpy(dtype=np.float64)
    flags: list[dict[str, Any]] = []
    min_p = max(3, window // 2)

    for curve in target_curves:
        s = df[curve].astype(np.float64)
        roll_mean = s.rolling(window=window, center=True, min_periods=min_p).mean()
        roll_std = s.rolling(window=window, center=True, min_periods=min_p).std()

        diff = (s - roll_mean).abs()
        valid_std = (roll_std > 1e-6) & roll_std.notna()

        z_scores = pd.Series(0.0, index=s.index, dtype=np.float64)
        z_scores.loc[valid_std] = diff.loc[valid_std] / roll_std.loc[valid_std]

        mask_spike = (z_scores > zscore_threshold) & s.notna()

        if np.any(mask_spike):
            spike_intervals = find_contiguous_intervals(mask_spike, depths)
            for start, end in spike_intervals:
                d_top = float(min(depths[start], depths[end]))
                d_base = float(max(depths[start], depths[end]))
                sub_vals = s.iloc[start : end + 1].to_numpy()
                sub_z = z_scores.iloc[start : end + 1].to_numpy()
                sub_m = roll_mean.iloc[start : end + 1].to_numpy()

                max_idx = int(np.argmax(sub_z))
                peak_val = float(sub_vals[max_idx])
                peak_z = float(sub_z[max_idx])
                local_mean = float(sub_m[max_idx])
                n_samples = end - start + 1

                flags.append({
                    "well_name": well_name,
                    "depth_top": d_top,
                    "depth_base": d_base,
                    "curve": curve,
                    "issue": "SPIKE",
                    "severity": "WARNING",
                    "method": "rolling_zscore",
                    "value": peak_val,
                    "threshold": f"|z-score| > {zscore_threshold:.1f} (window={window})",
                    "description": (
                        f"Spike detected on {curve} across {d_top:.2f}-{d_base:.2f} ft: "
                        f"peak value is {peak_val:.2f} (local mean: {local_mean:.2f}, "
                        f"z-score: {peak_z:.1f} > {zscore_threshold:.1f}, {n_samples} samples)."
                    ),
                })

    if not flags:
        return empty_qc_dataframe()

    res_df = pd.DataFrame(flags)[QC_COLS]
    res_df["depth_top"] = res_df["depth_top"].astype(np.float64)
    res_df["depth_base"] = res_df["depth_base"].astype(np.float64)
    res_df["value"] = res_df["value"].astype(np.float64)
    return res_df.sort_values(by=["depth_top", "depth_base", "curve"], ignore_index=True)
