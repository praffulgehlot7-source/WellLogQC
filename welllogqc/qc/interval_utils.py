"""
interval_utils.py
-----------------
Shared utilities, data structures, and interval-grouping algorithms for QC modules.
"""

from typing import Any
import numpy as np
import pandas as pd

QC_COLS: list[str] = [
    "well_name",
    "depth_top",
    "depth_base",
    "curve",
    "issue",
    "severity",
    "method",
    "value",
    "threshold",
    "description",
]

LABEL_MAP: dict[int, str] = {
    0: "Acceptable",
    1: "Questionable",
    2: "Unreliable",
    -1: "Unreviewed",
}


def empty_qc_dataframe() -> pd.DataFrame:
    """Creates an empty DataFrame matching the standard QC schema.

    Returns:
        pd.DataFrame with QC_COLS columns and correct dtypes.
    """
    df = pd.DataFrame(columns=QC_COLS)
    df["well_name"] = df["well_name"].astype(str)
    df["depth_top"] = df["depth_top"].astype(np.float64)
    df["depth_base"] = df["depth_base"].astype(np.float64)
    df["curve"] = df["curve"].astype(str)
    df["issue"] = df["issue"].astype(str)
    df["severity"] = df["severity"].astype(str)
    df["method"] = df["method"].astype(str)
    df["value"] = df["value"].astype(np.float64)
    df["threshold"] = df["threshold"].astype(str)
    df["description"] = df["description"].astype(str)
    return df


def find_contiguous_intervals(
    mask: np.ndarray | pd.Series,
    depths: np.ndarray | pd.Index,
    max_step: float | None = None,
) -> list[tuple[int, int]]:
    """Identifies contiguous intervals (start_index, end_index inclusive) where mask is True.

    Optionally splits intervals if depth spacing between consecutive samples exceeds max_step.

    Args:
        mask: 1D boolean array or Series indicating flagged samples.
        depths: 1D float array or Index of depth values corresponding to mask.
        max_step: Optional maximum allowable depth step to consider consecutive samples contiguous.

    Returns:
        List of (start_idx, end_idx) tuples, 0-indexed and inclusive.
    """
    m = np.asarray(mask, dtype=bool)
    d = np.asarray(depths, dtype=np.float64)
    n = len(m)
    if n == 0 or not np.any(m):
        return []

    intervals: list[tuple[int, int]] = []
    in_interval = False
    start_idx = 0

    for i in range(n):
        if m[i]:
            if not in_interval:
                in_interval = True
                start_idx = i
            else:
                if max_step is not None and abs(d[i] - d[i - 1]) > max_step:
                    intervals.append((start_idx, i - 1))
                    start_idx = i
        else:
            if in_interval:
                in_interval = False
                intervals.append((start_idx, i - 1))

    if in_interval:
        intervals.append((start_idx, n - 1))

    return intervals
