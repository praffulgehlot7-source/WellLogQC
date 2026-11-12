"""
welllogqc.ml.problem_definition
===============================
Problem definition, class labels, severity mapping, and target conversion for ML.
"""

from typing import Dict, List
import numpy as np
import pandas as pd

LABEL_MAP: Dict[int, str] = {
    0: 'Acceptable',
    1: 'Questionable',
    2: 'Unreliable',
    -1: 'Unreviewed',
}

SEVERITY_TO_LABEL: Dict[str, int] = {
    'CRITICAL': 2,
    'WARNING': 1,
    'INFO': 0,
}

QC_RISK_CLASSES: List[int] = [0, 1, 2]

BINARY_MAP: Dict[int, str] = {
    0: 'No_Review',
    1: 'Review_Needed',
}


def flags_to_labels(
    df: pd.DataFrame,
    qc_flags: pd.DataFrame,
    mode: str = 'binary'
) -> pd.Series:
    """Convert QC flags DataFrame to per-depth-sample label Series.

    Maps interval-based QC flags onto depth sample points of a well log DataFrame.
    In binary mode, any depth sample falling within an interval flagged by a QC check
    is labelled as 1 ('Review_Needed'), otherwise 0 ('No_Review').
    In multiclass mode, each depth sample is assigned the maximum severity value across
    all overlapping flagged intervals using SEVERITY_TO_LABEL (0: INFO/None, 1: WARNING,
    2: CRITICAL).

    Args:
        df: Well log DataFrame whose index represents depth (DEPT).
        qc_flags: DataFrame of QC flags containing at least 'depth_top' and 'depth_base'
            columns, and 'severity' for multiclass mode.
        mode: Label conversion mode, either 'binary' or 'multiclass'. Defaults to 'binary'.

    Returns:
        pd.Series: Per-sample integer labels (dtype int64) indexed identically to df.
            Default value is 0.

    Raises:
        ValueError: If mode is not 'binary' or 'multiclass', or if required columns
            are missing in qc_flags when qc_flags is non-empty.
    """
    if mode not in ('binary', 'multiclass'):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'binary' or 'multiclass'.")

    labels = pd.Series(0, index=df.index, dtype=np.int64, name='label')
    if df.empty or qc_flags is None or qc_flags.empty:
        return labels

    if 'depth_top' not in qc_flags.columns or 'depth_base' not in qc_flags.columns:
        raise ValueError("qc_flags must contain 'depth_top' and 'depth_base' columns.")

    if mode == 'multiclass' and 'severity' not in qc_flags.columns:
        raise ValueError("qc_flags must contain 'severity' column for multiclass mode.")

    depth_values = df.index.to_numpy(dtype=np.float64)
    labels_arr = labels.to_numpy(dtype=np.int64).copy()

    for row in qc_flags.itertuples(index=False):
        d_top = float(row.depth_top)
        d_base = float(row.depth_base)
        top = min(d_top, d_base)
        base = max(d_top, d_base)

        mask = (depth_values >= top) & (depth_values <= base)
        if not np.any(mask):
            continue

        if mode == 'binary':
            labels_arr[mask] = 1
        else:
            sev = str(getattr(row, 'severity', 'INFO')).strip().upper()
            val = SEVERITY_TO_LABEL.get(sev, 1)
            labels_arr[mask] = np.maximum(labels_arr[mask], val)

    return pd.Series(labels_arr, index=df.index, dtype=np.int64, name='label')
