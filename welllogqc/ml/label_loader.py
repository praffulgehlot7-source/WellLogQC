"""
welllogqc.ml.label_loader
=========================
Module for loading, validating, converting, and checking quality of QC label datasets.
"""

from pathlib import Path
from typing import Any, Dict, List, Set, Union
import numpy as np
import pandas as pd

REQUIRED_LABEL_COLUMNS: List[str] = [
    'well_name',
    'depth_top',
    'depth_base',
    'label',
    'reviewer',
    'date',
    'notes',
]

VALID_LABELS: Set[int] = {-1, 0, 1, 2}


def load_labels(label_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate an interval QC label CSV file.

    Reads a CSV file containing reviewed QC intervals. Validates schema, label domain
    {-1, 0, 1, 2}, and geometric consistency (depth_top < depth_base).

    Args:
        label_path: Path to the CSV file containing labels.

    Returns:
        pd.DataFrame: Validated DataFrame with canonical columns:
            ['well_name', 'depth_top', 'depth_base', 'label', 'reviewer', 'date', 'notes'].

    Raises:
        ValueError: If file does not exist, required columns are missing, label values
            are outside {-1, 0, 1, 2}, or depth_top >= depth_base.
    """
    path = Path(label_path)
    if not path.is_file():
        raise ValueError(f"Label file not found at: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV at {path}: {exc}") from exc

    missing = [col for col in REQUIRED_LABEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Label CSV is missing required columns: {missing}. Present: {list(df.columns)}")

    if df.empty:
        df_empty = pd.DataFrame(columns=REQUIRED_LABEL_COLUMNS)
        df_empty['depth_top'] = df_empty['depth_top'].astype(np.float64)
        df_empty['depth_base'] = df_empty['depth_base'].astype(np.float64)
        df_empty['label'] = df_empty['label'].astype(np.int64)
        return df_empty

    try:
        df['depth_top'] = df['depth_top'].astype(np.float64)
        df['depth_base'] = df['depth_base'].astype(np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("depth_top and depth_base must be convertible to numeric float values.") from exc

    try:
        df['label'] = df['label'].astype(np.int64)
    except (ValueError, TypeError) as exc:
        raise ValueError("label column must contain integer values in {-1, 0, 1, 2}.") from exc

    invalid_labels = set(df['label'].unique()) - VALID_LABELS
    if invalid_labels:
        raise ValueError(
            f"Invalid label values found: {invalid_labels}. Allowed labels are {-1, 0, 1, 2}."
        )

    invalid_depths = df[df['depth_top'] >= df['depth_base']]
    if not invalid_depths.empty:
        first_bad = invalid_depths.iloc[0]
        raise ValueError(
            f"depth_top must be strictly less than depth_base. Found {len(invalid_depths)} "
            f"violating rows (e.g. well '{first_bad['well_name']}', top={first_bad['depth_top']}, "
            f"base={first_bad['depth_base']})."
        )

    return df[REQUIRED_LABEL_COLUMNS].copy()


def labels_to_series(
    labels_df: pd.DataFrame,
    depth_index: pd.Index,
    well_name: str
) -> pd.Series:
    """Map interval labels for a well to a continuous per-sample Series.

    Translates interval-based annotations to discrete depth points matching
    the depth index of a well log. Samples not covered by any annotation
    receive the unreviewed code -1. Where reviewed intervals overlap, the worst
    severity (highest label value) takes precedence.

    Args:
        labels_df: Validated DataFrame of label intervals.
        depth_index: Float Index representing well log depth (ft).
        well_name: Identifier of the well to extract labels for.

    Returns:
        pd.Series: Integer label Series (dtype int64) matching depth_index.
            Uncovered samples have value -1.
    """
    series_arr = np.full(len(depth_index), -1, dtype=np.int64)
    if labels_df.empty or len(depth_index) == 0:
        return pd.Series(series_arr, index=depth_index, dtype=np.int64, name='label')

    well_rows = labels_df[labels_df['well_name'].astype(str) == str(well_name)]
    if well_rows.empty:
        return pd.Series(series_arr, index=depth_index, dtype=np.int64, name='label')

    depth_arr = depth_index.to_numpy(dtype=np.float64)

    for row in well_rows.itertuples(index=False):
        top = float(row.depth_top)
        base = float(row.depth_base)
        lbl = int(row.label)

        mask = (depth_arr >= top) & (depth_arr <= base)
        if not np.any(mask):
            continue

        if lbl in (0, 1, 2):
            curr = series_arr[mask]
            series_arr[mask] = np.where(curr == -1, lbl, np.maximum(curr, lbl))
        else:
            series_arr[mask] = np.where(series_arr[mask] == -1, -1, series_arr[mask])

    return pd.Series(series_arr, index=depth_index, dtype=np.int64, name='label')


def check_label_quality(labels_df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect QC label dataset quality and consistency.

    Computes class distribution counts, unreviewed fraction, list of covered wells,
    and checks for interval overlaps per well.

    Args:
        labels_df: DataFrame of label intervals.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'class_distribution': Dict[int, int] mapping label to count.
            - 'unreviewed_fraction': float fraction of intervals with label == -1.
            - 'wells_covered': List[str] sorted unique well names.
            - 'any_overlap': bool indicating if any intervals overlap within any well.
    """
    if labels_df.empty:
        return {
            'class_distribution': {lbl: 0 for lbl in sorted(VALID_LABELS)},
            'unreviewed_fraction': 0.0,
            'wells_covered': [],
            'any_overlap': False,
        }

    total_rows = len(labels_df)
    class_dist: Dict[int, int] = {}
    for lbl in sorted(VALID_LABELS):
        class_dist[lbl] = int((labels_df['label'] == lbl).sum())

    unrev_count = class_dist.get(-1, 0)
    unrev_frac = float(unrev_count / total_rows) if total_rows > 0 else 0.0
    wells = sorted(labels_df['well_name'].dropna().astype(str).unique().tolist())

    any_overlap = False
    for well in wells:
        sub = labels_df[labels_df['well_name'].astype(str) == well].copy()
        if len(sub) <= 1:
            continue
        sub = sub.sort_values(by=['depth_top', 'depth_base']).reset_index(drop=True)
        tops = sub['depth_top'].to_numpy(dtype=np.float64)
        bases = sub['depth_base'].to_numpy(dtype=np.float64)

        for i in range(len(tops) - 1):
            if tops[i + 1] < bases[i]:
                any_overlap = True
                break
        if any_overlap:
            break

    return {
        'class_distribution': class_dist,
        'unreviewed_fraction': unrev_frac,
        'wells_covered': wells,
        'any_overlap': any_overlap,
    }
