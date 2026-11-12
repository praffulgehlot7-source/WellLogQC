"""
qc_reporter.py
--------------
Orchestrates the complete QC pipeline across all individual check modules, aggregates
detected anomalies into a unified flag DataFrame, and produces diagnostic summaries.
"""

from typing import Any
import numpy as np
import pandas as pd

from welllogqc.config.defaults import (
    CALI_BS_THRESHOLD,
    CALI_BS_TIGHT,
    SPIKE_WINDOW,
    SPIKE_ZSCORE_THRESHOLD,
    STUCK_WINDOW,
)
from welllogqc.ingestion.las_reader import IngestionResult
from welllogqc.qc.caliper_qc import check_caliper
from welllogqc.qc.consistency_checks import (
    check_density_neutron,
    check_null_intervals,
    check_stuck_curve,
)
from welllogqc.qc.depth_checks import check_depth
from welllogqc.qc.interval_utils import QC_COLS, empty_qc_dataframe
from welllogqc.qc.range_checks import check_ranges
from welllogqc.qc.spike_detection import detect_spikes


def run_all_qc(
    ingestion_result: IngestionResult | pd.DataFrame,
    washout_threshold: float = CALI_BS_THRESHOLD,
    tight_threshold: float = CALI_BS_TIGHT,
    zscore_threshold: float = SPIKE_ZSCORE_THRESHOLD,
    spike_window: int = SPIKE_WINDOW,
    stuck_window: int = STUCK_WINDOW,
    curve_ranges: dict[str, tuple[float, float, str]] | None = None,
) -> pd.DataFrame:
    """Runs all automated QC checks against the ingested well dataset.

    Executes depth integrity checks, range boundary checks, rolling z-score spike
    detection, caliper borehole enlargement/constriction analysis, density-neutron
    crossover identification, stuck-curve scans, and null interval tracking. Checks
    requiring curves that are not present are safely skipped without error.

    Args:
        ingestion_result: IngestionResult instance or depth-indexed pd.DataFrame.
        washout_threshold: Caliper enlargement above bit size (in) to flag washout.
        tight_threshold: Caliper constriction below bit size (in) to flag tight hole.
        zscore_threshold: Rolling z-score threshold for spike detection.
        spike_window: Window size (samples) for spike detection.
        stuck_window: Consecutive sample length to flag stuck tool sensor.
        curve_ranges: Optional dictionary of canonical curve ranges. Defaults to config.

    Returns:
        pd.DataFrame containing all aggregated QC flags, sorted ascending by depth_top.

    Raises:
        ValueError: If ingestion_result is neither an IngestionResult nor pd.DataFrame.
    """
    if isinstance(ingestion_result, IngestionResult):
        df = ingestion_result.df
        well_name = ingestion_result.well_name
    elif isinstance(ingestion_result, pd.DataFrame):
        df = ingestion_result
        well_name = getattr(df, "name", "UNKNOWN")
    else:
        raise ValueError(
            f"Expected IngestionResult or pd.DataFrame, got {type(ingestion_result).__name__}"
        )

    if df.empty:
        return empty_qc_dataframe()

    check_results: list[pd.DataFrame] = [
        check_depth(df, well_name=well_name),
        check_ranges(df, well_name=well_name, curve_ranges=curve_ranges),
        detect_spikes(
            df,
            well_name=well_name,
            zscore_threshold=zscore_threshold,
            window=spike_window,
        ),
        check_caliper(
            df,
            well_name=well_name,
            washout_threshold=washout_threshold,
            tight_threshold=tight_threshold,
        ),
        check_density_neutron(df, well_name=well_name),
        check_stuck_curve(df, well_name=well_name, window=stuck_window),
        check_null_intervals(df, well_name=well_name),
    ]

    non_empty = [res for res in check_results if not res.empty]
    if not non_empty:
        return empty_qc_dataframe()

    unified = pd.concat(non_empty, ignore_index=True)
    unified["depth_top"] = unified["depth_top"].astype(np.float64)
    unified["depth_base"] = unified["depth_base"].astype(np.float64)
    unified["value"] = unified["value"].astype(np.float64)

    return unified.sort_values(
        by=["depth_top", "depth_base", "curve", "issue"],
        ignore_index=True,
    )[QC_COLS]


def summarise_flags(flags: pd.DataFrame) -> dict[str, Any]:
    """Generates a summary breakdown of QC flags by severity and issue category.

    Args:
        flags: pd.DataFrame conforming to the standard QC_COLS schema.

    Returns:
        Dictionary with keys:
            - 'total': total number of QC flag intervals (int)
            - 'by_severity': count of flags grouped by severity category (dict)
            - 'by_issue': count of flags grouped by issue type (dict)

    Raises:
        ValueError: If flags is not a pandas DataFrame.
    """
    if not isinstance(flags, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(flags).__name__}")

    if flags.empty:
        return {
            "total": 0,
            "by_severity": {},
            "by_issue": {},
        }

    by_severity = {str(k): int(v) for k, v in flags["severity"].value_counts().items()}
    by_issue = {str(k): int(v) for k, v in flags["issue"].value_counts().items()}

    return {
        "total": int(len(flags)),
        "by_severity": by_severity,
        "by_issue": by_issue,
    }
