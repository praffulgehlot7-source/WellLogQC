"""
welllogqc.qc
------------
Automated quality control checks and reporting for well logs.
"""

from welllogqc.qc.caliper_qc import check_caliper
from welllogqc.qc.consistency_checks import (
    check_density_neutron,
    check_null_intervals,
    check_stuck_curve,
)
from welllogqc.qc.depth_checks import check_depth
from welllogqc.qc.interval_utils import (
    LABEL_MAP,
    QC_COLS,
    empty_qc_dataframe,
    find_contiguous_intervals,
)
from welllogqc.qc.qc_reporter import run_all_qc, summarise_flags
from welllogqc.qc.range_checks import check_ranges
from welllogqc.qc.spike_detection import detect_spikes

__all__ = [
    "QC_COLS",
    "LABEL_MAP",
    "empty_qc_dataframe",
    "find_contiguous_intervals",
    "check_depth",
    "check_ranges",
    "detect_spikes",
    "check_caliper",
    "check_density_neutron",
    "check_stuck_curve",
    "check_null_intervals",
    "run_all_qc",
    "summarise_flags",
]
