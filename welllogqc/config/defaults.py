"""
defaults.py
-----------
Central store for all user-configurable thresholds and assumptions.

ENGINEERING RATIONALE
---------------------
Every threshold here is a *starting point*, not a universal geological truth.
Different basins, formations, and logging tools require different values.
The user must be able to override every entry without touching source code.

Nothing in this file is silently applied. Each threshold is referenced
explicitly by the module that uses it, and every flag produced by those
modules records which threshold triggered it.
"""

# ---------------------------------------------------------------------------
# Curve physical plausibility ranges
# ---------------------------------------------------------------------------
# Format: "CANONICAL_NAME": (min_valid, max_valid, unit)
# Values outside these ranges raise a QC flag of severity "RANGE_VIOLATION".
# These are conservative engineering bounds, NOT geological cutoffs.
CURVE_RANGES: dict[str, tuple[float, float, str]] = {
    "GR":   (0.0,   300.0,  "GAPI"),
    "CALI": (4.0,   20.0,   "in"),
    "RHOB": (1.0,   3.5,    "g/cm3"),
    "NPHI": (-0.15, 1.0,    "v/v"),
    "RT":   (0.01,  50000.0,"ohm.m"),
    "DT":   (40.0,  240.0,  "us/ft"),
    "PEF":  (0.5,   10.0,   "b/e"),
    "BS":   (4.0,   20.0,   "in"),
}

# ---------------------------------------------------------------------------
# Caliper / borehole quality thresholds
# ---------------------------------------------------------------------------
# CALI_BS_THRESHOLD: Maximum allowed (CALI - BS) deviation in inches before
#   flagging a washout or tight-hole condition.
# CALI_BS_TIGHT: Minimum allowed (CALI - BS); negative values indicate
#   possible tool stick or caving artifact.
CALI_BS_THRESHOLD: float = 2.0   # inches  — washout flag
CALI_BS_TIGHT:     float = -0.5  # inches  — tight-hole flag

# ---------------------------------------------------------------------------
# Spike / discontinuity detection
# ---------------------------------------------------------------------------
# SPIKE_ZSCORE_THRESHOLD: How many rolling standard deviations above the
#   local mean constitutes a spike. Lower values = more sensitive.
# SPIKE_WINDOW: Number of depth samples used for the rolling statistics.
# STUCK_WINDOW: Minimum run of identical values to flag as "stuck/constant".
# STUCK_TOLERANCE: Floating-point equality tolerance for stuck detection.
SPIKE_ZSCORE_THRESHOLD: float = 4.0
SPIKE_WINDOW:           int   = 11    # samples (odd number recommended)
STUCK_WINDOW:           int   = 10    # consecutive identical samples
STUCK_TOLERANCE:        float = 1e-6  # floating-point equality tolerance for stuck detection

# ---------------------------------------------------------------------------
# Gamma-ray petrophysics
# ---------------------------------------------------------------------------
# GR_CLEAN and GR_SHALE are the baseline values used for the linear IGR
# calculation: IGR = (GR - GR_CLEAN) / (GR_SHALE - GR_CLEAN)
# These MUST be set by the user for each well / formation.
# The defaults below are placeholders that will trigger a warning if used
# without explicit user confirmation.
GR_CLEAN:  float = 15.0   # GAPI — typical clean sand / carbonate
GR_SHALE:  float = 120.0  # GAPI — typical shale baseline

# ---------------------------------------------------------------------------
# Density porosity assumptions
# ---------------------------------------------------------------------------
# These lithology / fluid assumptions must be stated explicitly in every
# report. They are NOT applied silently.
RHOMAT:   float = 2.65   # g/cm3 — quartz sandstone matrix
RHOFLUID: float = 1.00   # g/cm3 — fresh water (change to 1.10 for brine)

# ---------------------------------------------------------------------------
# Net-to-gross cutoffs (preliminary, rule-based)
# ---------------------------------------------------------------------------
# These are applied ONLY when the user explicitly enables NTG calculation
# and has confirmed the relevant petrophysical curves are valid.
VSH_CUTOFF:   float = 0.5   # Vsh above this → non-reservoir candidate
PHID_CUTOFF:  float = 0.05  # phiD below this → non-reservoir candidate

# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------
# NULL_SUBSTITUTE: Value written by logging tools to represent absent data.
# lasio reads the LAS null value from the header; this is the fallback.
NULL_FALLBACK: float = -999.25
