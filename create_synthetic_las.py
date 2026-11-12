"""
create_synthetic_las.py
=======================
Generates a realistic-but-clearly-labelled set of synthetic LAS files for
software development and testing.

PURPOSE
-------
These files exist solely to allow testing of code before real well data is
available. They are NEVER to be used as proof of real-world model performance,
geological interpretation, or reservoir characterisation.

Every file produced by this script carries the header comment:
    SYNTHETIC DATA — FOR SOFTWARE TESTING ONLY

WHAT IS GENERATED
-----------------
Five wells are created, each with slightly different characteristics so that
the ingestion, QC, and feature-engineering modules encounter realistic edge
cases:

  WELL_A  (training) — full curve suite, good borehole, clean data
  WELL_B  (training) — full curve suite, washout interval, a few spikes
  WELL_C  (training) — missing DT and PEF curves
  WELL_D  (validation) — missing BS, moderate spikes, tight-hole section
  WELL_E  (test)     — full suite, deliberately held back, NOT inspected here

HOW DEPTH IS STRUCTURED
------------------------
Each well spans 3000 ft (e.g. 8000–11000 ft MD), sampled at 0.5 ft intervals
→ 6001 depth rows per well.

GEOLOGICAL SCENARIO (synthetic only)
--------------------------------------
Four broad layers are modelled per well:
  1. Shale cap    ~200 ft   — high GR, low resistivity, low porosity
  2. Sand body    ~500 ft   — low GR, moderate-high resistivity, higher phi
  3. Interbedded  ~300 ft   — alternating GR
  4. Tight shale  ~2000 ft  — high GR, very low resistivity

Noise and defects are added on top to produce realistic-looking logs.
"""

import numpy as np
import pandas as pd
import lasio
import os
from pathlib import Path

# ── reproducibility ────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ── output paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT  = SCRIPT_DIR / "data" / "raw"

WELL_PATHS = {
    "WELL_A": DATA_ROOT / "training"   / "WELL_A.las",
    "WELL_B": DATA_ROOT / "training"   / "WELL_B.las",
    "WELL_C": DATA_ROOT / "training"   / "WELL_C.las",
    "WELL_D": DATA_ROOT / "validation" / "WELL_D.las",
    "WELL_E": DATA_ROOT / "test"       / "WELL_E.las",
}

# ── depth grid ──────────────────────────────────────────────────────────────
DEPTH_START = 8000.0   # ft MD
DEPTH_STOP  = 11000.0  # ft MD
DEPTH_STEP  = 0.5      # ft
DEPTH       = np.arange(DEPTH_START, DEPTH_STOP + DEPTH_STEP, DEPTH_STEP)
N           = len(DEPTH)

LAYER_TOPS = {           # approximate MD (ft)
    "shale_cap":    8000,
    "sand_body":    8200,
    "interbedded":  8700,
    "tight_shale":  9000,
}


# ── helpers ─────────────────────────────────────────────────────────────────

def _layer_mask():
    """Return integer layer index for each depth sample."""
    masks = np.zeros(N, dtype=int)   # 0 = shale_cap
    masks[DEPTH >= LAYER_TOPS["sand_body"]]   = 1
    masks[DEPTH >= LAYER_TOPS["interbedded"]] = 2
    masks[DEPTH >= LAYER_TOPS["tight_shale"]] = 3
    return masks


def _base_gr(mask):
    """Gamma ray base signal by layer (GAPI)."""
    base = np.where(mask == 0, 110.0,   # shale cap
           np.where(mask == 1,  20.0,   # sand body
           np.where(mask == 2,  65.0,   # interbedded
                                100.0)))# tight shale
    # interbedded: add sinusoidal variation
    sine_zone = mask == 2
    base[sine_zone] += 45.0 * np.sin(
        2 * np.pi * (DEPTH[sine_zone] - LAYER_TOPS["interbedded"]) / 20.0
    )
    return base


def _base_rhob(mask):
    """Bulk density base signal (g/cm3)."""
    return np.where(mask == 0, 2.50,
           np.where(mask == 1, 2.30,
           np.where(mask == 2, 2.45,
                               2.60)))


def _base_nphi(mask):
    """Neutron porosity base signal (v/v)."""
    return np.where(mask == 0, 0.28,
           np.where(mask == 1, 0.18,
           np.where(mask == 2, 0.22,
                               0.08)))


def _base_rt(mask):
    """True resistivity base signal (ohm.m)."""
    return np.where(mask == 0,   1.5,
           np.where(mask == 1, 150.0,
           np.where(mask == 2,  20.0,
                                 2.5)))


def _base_dt(mask):
    """Sonic travel time base signal (us/ft)."""
    return np.where(mask == 0,  95.0,
           np.where(mask == 1,  75.0,
           np.where(mask == 2,  85.0,
                                60.0)))


def _base_pef(mask):
    """Photoelectric factor base signal (b/e)."""
    return np.where(mask == 0, 3.5,
           np.where(mask == 1, 1.8,
           np.where(mask == 2, 2.5,
                               2.6)))


def _noise(scale, size=None):
    return RNG.normal(0, scale, size=size or N)


def _add_spikes(arr, n_spikes, magnitude_range=(3.0, 8.0)):
    """Inject n_spikes point anomalies into arr (in-place clone)."""
    out   = arr.copy()
    idxs  = RNG.integers(50, N - 50, size=n_spikes)
    signs = RNG.choice([-1, 1], size=n_spikes)
    mags  = RNG.uniform(*magnitude_range, size=n_spikes)
    out[idxs] += signs * mags * np.std(arr)
    return out


def _add_washout(cali, bs, start_ft, end_ft, max_deviation=3.5):
    """Widen caliper in a washout zone."""
    out  = cali.copy()
    zone = (DEPTH >= start_ft) & (DEPTH <= end_ft)
    out[zone] += RNG.uniform(1.5, max_deviation, size=zone.sum())
    return out


def _add_stuck_interval(arr, start_ft, end_ft):
    """Replace an interval with the constant value at start."""
    out  = arr.copy()
    zone = (DEPTH >= start_ft) & (DEPTH <= end_ft)
    out[zone] = out[np.argmax(zone)]
    return out


def _null_interval(arr, start_ft, end_ft, null=-999.25):
    """Set an interval to null (simulates no-data zone)."""
    out  = arr.copy()
    zone = (DEPTH >= start_ft) & (DEPTH <= end_ft)
    out[zone] = null
    return out


def _build_las(well_name: str, api: str, curves: dict[str, dict]) -> lasio.LASFile:
    """
    Construct a lasio LASFile from a dict of curve definitions.

    Parameters
    ----------
    well_name : str
    api       : str  — synthetic API number
    curves    : dict[str, dict]
        keys   = curve mnemonic
        values = {"data": np.ndarray, "unit": str, "descr": str}
    """
    las = lasio.LASFile()

    # -- Well section
    las.well["WELL"]  = lasio.HeaderItem("WELL",  value=well_name,
                                         descr="Well name")
    las.well["API"]   = lasio.HeaderItem("API",   value=api,
                                         descr="API number (synthetic)")
    las.well["FLD"]   = lasio.HeaderItem("FLD",   value="SYNTHETIC FIELD",
                                         descr="Field name")
    las.well["COMP"]  = lasio.HeaderItem("COMP",  value="TEST COMPANY",
                                         descr="Company")
    las.well["SRVC"]  = lasio.HeaderItem("SRVC",  value="SYNTHETIC SERVICE CO",
                                         descr="Service company")
    las.well["DATE"]  = lasio.HeaderItem("DATE",  value="2024-01-01",
                                         descr="Log date")
    las.well["UWI"]   = lasio.HeaderItem("UWI",   value=api,
                                         descr="Unique well identifier")
    las.well["STRT"]  = lasio.HeaderItem("STRT",  value=DEPTH_START,
                                         unit="FT",
                                         descr="Start depth")
    las.well["STOP"]  = lasio.HeaderItem("STOP",  value=DEPTH[-1],
                                         unit="FT",
                                         descr="Stop depth")
    las.well["STEP"]  = lasio.HeaderItem("STEP",  value=DEPTH_STEP,
                                         unit="FT",
                                         descr="Depth increment")
    las.well["NULL"]  = lasio.HeaderItem("NULL",  value=-999.25,
                                         descr="Null value")

    # -- Params section — document synthetic nature explicitly
    las.params["REMARK"] = lasio.HeaderItem(
        "REMARK",
        value="SYNTHETIC DATA — FOR SOFTWARE TESTING ONLY. "
              "Do not use for geological or reservoir interpretation.",
        descr="Data provenance"
    )

    # -- Curves
    las.append_curve("DEPT", DEPTH, unit="FT", descr="Measured depth")
    for mnem, info in curves.items():
        las.append_curve(mnem, info["data"],
                         unit=info["unit"], descr=info["descr"])
    return las


# ── well factories ───────────────────────────────────────────────────────────

def make_well_a() -> lasio.LASFile:
    """
    WELL_A — Training well.
    Full curve suite, clean borehole, minimal defects.
    Purpose: baseline for module smoke tests.
    """
    mask = _layer_mask()
    bs   = np.full(N, 8.5)   # 8.5-inch bit throughout

    curves = {
        "GR":   {"data": np.clip(_base_gr(mask)   + _noise(4.0), 0, 300),
                 "unit": "GAPI", "descr": "Gamma Ray"},
        "CALI": {"data": np.clip(bs + RNG.uniform(0.2, 0.6, N), 4, 20),
                 "unit": "IN",   "descr": "Caliper"},
        "BS":   {"data": bs,
                 "unit": "IN",   "descr": "Bit size"},
        "RHOB": {"data": np.clip(_base_rhob(mask)  + _noise(0.02), 1.0, 3.5),
                 "unit": "G/C3", "descr": "Bulk density"},
        "NPHI": {"data": np.clip(_base_nphi(mask)  + _noise(0.01), -0.1, 0.6),
                 "unit": "V/V",  "descr": "Neutron porosity"},
        "RT":   {"data": np.clip(_base_rt(mask)    + _noise(1.0),  0.01, 50000),
                 "unit": "OHMM", "descr": "True resistivity"},
        "DT":   {"data": np.clip(_base_dt(mask)    + _noise(1.0),  40, 240),
                 "unit": "US/F", "descr": "Sonic travel time"},
        "PEF":  {"data": np.clip(_base_pef(mask)   + _noise(0.05), 0.5, 10),
                 "unit": "B/E",  "descr": "Photoelectric factor"},
    }
    return _build_las("WELL_A", "42-000-00001-00-00", curves)


def make_well_b() -> lasio.LASFile:
    """
    WELL_B — Training well.
    Full curve suite, washout 8800–9000 ft, 15 GR spikes, 8 RHOB spikes.
    Purpose: test caliper QC and spike-detection modules.
    """
    mask = _layer_mask()
    bs   = np.full(N, 8.5)
    cali = np.clip(bs + RNG.uniform(0.2, 0.6, N), 4, 20)
    cali = _add_washout(cali, bs, 8800, 9000)

    gr   = _add_spikes(_base_gr(mask)   + _noise(4.0), n_spikes=15)
    rhob = _add_spikes(_base_rhob(mask) + _noise(0.02), n_spikes=8,
                       magnitude_range=(2.0, 5.0))

    curves = {
        "GR":   {"data": np.clip(gr,   0, 300),
                 "unit": "GAPI", "descr": "Gamma Ray"},
        "CALI": {"data": np.clip(cali, 4,  20),
                 "unit": "IN",   "descr": "Caliper"},
        "BS":   {"data": bs,
                 "unit": "IN",   "descr": "Bit size"},
        "RHOB": {"data": np.clip(rhob, 1.0, 3.5),
                 "unit": "G/C3", "descr": "Bulk density"},
        "NPHI": {"data": np.clip(_base_nphi(mask) + _noise(0.01), -0.1, 0.6),
                 "unit": "V/V",  "descr": "Neutron porosity"},
        "RT":   {"data": np.clip(_base_rt(mask)   + _noise(1.0),  0.01, 50000),
                 "unit": "OHMM", "descr": "True resistivity"},
        "DT":   {"data": np.clip(_base_dt(mask)   + _noise(1.0),  40, 240),
                 "unit": "US/F", "descr": "Sonic travel time"},
        "PEF":  {"data": np.clip(_base_pef(mask)  + _noise(0.05), 0.5, 10),
                 "unit": "B/E",  "descr": "Photoelectric factor"},
    }
    return _build_las("WELL_B", "42-000-00002-00-00", curves)


def make_well_c() -> lasio.LASFile:
    """
    WELL_C — Training well.
    Missing DT and PEF. No bit-size curve.
    Purpose: test graceful handling of absent curves.
    """
    mask = _layer_mask()
    bs   = np.full(N, 9.875)
    cali = np.clip(bs + RNG.uniform(0.1, 0.8, N), 4, 20)

    # Stuck interval on GR: 9400-9420 ft
    gr = _base_gr(mask) + _noise(5.0)
    gr = _add_stuck_interval(gr, 9400, 9420)

    curves = {
        "GR":   {"data": np.clip(gr, 0, 300),
                 "unit": "GAPI", "descr": "Gamma Ray"},
        "CALI": {"data": np.clip(cali, 4, 20),
                 "unit": "IN",   "descr": "Caliper"},
        # No BS — tests missing-BS handling
        "RHOB": {"data": np.clip(_base_rhob(mask) + _noise(0.025), 1.0, 3.5),
                 "unit": "G/C3", "descr": "Bulk density"},
        "NPHI": {"data": np.clip(_base_nphi(mask) + _noise(0.012), -0.1, 0.6),
                 "unit": "V/V",  "descr": "Neutron porosity"},
        "ILD":  {"data": np.clip(_base_rt(mask)   + _noise(2.0),  0.01, 50000),
                 "unit": "OHMM", "descr": "Induction deep (alias for RT)"},
        # No DT, no PEF — tests missing-curve reporting
    }
    return _build_las("WELL_C", "42-000-00003-00-00", curves)


def make_well_d() -> lasio.LASFile:
    """
    WELL_D — Validation well.
    No BS curve, moderate spikes, tight-hole section (CALI < BS equivalent),
    null data gap on RHOB 9600–9620 ft.
    Purpose: test validation metrics and edge-case QC flags.
    """
    mask = _layer_mask()
    # No BS available, but we know the bit was 8.5 in — reflected in CALI shape
    cali_base = 8.5 + RNG.uniform(0.1, 0.5, N)
    # tight-hole: CALI dips below nominal bit size at 9200-9250 ft
    tight = (DEPTH >= 9200) & (DEPTH <= 9250)
    cali_base[tight] = 8.5 - RNG.uniform(0.1, 0.4, cali_base[tight].shape[0])

    rhob = _base_rhob(mask) + _noise(0.03)
    rhob = _null_interval(rhob, 9600, 9620)   # simulate dropout / tool issue

    gr = _add_spikes(_base_gr(mask) + _noise(5.0), n_spikes=10)

    curves = {
        "GR":   {"data": np.clip(gr,   0, 300),
                 "unit": "GAPI", "descr": "Gamma Ray"},
        "CALI": {"data": np.clip(cali_base, 4, 20),
                 "unit": "IN",   "descr": "Caliper"},
        # No BS
        "RHOB": {"data": rhob,        # contains nulls — left as-is
                 "unit": "G/C3", "descr": "Bulk density"},
        "NPHI": {"data": np.clip(_base_nphi(mask) + _noise(0.015), -0.1, 0.6),
                 "unit": "V/V",  "descr": "Neutron porosity"},
        "LLD":  {"data": np.clip(_base_rt(mask)   + _noise(1.5),  0.01, 50000),
                 "unit": "OHMM", "descr": "Laterolog deep (alias for RT)"},
        "DT":   {"data": np.clip(_base_dt(mask)   + _noise(2.0),  40, 240),
                 "unit": "US/F", "descr": "Sonic travel time"},
        # No PEF
    }
    return _build_las("WELL_D", "42-000-00004-00-00", curves)


def make_well_e() -> lasio.LASFile:
    """
    WELL_E — HELD-OUT TEST WELL.
    Full suite. Different noise seed. Slightly different layer depths.
    This file is written but NOT described or inspected in this script.
    It exists only in data/raw/test/ and must not be opened until Stage 9.
    """
    rng2 = np.random.default_rng(seed=99)   # different seed
    # Shift layer boundaries slightly
    local_tops = {
        "shale_cap":   8000,
        "sand_body":   8250,
        "interbedded": 8800,
        "tight_shale": 9100,
    }
    mask = np.zeros(N, dtype=int)
    mask[DEPTH >= local_tops["sand_body"]]   = 1
    mask[DEPTH >= local_tops["interbedded"]] = 2
    mask[DEPTH >= local_tops["tight_shale"]] = 3

    bs   = np.full(N, 8.5)
    cali = np.clip(bs + rng2.uniform(0.2, 0.7, N), 4, 20)
    # One washout zone
    zone = (DEPTH >= 9050) & (DEPTH <= 9200)
    cali[zone] += rng2.uniform(1.0, 2.5, zone.sum())

    curves = {
        "GR":   {"data": np.clip(_base_gr(mask)   + rng2.normal(0,4.0,N), 0, 300),
                 "unit": "GAPI", "descr": "Gamma Ray"},
        "CALI": {"data": cali,
                 "unit": "IN",   "descr": "Caliper"},
        "BS":   {"data": bs,
                 "unit": "IN",   "descr": "Bit size"},
        "RHOB": {"data": np.clip(_base_rhob(mask) + rng2.normal(0,0.02,N), 1.0, 3.5),
                 "unit": "G/C3", "descr": "Bulk density"},
        "NPHI": {"data": np.clip(_base_nphi(mask) + rng2.normal(0,0.01,N), -0.1, 0.6),
                 "unit": "V/V",  "descr": "Neutron porosity"},
        "RT":   {"data": np.clip(_base_rt(mask)   + rng2.normal(0,1.0,N),  0.01, 50000),
                 "unit": "OHMM", "descr": "True resistivity"},
        "DT":   {"data": np.clip(_base_dt(mask)   + rng2.normal(0,1.0,N),  40, 240),
                 "unit": "US/F", "descr": "Sonic travel time"},
        "PEF":  {"data": np.clip(_base_pef(mask)  + rng2.normal(0,0.05,N), 0.5, 10),
                 "unit": "B/E",  "descr": "Photoelectric factor"},
    }
    return _build_las("WELL_E", "42-000-00005-00-00", curves)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    factories = {
        "WELL_A": make_well_a,
        "WELL_B": make_well_b,
        "WELL_C": make_well_c,
        "WELL_D": make_well_d,
        "WELL_E": make_well_e,
    }

    for name, factory in factories.items():
        out_path = WELL_PATHS[name]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        las = factory()
        las.write(str(out_path), version=2.0)
        n_curves = len(las.curves) - 1   # subtract DEPT
        print(f"  Written {out_path.name}  ({n_curves} curves, {N:,} rows)")

    print("\nAll synthetic LAS files written.")
    print("REMINDER: These files are SYNTHETIC — for software testing only.")
    print(f"  Training:   {DATA_ROOT / 'training'}")
    print(f"  Validation: {DATA_ROOT / 'validation'}")
    print(f"  Test (held-out): {DATA_ROOT / 'test'}")


if __name__ == "__main__":
    main()
