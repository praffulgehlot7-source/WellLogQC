"""
tests/test_environmental_correction.py
======================================
Tests for automated borehole environmental corrections:
- Standoff calculation
- Bulk density mudcake / pad standoff compensation
- Deep resistivity borehole fluid departure compensation
- Integrated petrophysical recalculation (phiD, phiE, Sw)
"""

import pytest
import numpy as np
import pandas as pd
from welllogqc.petrophysics.environmental_correction import (
    compute_borehole_standoff,
    correct_density_washout,
    correct_resistivity_washout,
    apply_automated_environmental_corrections,
)


def test_compute_borehole_standoff():
    cali = pd.Series([8.5, 9.5, 11.0, 8.2])
    bs = 8.5
    standoff = compute_borehole_standoff(cali, bs)
    assert np.isclose(standoff.iloc[0], 0.0)
    assert np.isclose(standoff.iloc[1], 1.0)
    assert np.isclose(standoff.iloc[2], 2.5)
    assert np.isclose(standoff.iloc[3], -0.3)


def test_correct_density_on_gauge():
    # When CALI == BS, density should remain unchanged
    rhob = pd.Series([2.30, 2.35, 2.40])
    cali = pd.Series([8.5, 8.5, 8.5])
    bs = 8.5
    corr_rhob, delta_rho = correct_density_washout(rhob, cali, bs)
    assert np.allclose(corr_rhob, rhob)
    assert np.allclose(delta_rho, 0.0)


def test_correct_density_washout():
    # In a washout, RHOB is artificially depressed by mud; correction should restore it
    rhob = pd.Series([2.05, 1.95])
    cali = pd.Series([10.0, 11.5])  # +1.5 in and +3.0 in washout
    bs = 8.5
    corr_rhob, delta_rho = correct_density_washout(rhob, cali, bs)
    assert (corr_rhob > rhob).all()
    assert (delta_rho > 0.0).all()
    assert (corr_rhob <= 2.65).all()


def test_correct_resistivity_washout():
    # In a washout with resistive rock and conductive mud, apparent Rt is depressed
    rt = pd.Series([50.0, 100.0])
    cali = pd.Series([10.5, 11.5])
    bs = 8.5
    rm = 0.8
    corr_rt, mult = correct_resistivity_washout(rt, cali, bs, rm=rm)
    assert (corr_rt > rt).all()
    assert (mult > 1.0).all()
    assert (mult <= 1.85).all()


def test_apply_automated_environmental_corrections_pipeline():
    depths = np.linspace(7000.0, 7100.0, 101)
    df = pd.DataFrame({
        "GR": np.full(101, 35.0),
        "CALI": np.full(101, 10.5),  # 2.0 in washout throughout
        "BS": np.full(101, 8.5),
        "RHOB": np.full(101, 2.10),
        "RT": np.full(101, 40.0),
        "DT": np.full(101, 85.0),
    }, index=depths)

    pkg = apply_automated_environmental_corrections(df, bs_default=8.5)
    assert pkg["corrections_applied"] is True
    assert "RHOB_CORR" in pkg["corrected_df"].columns
    assert "RT_CORR" in pkg["corrected_df"].columns
    assert "PHID_CORR" in pkg["corrected_df"].columns
    assert "SW_CORR" in pkg["corrected_df"].columns

    # Porosity before vs after: uncorrected had artificially high porosity due to pad standoff
    raw_phid = pkg["petro_raw"]["PHID"]
    corr_phid = pkg["petro_corr"]["PHID"]
    assert (corr_phid < raw_phid).all()  # corrected density is higher, so porosity is realistically reduced
    assert pkg["mean_density_correction_g_cm3"] > 0.0
    assert pkg["mean_resistivity_boost_pct"] > 0.0


def test_series_bs_partial_washout_broadcasting():
    # Exactly replicates user scenario: 401 points, 70 washout points, BS is a pd.Series
    depths = np.linspace(7000.0, 7400.0, 401)
    cali = np.full(401, 8.5)
    cali[:70] = 10.5  # Exactly 70 washout levels!
    
    df = pd.DataFrame({
        "GR": np.full(401, 35.0),
        "CALI": cali,
        "BS": pd.Series(np.full(401, 8.5), index=depths),
        "RHOB": np.full(401, 2.15),
        "RT": np.full(401, 40.0),
        "DT": np.full(401, 85.0),
    }, index=depths)

    # Should run with zero ValueError or shape mismatch
    pkg = apply_automated_environmental_corrections(df)
    assert pkg["corrections_applied"] is True
    assert pkg["washout_levels_count"] == 70
    assert len(pkg["corrected_df"]) == 401
    assert "RT_CORR" in pkg["corrected_df"].columns
    assert "RHOB_CORR" in pkg["corrected_df"].columns
