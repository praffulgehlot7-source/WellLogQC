"""
tests/test_cross_correlation.py
===============================
Tests for multi-curve petrophysical cross-correlation and cross-examination:
- Clean oil pay confirmation
- Genuine gas crossover vs washout pad standoff artifact disambiguation
- Radioactive arkosic sand pay recovery via SP cross-referencing
- Saline aquifer identification
- Error minimization metrics and executive conclusions
"""

import pytest
import numpy as np
import pandas as pd
from welllogqc.petrophysics.cross_correlation import run_multicurve_cross_correlation


def test_multicurve_clean_oil_pay():
    depths = np.linspace(8000.0, 8020.0, 21)
    df = pd.DataFrame({
        "GR": np.full(21, 30.0),       # Clean quartz sand
        "RT": np.full(21, 65.0),       # High hydrocarbon resistivity
        "CALI": np.full(21, 8.5),      # On-gauge
        "BS": np.full(21, 8.5),
        "RHOB": np.full(21, 2.25),     # ~24% porosity
        "NPHI": np.full(21, 0.24),     # Liquid tracking
        "SP": np.full(21, -45.0),      # Strong negative deflection (permeable)
    }, index=depths)

    res = run_multicurve_cross_correlation(df, bs_default=8.5)
    cc_df = res["cross_corr_df"]
    assert (cc_df["Classification"] == "CONFIRMED_OIL_PAY").all()
    assert (cc_df["Confidence"] == "VERY_HIGH").all()
    assert res["summary_metrics"]["confirmed_clean_oil_pay_ft"] > 0.0


def test_genuine_gas_crossover_on_gauge():
    depths = np.linspace(8050.0, 8060.0, 11)
    df = pd.DataFrame({
        "GR": np.full(11, 28.0),
        "RT": np.full(11, 80.0),
        "CALI": np.full(11, 8.5),      # On-gauge borehole
        "BS": np.full(11, 8.5),
        "RHOB": np.full(11, 2.15),     # ~30% density porosity
        "NPHI": np.full(11, 0.12),     # Low neutron -> prominent crossover
        "SP": np.full(11, -50.0),
    }, index=depths)

    res = run_multicurve_cross_correlation(df, bs_default=8.5)
    cc_df = res["cross_corr_df"]
    assert (cc_df["Classification"] == "CONFIRMED_GAS_PAY").all()
    assert res["summary_metrics"]["confirmed_gas_pay_ft"] > 0.0


def test_false_gas_crossover_debunked_in_washout():
    # In a severe washout, pad standoff makes density and neutron look like crossover
    depths = np.linspace(8100.0, 8110.0, 11)
    df = pd.DataFrame({
        "GR": np.full(11, 35.0),
        "RT": np.full(11, 25.0),
        "CALI": np.full(11, 11.2),     # +2.7 in washout!
        "BS": np.full(11, 8.5),
        "RHOB": np.full(11, 2.05),     # Apparent low density due to mud
        "NPHI": np.full(11, 0.15),
        "SP": np.full(11, -30.0),
    }, index=depths)

    res = run_multicurve_cross_correlation(df, bs_default=8.5)
    cc_df = res["cross_corr_df"]
    # Should debunk false gas and flag as washout artifact
    assert (cc_df["Classification"] == "WASHOUT_ARTIFACT").all()
    assert res["summary_metrics"]["false_gas_alarms_debunked_ft"] > 0.0


def test_radioactive_arkosic_sand_recovery():
    # High GR usually gets rejected as shale, but SP deflection and RT prove it is pay!
    depths = np.linspace(8200.0, 8210.0, 11)
    df = pd.DataFrame({
        "GR": np.full(11, 95.0),       # High GR (feldspathic minerals)
        "RT": np.full(11, 45.0),       # Resistive pay
        "CALI": np.full(11, 8.6),      # On-gauge
        "BS": np.full(11, 8.5),
        "RHOB": np.full(11, 2.28),     # 22% porosity
        "NPHI": np.full(11, 0.22),
        "SP": np.full(11, -48.0),      # Strong negative SP confirms permeable sand matrix!
    }, index=depths)

    res = run_multicurve_cross_correlation(df, bs_default=8.5)
    cc_df = res["cross_corr_df"]
    assert (cc_df["Classification"] == "RADIOACTIVE_ARKOSIC_PAY").all()
    assert res["summary_metrics"]["radioactive_pay_recovered_ft"] > 0.0


def test_saline_aquifer_identification():
    depths = np.linspace(8300.0, 8310.0, 11)
    df = pd.DataFrame({
        "GR": np.full(11, 30.0),       # Clean sand
        "RT": np.full(11, 1.2),        # Extremely conductive (saline water wet)
        "CALI": np.full(11, 8.5),
        "BS": np.full(11, 8.5),
        "RHOB": np.full(11, 2.25),     # High porosity
        "NPHI": np.full(11, 0.25),
        "SP": np.full(11, -55.0),
    }, index=depths)

    res = run_multicurve_cross_correlation(df, bs_default=8.5)
    cc_df = res["cross_corr_df"]
    assert (cc_df["Classification"] == "SALINE_WATER_AQUIFER").all()
