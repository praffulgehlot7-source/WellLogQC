"""
tests/test_curve_guide.py
=========================
Unit tests for the Petrophysical Reference Guide and Hover Point Evaluator.
"""

import numpy as np
import pandas as pd
import pytest

from welllogqc.advisory.curve_guide import (
    evaluate_point_status,
    generate_hover_customdata,
    get_curve_guide_for_well,
    get_all_curve_guides,
    get_guide_comparison_dataframe,
    CURVE_KNOWLEDGE_BASE,
)


def test_evaluate_point_status_gr_143():
    """Verify that GR at 143 gAPI triggers an elevated shale warning explaining its meaning."""
    status, meaning = evaluate_point_status("GR", 143.0)
    assert "Very High" in status
    assert "Shale" in status
    assert "143.0 gAPI" in meaning
    assert "clean" in meaning.lower()
    assert "zero" in meaning.lower() or "impermeable" in meaning.lower()


def test_evaluate_point_status_gr_clean():
    """Verify that GR at 25 gAPI is classified as clean sandstone."""
    status, meaning = evaluate_point_status("GR", 25.0)
    assert "Optimal" in status
    assert "Clean" in status
    assert "permeability" in meaning.lower()


def test_evaluate_point_status_caliper_washout():
    """Verify that caliper > bit size by 2.0 in flags severe breakout."""
    status, meaning = evaluate_point_status("CALI", 10.5, aux={"BS": 8.5})
    assert "Washout" in status or "Breakout" in status
    assert "+2.00 in" in meaning
    assert "pad" in meaning.lower() or "density" in meaning.lower()


def test_evaluate_point_status_caliper_ongauge():
    """Verify that on-gauge caliper is marked optimal."""
    status, meaning = evaluate_point_status("CALI", 8.55, aux={"BS": 8.5})
    assert "Optimal" in status
    assert "On-Gauge" in status


def test_evaluate_point_status_rt():
    """Verify deep resistivity interpretation for pay vs brine."""
    status_pay, meaning_pay = evaluate_point_status("RT", 85.0)
    assert "Optimal" in status_pay
    assert "Hydrocarbon" in status_pay
    assert "85.0 Ohm.m" in meaning_pay

    status_wet, meaning_wet = evaluate_point_status("RT", 1.2)
    assert "Low" in status_wet
    assert "Water" in meaning_wet.lower() or "saline" in meaning_wet.lower()


def test_evaluate_point_status_porosity():
    """Verify density porosity evaluation."""
    status_high, meaning_high = evaluate_point_status("PHID", 0.24)
    assert "Optimal" in status_high
    assert "24.0%" in meaning_high

    status_tight, meaning_tight = evaluate_point_status("PHID", 0.05)
    assert "Tight" in status_tight


def test_evaluate_point_status_water_saturation():
    """Verify Archie Sw evaluation."""
    status_pay, meaning_pay = evaluate_point_status("SW", 0.22)
    assert "Optimal" in status_pay
    assert "22.0%" in meaning_pay
    assert "Hydrocarbon" in meaning_pay

    status_wet, meaning_wet = evaluate_point_status("SW", 0.85)
    assert "Water" in status_wet


def test_generate_hover_customdata():
    """Verify batch customdata array generation for Plotly traces."""
    gr_vals = pd.Series([22.0, 55.0, 80.0, 143.0])
    customdata, template = generate_hover_customdata("GR", gr_vals)
    assert customdata.shape == (4, 2)
    assert "Optimal Status" in template
    assert "Meaning" in template
    # Point 4 (143 gAPI)
    assert "Very High" in customdata[3, 0]
    assert "143.0 gAPI" in customdata[3, 1]


def test_get_curve_guide_for_well():
    """Verify curve guide filtering for available well curves."""
    curves = ["GR", "RHOB", "NPHI", "RT", "CALI", "UNKNOWN_LOG"]
    guides = get_curve_guide_for_well(curves)
    guide_names = [g["curve_symbol"] for g in guides]
    assert "GR" in guide_names
    assert "RHOB" in guide_names
    assert "NPHI" in guide_names
    assert "RT" in guide_names
    assert "CALI" in guide_names
    assert "UNKNOWN_LOG" not in guide_names


def test_get_guide_comparison_dataframe():
    """Verify reference table generation."""
    df = get_guide_comparison_dataframe(["GR", "RT"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 2
    assert "Curve" in df.columns
    assert "Ideal Commercial Reservoir Range" in df.columns
    assert "Authoritative Literature Source" in df.columns
