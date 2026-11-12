"""
tests/test_gpu_and_advisory.py
==============================
Unit tests for GPU telemetry, ideal reservoir benchmarking,
and the petrophysical decision advisory engine.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from welllogqc.service.gpu_detector import detect_gpu
from welllogqc.advisory.ideal_reservoir import (
    IDEAL_RESERVOIR_BENCHMARKS,
    compute_ideal_reservoir_gap_analysis,
    get_ideal_benchmark_comparison_table,
)
from welllogqc.advisory.decision_engine import (
    generate_interval_decisions,
    generate_actionable_advisories
)
from welllogqc.ingestion.web_fetcher import list_public_benchmark_sources

def test_gpu_detection():
    gpu = detect_gpu()
    assert isinstance(gpu, dict)
    assert "gpu_available" in gpu
    assert "device_name" in gpu
    assert "total_memory_mb" in gpu
    assert "used_memory_mb" in gpu
    # On this machine, NVIDIA GPU was verified
    assert gpu["gpu_available"] is True
    assert "GeForce" in gpu["device_name"]

def test_ideal_reservoir_benchmarking():
    depth = np.linspace(8000, 8500, 1001)
    df = pd.DataFrame({
        "GR": np.full(1001, 25.0),
        "CALI": np.full(1001, 8.4),
        "BS": np.full(1001, 8.5),
        "RHOB": np.full(1001, 2.22),
        "RT": np.full(1001, 75.0)
    }, index=depth)
    
    vsh = pd.Series(np.full(1001, 0.08), index=depth)
    phid = pd.Series(np.full(1001, 0.24), index=depth)
    sw = pd.Series(np.full(1001, 0.20), index=depth)

    res = compute_ideal_reservoir_gap_analysis(df, vsh=vsh, phid=phid, sw=sw)
    assert res["composite_reservoir_proximity_pct"] > 85.0
    assert "Tier 1" in res["reservoir_tier"]
    assert res["gauge_stability_index"] >= 90.0

    comp_df = get_ideal_benchmark_comparison_table(res)
    assert isinstance(comp_df, pd.DataFrame)
    assert "Key Comparison Notes & Concluding Remarks" in comp_df.columns
    assert "Given Well Data" in comp_df.columns
    assert "Ideal Benchmark Standard" in comp_df.columns
    assert len(comp_df) >= 4
    for note in comp_df["Key Comparison Notes & Concluding Remarks"]:
        assert len(note) > 15

def test_interval_decisions():
    depth = np.linspace(8000, 8500, 1001)
    df = pd.DataFrame({"CALI": 8.5, "BS": 8.5}, index=depth)
    qc_flags = pd.DataFrame([
        {"top": 8100.0, "base": 8150.0, "issue": "WASHOUT", "severity": "WARNING"},
        {"top": 8200.0, "base": 8220.0, "issue": "STUCK_TOOL", "severity": "CRITICAL"}
    ])
    
    decisions = generate_interval_decisions(df, qc_flags=qc_flags)
    assert len(decisions) == 2
    assert "REJECTED_WASHOUT" in decisions["decision"].values
    assert "RE_LOG_RECOMMENDED" in decisions["decision"].values

def test_actionable_advisories():
    depth = np.linspace(8000, 8500, 1001)
    df = pd.DataFrame({"CALI": 8.5, "BS": 8.5}, index=depth)
    qc_flags = pd.DataFrame([
        {"top": 8100.0, "base": 8150.0, "issue": "WASHOUT", "severity": "WARNING"}
    ])
    prox = {"reservoir_tier": "Tier 1: World-Class Commercial Pay", "composite_reservoir_proximity_pct": 92.0}
    
    adv = generate_actionable_advisories(df, qc_flags=qc_flags, proximity_analysis=prox)
    assert len(adv) >= 2
    categories = [a["category"] for a in adv]
    assert "Drilling & Borehole Geomechanics" in categories
    assert "Petrophysics & Formation Evaluation" in categories

def test_web_fetcher_sources():
    sources = list_public_benchmark_sources()
    assert len(sources) >= 2
    for s in sources:
        assert s["url"].startswith("http")
        assert s["default_filename"].endswith(".las")

def test_train_qc_model_xgboost_gpu():
    from welllogqc.ingestion.las_reader import LASReader
    from welllogqc.qc.qc_reporter import run_all_qc
    from welllogqc.ml.problem_definition import flags_to_labels
    from welllogqc.ml.train import train_qc_model
    from welllogqc.ml.inference import predict_well

    reader = LASReader()
    well_a = reader.read("data/raw/training/WELL_A.las", role="training")
    well_b = reader.read("data/raw/training/WELL_B.las", role="training")

    flags_a = run_all_qc(well_a)
    flags_b = run_all_qc(well_b)
    labels_a = flags_to_labels(well_a.df, flags_a, mode="binary")
    labels_b = flags_to_labels(well_b.df, flags_b, mode="binary")

    res = train_qc_model(
        training_data=[(well_a, labels_a), (well_b, labels_b)],
        model_type="xgboost_gpu",
        feature_window=11,
        save_dir="models",
        random_state=42
    )

    assert res.model is not None
    assert "xgb" in type(res.model).__name__.lower()
    assert res.train_metrics["f1"] >= 0.0

    preds = predict_well(
        ingestion_result=well_a,
        model=res.model,
        preprocessor=res.preprocessor,
        decision_threshold=0.5
    )
    assert "QC_RISK_SCORE" in preds.columns
    assert len(preds) == len(well_a.df)

def test_gpu_service_endpoints():
    from welllogqc.service.server import (
        health_check,
        get_engine_status,
        get_ideal_benchmark,
        run_gpu_inference,
        api_load_model,
        PredictRequest,
        LoadModelRequest
    )

    hc = health_check()
    assert hc["status"] == "ONLINE"
    assert hc["gpu"]["gpu_available"] is True

    status = get_engine_status()
    assert "device_mode" in status
    assert "active_model_loaded" in status

    bench = get_ideal_benchmark()
    assert "reservoir_type" in bench
    assert "Wilcox" in bench["reservoir_type"]

    req = PredictRequest(
        well_name="TEST_WELL",
        features=[{"CALI": 8.5, "GR": 45.0, "RHOB": 2.25}, {"CALI": 9.2, "GR": 95.0, "RHOB": 2.45}],
        threshold=0.5
    )
    pred_res = run_gpu_inference(req)
    assert pred_res["well_name"] == "TEST_WELL"
    assert pred_res["sample_count"] == 2
    assert len(pred_res["risk_scores"]) == 2
    assert "device_used" in pred_res
    assert "model_name" in pred_res

    # Test load_model endpoint
    load_res = api_load_model(LoadModelRequest())
    assert load_res["success"] is True
    assert "model_name" in load_res
