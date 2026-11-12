"""
tests/test_pipeline.py
----------------------
Automated unit and regression tests for core pipeline modules:
- Ingestion & Curve Mapping
- Conventional QC
- Petrophysical Calculations (Vsh, Porosity, NTG)
- Feature Engineering Matrix
- Model Training & Inference

All tests use synthetic well datasets for reproducibility.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from welllogqc.ingestion.curve_mapper import CurveMapper
from welllogqc.ingestion.las_reader import LASReader
from welllogqc.qc.qc_reporter import run_all_qc, summarise_flags
from welllogqc.petrophysics.vsh import calc_vsh_linear
from welllogqc.petrophysics.porosity import calc_density_porosity
from welllogqc.petrophysics.ntg import calc_ntg
from welllogqc.features.feature_registry import build_feature_matrix
from welllogqc.ml.problem_definition import flags_to_labels
from welllogqc.ml.train import train_qc_model
from welllogqc.ml.inference import predict_well


@pytest.fixture(scope="module")
def well_a_result():
    reader = LASReader()
    return reader.read("data/raw/training/WELL_A.las", role="training")


@pytest.fixture(scope="module")
def well_b_result():
    reader = LASReader()
    return reader.read("data/raw/training/WELL_B.las", role="training")


class TestIngestion:
    def test_curve_mapper_resolves_common_aliases(self):
        mapper = CurveMapper()
        mapped = mapper.map(["DEPT", "GRC", "HCAL", "DEN", "CNCF", "ILD"])
        assert mapped["DEPT"] == "DEPT"
        assert mapped["GRC"] == "GR"
        assert mapped["HCAL"] == "CALI"
        assert mapped["DEN"] == "RHOB"
        assert mapped["CNCF"] == "NPHI"
        assert mapped["ILD"] == "RT"

    def test_las_reader_depth_index(self, well_a_result):
        assert well_a_result.df.index.name == "DEPT"
        assert np.all(np.diff(well_a_result.df.index.to_numpy()) > 0)
        assert well_a_result.depth_start == 8000.0
        assert well_a_result.depth_stop == 11000.0


class TestQC:
    def test_qc_schema_and_flags(self, well_b_result):
        flags = run_all_qc(well_b_result)
        required_cols = {"well_name", "depth_top", "depth_base", "curve", "issue", "severity", "description"}
        assert required_cols.issubset(flags.columns)
        assert len(flags) > 0
        summary = summarise_flags(flags)
        assert "total" in summary
        assert summary["total"] == len(flags)


class TestPetrophysics:
    def test_vsh_linear_bounds(self, well_a_result):
        vsh = calc_vsh_linear(well_a_result.df, gr_clean=15.0, gr_shale=120.0)
        assert vsh.name == "VSH_LINEAR"
        assert vsh.min() >= 0.0
        assert vsh.max() <= 1.0

    def test_density_porosity(self, well_a_result):
        phid = calc_density_porosity(well_a_result.df, rho_matrix=2.65, rho_fluid=1.00)
        assert phid.name == "PHID"
        assert phid.dropna().shape[0] > 0

    def test_ntg_calculation(self, well_a_result):
        df = well_a_result.df.copy()
        df["VSH_LINEAR"] = calc_vsh_linear(df, gr_clean=15.0, gr_shale=120.0)
        df["PHID"] = calc_density_porosity(df)
        ntg = calc_ntg(df, gross_top=8200.0, gross_base=8700.0)
        assert 0.0 <= ntg["ntg"] <= 1.0
        assert ntg["gross_ft"] > 0
        assert "summary" in ntg


class TestMLPipeline:
    def test_train_and_predict(self, well_a_result, well_b_result):
        flags_a = run_all_qc(well_a_result)
        flags_b = run_all_qc(well_b_result)

        labels_a = flags_to_labels(well_a_result.df, flags_a, mode="binary")
        labels_b = flags_to_labels(well_b_result.df, flags_b, mode="binary")

        train_res = train_qc_model(
            training_data=[(well_a_result, labels_a), (well_b_result, labels_b)],
            model_type="random_forest",
            feature_window=11,
            save_dir="models",
            random_state=42,
        )
        assert train_res.model is not None
        assert train_res.preprocessor is not None

        preds = predict_well(
            ingestion_result=well_a_result,
            model=train_res.model,
            preprocessor=train_res.preprocessor,
            decision_threshold=0.5,
        )
        assert "QC_RISK_SCORE" in preds.columns
        assert "QC_RISK_LABEL" in preds.columns
        assert preds["QC_RISK_SCORE"].min() >= 0.0
        assert preds["QC_RISK_SCORE"].max() <= 1.0
