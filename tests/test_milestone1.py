"""
tests/test_milestone1.py
------------------------
Milestone 1 acceptance tests.

These tests verify:
1. The project configuration files are valid and loadable.
2. The synthetic LAS files exist and have correct structure.
3. The shared pytest fixtures in conftest.py work correctly.

ENGINEERING NOTE
----------------
These tests do NOT test any petrophysical logic — that begins in Milestone 2.
They are purely structural: "does the scaffold we built actually hold together?"
"""

import pathlib
import yaml
import pytest
import numpy as np
import pandas as pd


# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = pathlib.Path(__file__).parent.parent
CONFIG_DIR    = PROJECT_ROOT / "welllogqc" / "config"
DATA_TRAINING = PROJECT_ROOT / "data" / "raw" / "training"
DATA_VALID    = PROJECT_ROOT / "data" / "raw" / "validation"
DATA_TEST     = PROJECT_ROOT / "data" / "raw" / "test"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration files
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigFiles:
    """Verify that config files exist, parse correctly, and contain required keys."""

    def test_defaults_module_importable(self):
        """welllogqc.config.defaults must import without errors."""
        import welllogqc.config.defaults as cfg
        assert cfg is not None

    def test_defaults_has_curve_ranges(self):
        """CURVE_RANGES must define at least GR, RHOB, NPHI, RT."""
        from welllogqc.config.defaults import CURVE_RANGES
        required = {"GR", "RHOB", "NPHI", "RT"}
        assert required.issubset(CURVE_RANGES.keys()), (
            f"Missing entries in CURVE_RANGES: {required - CURVE_RANGES.keys()}"
        )

    def test_defaults_range_tuples_valid(self):
        """Every entry in CURVE_RANGES must be (min, max, unit) with min < max."""
        from welllogqc.config.defaults import CURVE_RANGES
        for name, (lo, hi, unit) in CURVE_RANGES.items():
            assert lo < hi, f"{name}: min ({lo}) >= max ({hi})"
            assert isinstance(unit, str) and len(unit) > 0, (
                f"{name}: unit must be a non-empty string"
            )

    def test_gr_references_sensible(self):
        """GR_CLEAN must be strictly less than GR_SHALE."""
        from welllogqc.config.defaults import GR_CLEAN, GR_SHALE
        assert GR_CLEAN < GR_SHALE, (
            f"GR_CLEAN ({GR_CLEAN}) must be < GR_SHALE ({GR_SHALE})"
        )

    def test_density_assumptions_physical(self):
        """Matrix density must be > fluid density (physical constraint)."""
        from welllogqc.config.defaults import RHOMAT, RHOFLUID
        assert RHOMAT > RHOFLUID, (
            f"RHOMAT ({RHOMAT}) must be > RHOFLUID ({RHOFLUID})"
        )

    def test_curve_aliases_yaml_parses(self):
        """curve_aliases.yaml must be valid YAML and contain canonical names."""
        alias_path = CONFIG_DIR / "curve_aliases.yaml"
        assert alias_path.exists(), f"Missing: {alias_path}"
        with open(alias_path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "curve_aliases.yaml must parse to a dict"
        required_canonicals = {"GR", "CALI", "RHOB", "NPHI", "RT", "DEPT"}
        assert required_canonicals.issubset(data.keys()), (
            f"Missing canonical names: {required_canonicals - data.keys()}"
        )

    def test_curve_aliases_are_lists(self):
        """Every canonical name must map to a non-empty list of aliases."""
        alias_path = CONFIG_DIR / "curve_aliases.yaml"
        with open(alias_path) as f:
            data = yaml.safe_load(f)
        for canon, aliases in data.items():
            assert isinstance(aliases, list) and len(aliases) > 0, (
                f"'{canon}' must map to a non-empty list, got: {aliases}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic LAS files
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticLASFiles:
    """Verify that generated LAS files exist and have the expected structure."""

    def test_training_wells_exist(self):
        for name in ["WELL_A.las", "WELL_B.las", "WELL_C.las"]:
            assert (DATA_TRAINING / name).exists(), f"Missing training file: {name}"

    def test_validation_well_exists(self):
        assert (DATA_VALID / "WELL_D.las").exists(), "Missing WELL_D.las"

    def test_test_well_exists(self):
        assert (DATA_TEST / "WELL_E.las").exists(), "Missing WELL_E.las"

    def test_well_a_curve_count(self, synthetic_las_path):
        """WELL_A should have exactly 8 curves (excluding DEPT)."""
        import lasio
        las = lasio.read(str(synthetic_las_path))
        n_curves = len(las.curves) - 1
        assert n_curves == 8, f"Expected 8 curves, got {n_curves}"

    def test_well_a_depth_monotonic(self, synthetic_las_path):
        """Depth must be strictly monotonically increasing."""
        import lasio
        las = lasio.read(str(synthetic_las_path))
        depth = las.index
        diffs = np.diff(depth)
        assert np.all(diffs > 0), "WELL_A depth is not strictly monotonic"

    def test_well_a_null_value(self, synthetic_las_path):
        """LAS null value must be -999.25."""
        import lasio
        las = lasio.read(str(synthetic_las_path))
        assert las.well["NULL"].value == -999.25

    def test_well_a_depth_range(self, synthetic_las_path):
        """Depth must span 8000–11000 ft."""
        import lasio
        las = lasio.read(str(synthetic_las_path))
        assert abs(las.well["STRT"].value - 8000.0) < 0.01
        assert abs(las.well["STOP"].value - 11000.0) < 0.1

    def test_well_c_missing_dt_pef(self):
        """WELL_C must NOT contain DT or PEF (tests missing-curve handling later)."""
        import lasio
        las = lasio.read(str(DATA_TRAINING / "WELL_C.las"))
        mnemonics = [c.mnemonic for c in las.curves]
        assert "DT"  not in mnemonics, "WELL_C should not have DT"
        assert "PEF" not in mnemonics, "WELL_C should not have PEF"
        assert "ILD" in mnemonics,     "WELL_C should use ILD (alias for RT)"

    def test_well_d_has_null_rhob_interval(self):
        """WELL_D RHOB should contain nulls in the 9600-9620 ft gap."""
        import lasio
        las = lasio.read(str(DATA_VALID / "WELL_D.las"))
        df  = las.df()
        # lasio replaces null values with NaN
        null_zone = df.loc[9600:9620, "RHOB"]
        assert null_zone.isna().any(), (
            "WELL_D RHOB should have NaN values in the 9600-9620 ft interval"
        )

    def test_well_a_dataframe_shape(self, synthetic_las_path):
        """DataFrame must have 6001 rows and 8 columns."""
        import lasio
        las = lasio.read(str(synthetic_las_path))
        df = las.df()
        assert df.shape == (6001, 8), f"Unexpected shape: {df.shape}"

    def test_well_a_gr_within_physical_range(self, synthetic_las_path):
        """GR values must stay within 0–300 GAPI (physical plausibility)."""
        import lasio
        las = lasio.read(str(synthetic_las_path))
        gr = las["GR"]
        assert gr.min() >= 0.0,   f"GR below 0: min={gr.min():.2f}"
        assert gr.max() <= 300.0, f"GR above 300: max={gr.max():.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Shared fixtures (conftest.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestConfixtures:
    """Verify that the shared pytest fixtures work correctly."""

    def test_small_depth_fixture(self, small_depth):
        assert len(small_depth) == 100
        assert small_depth[0] == pytest.approx(8000.0)
        assert np.all(np.diff(small_depth) > 0), "small_depth must be monotonic"

    def test_clean_df_fixture(self, clean_df):
        assert isinstance(clean_df, pd.DataFrame)
        assert set(clean_df.columns) == {"GR", "RHOB", "NPHI"}
        assert not clean_df.isnull().any().any(), (
            "clean_df fixture must not contain NaN values"
        )
        # Physically plausible ranges
        assert clean_df["GR"].between(0, 300).all()
        assert clean_df["RHOB"].between(1.0, 3.5).all()
        assert clean_df["NPHI"].between(-0.15, 1.0).all()
