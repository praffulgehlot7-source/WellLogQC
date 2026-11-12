"""
las_reader.py
-------------
Parses LAS log files, maps mnemonics to canonical names, validates depth monotonicity,
handles missing value sentinels, and packages results into an IngestionResult.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import lasio
import numpy as np
import pandas as pd

from welllogqc.config.defaults import NULL_FALLBACK
from welllogqc.ingestion.curve_mapper import CurveMapper


@dataclass
class IngestionResult:
    """Encapsulates the ingested well log data and associated metadata.

    Attributes:
        well_name: Name of the well extracted from LAS header or file name.
        df: Canonical depth-indexed DataFrame with index name 'DEPT' and float64 columns.
        role: Dataset role assigned to this well ('training', 'blind_test', etc.).
        las_path: Absolute or relative file path to the ingested LAS file.
        curve_map: Mapping of raw LAS mnemonics to canonical curve names.
        unknown_curves: List of LAS mnemonics that could not be mapped.
        available_curves: List of canonical curve names present in the DataFrame.
        missing_curves: List of expected canonical curve names not present in the DataFrame.
        depth_start: First depth value in the dataset (ft).
        depth_stop: Last depth value in the dataset (ft).
        depth_step: Nominal depth step between consecutive samples (ft).
        null_value: LAS null sentinel value read from the header.
        null_counts: Dictionary mapping each canonical curve name to its null sample count.
        warnings: List of diagnostic warning strings encountered during ingestion.
    """

    well_name: str
    df: pd.DataFrame
    role: str
    las_path: str
    curve_map: dict[str, str]
    unknown_curves: list[str]
    available_curves: list[str]
    missing_curves: list[str]
    depth_start: float
    depth_stop: float
    depth_step: float
    null_value: float
    null_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)


class LASReader:
    """Ingests LAS files and builds canonical, depth-indexed DataFrames.

    Validates depth monotonicity and step consistency, substitutes LAS null
    sentinels with np.nan, and standardises curve mnemonics.

    Attributes:
        curve_mapper: CurveMapper instance used to resolve mnemonics.
    """

    EXPECTED_CURVES: list[str] = [
        "GR",
        "CALI",
        "BS",
        "RHOB",
        "NPHI",
        "RT",
        "RS",
        "DT",
        "PEF",
    ]

    def __init__(self, alias_path: str | Path | None = None) -> None:
        """Initialises the LASReader.

        Args:
            alias_path: Optional path to curve_aliases.yaml. Defaults to package config.
        """
        self.curve_mapper = CurveMapper(alias_path=alias_path)

    def read(
        self,
        las_path: str | Path,
        role: str = "training",
        manual_map: dict[str, str] | None = None,
    ) -> IngestionResult:
        """Reads a LAS file, validates data, and constructs an IngestionResult.

        Args:
            las_path: Path to the LAS file to ingest.
            role: Dataset role identifier, e.g. 'training' or 'blind_test'.
            manual_map: Optional dictionary of raw mnemonic -> canonical curve overrides.

        Returns:
            IngestionResult containing canonical DataFrame and data quality metadata.

        Raises:
            FileNotFoundError: If the LAS file does not exist.
            ValueError: If the LAS file cannot be parsed or lacks a valid depth curve.
        """
        path = Path(las_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LAS file not found: {path}")

        try:
            las = lasio.read(str(path))
        except Exception as e:
            raise ValueError(f"Failed to parse LAS file {path}: {e}") from e

        return self._parse_las(
            las,
            well_fallback=path.stem,
            role=role,
            source_path=str(path),
            manual_map=manual_map
        )

    def read_bytes(
        self,
        content_bytes: bytes,
        filename: str = "uploaded_well.las",
        role: str = "test",
        manual_map: dict[str, str] | None = None,
    ) -> IngestionResult:
        """Reads wireline data from in-memory raw bytes (supports both LAS and CSV)."""
        stem = Path(filename).stem
        if filename.lower().endswith(".csv"):
            return self._parse_csv_bytes(content_bytes, well_fallback=stem, role=role, filename=filename, manual_map=manual_map)

        text = None
        for enc in ["utf-8", "latin-1", "cp1252", "ascii"]:
            try:
                text = content_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = content_bytes.decode("utf-8", errors="replace")

        try:
            las = lasio.read(io.StringIO(text))
        except Exception as e:
            raise ValueError(f"Failed to parse LAS bytes from {filename}: {e}") from e

        return self._parse_las(
            las,
            well_fallback=stem,
            role=role,
            source_path=filename,
            manual_map=manual_map
        )

    def _parse_csv_bytes(
        self,
        content_bytes: bytes,
        well_fallback: str,
        role: str,
        filename: str,
        manual_map: dict[str, str] | None = None,
    ) -> IngestionResult:
        df_raw = pd.read_csv(io.BytesIO(content_bytes))
        depth_col = None
        for c in df_raw.columns:
            if str(c).strip().upper() in ["DEPT", "DEPTH", "MD", "DEPTH_MD", "INDEX"]:
                depth_col = c
                break
        if depth_col is None:
            depth_col = df_raw.columns[0]

        df_raw = df_raw.rename(columns={depth_col: "DEPT"}).set_index("DEPT")
        raw_mnemonics = [str(c) for c in df_raw.columns]
        raw_map = self.curve_mapper.map(raw_mnemonics)
        if manual_map:
            for k, v in manual_map.items():
                k_upper = str(k).strip().upper()
                for m in raw_mnemonics:
                    if str(m).strip().upper() == k_upper:
                        raw_map[m] = str(v).strip().upper()

        curve_map = {m: c for m, c in raw_map.items() if c is not None}
        unknown_curves = [m for m, c in raw_map.items() if c is None]
        df_clean = df_raw.rename(columns=curve_map).apply(pd.to_numeric, errors="coerce").sort_index()

        available_curves = [c for c in self.EXPECTED_CURVES if c in df_clean.columns] + [
            c for c in df_clean.columns if c not in self.EXPECTED_CURVES
        ]
        missing_curves = [c for c in self.EXPECTED_CURVES if c not in df_clean.columns]
        null_counts = {col: int(df_clean[col].isna().sum()) for col in df_clean.columns}

        return IngestionResult(
            well_name=well_fallback,
            df=df_clean,
            role=role,
            las_path=filename,
            curve_map=curve_map,
            unknown_curves=unknown_curves,
            available_curves=available_curves,
            missing_curves=missing_curves,
            depth_start=float(df_clean.index[0]) if len(df_clean) > 0 else 0.0,
            depth_stop=float(df_clean.index[-1]) if len(df_clean) > 0 else 0.0,
            depth_step=float(np.median(np.diff(df_clean.index))) if len(df_clean) > 1 else 0.5,
            null_value=-999.25,
            null_counts=null_counts,
            warnings=[],
        )

    def _parse_las(
        self,
        las: lasio.LASFile,
        well_fallback: str,
        role: str,
        source_path: str,
        manual_map: dict[str, str] | None = None,
    ) -> IngestionResult:
        # Extract well name from header or fallback to filename stem
        well_name = ""
        if "WELL" in las.well and las.well["WELL"].value is not None:
            well_name = str(las.well["WELL"].value).strip()
        if not well_name:
            well_name = well_fallback

        # Extract null sentinel from header
        null_value = NULL_FALLBACK
        if "NULL" in las.well and las.well["NULL"].value is not None:
            try:
                null_value = float(las.well["NULL"].value)
            except (ValueError, TypeError):
                null_value = NULL_FALLBACK

        # Extract nominal step from header if present
        header_step: float | None = None
        if "STEP" in las.well and las.well["STEP"].value is not None:
            try:
                header_step = abs(float(las.well["STEP"].value))
            except (ValueError, TypeError):
                header_step = None

        warnings: list[str] = []

        # Resolve curve mnemonics
        raw_mnemonics = [item.mnemonic for item in las.curves]
        raw_map = self.curve_mapper.map(raw_mnemonics)

        if manual_map:
            for k, v in manual_map.items():
                # Case-insensitive manual mapping match
                k_upper = str(k).strip().upper()
                for m in raw_mnemonics:
                    if str(m).strip().upper() == k_upper:
                        raw_map[m] = str(v).strip().upper()

        curve_map: dict[str, str] = {m: c for m, c in raw_map.items() if c is not None}
        unknown_curves: list[str] = [m for m, c in raw_map.items() if c is None]

        # Identify depth mnemonic
        dept_mnemonic: str | None = None
        for m, c in curve_map.items():
            if c == "DEPT":
                dept_mnemonic = m
                break

        if dept_mnemonic is None:
            # Fallback to the 0th curve if it looks like depth
            if len(las.curves) > 0:
                first_m = las.curves[0].mnemonic
                if first_m.strip().upper() in ["DEPT", "DEPTH", "MD", "DEPTH_MD", "INDEX"]:
                    dept_mnemonic = first_m
                    curve_map[dept_mnemonic] = "DEPT"
                    if dept_mnemonic in unknown_curves:
                        unknown_curves.remove(dept_mnemonic)

        if dept_mnemonic is None:
            raise ValueError(
                f"Could not identify depth curve in {source_path}. Raw curves: {raw_mnemonics}"
            )

        # Extract depth array
        raw_dept = np.asarray(las[dept_mnemonic], dtype=np.float64)

        # Clean null sentinel in depth
        depth_null_mask = (
            np.isnan(raw_dept)
            | np.isclose(raw_dept, null_value, atol=1e-3)
            | np.isclose(raw_dept, -999.25, atol=1e-3)
            | np.isclose(raw_dept, -9999.0, atol=1e-3)
        )
        if np.any(depth_null_mask):
            warnings.append(
                f"Found {np.sum(depth_null_mask)} null values in depth curve {dept_mnemonic}."
            )

        # Extract and clean curve data
        curve_data: dict[str, np.ndarray] = {}
        for m, canonical in curve_map.items():
            if canonical == "DEPT":
                continue
            raw_vals = np.asarray(las[m], dtype=np.float64)
            cleaned = raw_vals.copy()
            # Replace null sentinels with NaN
            val_null_mask = np.isnan(cleaned)
            if not np.isnan(null_value):
                val_null_mask |= np.isclose(cleaned, null_value, atol=1e-3)
            val_null_mask |= np.isclose(cleaned, -999.25, atol=1e-3)
            val_null_mask |= np.isclose(cleaned, -9999.0, atol=1e-3)
            cleaned[val_null_mask] = np.nan
            curve_data[canonical] = cleaned

        # Build depth-indexed DataFrame
        dept_index = pd.Index(raw_dept, name="DEPT", dtype=np.float64)
        df = pd.DataFrame(curve_data, index=dept_index, dtype=np.float64)

        # Depth validation and sorting
        depths = df.index.to_numpy(dtype=np.float64)
        if len(depths) > 1:
            diffs = np.diff(depths)
            if np.all(diffs < 0):
                warnings.append("Depth values were in descending order (logged bottom-up); sorted ascending.")
                df = df.iloc[::-1]
                depths = df.index.to_numpy(dtype=np.float64)
                diffs = np.diff(depths)
            elif np.any(diffs < 0):
                warnings.append("Non-monotonic depth values detected in file (depth reversals present).")

            if df.index.has_duplicates:
                warnings.append(
                    f"Duplicate depth indices detected ({df.index.duplicated().sum()} duplicate samples)."
                )

            pos_diffs = diffs[diffs > 0]
            if len(pos_diffs) > 0:
                median_step = float(np.median(pos_diffs))
                nominal_step = header_step if header_step is not None and header_step > 0 else median_step
                nominal_step = round(nominal_step, 4)
                max_dev = float(np.max(np.abs(pos_diffs - nominal_step)))
                if max_dev > 0.005:
                    warnings.append(
                        f"Variable depth step detected: nominal step is {nominal_step:.4f} ft, "
                        f"maximum deviation is {max_dev:.4f} ft."
                    )
            else:
                nominal_step = header_step if header_step else 0.0
        else:
            nominal_step = header_step if header_step else 0.0

        depth_start = float(df.index[0]) if len(df) > 0 else 0.0
        depth_stop = float(df.index[-1]) if len(df) > 0 else 0.0
        depth_step = float(nominal_step)

        # Canonical curve availability
        available_curves = [c for c in self.EXPECTED_CURVES if c in df.columns] + [
            c for c in df.columns if c not in self.EXPECTED_CURVES
        ]
        missing_curves = [c for c in self.EXPECTED_CURVES if c not in df.columns]

        # Null counts per curve
        null_counts = {col: int(df[col].isna().sum()) for col in df.columns}

        return IngestionResult(
            well_name=well_name,
            df=df,
            role=role,
            las_path=str(source_path),
            curve_map=curve_map,
            unknown_curves=unknown_curves,
            available_curves=available_curves,
            missing_curves=missing_curves,
            depth_start=depth_start,
            depth_stop=depth_stop,
            depth_step=depth_step,
            null_value=null_value,
            null_counts=null_counts,
            warnings=warnings,
        )
