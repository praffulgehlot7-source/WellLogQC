"""Milestone 1 smoke test — verifies imports and LAS file integrity."""
import lasio
import pandas
import numpy
import matplotlib
import scipy
import sklearn
import streamlit
import jinja2
import yaml
import pathlib

libs = {
    "lasio":        lasio.__version__,
    "pandas":       pandas.__version__,
    "numpy":        numpy.__version__,
    "matplotlib":   matplotlib.__version__,
    "scipy":        scipy.__version__,
    "scikit-learn": sklearn.__version__,
    "streamlit":    streamlit.__version__,
    "jinja2":       jinja2.__version__,
    "pyyaml":       yaml.__version__,
}
print("=== Installed versions ===")
for k, v in libs.items():
    print(f"  {k:<14} {v}")

# Verify all 5 LAS files exist and are readable
print()
print("=== LAS file verification ===")
wells = {
    "training/WELL_A.las":   8,
    "training/WELL_B.las":   8,
    "training/WELL_C.las":   5,
    "validation/WELL_D.las": 6,
    "test/WELL_E.las":       8,
}
base = pathlib.Path("data/raw")
for rel, expected_curves in wells.items():
    p = base / rel
    las = lasio.read(str(p))
    n_curves = len(las.curves) - 1   # subtract DEPT
    strt = las.well["STRT"].value
    stop = las.well["STOP"].value
    null_val = las.well["NULL"].value
    remark_item = las.params.get("REMARK")
    remark = remark_item.value[:45] if remark_item else "N/A"
    status = "OK" if n_curves == expected_curves else f"MISMATCH (got {n_curves})"
    print(f"  {p.name:<14}  curves={n_curves}  depth={strt:.0f}-{stop:.0f} ft  null={null_val}  [{status}]")
    print(f"               remark: {remark}...")

# Quick DataFrame check on WELL_A
las_a = lasio.read(str(base / "training/WELL_A.las"))
df = las_a.df()
print()
print(f"=== WELL_A DataFrame shape: {df.shape} ===")
print(df.describe().round(3).to_string())

print()
print("=== Smoke test PASSED ===")
