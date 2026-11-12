"""
smoke_petro_plot.py
-------------------
Smoke test verifying petrophysics, feature engineering, and log plotting pipelines.
"""

import sys
import numpy as np
import pandas as pd

try:
    from welllogqc.ingestion.las_reader import LASReader
    reader = LASReader()
    result = reader.read("data/raw/training/WELL_A.las")
    df = result.df
except (ImportError, ModuleNotFoundError, AttributeError):
    import lasio
    las = lasio.read("data/raw/training/WELL_A.las")
    df = las.df().replace(-999.25, np.nan)
    df.index.name = "DEPT"

from welllogqc.petrophysics.vsh import calc_vsh_linear
from welllogqc.petrophysics.porosity import calc_density_porosity
from welllogqc.petrophysics.saturation import calc_archie_sw
from welllogqc.petrophysics.ntg import calc_ntg
from welllogqc.features.feature_registry import build_feature_matrix, FEATURE_REGISTRY
from welllogqc.plotting.log_plot import plot_well_log

print("=== Running Petrophysics & Plotting Smoke Test ===")
print("Input df shape:", df.shape)
print("Input df columns:", list(df.columns))

# 1. Petrophysics calculation
vsh = calc_vsh_linear(df, gr_clean=15.0, gr_shale=120.0)
phid = calc_density_porosity(df)
print("VSH range:", float(vsh.min()), "-", float(vsh.max()))
print("PHID range:", float(phid.min()), "-", float(phid.max()))

# Also test Archie Sw if RT is available
if "RT" in df.columns:
    df_sw = df.copy()
    df_sw["PHID"] = phid
    sw = calc_archie_sw(df_sw, rw=0.05, phi_col="PHID")
    print("SW range:", float(sw.dropna().min()), "-", float(sw.dropna().max()))
else:
    sw = None

df2 = df.copy()
df2["VSH_LINEAR"] = vsh
df2["PHID"] = phid
ntg = calc_ntg(df2, gross_top=8200.0, gross_base=8700.0)
print("NTG:", ntg["ntg"], "  Net ft:", ntg["net_ft"])
print("Classes:", ntg["summary"])

# 2. Feature Matrix
feat_df = build_feature_matrix(df)
print("Feature matrix shape:", feat_df.shape)
print("Feature columns:", list(feat_df.columns)[:10], "...")
print("Total registered features in FEATURE_REGISTRY:", len(FEATURE_REGISTRY))

# 3. Plotting
fig = plot_well_log(
    df,
    "WELL_A",
    extra_curves={"VSH": vsh, "PHID": phid},
    save_path="outputs/test_plot.png"
)
print("Plot created, axes:", len(fig.axes))
print("Smoke test completed successfully!")
