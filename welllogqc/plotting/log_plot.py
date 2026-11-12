"""
log_plot.py
-----------
Multi-track petrophysical well-log plotting and QC anomaly overlay.
"""

from typing import Any
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from welllogqc.config.defaults import CURVE_RANGES

# Shared QC data structures
QC_COLS = [
    "well_name", "depth_top", "depth_base", "curve", "issue",
    "severity", "method", "value", "threshold", "description"
]

LABEL_MAP = {
    0: "Acceptable",
    1: "Questionable",
    2: "Unreliable",
    -1: "Unreviewed"
}

# Standard canonical curve presentation ordering
_CANONICAL_PLOT_ORDER: list[str] = [
    "CALI", "BS", "GR", "RT", "RHOB", "NPHI", "DT", "PEF"
]


def plot_well_log(
    df: pd.DataFrame,
    well_name: str,
    curves_to_plot: list[str] | None = None,
    qc_flags: pd.DataFrame | None = None,
    extra_curves: dict[str, pd.Series] | None = None,
    depth_range: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None
) -> matplotlib.figure.Figure:
    """Create a publication-quality multi-track well-log display with QC flag overlays.

    Features:
      - Automatically groups CALI and BS into a single borehole geometry track
        when both are requested and present.
      - GR track includes green shading between curve and minimum valid baseline.
      - RT resistivity track plotted on a 4-decade logarithmic x-scale.
      - Depth axis is inverted (increasing downward) and shared across all tracks.
      - QC flag intervals are overlaid across tracks:
          CRITICAL: red band (alpha=0.3)
          WARNING: orange band (alpha=0.2)
          INFO: blue band (alpha=0.1)
      - Appends extra calculated Series (e.g. VSH, PHID, SW) as dedicated tracks on the right.
      - Defensive: never crashes if a curve is absent; absent tracks are simply skipped.

    Args:
        df: pandas DataFrame containing well log curves, indexed by depth ('DEPT' in ft).
        well_name: Name/identifier of the well to display in the plot title.
        curves_to_plot: Optional list of curve names to include. If None, all canonical
            curves present in df are plotted.
        qc_flags: Optional DataFrame containing QC flags conforming to QC_COLS.
        extra_curves: Optional dictionary mapping curve names to pd.Series for additional
            petrophysical tracks (e.g. {'VSH': vsh_series, 'PHID': phid_series}).
        depth_range: Optional tuple of (top, base) depths in ft to restrict the display.
        figsize: Optional tuple of (width, height) in inches. Defaults to dynamic sizing
            based on the number of tracks.
        save_path: Optional file path to export the figure as a high-resolution PNG image.

    Returns:
        matplotlib.figure.Figure: The rendered Matplotlib Figure object.
    """
    # 1. Determine which tracks to construct
    tracks: list[dict[str, Any]] = []

    if curves_to_plot is not None:
        target_curves = [c for c in curves_to_plot if c in df.columns]
        # Check if CALI and BS should be combined
        if "CALI" in target_curves and "BS" in target_curves:
            tracks.append({
                "type": "combined_cali_bs",
                "curves": ["CALI", "BS"],
                "title": "CALI / BS",
                "unit": "in",
            })
            remaining = [c for c in target_curves if c not in ("CALI", "BS")]
        else:
            remaining = target_curves

        for c in remaining:
            tracks.append({
                "type": "single",
                "curves": [c],
                "title": c,
                "unit": CURVE_RANGES.get(c, (0, 0, ""))[2] or "",
            })
    else:
        # Canonical automatic ordering
        has_cali = "CALI" in df.columns
        has_bs = "BS" in df.columns

        if has_cali and has_bs:
            tracks.append({
                "type": "combined_cali_bs",
                "curves": ["CALI", "BS"],
                "title": "CALI / BS",
                "unit": "in",
            })
        elif has_cali:
            tracks.append({
                "type": "single",
                "curves": ["CALI"],
                "title": "CALI",
                "unit": "in",
            })
        elif has_bs:
            tracks.append({
                "type": "single",
                "curves": ["BS"],
                "title": "BS",
                "unit": "in",
            })

        for c in ["GR", "RT", "RHOB", "NPHI", "DT", "PEF"]:
            if c in df.columns:
                unit = CURVE_RANGES.get(c, (0, 0, ""))[2] or ""
                tracks.append({
                    "type": "single",
                    "curves": [c],
                    "title": c,
                    "unit": unit,
                })

        # Other numeric curves present in df not already accounted for
        already_tracked = {c for t in tracks for c in t["curves"]}
        for c in df.columns:
            if c not in already_tracked and pd.api.types.is_numeric_dtype(df[c]):
                if not (c.startswith("MISSING_") or "_ROLL_" in c or "_GRAD" in c):
                    tracks.append({
                        "type": "single",
                        "curves": [c],
                        "title": c,
                        "unit": "",
                    })

    # Extra curves (appended on the right)
    extra_items: list[tuple[str, pd.Series]] = []
    if extra_curves is not None:
        for ename, eseries in extra_curves.items():
            if isinstance(eseries, pd.Series) and len(eseries) > 0:
                extra_items.append((ename, eseries))

    num_tracks = len(tracks) + len(extra_items)

    # Fallback if no curves to display
    if num_tracks == 0:
        fig, ax = plt.subplots(figsize=figsize or (6.0, 8.0))
        ax.text(
            0.5, 0.5,
            f"No curves available to display for {well_name}",
            ha="center", va="center", fontsize=12, color="gray"
        )
        ax.axis("off")
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    # 2. Setup Figure and Axes
    if figsize is None:
        figsize = (max(2.8 * num_tracks, 6.0), 11.0)

    fig, axes_raw = plt.subplots(
        nrows=1,
        ncols=num_tracks,
        sharey=True,
        figsize=figsize
    )
    axes: list[plt.Axes] = [axes_raw] if num_tracks == 1 else list(axes_raw)

    # 3. Determine Depth Boundaries
    if depth_range is not None:
        z_top = float(min(depth_range[0], depth_range[1]))
        z_base = float(max(depth_range[0], depth_range[1]))
    else:
        dept_vals = df.index.to_series().astype(np.float64)
        z_top = float(dept_vals.min()) if len(dept_vals) > 0 else 0.0
        z_base = float(dept_vals.max()) if len(dept_vals) > 0 else 1000.0

    # 4. Render Primary Curve Tracks
    for i, trk in enumerate(tracks):
        ax = axes[i]
        trk_type = trk["type"]

        if trk_type == "combined_cali_bs":
            cali_data = df["CALI"].astype(np.float64)
            bs_data = df["BS"].astype(np.float64)
            ax.plot(cali_data, df.index, color="black", linewidth=1.2, label="CALI")
            ax.plot(bs_data, df.index, color="blue", linestyle="--", linewidth=1.2, label="BS")
            ax.set_xlim(4.0, 20.0)
            ax.set_xlabel("CALI / BS (in)", fontsize=9, fontweight="bold")
            ax.legend(loc="lower left", fontsize=8, framealpha=0.7)

        elif trk_type == "single":
            cname = trk["curves"][0]
            cdata = df[cname].astype(np.float64)
            unit_str = f" ({trk['unit']})" if trk["unit"] else ""

            if cname == "GR":
                ax.plot(cdata, df.index, color="green", linewidth=1.2, label="GR")
                valid_gr = cdata.dropna()
                min_val = 0.0
                ax.fill_betweenx(
                    df.index, min_val, cdata,
                    where=(cdata >= min_val),
                    color="lightgreen", alpha=0.35
                )
                max_gr = max(150.0, float(valid_gr.quantile(0.99)) if len(valid_gr) > 0 else 150.0)
                ax.set_xlim(0.0, max_gr)
                ax.set_xlabel(f"GR{unit_str}", color="darkgreen", fontsize=9, fontweight="bold")

            elif cname == "RT":
                ax.plot(cdata, df.index, color="red", linewidth=1.2, label="RT")
                ax.set_xscale("log")
                ax.set_xlim(0.2, 2000.0)
                ax.set_xlabel(f"RT{unit_str}", color="red", fontsize=9, fontweight="bold")

            elif cname == "RHOB":
                ax.plot(cdata, df.index, color="red", linewidth=1.2, label="RHOB")
                ax.set_xlim(1.95, 2.95)
                ax.set_xlabel(f"RHOB{unit_str}", color="red", fontsize=9, fontweight="bold")

            elif cname == "NPHI":
                ax.plot(cdata, df.index, color="blue", linewidth=1.2, label="NPHI")
                ax.set_xlim(0.45, -0.15)  # Inverted standard petrophysical scale
                ax.set_xlabel(f"NPHI{unit_str}", color="blue", fontsize=9, fontweight="bold")

            elif cname == "DT":
                ax.plot(cdata, df.index, color="purple", linewidth=1.2, label="DT")
                ax.set_xlim(40.0, 140.0)
                ax.set_xlabel(f"DT{unit_str}", color="purple", fontsize=9, fontweight="bold")

            elif cname == "PEF":
                ax.plot(cdata, df.index, color="darkorange", linewidth=1.2, label="PEF")
                ax.set_xlim(0.0, 10.0)
                ax.set_xlabel(f"PEF{unit_str}", color="darkorange", fontsize=9, fontweight="bold")

            elif cname == "CALI":
                ax.plot(cdata, df.index, color="black", linewidth=1.2, label="CALI")
                ax.set_xlim(4.0, 20.0)
                ax.set_xlabel(f"CALI{unit_str}", fontsize=9, fontweight="bold")

            elif cname == "BS":
                ax.plot(cdata, df.index, color="blue", linestyle="--", linewidth=1.2, label="BS")
                ax.set_xlim(4.0, 20.0)
                ax.set_xlabel(f"BS{unit_str}", fontsize=9, fontweight="bold")

            else:
                ax.plot(cdata, df.index, color="navy", linewidth=1.2, label=cname)
                valid = cdata.dropna()
                if len(valid) > 0:
                    span = valid.max() - valid.min()
                    pad = span * 0.05 if span > 0 else 1.0
                    ax.set_xlim(valid.min() - pad, valid.max() + pad)
                ax.set_xlabel(f"{cname}{unit_str}", fontsize=9, fontweight="bold")

    # 5. Render Extra Petrophysical Curves
    extra_start_idx = len(tracks)
    for j, (ename, eseries) in enumerate(extra_items):
        ax = axes[extra_start_idx + j]
        s_vals = eseries.astype(np.float64)
        s_idx = eseries.index.astype(np.float64)

        uname = ename.upper()
        if "VSH" in uname:
            ax.plot(s_vals, s_idx, color="saddlebrown", linewidth=1.2, label=ename)
            ax.fill_betweenx(
                s_idx, 0.0, s_vals,
                where=(s_vals >= 0.0),
                color="saddlebrown", alpha=0.3
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel(f"{ename} (v/v)", color="saddlebrown", fontsize=9, fontweight="bold")

        elif "PHI" in uname:
            ax.plot(s_vals, s_idx, color="blue", linewidth=1.2, label=ename)
            ax.fill_betweenx(
                s_idx, 0.0, s_vals,
                where=(s_vals >= 0.0),
                color="blue", alpha=0.25
            )
            ax.set_xlim(0.0, 0.45)
            ax.set_xlabel(f"{ename} (v/v)", color="blue", fontsize=9, fontweight="bold")

        elif "SW" in uname:
            ax.plot(s_vals, s_idx, color="teal", linewidth=1.2, label=ename)
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel(f"{ename} (v/v)", color="teal", fontsize=9, fontweight="bold")

        else:
            ax.plot(s_vals, s_idx, color="darkcyan", linewidth=1.2, label=ename)
            valid = s_vals.dropna()
            if len(valid) > 0:
                span = valid.max() - valid.min()
                pad = span * 0.05 if span > 0 else 1.0
                ax.set_xlim(valid.min() - pad, valid.max() + pad)
            ax.set_xlabel(ename, fontsize=9, fontweight="bold")

    # 6. Apply Common Axis Properties (Invert Depth, Position X-labels on Top)
    for ax in axes:
        ax.set_ylim(z_base, z_top)  # Depth increases downward
        ax.grid(True, which="both", linestyle=":", color="gray", alpha=0.55)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=9)

    axes[0].set_ylabel("Depth (ft)", fontsize=11, fontweight="bold")

    # 7. Overlay QC Flag Bands
    if qc_flags is not None and isinstance(qc_flags, pd.DataFrame) and not qc_flags.empty:
        for _, row in qc_flags.iterrows():
            d_top = float(row.get("depth_top", np.nan))
            d_base = float(row.get("depth_base", np.nan))
            if np.isnan(d_top) or np.isnan(d_base):
                continue

            sev = str(row.get("severity", "INFO")).upper()
            if "CRIT" in sev:
                color, alpha = "red", 0.30
            elif "WARN" in sev:
                color, alpha = "orange", 0.20
            else:
                color, alpha = "blue", 0.10

            for ax in axes:
                if abs(d_top - d_base) < 1e-4:
                    ax.axhline(
                        d_top,
                        color=color,
                        alpha=min(alpha * 2.5, 0.85),
                        linewidth=1.5,
                        zorder=2
                    )
                else:
                    ax.axhspan(
                        min(d_top, d_base),
                        max(d_top, d_base),
                        color=color,
                        alpha=alpha,
                        zorder=1
                    )

    fig.suptitle(f"Well Log QC Display: {well_name}", fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(top=0.88, bottom=0.06, wspace=0.15)

    # 8. Save image if path provided
    if save_path:
        out_dir = os.path.dirname(os.path.abspath(save_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
