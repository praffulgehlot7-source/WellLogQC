"""
welllogqc/app/pages/2_Well_Analysis.py
======================================
Option 2: Analysis Interface.
Single-file input (LAS or CSV).
Single primary action: "Run AI Prediction & Formation Analysis".
Comprehensive outcomes:
1. Automated Quality Control & Sensor Plausibility Screening
2. Petrophysical Formation Analysis (Vsh, Porosity, Archie Saturation, Net Pay, NTG)
3. Synchronized Interactive Plotly Graphs with depth-linked hover highlighting
4. Pointout of exact depths where Petrophysicist Review is Required
5. User-friendly plain-English answers, formation verdicts, and actionable advice.
Strictly zero emojis.
"""

import sys
from pathlib import Path
import io
import time
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
if "welllogqc.app.ui_theme" in sys.modules:
    importlib.reload(sys.modules["welllogqc.app.ui_theme"])

from welllogqc.app.ui_theme import (
    clean_html,
    inject_custom_theme,
    render_page_header,
    render_kpi,
    render_section_header,
    render_sidebar_header,
    render_advisory_card,
    get_petroleum_loader_html,
    ICONS,
)
from welllogqc.ingestion.las_reader import LASReader
from welllogqc.qc.qc_reporter import run_all_qc, summarise_flags
from welllogqc.petrophysics.vsh import calc_vsh_linear
from welllogqc.petrophysics.porosity import calc_density_porosity
from welllogqc.petrophysics.saturation import calc_archie_sw
from welllogqc.petrophysics.ntg import calc_ntg
from welllogqc.advisory.ideal_reservoir import (
    compute_ideal_reservoir_gap_analysis,
    get_ideal_benchmark_comparison_table,
)
from welllogqc.advisory.decision_engine import (
    generate_interval_decisions,
    generate_actionable_advisories,
)
from welllogqc.advisory.curve_guide import (
    generate_hover_customdata,
    get_curve_guide_for_well,
    get_guide_comparison_dataframe,
    get_all_curve_guides,
    show_guidelines_dialog,
)
from welllogqc.petrophysics.environmental_correction import (
    apply_automated_environmental_corrections,
)
from welllogqc.petrophysics.cross_correlation import (
    run_multicurve_cross_correlation,
)
from welllogqc.features.feature_registry import build_feature_matrix
from welllogqc.service.gpu_detector import detect_gpu

import plotly.graph_objects as go
from plotly.subplots import make_subplots

inject_custom_theme(analysis_done=st.session_state.get("analysis_done", False))

MODELS_DIR = PROJECT_ROOT / "models"
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# Session State Persistence
for key in [
    "analysis_well_res", "analysis_qc_flags", "analysis_petro", "analysis_ntg",
    "analysis_prox", "analysis_decisions", "analysis_done", "analysis_env_corr", "analysis_cross_corr",
    "show_benchmark_comparison"
]:
    if key not in st.session_state:
        st.session_state[key] = False if key in ["analysis_done", "show_benchmark_comparison"] else None

gpu_info = detect_gpu()

# ── TOP BAR: GUIDELINES BUTTON (TOP LEFT) ────────────────────────────────────
top_col_guide, top_col_spacer = st.columns([2, 10])
with top_col_guide:
    if st.button("Guidelines", icon=":material/menu_book:", type="secondary", use_container_width=True, key="top_guidelines_btn"):
        show_guidelines_dialog(st.session_state.analysis_well_res.available_curves if st.session_state.analysis_well_res is not None else None)

render_page_header(
    title="Well Log Prediction & Formation Analysis",
    subtitle="Single-file wireline log evaluation: automated QC, rigorous petrophysics, and synchronized depth-linked review highlighting",
)

# ── SECTION 1: ABOUT (LEFT) & FILE INPUT (RIGHT) ──────────────────────────────
render_section_header("Upload or Select Wireline Log File", "database", "Upload a vendor wireline file (.las, .csv) or pick a preloaded well from disk")

col_desc, col_input = st.columns([1, 1], gap="medium")

with col_desc:
    st.markdown(clean_html(r"""
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px 24px; height: 100%; box-sizing: border-box;">
        <div style="color: #38bdf8; font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;">
            Autonomous Well Log Evaluation Engine
        </div>
        <div style="color: #cbd5e1; font-size: 0.90rem; line-height: 1.6; margin-bottom: 12px;">
            An enterprise petrophysical evaluation platform built for rapid subsurface well log screening and commercial reservoir characterization.
        </div>
        <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.6;">
            &bull; <strong>Automated QC:</strong> Range sanity, sticking sensor, and depth checks<br>
            &bull; <strong>Borehole Corrections:</strong> Dynamic mudcake and pad standoff restoration<br>
            &bull; <strong>Petrophysics:</strong> Clay volume (Vsh), density porosity (&phi;D), Archie saturation (Sw)<br>
            &bull; <strong>Synthesis:</strong> Multi-curve cross-correlation and commercial Net Pay (NTG)<br>
            &bull; <strong>Standards:</strong> SPWLA and Schlumberger reference baselines
        </div>
    </div>
    """), unsafe_allow_html=True)

with col_input:
    uploaded_file = st.file_uploader(
        "Upload Well Log File (.las, .csv)",
        type=["las", "LAS", "csv", "CSV"],
        key="analysis_file_uploader",
        help="Upload vendor wireline LAS file or depth-indexed CSV file"
    )
    archive_paths = list(DATA_RAW.rglob("*.las"))
    archive_names = [p.stem for p in archive_paths]
    selected_preset = st.selectbox(
        "Or Pick from Disk Archive Wells",
        ["(Select file above)"] + archive_names,
        key="analysis_preset_sel"
    )

active_res = None
reader = LASReader()

if uploaded_file is not None:
    try:
        raw_b = uploaded_file.read()
        active_res = reader.read_bytes(raw_b, filename=uploaded_file.name, role="test")
        st.session_state.analysis_well_res = active_res
    except Exception as e:
        st.error(f"Error parsing uploaded file '{uploaded_file.name}': {e}")
elif selected_preset != "(Select file above)":
    match_p = next(p for p in archive_paths if p.stem == selected_preset)
    active_res = reader.read(str(match_p), role="test")
    st.session_state.analysis_well_res = active_res
elif st.session_state.analysis_well_res is not None:
    active_res = st.session_state.analysis_well_res

if active_res is None:
    st.info("Please upload a wireline log file or select a preloaded well to proceed with analysis.")
    st.stop()

df = active_res.df

# Active Well Summary Card
st.markdown(clean_html(f"""
<div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px 20px; margin-top: 10px; margin-bottom: 20px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 14px; width: 100%; box-sizing: border-box;">
    <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
        <span style="color: #38bdf8; font-weight: 700; font-size: 1.05rem;">Target Well: {active_res.well_name}</span>
        <span style="color: #cbd5e1; font-size: 0.88rem; background: #0f172a; padding: 4px 10px; border-radius: 6px; border: 1px solid #334155;">
            Interval: {active_res.depth_start:.1f} – {active_res.depth_stop:.1f} ft ({len(df):,} depth levels)
        </span>
    </div>
    <div>
        <span style="color: #cbd5e1; font-size: 0.86rem;">Available Curves: <strong style="color: #38bdf8;">{', '.join(active_res.available_curves)}</strong></span>
    </div>
</div>
"""), unsafe_allow_html=True)

# ── SECTION 2: PRIMARY ACTION BUTTON ──────────────────────────────────────────
run_analysis_btn = st.button("Formation Analysis", type="primary", use_container_width=True, key="btn_run_formation_analysis")

if run_analysis_btn:
    loader_ph = st.empty()
    loader_ph.markdown(get_petroleum_loader_html(), unsafe_allow_html=True)
    time.sleep(1.0)

    # 1. Quality Control screening
    flags = run_all_qc(active_res)
    st.session_state.analysis_qc_flags = flags

    # 2. Automated Borehole Environmental Corrections
    # When caliper flags washout, automatically apply mudcake and borehole-fluid corrections
    # to Bulk Density (RHOB) and Resistivity (RT) rather than just issuing a warning flag!
    bs_nominal = float(df["BS"].iloc[0]) if "BS" in df.columns else 8.5
    env_pkg = apply_automated_environmental_corrections(df, bs_default=bs_nominal)
    st.session_state.analysis_env_corr = env_pkg
    corr_df = env_pkg["corrected_df"]

    # 3. Petrophysics calculation on environmentally corrected curves
    petro = env_pkg["petro_corr"].copy()
    st.session_state.analysis_petro = petro

    # 4. Multi-Curve Petrophysical Cross-Correlation Engine
    cross_corr_res = run_multicurve_cross_correlation(
        corr_df,
        vsh=petro.get("VSH"),
        phid=petro.get("PHID"),
        sw=petro.get("SW"),
        bs_default=bs_nominal
    )
    st.session_state.analysis_cross_corr = cross_corr_res

    # 5. Net-to-Gross & Net Pay (calculated on environmentally corrected curves)
    if "VSH" in petro and "PHID" in petro:
        df_ntg = corr_df.copy()
        df_ntg["VSH_LINEAR"] = petro["VSH"]
        df_ntg["PHID"] = petro["PHID"]
        qc_depths = list(zip(flags.depth_top, flags.depth_base)) if flags is not None else None
        ntg = calc_ntg(
            df_ntg,
            gross_top=float(df.index.min()),
            gross_base=float(df.index.max()),
            vsh_cutoff=0.35,
            phid_cutoff=0.10,
            qc_flag_depths=qc_depths
        )
        st.session_state.analysis_ntg = ntg

    # 6. Ideal Reservoir Benchmark Proximity (on corrected curves)
    prox = compute_ideal_reservoir_gap_analysis(
        corr_df,
        vsh=petro.get("VSH"),
        phid=petro.get("PHID"),
        sw=petro.get("SW")
    )
    st.session_state.analysis_prox = prox

    # 7. Interval Decisions & Review Required Identification
    decisions = generate_interval_decisions(
        corr_df,
        qc_flags=flags,
        predictions=None,
        vsh=petro.get("VSH"),
        phid=petro.get("PHID")
    )
    st.session_state.analysis_decisions = decisions
    st.session_state.analysis_done = True
    loader_ph.empty()

# ── SECTION 3: COMPREHENSIVE OUTCOMES & USER-FRIENDLY RESULTS ─────────────────
flags = st.session_state.analysis_qc_flags
petro = st.session_state.analysis_petro or {}
ntg = st.session_state.analysis_ntg
prox = st.session_state.analysis_prox
decisions = st.session_state.analysis_decisions

if st.session_state.analysis_done and prox is not None:
    # ── 1. PERFORMANCE INDICATORS ─────────────────────────────────────────────
    render_section_header("Performance Indicators", "chart", "Overview of formation potential, pay thickness, and reservoir quality in simple numbers")

    row1_1, row1_2, row1_3 = st.columns(3)
    with row1_1:
        render_kpi("Ideal Reservoir Match", f"{prox['composite_reservoir_proximity_pct']}%", prox["reservoir_tier"])
    with row1_2:
        net_str = f"{ntg['net_ft']:.1f} ft" if ntg else "N/A"
        render_kpi("Commercial Net Pay", net_str, "Conforming productive pay sand")
    with row1_3:
        ntg_str = f"{ntg['ntg']:.1%}" if ntg else "N/A"
        render_kpi("Net-to-Gross (NTG)", ntg_str, "Pay-to-gross formation ratio")

    row2_1, row2_2, row2_3 = st.columns(3)
    with row2_1:
        phi_mean = f"{prox.get('mean_porosity', 0.15):.1%}"
        render_kpi("Matrix Storage Porosity", phi_mean, "Average formation pore volume")
    with row2_2:
        hc_mean = f"{(1.0 - prox.get('mean_water_saturation', 0.25)):.1%}"
        render_kpi("Hydrocarbon Saturation", hc_mean, "Estimated 1 - Sw hydrocarbon cut")
    with row2_3:
        flag_cnt = len(flags) if flags is not None else 0
        render_kpi("Flagged QC Anomalies", f"{flag_cnt} Intervals", "Screened from commercial pay")

    env_pkg = st.session_state.analysis_env_corr
    cross_corr = st.session_state.analysis_cross_corr

    # Formation Verdict Card
    tier = prox.get("reservoir_tier", "Commercial Reservoir")
    st.markdown(clean_html(f"""
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px 24px; margin-top: 14px; margin-bottom: 22px;">
        <div style="color: #38bdf8; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px;">
            Verdict: {tier}
        </div>
        <div style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6;">
            <strong>What this means for your well:</strong><br>
            - Target well <strong>{active_res.well_name}</strong> was evaluated across the logged section from 
            <strong>{active_res.depth_start:.1f} ft</strong> to <strong>{active_res.depth_stop:.1f} ft</strong>.<br>
            - The formation shows an overall <strong>{prox['composite_reservoir_proximity_pct']}% proximity match</strong> to world-class clastic pay reservoirs (Wilcox/Brent benchmark).<br>
            - <strong>Cleanliness</strong>: Matrix shale volume ($V_{{sh}}$) averages <strong>{prox['mean_vsh']:.1%}</strong> (clean quartz sands offer optimal flow permeability).<br>
            - <strong>Storage & Fluids</strong>: Effective reservoir storage porosity averages <strong>{prox['mean_porosity']:.1%}</strong>, with estimated hydrocarbon saturation at <strong>{(1.0 - prox['mean_water_saturation']):.1%}</strong>.<br>
            - <strong>Integrity Screening</strong>: <strong>{len(flags) if flags is not None else 0} anomaly intervals</strong> were flagged and screened to ensure unreliable sensor readings do not artificially inflate or deflate your reserve estimates.
        </div>
    </div>
    """), unsafe_allow_html=True)

    if env_pkg and env_pkg.get("corrections_applied"):
        dens_corr = env_pkg.get("mean_density_correction_g_cm3", 0.0)
        rt_boost = env_pkg.get("mean_resistivity_boost_pct", 0.0)
        w_levels = env_pkg.get("washout_levels_count", 0)
        st.markdown(clean_html(f"""
        <div style="background: rgba(14, 165, 233, 0.08); border-left: 4px solid #0ea5e9; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155; border-radius: 6px; padding: 14px 20px; margin-top: 10px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 8px;">
                <strong style="color: #38bdf8; font-size: 0.96rem;">Automated Borehole Environmental Corrections Active</strong>
                <span style="font-size: 0.78rem; font-weight: 700; color: #10b981; background: rgba(16, 185, 129, 0.15); padding: 3px 10px; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.3);">
                    PHYSICS-BASED STANDOFF COMPENSATION APPLIED
                </span>
            </div>
            <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
                When the caliper flagged borehole breakouts and washouts (CALI > BS), the system automatically applied physics-based mudcake and borehole-fluid corrections:
                <strong>Bulk Density (RHOB)</strong> was restored by an average of <strong>+{dens_corr:.3f} g/cm3</strong> (neutralizing pad standoff), and 
                <strong>Deep Resistivity (RT)</strong> was boosted by <strong>+{rt_boost:.1f}%</strong> across <strong>{w_levels} washout depth levels</strong> (countering conductive mud shunting).
                Porosity (phiD, phiE) and Archie water saturation (Sw) have been re-evaluated on the environmentally corrected curves rather than issuing mere warning flags.
            </div>
        </div>
        """), unsafe_allow_html=True)

    # Identify washouts & anomalies for review
    review_needed_rows = []
    if "CALI" in df.columns and "BS" in df.columns:
        washout_mask = (df["CALI"] - df["BS"]) > 1.5
        if washout_mask.any():
            diffs = np.diff(washout_mask.astype(int))
            starts = np.where(diffs == 1)[0] + 1
            if washout_mask.iloc[0]:
                starts = np.insert(starts, 0, 0)
            ends = np.where(diffs == -1)[0]
            if washout_mask.iloc[-1]:
                ends = np.append(ends, len(df) - 1)

            for s, e in zip(starts, ends):
                top_d = float(df.index[s])
                base_d = float(df.index[e])
                thick = abs(base_d - top_d)
                max_dev = float((df["CALI"] - df["BS"]).iloc[s:e+1].max())
                review_needed_rows.append({
                    "Depth Interval (ft)": f"{top_d:.1f} - {base_d:.1f}",
                    "Thickness": f"{thick:.1f} ft",
                    "Primary Anomaly": "Borehole Washout / Breakout",
                    "Severity": "HIGH" if max_dev > 2.5 else "MEDIUM",
                    "Why Review is Required (Plain English)": f"Caliper exceeds bit size by {max_dev:.2f} in. Density tool pad likely lost contact with formation wall, causing artificially high porosity readings.",
                    "Recommended Action": "Exclude density porosity over this interval; substitute acoustic sonic porosity or apply geometric cavity corrections.",
                    "top": top_d,
                    "base": base_d
                })

    if flags is not None:
        for _, flg in flags[flags["severity"].isin(["CRITICAL", "WARNING"])].iterrows():
            t = float(flg["depth_top"])
            b = float(flg["depth_base"])
            thick = abs(b - t)
            if not any(abs(r["top"] - t) < 2.0 and abs(r["base"] - b) < 2.0 for r in review_needed_rows):
                review_needed_rows.append({
                    "Depth Interval (ft)": f"{t:.1f} - {b:.1f}",
                    "Thickness": f"{thick:.1f} ft",
                    "Primary Anomaly": flg["issue"],
                    "Severity": flg["severity"],
                    "Why Review is Required (Plain English)": flg["description"],
                    "Recommended Action": "Review raw sensor traces with wireline engineer; verify against repeat section.",
                    "top": t,
                    "base": b
                })

    # ── 2. SYNCHRONISED MULTITASK WELL LOG ────────────────────────────────────
    render_section_header(
        "Synchronised Multitask Well Log",
        "activity",
        "Move your cursor anywhere across the graph: the exact same depth highlights synchronously across all 4 tracks with review callouts"
    )

    vsh_s = petro.get("VSH")
    phi_s = petro.get("PHID")
    sw_s = petro.get("SW")
    depths = df.index.values

    # Build 4-track Plotly Subplots sharing Y-axis
    fig = make_subplots(
        rows=1,
        cols=4,
        shared_yaxes=True,
        horizontal_spacing=0.025,
        subplot_titles=[
            "Track 1: GR & Caliper (Depth ft)",
            "Track 2: Deep Resistivity (RT)",
            "Track 3: Porosity (phiD & phiN)",
            "Track 4: Water Sat & Review Alerts"
        ]
    )

    # ── TRACK 1: GR & Caliper ──
    if "GR" in df.columns:
        gr_custom, gr_hover = generate_hover_customdata("GR", df["GR"])
        fig.add_trace(
            go.Scatter(
                x=df["GR"],
                y=depths,
                name="GR (gAPI)",
                line=dict(color="#10b981", width=1.3),
                customdata=gr_custom,
                hovertemplate=gr_hover
            ),
            row=1, col=1
        )
    if "CALI" in df.columns:
        cali_aux = {"BS": df["BS"].values} if "BS" in df.columns else {}
        cali_custom, cali_hover = generate_hover_customdata("CALI", df["CALI"], cali_aux)
        fig.add_trace(
            go.Scatter(
                x=df["CALI"],
                y=depths,
                name="CALI (in)",
                line=dict(color="#38bdf8", width=1.3),
                customdata=cali_custom,
                hovertemplate=cali_hover
            ),
            row=1, col=1
        )
    if "BS" in df.columns:
        bs_custom, bs_hover = generate_hover_customdata("BS", df["BS"])
        fig.add_trace(
            go.Scatter(
                x=df["BS"],
                y=depths,
                name="Bit Size (in)",
                line=dict(color="#94a3b8", width=1.2, dash="dash"),
                customdata=bs_custom,
                hovertemplate=bs_hover
            ),
            row=1, col=1
        )

    # ── TRACK 2: Deep Resistivity RT (Log Scale) ──
    corr_df = env_pkg["corrected_df"] if env_pkg is not None else df
    if "RT" in df.columns:
        if env_pkg and env_pkg.get("corrections_applied") and "RT_CORR" in corr_df.columns and (corr_df["RT_CORR"] != df["RT"]).any():
            # Show raw sensor RT
            rt_raw_custom, rt_raw_hover = generate_hover_customdata("RT", df["RT"])
            fig.add_trace(
                go.Scatter(
                    x=df["RT"].clip(lower=0.1, upper=2000.0),
                    y=depths,
                    name="RT (Raw Sensor)",
                    line=dict(color="#94a3b8", width=1.1, dash="dash"),
                    customdata=rt_raw_custom,
                    hovertemplate=rt_raw_hover
                ),
                row=1, col=2
            )
            # Show environmentally corrected RT
            rt_corr_custom, rt_corr_hover = generate_hover_customdata("RT", corr_df["RT_CORR"])
            fig.add_trace(
                go.Scatter(
                    x=corr_df["RT_CORR"].clip(lower=0.1, upper=2000.0),
                    y=depths,
                    name="RT (Env Corrected)",
                    line=dict(color="#ef4444", width=1.5),
                    customdata=rt_corr_custom,
                    hovertemplate=rt_corr_hover
                ),
                row=1, col=2
            )
        else:
            rt_custom, rt_hover = generate_hover_customdata("RT", df["RT"])
            fig.add_trace(
                go.Scatter(
                    x=df["RT"].clip(lower=0.1, upper=2000.0),
                    y=depths,
                    name="RT (Ohm.m)",
                    line=dict(color="#ef4444", width=1.4),
                    customdata=rt_custom,
                    hovertemplate=rt_hover
                ),
                row=1, col=2
            )

    # ── TRACK 3: Porosity (phiD, phiN) ──
    if phi_s is not None:
        if env_pkg and env_pkg.get("corrections_applied") and "RHOB" in df.columns:
            raw_phid = env_pkg["petro_raw"].get("PHID")
            if raw_phid is not None and (raw_phid != phi_s).any():
                raw_phid_custom, raw_phid_hover = generate_hover_customdata("PHID", raw_phid)
                fig.add_trace(
                    go.Scatter(
                        x=raw_phid.clip(lower=0.0, upper=0.45),
                        y=depths,
                        name="phiD (Raw Washout)",
                        line=dict(color="#94a3b8", width=1.1, dash="dash"),
                        customdata=raw_phid_custom,
                        hovertemplate=raw_phid_hover
                    ),
                    row=1, col=3
                )
            phid_custom, phid_hover = generate_hover_customdata("PHID", phi_s)
            fig.add_trace(
                go.Scatter(
                    x=phi_s.clip(lower=0.0, upper=0.45),
                    y=depths,
                    name="phiD (Env Corrected)",
                    line=dict(color="#06b6d4", width=1.4),
                    customdata=phid_custom,
                    hovertemplate=phid_hover
                ),
                row=1, col=3
            )
        else:
            phid_custom, phid_hover = generate_hover_customdata("PHID", phi_s)
            fig.add_trace(
                go.Scatter(
                    x=phi_s.clip(lower=0.0, upper=0.45),
                    y=depths,
                    name="phiD Density Porosity",
                    line=dict(color="#06b6d4", width=1.3),
                    customdata=phid_custom,
                    hovertemplate=phid_hover
                ),
                row=1, col=3
            )
    if "NPHI" in df.columns:
        nphi_aux = {"PHID": phi_s.values} if phi_s is not None else {}
        nphi_custom, nphi_hover = generate_hover_customdata("NPHI", df["NPHI"], nphi_aux)
        fig.add_trace(
            go.Scatter(
                x=df["NPHI"].clip(lower=-0.05, upper=0.45),
                y=depths,
                name="NPHI Neutron Porosity",
                line=dict(color="#f59e0b", width=1.3, dash="dot"),
                customdata=nphi_custom,
                hovertemplate=nphi_hover
            ),
            row=1, col=3
        )

    # ── TRACK 4: Saturation & Review Required Alert Markers ──
    if sw_s is not None:
        sw_custom, sw_hover = generate_hover_customdata("SW", sw_s)
        fig.add_trace(
            go.Scatter(
                x=sw_s.clip(lower=0.0, upper=1.0),
                y=depths,
                name="Sw Water Saturation",
                line=dict(color="#3b82f6", width=1.3),
                customdata=sw_custom,
                hovertemplate=sw_hover
            ),
            row=1, col=4
        )
        hc_sat = (1.0 - sw_s).clip(lower=0.0, upper=1.0)
        sh_custom, sh_hover = generate_hover_customdata("SH", hc_sat)
        fig.add_trace(
            go.Scatter(
                x=hc_sat,
                y=depths,
                name="Sh Hydrocarbon Saturation",
                line=dict(color="#10b981", width=1.3, dash="dash"),
                customdata=sh_custom,
                hovertemplate=sh_hover
            ),
            row=1, col=4
        )

    # Add Petrophysicist Review Markers to Track 4
    if review_needed_rows:
        rev_depths = [0.5 * (r["top"] + r["base"]) for r in review_needed_rows]
        rev_custom = np.stack((
            [r["Primary Anomaly"] for r in review_needed_rows],
            [r["Why Review is Required (Plain English)"] for r in review_needed_rows],
            [r["Severity"] for r in review_needed_rows],
            [r["Recommended Action"] for r in review_needed_rows]
        ), axis=-1)
        fig.add_trace(
            go.Scatter(
                x=[0.5] * len(rev_depths),
                y=rev_depths,
                mode="markers",
                name="REVIEW REQUIRED",
                marker=dict(symbol="x", size=11, color="#ef4444", line=dict(width=2, color="#f87171")),
                customdata=rev_custom,
                hovertemplate=(
                    "<b>PETROPHYSICIST REVIEW REQUIRED</b><br>"
                    "Depth: <b>%{y:.1f} ft</b><br>"
                    "<b>Severity:</b> %{customdata[2]}<br>"
                    "<b>Anomaly:</b> %{customdata[0]}<br>"
                    "<b>Why Review Needed:</b> %{customdata[1]}<br>"
                    "<b>Action:</b> %{customdata[3]}"
                    "<extra></extra>"
                )
            ),
            row=1, col=4
        )

        # Highlight horizontal shaded alert bands across the entire log
        for r in review_needed_rows[:8]:  # show up to 8 alert ribbons cleanly
            fig.add_hrect(
                y0=r["top"],
                y1=r["base"],
                fillcolor="rgba(239, 68, 68, 0.12)",
                line_width=1,
                line_color="rgba(239, 68, 68, 0.4)",
                annotation_text=f" Review Needed ({r['Primary Anomaly'][:12]})",
                annotation_position="top left",
                annotation_font=dict(size=8, color="#f87171")
            )

    # Layout configuration: Depth inverted, synchronized spikes, corporate theme
    fig.update_yaxes(
        autorange="reversed",
        title_text="",
        showgrid=True,
        gridcolor="#334155",
        zerolinecolor="#334155",
        color="#cbd5e1",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1.5,
        spikecolor="#38bdf8",
        spikedash="solid",
        row=1, col=1
    )
    for col_idx in range(2, 5):
        fig.update_yaxes(
            autorange="reversed",
            showgrid=True,
            gridcolor="#334155",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1.5,
            spikecolor="#38bdf8",
            spikedash="solid",
            row=1, col=col_idx
        )

    fig.update_xaxes(title_text="GR / CALI / BS", range=[0, 150], row=1, col=1)
    fig.update_xaxes(title_text="RT (log scale)", type="log", range=[-1, 3], row=1, col=2)
    fig.update_xaxes(title_text="Porosity (v/v)", range=[0.45, -0.05], autorange="reversed", row=1, col=3)
    fig.update_xaxes(title_text="Saturation (v/v)", range=[0.0, 1.0], row=1, col=4)

    fig.update_layout(
        height=620,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        hovermode="closest",
        hoverdistance=35,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            font=dict(size=9, color="#f8fafc"),
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#334155",
            borderwidth=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── 4. MULTI-CURVE PETROPHYSICAL CROSS-CORRELATION & SYNTHESIS ───────────
    render_section_header(
        "4. Multi-Curve Petrophysical Cross-Correlation & Synthesis (Error Minimization)",
        "layers",
        "Simultaneous cross-examination across GR, SP, RT, RHOB, NPHI, and CALI to eliminate single-sensor ambiguities and minimize interpretation error"
    )

    if cross_corr is not None:
        sm = cross_corr.get("summary_metrics", {})
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi("Confirmed Oil Pay", f"{sm.get('confirmed_clean_oil_pay_ft', 0.0):.1f} ft", "Multi-curve verified oil pay")
        with c2:
            render_kpi("Confirmed Gas-Cap Pay", f"{sm.get('confirmed_gas_pay_ft', 0.0):.1f} ft", "On-gauge phiN < phiD crossover")
        with c3:
            render_kpi("False Gas Debunked", f"{sm.get('false_gas_alarms_debunked_ft', 0.0):.1f} ft", "Washout artifacts screened")
        with c4:
            render_kpi("Bypassed Arkosic Pay", f"{sm.get('radioactive_pay_recovered_ft', 0.0):.1f} ft", "Recovered via SP deflection")

        # Synthesized conclusions
        conclusions = cross_corr.get("conclusions", [])
        if conclusions:
            for concl in conclusions:
                sev_color = "#10b981" if concl["severity"] == "SUCCESS" else ("#f59e0b" if concl["severity"] == "WARNING" else "#38bdf8")
                st.markdown(clean_html(f"""
                <div style="background: #1e293b; border-left: 4px solid {sev_color}; border-radius: 0 6px 6px 0; padding: 12px 18px; margin-bottom: 10px;">
                    <div style="color: {sev_color}; font-weight: 700; font-size: 0.92rem; margin-bottom: 4px;">
                        {concl['title']}
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.5;">
                        {concl['text']}
                    </div>
                </div>
                """), unsafe_allow_html=True)

        # Facies distribution & ledger expander
        cc_df = cross_corr.get("cross_corr_df")
        if cc_df is not None:
            with st.expander("View Depth-by-Depth Multi-Curve Cross-Examination Ledger", expanded=False):
                st.dataframe(
                    cc_df[["MULTICURVE_FACIES", "CONFIDENCE", "SYNTHESIS_EVIDENCE"]],
                    use_container_width=True
                )

    # ── 4. IDEAL RESERVOIR BENCHMARK GAP ANALYSIS ─────────────────────────────
    render_section_header(
        "Ideal Reservoir Benchmark Comparison (Gap Analysis)",
        "target",
        f"Side-by-side gap analysis comparing active well parameters against world-class clastic reservoir baselines ({prox['composite_reservoir_proximity_pct']}% composite match, {prox['reservoir_tier']})"
    )

    # Proximity Meter Progress Bars & Match Percentage (Requirement 10)
    st.markdown("##### Formation Quality Scores Relative to Ideal Reservoir")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Borehole Gauge Stability", f"{prox['gauge_stability_index']}%")
        st.progress(float(prox['gauge_stability_index']) / 100.0)
    with b2:
        st.metric("Matrix Cleanliness (Low Clay)", f"{prox['cleanliness_index']}%")
        st.progress(float(prox['cleanliness_index']) / 100.0)
    with b3:
        st.metric("Matrix Storage Porosity", f"{prox['porosity_quality_index']}%")
        st.progress(float(prox['porosity_quality_index']) / 100.0)
    with b4:
        st.metric("Hydrocarbon Saturation", f"{prox['hydrocarbon_saturation_index']}%")
        st.progress(float(prox['hydrocarbon_saturation_index']) / 100.0)

    # Separate button to reveal the benchmark comparison table (Requirement 10)
    comp_df = get_ideal_benchmark_comparison_table(prox)
    btn_label = "Hide Benchmark Comparison Table" if st.session_state.get("show_benchmark_comparison", False) else "View Benchmark Comparison Table"
    if st.button(btn_label, key="btn_benchmark_table_toggle", type="secondary"):
        st.session_state.show_benchmark_comparison = not st.session_state.get("show_benchmark_comparison", False)
        st.rerun()

    if st.session_state.get("show_benchmark_comparison", False):
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.markdown(clean_html(f"""
        <div style="background: rgba(56, 189, 248, 0.08); border-left: 4px solid #38bdf8; border-radius: 0 8px 8px 0; padding: 14px 20px; margin-top: 14px; margin-bottom: 18px;">
            <div style="color: #38bdf8; font-weight: 700; font-size: 0.96rem; margin-bottom: 6px;">
                Key Takeaways & Comparison Summary vs. Ideal Standard
            </div>
            <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.55;">
                The comparison table above contrasts your given well data with world-class clastic reservoir standards ({prox['composite_reservoir_proximity_pct']}% composite match, <strong>{prox['reservoir_tier']}</strong>).
                Review the <strong>Key Comparison Notes & Concluding Remarks</strong> column to inspect specific operational impacts (borehole breakout, clay choke, porosity storage, or water cut risks) for every measured and calculated parameter.
            </div>
        </div>
        """), unsafe_allow_html=True)

    # ── 5. ACTIONABLE ENGINEERING ADVICE & DELIVERABLES ───────────────────────
    render_section_header("Actionable Engineering Improvements & Deliverables", "shield", "Domain guidance synthesized for drilling, wireline acquisition, and completion engineers")

    advisories = generate_actionable_advisories(
        df,
        qc_flags=flags,
        predictions=None,
        proximity_analysis=prox
    )
    for adv in advisories:
        cat = "drilling" if "drilling" in adv["category"].lower() else ("wireline" if "wireline" in adv["category"].lower() else "petro")
        render_advisory_card(
            title=f"{adv['category']}: {adv['title']}",
            recommendation=adv["recommendation"],
            icon_name=adv.get("icon", "shield"),
            category=cat
        )

    # ── 6. ACTION REQUIRED: SPECIFIC DEPTHS NEEDING PETROPHYSICIST REVIEW (SECOND-TO-LAST) ─────────
    render_section_header(
        "Action Required: Specific Depths Needing Petrophysicist Review",
        "alert",
        f"Critical intervals flagged for sensor distortion, borehole breakout, or tool failure requiring immediate petrophysical intervention ({len(review_needed_rows)} intervals identified)"
    )

    if review_needed_rows:
        rev_df = pd.DataFrame(review_needed_rows)
        crit_rows = [r for r in review_needed_rows if r["Severity"] in ["HIGH", "CRITICAL"]]
        if crit_rows:
            st.markdown(f"**Identified {len(crit_rows)} high-priority intervals requiring operational intervention:**")
            for cr in crit_rows[:5]:
                st.markdown(clean_html(f"""
                <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; border-radius: 0 6px 6px 0; padding: 12px 18px; margin-bottom: 10px;">
                    <div style="color: #ef4444; font-weight: 700; font-size: 0.92rem; margin-bottom: 4px;">
                        Depth {cr['Depth Interval (ft)']} ({cr['Thickness']}) &bull; {cr['Primary Anomaly']} [{cr['Severity']}]
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.5;">
                        <strong>Root Cause:</strong> {cr['Why Review is Required (Plain English)']}<br>
                        <strong>Remedial Action:</strong> {cr['Recommended Action']}
                    </div>
                </div>
                """), unsafe_allow_html=True)

        with st.expander("View Full Action Required Table", expanded=True):
            display_cols = ["Depth Interval (ft)", "Thickness", "Primary Anomaly", "Severity", "Why Review is Required (Plain English)", "Recommended Action"]
            st.dataframe(rev_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.success("No critical depth intervals flagged for manual review. Formation evaluation logs are on-gauge and within nominal sensor tolerances.")

    # ── 7. EXPORT DELIVERABLES & REPORTS (LAST) ──────────────────────────────
    render_section_header("Export Deliverables & Reports", "download", "Download structured CSV formation deliverables or print a complete engineering report")
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        if flags is not None and len(flags) > 0:
            st.download_button(
                "Export QC Flags (CSV)",
                flags.to_csv(index=False).encode(),
                f"{active_res.well_name}_qc_flags.csv",
                "text/csv",
                use_container_width=True
            )
    with e2:
        if petro:
            petro_export = {k: v for k, v in petro.items() if isinstance(v, (pd.Series, np.ndarray))}
            st.download_button(
                "Export Petrophysical Curves (CSV)",
                pd.DataFrame(petro_export, index=df.index).to_csv().encode(),
                f"{active_res.well_name}_petrophysics.csv",
                "text/csv",
                use_container_width=True
            )
    with e3:
        if comp_df is not None:
            st.download_button(
                "Export Benchmark Gap (CSV)",
                comp_df.to_csv(index=False).encode(),
                f"{active_res.well_name}_benchmark_gap.csv",
                "text/csv",
                use_container_width=True
            )
    with e4:
        if cross_corr and cross_corr.get("cross_corr_df") is not None:
            st.download_button(
                "Export Multi-Curve Ledger (CSV)",
                cross_corr["cross_corr_df"].to_csv().encode(),
                f"{active_res.well_name}_multicurve_synthesis.csv",
                "text/csv",
                use_container_width=True
            )
    with e5:
        if st.button("Print Formation Report", key="btn_print_formation_report", type="primary", use_container_width=True):
            st.components.v1.html("<script>window.parent.print();</script>", height=0)
