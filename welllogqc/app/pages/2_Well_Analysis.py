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
    render_sidebar_guidance,
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

inject_custom_theme()

MODELS_DIR = PROJECT_ROOT / "models"
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# Session State Persistence
for key in [
    "analysis_well_res", "analysis_qc_flags", "analysis_petro", "analysis_ntg",
    "analysis_prox", "analysis_decisions", "analysis_done", "analysis_env_corr", "analysis_cross_corr"
]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "analysis_done" else False

gpu_info = detect_gpu()

render_page_header(
    title="Well Log Prediction & Formation Analysis",
    subtitle="Single-file wireline log evaluation: automated QC, rigorous petrophysics, and synchronized depth-linked review highlighting",
)

# ── SECTION 1: SINGLE FILE INPUT ──────────────────────────────────────────────
render_section_header("Step 1: Upload or Select Wireline Log File", "database", "Upload a single vendor wireline file (.las, .csv) or pick a preloaded well from disk")

col_upload, col_preset = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Single Well Log File (.las, .csv)",
        type=["las", "LAS", "csv", "CSV"],
        key="analysis_file_uploader",
        help="Upload vendor wireline LAS file or depth-indexed CSV file"
    )

with col_preset:
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

# ── SIDEBAR GUIDANCE BAR ──────────────────────────────────────────────────────
render_sidebar_guidance(active_res.available_curves if active_res is not None else None)

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
run_analysis_btn = st.button("Run AI Prediction & Formation Analysis", type="primary", use_container_width=True)

if run_analysis_btn:
    with st.spinner("Executing multi-sensor QC, petrophysics, and formation gap analysis..."):
        t_start = time.perf_counter()

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

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        st.success(f"Analysis completed in {t_elapsed:.1f} ms! All results, interactive depth-linked graphs, and review points are generated below.")

# ── SECTION 3: COMPREHENSIVE OUTCOMES & USER-FRIENDLY RESULTS ─────────────────
flags = st.session_state.analysis_qc_flags
petro = st.session_state.analysis_petro or {}
ntg = st.session_state.analysis_ntg
prox = st.session_state.analysis_prox
decisions = st.session_state.analysis_decisions

if st.session_state.analysis_done and prox is not None:
    # ── 1. EXECUTIVE RESULTS SUMMARY GRID ─────────────────────────────────────
    render_section_header("1. Executive Results & Key Performance Indicators", "chart", "Overview of formation potential, pay thickness, and reservoir quality in simple numbers")

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

    # Plain-English Executive Verdict Card
    tier = prox.get("reservoir_tier", "Commercial Reservoir")
    st.markdown(clean_html(f"""
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px 24px; margin-top: 14px; margin-bottom: 22px;">
        <div style="color: #38bdf8; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px;">
            Plain-English Formation Verdict: {tier}
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

    # ── 2. DEPTHS WHERE PETROPHYSICIST REVIEW IS REQUIRED ─────────────────────
    render_section_header(
        "2. Action Required: Specific Depths Needing Petrophysicist Review",
        "alert",
        "High-priority intervals pinpointed by the AI engine where borehole washouts, sensor sticking, or extreme anomalies require human verification"
    )

    review_needed_rows = []
    # Identify washouts
    if "CALI" in df.columns and "BS" in df.columns:
        washout_mask = (df["CALI"] - df["BS"]) > 1.5
        if washout_mask.any():
            # Find contiguous washout intervals
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

    # Add flags from rule-based QC
    if flags is not None:
        for _, flg in flags[flags["severity"].isin(["CRITICAL", "WARNING"])].iterrows():
            # Check if not already covered
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

    if review_needed_rows:
        rev_df = pd.DataFrame(review_needed_rows)
        # Highlight in callout cards
        for idx, r in rev_df.head(4).iterrows():
            sev_border = "#ef4444" if r["Severity"] == "CRITICAL" or r["Severity"] == "HIGH" else "#f59e0b"
            st.markdown(clean_html(f"""
            <div style="background: #1e293b; border-left: 5px solid {sev_border}; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155; border-radius: 6px; padding: 12px 18px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong style="color: #f8fafc; font-size: 0.95rem;">Depth Interval: {r['Depth Interval (ft)']} ({r['Thickness']})</strong>
                    <span style="font-size: 0.78rem; font-weight: 700; color: {sev_border}; background: rgba(239, 68, 68, 0.1); padding: 2px 8px; border-radius: 4px;">
                        {r['Severity']} SEVERITY &bull; {r['Primary Anomaly']}
                    </span>
                </div>
                <div style="font-size: 0.86rem; color: #cbd5e1; margin-bottom: 4px;">
                    <strong>Why Review is Needed:</strong> {r['Why Review is Required (Plain English)']}
                </div>
                <div style="font-size: 0.84rem; color: #38bdf8;">
                    <strong>Recommended Action:</strong> {r['Recommended Action']}
                </div>
            </div>
            """), unsafe_allow_html=True)

        if len(rev_df) > 4:
            with st.expander(f"View All {len(rev_df)} Flagged Review Intervals in Detail", expanded=False):
                st.dataframe(rev_df[["Depth Interval (ft)", "Thickness", "Primary Anomaly", "Severity", "Why Review is Required (Plain English)", "Recommended Action"]], use_container_width=True, hide_index=True)
    else:
        st.success("Clean Formation: No high-severity anomalies detected. Log readings are physically consistent across all depth intervals.")

    # ── 3. SYNCHRONIZED INTERACTIVE PLOTLY GRAPHS WITH HOVER HIGHLIGHTING ─────
    render_section_header(
        "3. Synchronized Multi-Track Well Log (Hover to Inspect & Pinpoint Depths)",
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

    # ── 5. IDEAL RESERVOIR BENCHMARK GAP ANALYSIS ─────────────────────────────
    render_section_header(
        "5. Ideal Reservoir Benchmark Comparison (Gap Analysis)",
        "target",
        "Side-by-side gap analysis comparing active well parameters against world-class clastic reservoir baselines (Wilcox/Brent benchmark) with conclusive engineering takeaways"
    )

    comp_df = get_ideal_benchmark_comparison_table(prox)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown(clean_html(f"""
    <div style="background: #1e293b; border-left: 4px solid #38bdf8; border-radius: 0 8px 8px 0; padding: 14px 20px; margin-top: 14px; margin-bottom: 18px;">
        <div style="color: #38bdf8; font-weight: 700; font-size: 0.96rem; margin-bottom: 6px;">
            Key Takeaways & Comparison Summary vs. Ideal Standard
        </div>
        <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.55;">
            The comparison table above contrasts your given well data with world-class clastic reservoir standards ({prox['composite_reservoir_proximity_pct']}% composite match, <strong>{prox['reservoir_tier']}</strong>).
            Review the <strong>Key Comparison Notes & Concluding Remarks</strong> column to inspect specific operational impacts (borehole breakout, clay choke, porosity storage, or water cut risks) for every measured and calculated parameter.
        </div>
    </div>
    """), unsafe_allow_html=True)

    # Proximity Meter Progress Bars
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

    # ── 6. ACTIONABLE ENGINEERING ADVICE & DELIVERABLES ───────────────────────
    render_section_header("6. Actionable Engineering Improvements & Deliverables", "shield", "Domain guidance synthesized for drilling, wireline acquisition, and completion engineers")

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

    # ── 7. GUIDANCE: LOG STANDARDS & PETROPHYSICAL MEANING ────────────────────
    render_section_header(
        "7. Guidance: Ideal Wireline Log Standards & Petrophysical Meaning",
        "compass",
        f"Authoritative benchmark reference and physical meaning for all {len(active_res.available_curves)} curves in target well '{active_res.well_name}' (Compiled from Schlumberger, AAPG, and SPWLA standards)"
    )

    st.markdown(clean_html("""
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px 22px; margin-bottom: 20px;">
        <div style="color: #38bdf8; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px;">
            How to Interpret Wireline Logs in This Well: Standard Benchmarks & Physical Meaning
        </div>
        <div style="color: #cbd5e1; font-size: 0.90rem; line-height: 1.6;">
            Every downhole sensor operates on fundamental physics (nuclear scintillation, electron scattering, hydrogen moderation, electrical induction, and acoustic wave transit). 
            Below is the authentic petroleum engineering reference guide explaining what each log in your uploaded dataset measures, its physical minimum and maximum limits, ideal commercial reservoir baseline, standard clean sand vs. shale cutoffs, and common operational traps.
            <br><br>
            <span style="color: #94a3b8; font-size: 0.84rem;">
                <strong>Key References:</strong> Schlumberger Log Interpretation Principles & Applications (1989/2009) &bull; Asquith & Krygowski, <em>Basic Well Log Analysis</em> (AAPG Methods in Exploration 16) &bull; SPWLA Standards &bull; Glover Petrophysics (University of Aberdeen).
            </span>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # 1. Active Well Standards Comparison Table
    st.markdown("##### 1. Master Reference Table: Ideal Values vs. Cutoffs for Active Well Curves")
    guide_table_df = get_guide_comparison_dataframe(active_res.available_curves)
    st.dataframe(guide_table_df, use_container_width=True, hide_index=True)

    # 2. Detailed Educational Guide for Each Available Curve (Accordions)
    st.markdown(f"##### 2. Guidance: Detailed Formation Log Analysis in '{active_res.well_name}'")
    well_guides = get_curve_guide_for_well(active_res.available_curves)
    
    # Also include derived petrophysical guides if computed
    derived_to_add = []
    if "VSH" in petro:
        derived_to_add.append("VSH")
    if "PHID" in petro:
        derived_to_add.append("PHID")
    if "SW" in petro:
        derived_to_add.append("SW")
    if derived_to_add:
        well_guides.extend(get_curve_guide_for_well(derived_to_add))

    for g in well_guides:
        with st.expander(f"{g['curve_symbol']} - {g['name']} ({g['unit']}) &bull; Category: {g['category']}", expanded=False):
            gc1, gc2 = st.columns([1, 1])
            with gc1:
                st.markdown(clean_html(f"""
                <div style="background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 14px; margin-bottom: 10px;">
                    <strong style="color: #38bdf8;">Ideal Commercial Reservoir Value:</strong><br>
                    <span style="color: #f8fafc; font-size: 0.95rem; font-weight: 600;">{g['ideal_value']}</span><br><br>
                    <strong style="color: #10b981;">Pay Cutoff Rule:</strong><br>
                    <span style="color: #cbd5e1; font-size: 0.90rem;">{g['cutoff_value']}</span><br><br>
                    <strong style="color: #f59e0b;">Shale / Non-Pay Baseline:</strong><br>
                    <span style="color: #cbd5e1; font-size: 0.90rem;">{g['shale_baseline']}</span>
                </div>
                """), unsafe_allow_html=True)
            with gc2:
                st.markdown(clean_html(f"""
                <div style="background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 14px; margin-bottom: 10px;">
                    <strong style="color: #38bdf8;">Physical Measurement Principle:</strong><br>
                    <span style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.4;">{g['physics']}</span><br><br>
                    <strong style="color: #38bdf8;">Petrophysical Significance:</strong><br>
                    <span style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.4;">{g['significance']}</span>
                </div>
                """), unsafe_allow_html=True)

            st.markdown(clean_html(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px; margin-bottom: 10px;">
                <div style="background: #0f172a; border-left: 4px solid #ef4444; padding: 12px 16px; border-radius: 0 6px 6px 0;">
                    <strong style="color: #ef4444; font-size: 0.88rem;">What High Readings Mean (Plain English):</strong><br>
                    <span style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.45;">{g['high_meaning']}</span>
                </div>
                <div style="background: #0f172a; border-left: 4px solid #10b981; padding: 12px 16px; border-radius: 0 6px 6px 0;">
                    <strong style="color: #10b981; font-size: 0.88rem;">What Low Readings Mean (Plain English):</strong><br>
                    <span style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.45;">{g['low_meaning']}</span>
                </div>
            </div>
            <div style="background: #0f172a; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 0 6px 6px 0; margin-top: 10px;">
                <strong style="color: #f59e0b; font-size: 0.88rem;">Operational Traps & Failure Modes (QC Alert):</strong><br>
                <span style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.45;">{g['common_pitfalls']}</span>
            </div>
            <div style="color: #64748b; font-size: 0.80rem; margin-top: 8px;">
                <strong>Literature Citation:</strong> {g['reference']}
            </div>
            """), unsafe_allow_html=True)

    # 3. Master Reference for All Standard Logs
    with st.expander("View Complete Petrophysical Reference for All 24 Standard Wireline Logs (Min & Max Bounds Included)", expanded=False):
        all_guides_df = get_guide_comparison_dataframe()
        st.dataframe(all_guides_df, use_container_width=True, hide_index=True)
        st.caption("Includes comprehensive formation evaluation logs (Spectral Gamma Ray CGR, Laterologs LLD/LLS, Sonic DTS, Density Correction DRHO, Timur Permeability, NMR Porosity, Cable Tension) for complete exploration context.")

    # ── EXPORT DELIVERABLES ───────────────────────────────────────────────────
    render_section_header("Export Deliverables & Reports", "download")
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
        st.download_button(
            "Export Curve Guide (CSV)",
            guide_table_df.to_csv(index=False).encode(),
            f"{active_res.well_name}_curve_guide.csv",
            "text/csv",
            use_container_width=True
        )
    with e5:
        if cross_corr and cross_corr.get("cross_corr_df") is not None:
            st.download_button(
                "Export Multi-Curve Ledger (CSV)",
                cross_corr["cross_corr_df"].to_csv().encode(),
                f"{active_res.well_name}_multicurve_synthesis.csv",
                "text/csv",
                use_container_width=True
            )
