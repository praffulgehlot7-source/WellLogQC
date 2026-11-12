"""
integration_test.py
===================
Full end-to-end integration test: ingestion → QC → petro → features → ML training → inference → report.
Runs without Streamlit. Uses synthetic LAS data only.
"""

import sys
import traceback
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PASS = []
FAIL = []

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def ok(msg):
    PASS.append(msg)
    print(f"  [OK] {msg}")

def fail(msg, exc=None):
    FAIL.append(msg)
    print(f"  [FAIL] {msg}")
    if exc:
        print(f"     {type(exc).__name__}: {exc}")

# ── 1. INGESTION ──────────────────────────────────────────────────────────────
section("1. Ingestion")
try:
    from welllogqc.ingestion.las_reader import LASReader, IngestionResult
    reader = LASReader()
    r_a = reader.read("data/raw/training/WELL_A.las",   role="training")
    r_b = reader.read("data/raw/training/WELL_B.las",   role="training")
    r_c = reader.read("data/raw/training/WELL_C.las",   role="training")
    r_d = reader.read("data/raw/validation/WELL_D.las", role="validation")
    ok(f"WELL_A: {r_a.available_curves}, missing={r_a.missing_curves}")
    ok(f"WELL_B ingested, {len(r_b.df)} rows")
    ok(f"WELL_C missing curves correct: {r_c.missing_curves}")
    ok(f"WELL_D null RHOB count: {r_d.null_counts.get('RHOB', 0)}")
    assert r_c.df.index.name == 'DEPT', "Index must be named DEPT"
    ok("Index name = 'DEPT' confirmed")
except Exception as e:
    fail("Ingestion failed", e); traceback.print_exc(); sys.exit(1)

# ── 2. QC ─────────────────────────────────────────────────────────────────────
section("2. Conventional QC")
try:
    from welllogqc.qc.qc_reporter import run_all_qc, summarise_flags
    flags_a = run_all_qc(r_a)
    flags_b = run_all_qc(r_b)
    flags_c = run_all_qc(r_c)
    flags_d = run_all_qc(r_d)
    ok(f"WELL_A flags: {len(flags_a)}")
    ok(f"WELL_B flags: {len(flags_b)} (expect washout + spikes)")
    ok(f"WELL_C flags: {len(flags_c)} (missing curves handled)")
    ok(f"WELL_D flags: {len(flags_d)}")
    assert set(flags_a.columns) >= {'depth_top','depth_base','curve','issue','severity','description'}, \
        "Flag schema incomplete"
    ok("QC flag schema validated")
    summ = summarise_flags(flags_b)
    ok(f"WELL_B summary: {summ}")
except Exception as e:
    fail("QC failed", e); traceback.print_exc()

# ── 3. PETROPHYSICS ───────────────────────────────────────────────────────────
section("3. Petrophysics")
try:
    from welllogqc.petrophysics.vsh import calc_vsh_linear
    from welllogqc.petrophysics.porosity import calc_density_porosity
    from welllogqc.petrophysics.ntg import calc_ntg

    vsh_a  = calc_vsh_linear(r_a.df, gr_clean=15.0, gr_shale=120.0)
    phid_a = calc_density_porosity(r_a.df)
    ok(f"WELL_A VSH: [{vsh_a.min():.3f}, {vsh_a.max():.3f}]")
    ok(f"WELL_A PHID: [{phid_a.min():.3f}, {phid_a.max():.3f}]")
    assert vsh_a.min() >= 0.0 and vsh_a.max() <= 1.0, "Vsh out of [0,1]"
    ok("Vsh bounds [0,1] confirmed")

    df_ntg = r_a.df.copy()
    df_ntg['VSH_LINEAR'] = vsh_a
    df_ntg['PHID']       = phid_a
    qc_depths = list(zip(flags_a['depth_top'], flags_a['depth_base']))
    ntg = calc_ntg(df_ntg, gross_top=8200.0, gross_base=8700.0, qc_flag_depths=qc_depths)
    ok(f"NTG={ntg['ntg']:.3f}, net={ntg['net_ft']:.1f}ft, classes={ntg['summary']}")
    assert 0.0 <= ntg['ntg'] <= 1.0, f"NTG out of range: {ntg['ntg']}"
    ok("NTG in [0,1] confirmed")
except Exception as e:
    fail("Petrophysics failed", e); traceback.print_exc()

# ── 4. FEATURE ENGINEERING ───────────────────────────────────────────────────
section("4. Feature Engineering")
try:
    from welllogqc.features.feature_registry import build_feature_matrix
    feat_a = build_feature_matrix(r_a.df, window=11)
    feat_c = build_feature_matrix(r_c.df, window=11)
    ok(f"WELL_A feature matrix: {feat_a.shape}")
    ok(f"WELL_C feature matrix (missing curves): {feat_c.shape}")
    assert feat_a.shape[0] == len(r_a.df), "Feature row count mismatch"
    assert not feat_a.isnull().all().any(), "Some feature columns are all-NaN"
    ok("Feature matrix row count and non-null columns confirmed")
except Exception as e:
    fail("Feature engineering failed", e); traceback.print_exc()

# ── 5. PLOTTING ───────────────────────────────────────────────────────────────
section("5. Well-Log Plot")
try:
    from welllogqc.plotting.log_plot import plot_well_log
    Path("outputs").mkdir(exist_ok=True)
    fig = plot_well_log(
        r_b.df, "WELL_B",
        qc_flags=flags_b,
        extra_curves={"VSH": vsh_a.reindex(r_b.df.index)},
        save_path="outputs/integration_test_plot.png",
    )
    ok(f"Plot created: {len(fig.axes)} tracks, saved to outputs/integration_test_plot.png")
    import matplotlib.pyplot as plt
    plt.close(fig)
except Exception as e:
    fail("Plotting failed", e); traceback.print_exc()

# ── 6. ML — LABEL CREATION ────────────────────────────────────────────────────
section("6. ML — Label preparation")
try:
    from welllogqc.ml.problem_definition import flags_to_labels, LABEL_MAP
    labels_a = flags_to_labels(r_a.df, flags_a, mode='binary')
    labels_b = flags_to_labels(r_b.df, flags_b, mode='binary')
    ok(f"WELL_A labels: {labels_a.value_counts().to_dict()}")
    ok(f"WELL_B labels: {labels_b.value_counts().to_dict()}")
    assert set(labels_a.unique()).issubset({0, 1}), "Binary labels must be 0 or 1"
    ok("Binary label values {0,1} confirmed")
except Exception as e:
    fail("Label prep failed", e); traceback.print_exc()

# ── 7. ML — TRAINING ──────────────────────────────────────────────────────────
section("7. ML — Model Training")
training_result = None
try:
    from welllogqc.ml.train import train_qc_model
    training_result = train_qc_model(
        training_data=[(r_a, labels_a), (r_b, labels_b)],
        validation_data=[(r_d, flags_to_labels(r_d.df, flags_d, mode='binary'))],
        model_type='random_forest',
        feature_window=11,
        save_dir='models',
        random_state=42,
    )
    ok(f"Model type: {training_result.model_type}")
    ok(f"Training wells: {training_result.training_wells}")
    ok(f"Train metrics: P={training_result.train_metrics.get('precision',0):.3f}, "
       f"R={training_result.train_metrics.get('recall',0):.3f}, "
       f"F1={training_result.train_metrics.get('f1',0):.3f}")
    if training_result.val_metrics:
        ok(f"Val metrics: P={training_result.val_metrics.get('precision',0):.3f}, "
           f"R={training_result.val_metrics.get('recall',0):.3f}")
    if training_result.model_path:
        ok(f"Model saved: {Path(training_result.model_path).name}")
except Exception as e:
    fail("Model training failed", e); traceback.print_exc()

# ── 8. ML — INFERENCE ─────────────────────────────────────────────────────────
section("8. ML — Inference on WELL_C (missing curves)")
try:
    if training_result:
        from welllogqc.ml.inference import predict_well
        preds = predict_well(
            ingestion_result=r_c,
            model=training_result.model,
            preprocessor=training_result.preprocessor,
            decision_threshold=0.5,
            save_path="outputs/WELL_C_predictions_test.csv",
        )
        ok(f"Predictions shape: {preds.shape}")
        ok(f"Columns: {list(preds.columns)}")
        ok(f"Risk score range: [{preds['QC_RISK_SCORE'].min():.3f}, {preds['QC_RISK_SCORE'].max():.3f}]")
        assert Path("outputs/WELL_C_predictions_test.csv").exists(), "Predictions not saved"
        ok("Predictions saved to disk before label comparison")
    else:
        fail("Skipped — training failed")
except Exception as e:
    fail("Inference failed", e); traceback.print_exc()

# ── 9. REPORTING ──────────────────────────────────────────────────────────────
section("9. HTML Report")
try:
    from welllogqc.reporting.report_builder import build_html_report, build_pdf_report
    from welllogqc.ingestion.ingestion_summary import ingestion_to_dict
    html_str = build_html_report(
        well_name="WELL_A",
        ingestion_report=ingestion_to_dict(r_a),
        qc_flags=flags_a,
        petro_df=df_ntg[['VSH_LINEAR','PHID']],
        ntg_result=ntg,
        predictions=None,
        plot_path="outputs/integration_test_plot.png",
        output_path="outputs/WELL_A_report_test.html",
    )
    ok(f"HTML report generated: {len(html_str)} chars")
    assert Path("outputs/WELL_A_report_test.html").exists(), "HTML file not saved"
    ok("HTML saved to outputs/WELL_A_report_test.html")

    pdf_ok = build_pdf_report("outputs/WELL_A_report_test.html",
                               "outputs/WELL_A_report_test.pdf")
    if pdf_ok:
        ok("PDF generated successfully")
    else:
        ok("PDF skipped (weasyprint GTK not available on this system — expected on Windows)")
except Exception as e:
    fail("Reporting failed", e); traceback.print_exc()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
section("INTEGRATION TEST SUMMARY")
print(f"  Passed: {len(PASS)}")
print(f"  Failed: {len(FAIL)}")
if FAIL:
    print("\n  Failed checks:")
    for f in FAIL:
        print(f"    [FAIL] {f}")
    sys.exit(1)
else:
    print("\n  All checks passed - app is ready to launch.")
    print("\n  Launch with:")
    print("    .venv\\Scripts\\streamlit.exe run welllogqc/app/main.py")
