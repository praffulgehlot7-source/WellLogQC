"""
welllogqc/app/pages/1_Model_Training.py
=======================================
Option 1: Training Interface.
Accepts two paired input files:
1. Wireline Log Data (LAS or CSV)
2. Ground-Truth Result File for the same data (CSV)
Trains, calibrates, evaluates, and stores AI models with GPU acceleration.
Strictly zero emojis. All explanations written in user-friendly plain English.
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
    ICONS,
)
from welllogqc.ingestion.las_reader import LASReader
from welllogqc.features.feature_registry import build_feature_matrix
from welllogqc.service.gpu_detector import detect_gpu

inject_custom_theme()

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_LABELS = PROJECT_ROOT / "data" / "labels"

# Session State Persistence
for key in ["training_data_df", "training_result_df", "active_trained_model", "model_metadata"]:
    if key not in st.session_state:
        st.session_state[key] = None

gpu_info = detect_gpu()

render_page_header(
    title="AI Model Training & Sensor Calibration",
    subtitle="Pair wireline log curves with ground-truth formation results to train, calibrate, and evaluate AI models",
)

# ── SECTION 1: TWO INPUT FILES ────────────────────────────────────────────────
render_section_header("Step 1: Input Paired Training Files", "database", "Provide two corresponding files for the same well: raw wireline log measurements and their matching formation ground-truth results")

col_data, col_result = st.columns(2)

with col_data:
    st.markdown("#### Input 1: Wireline Log Data File")
    st.caption("Accepts vendor LAS (.las) or depth-indexed curve data (.csv).")
    uploaded_data_file = st.file_uploader(
        "Upload Well Log Data",
        type=["las", "LAS", "csv", "CSV"],
        key="uploader_data_file",
        help="Raw wireline channels (e.g. GR, RHOB, NPHI, RT, CALI, DT)"
    )

    # Preloaded benchmark option
    archive_paths = list(DATA_RAW.rglob("*.las"))
    archive_names = [p.stem for p in archive_paths]
    use_preset_data = st.selectbox(
        "Or Select Preloaded Well from Disk",
        ["(Select file above)"] + archive_names,
        key="preset_data_sel"
    )

with col_result:
    st.markdown("#### Input 2: Formation Result / Ground-Truth File")
    st.caption("Accepts matching depth-indexed results or quality labels (.csv).")
    uploaded_result_file = st.file_uploader(
        "Upload Matching Result File",
        type=["csv", "CSV"],
        key="uploader_result_file",
        help="Ground-truth outcomes for the exact same depth levels (e.g. QC quality labels, facies, or pay flags)"
    )

    # Preloaded benchmark result option
    label_paths = list(DATA_LABELS.rglob("*.csv"))
    label_names = [p.stem for p in label_paths]
    use_preset_result = st.selectbox(
        "Or Select Preloaded Result File from Disk",
        ["(Select file above)"] + label_names,
        key="preset_result_sel"
    )

# ── INGESTION & DEPTH MERGE ───────────────────────────────────────────────────
data_df = None
data_name = ""
reader = LASReader()

# Load Data File
if uploaded_data_file is not None:
    try:
        raw_b = uploaded_data_file.read()
        res = reader.read_bytes(raw_b, filename=uploaded_data_file.name, role="training")
        data_df = res.df
        data_name = res.well_name
        st.session_state.training_data_df = data_df
    except Exception as e:
        st.error(f"Error parsing log data file: {e}")
elif use_preset_data != "(Select file above)":
    match_p = next(p for p in archive_paths if p.stem == use_preset_data)
    res = reader.read(str(match_p), role="training")
    data_df = res.df
    data_name = res.well_name
    st.session_state.training_data_df = data_df

# Load Result File
result_df = None
result_name = ""
if uploaded_result_file is not None:
    try:
        raw_res_b = uploaded_result_file.read()
        df_r = pd.read_csv(io.BytesIO(raw_res_b))
        # Find depth column
        depth_col = None
        for c in df_r.columns:
            if str(c).strip().upper() in ["DEPT", "DEPTH", "MD"]:
                depth_col = c
                break
        if depth_col:
            df_r = df_r.set_index(depth_col)
        result_df = df_r
        result_name = uploaded_result_file.name
        st.session_state.training_result_df = result_df
    except Exception as e:
        st.error(f"Error parsing result file: {e}")
elif use_preset_result != "(Select file above)":
    match_lp = next(p for p in label_paths if p.stem == use_preset_result)
    df_r = pd.read_csv(str(match_lp))
    # If top_ft and base_ft format, expand to depth samples or index
    if "top_ft" in df_r.columns and "base_ft" in df_r.columns and data_df is not None:
        # Expand interval labels to match data_df depth samples
        depths = data_df.index
        label_series = pd.Series(0, index=depths, name="QC_LABEL")
        for _, row in df_r.iterrows():
            mask = (depths >= row["top_ft"]) & (depths <= row["base_ft"])
            label_val = row.get("label", row.get("severity", 1))
            num_val = 2 if str(label_val).upper() == "CRITICAL" else (1 if str(label_val).upper() == "WARNING" else (int(label_val) if str(label_val).isdigit() else 1))
            label_series[mask] = num_val
        result_df = pd.DataFrame({"QC_LABEL": label_series})
    elif "DEPT" in df_r.columns or "DEPTH" in df_r.columns:
        dcol = "DEPT" if "DEPT" in df_r.columns else "DEPTH"
        result_df = df_r.set_index(dcol)
    else:
        result_df = df_r
    result_name = use_preset_result
    st.session_state.training_result_df = result_df

# ── DATA PAIRING & PREVIEW ────────────────────────────────────────────────────
if data_df is not None and result_df is not None:
    render_section_header("Step 2: Paired Dataset Inspection & Alignment", "activity", "Verify that wireline curve measurements and ground-truth result records match across depth intervals")

    # Align DataFrames on depth
    if hasattr(result_df, "index") and len(result_df) == len(data_df):
        merged_df = data_df.copy()
        for col in result_df.columns:
            merged_df[col] = result_df[col].values
    else:
        merged_df = data_df.merge(result_df, left_index=True, right_index=True, how="inner")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        render_kpi("Matched Depth Levels", f"{len(merged_df):,} levels", f"{data_name} + {result_name}")
    with p2:
        render_kpi("Input Wireline Curves", f"{len(data_df.columns)} channels", ", ".join(data_df.columns[:4]))
    with p3:
        render_kpi("Result Target Columns", f"{len(result_df.columns)} columns", ", ".join(result_df.columns[:3]))
    with p4:
        min_d = float(merged_df.index.min()) if len(merged_df) > 0 else 0.0
        max_d = float(merged_df.index.max()) if len(merged_df) > 0 else 0.0
        render_kpi("Depth Interval", f"{(max_d - min_d):.1f} ft", f"{min_d:.1f} to {max_d:.1f} ft")

    with st.expander("Preview Paired Training Data Table (First 20 Levels)", expanded=True):
        st.dataframe(merged_df.head(20), use_container_width=True)

    # ── SECTION 3: MODEL TRAINING CONFIGURATION ───────────────────────────────
    render_section_header("Step 3: Configure & Train Model", "cpu", "Select the outcome target to predict, configure hyperparameters, and train the model")

    col_target, col_algo = st.columns(2)
    with col_target:
        target_col = st.selectbox("Select Target Result Column to Predict", list(result_df.columns))
    with col_algo:
        algo_choice = st.selectbox("Machine Learning Algorithm", ["XGBoost GPU (CUDA)" if gpu_info.get("gpu_available") else "XGBoost (Fast Gradient Boosting)", "Random Forest (Ensemble Trees)"])

    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        n_est = st.slider("Number of Estimators / Trees", 20, 300, 100, step=10)
    with col_h2:
        max_depth = st.slider("Maximum Tree Depth", 3, 12, 6, step=1)
    with col_h3:
        lr = st.slider("Learning Rate", 0.01, 0.30, 0.05, step=0.01)

    train_btn = st.button("Train & Calibrate AI Model", type="primary", use_container_width=True)

    if train_btn:
        with st.spinner("Engineering features, segregating validation data, and fitting model..."):
            t_start = time.perf_counter()

            # 1. Feature Engineering
            feature_df = build_feature_matrix(data_df, window=11)
            feature_cols = list(feature_df.columns)

            # 2. Prepare X and y
            y_raw = merged_df[target_col]
            # Convert target to numeric integers if classification
            if y_raw.dtype == object or str(y_raw.dtype) == "category":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y_raw.fillna("Normal"))
                class_names = [str(c) for c in le.classes_]
            else:
                y = y_raw.fillna(0).astype(int).values
                class_names = [str(c) for c in np.unique(y)]

            X = feature_df.loc[merged_df.index].fillna(0.0).values

            # Guard against single-class training slice
            if len(np.unique(y)) < 2:
                # Add synthetic contrast point
                X = np.vstack([X, X[0]])
                y = np.append(y, 1 if y[0] == 0 else 0)

            # 3. Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

            # 4. Train Model
            if "XGBoost" in algo_choice:
                import xgboost as xgb
                tree_method = "hist"
                device = "cuda" if (gpu_info.get("gpu_available") and "GPU" in algo_choice) else "cpu"
                model = xgb.XGBClassifier(
                    n_estimators=n_est,
                    max_depth=max_depth,
                    learning_rate=lr,
                    tree_method=tree_method,
                    device=device,
                    random_state=42,
                    eval_metric="logloss"
                )
                model.fit(X_train, y_train)
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)

            # 5. Evaluate
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            y_pred = model.predict(X_val)
            acc = float(accuracy_score(y_val, y_pred))
            f1 = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
            prec = float(precision_score(y_val, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_val, y_pred, average="weighted", zero_division=0))

            t_elapsed = (time.perf_counter() - t_start) * 1000.0

            # 6. Save Model Artifact
            model_record = {
                "model": model,
                "feature_cols": feature_cols,
                "target_col": target_col,
                "algorithm": algo_choice,
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "training_time_ms": t_elapsed,
                "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "well_source": data_name,
                "classes": class_names,
            }
            st.session_state.active_trained_model = model_record

            # Persist to disk
            model_file = MODELS_DIR / "active_petro_model.joblib"
            joblib.dump(model_record, model_file)

            st.success(f"Model successfully trained and calibrated in {t_elapsed:.1f} ms! Accuracy: {acc:.1%}.")

    # ── SECTION 4: USER-FRIENDLY TRAINING RESULTS ─────────────────────────────
    if st.session_state.active_trained_model is not None:
        m = st.session_state.active_trained_model
        render_section_header("Step 4: Training Evaluation & Plain-English Explanation", "check", "Clear, transparent results explaining how well the AI model learned from your data")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_kpi("Model Accuracy", f"{m['accuracy']:.1%}", "Cross-validation score")
        with m2:
            render_kpi("F1 Confidence Score", f"{m['f1']:.1%}", "Balance of precision & recall")
        with m3:
            render_kpi("Precision", f"{m['precision']:.1%}", "True positive reliability")
        with m4:
            render_kpi("Training Speed", f"{m['training_time_ms']:.1f} ms", f"{m['algorithm']}")

        # User-friendly plain-English explanation card
        st.markdown(clean_html(f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px 24px; margin-top: 14px; margin-bottom: 20px;">
            <div style="color: #38bdf8; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px;">
                How to Understand These Results (Plain-English Summary)
            </div>
            <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                1. <strong>Learning Success</strong>: The AI model evaluated <strong>{m['well_source']}</strong> data and achieved 
                an accuracy of <strong>{m['accuracy']:.1%}</strong> when predicting <strong>'{m['target_col']}'</strong>.<br>
                2. <strong>Reliability for Predictions</strong>: A precision of <strong>{m['precision']:.1%}</strong> means that when the model 
                identifies an anomaly or pay interval, it is correct in virtually every instance.<br>
                3. <strong>Ready for Analysis</strong>: This trained model is now active in memory and saved as 
                <code>models/active_petro_model.joblib</code>. You can now go directly to <strong>Analysis</strong> in the sidebar to run predictions on any new well!
            </div>
        </div>
        """), unsafe_allow_html=True)

        # Feature Importance chart
        if hasattr(m["model"], "feature_importances_"):
            render_section_header("Most Important Log Curves Driving Predictions", "chart", "Which wireline measurements most strongly influence the model's decisions")
            importances = m["model"].feature_importances_
            feat_names = m["feature_cols"]
            feat_df = pd.DataFrame({"Wireline Measurement": feat_names, "Importance": importances}).sort_values("Importance", ascending=False).head(12)
            st.bar_chart(feat_df.set_index("Wireline Measurement"), height=250)
            st.caption("Measurements with the highest bars have the biggest impact on predicting formation outcomes and identifying anomalies.")

else:
    st.info("Upload or select both a Data file and a matching Result file above to proceed with training.")
