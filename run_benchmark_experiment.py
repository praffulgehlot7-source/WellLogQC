"""
run_benchmark_experiment.py
===========================
Executes a rigorous, reproducible benchmark experiment addressing the central research question:

"Can a reproducible Python workflow combining conventional well-log QC and engineering
features with machine learning improve the identification and prioritization of suspicious
well-log intervals compared with rule-based QC, when evaluated on previously unseen wells?"

Compares:
1. Method A: Conventional Rule-Based QC (baseline)
2. Method B: Machine Learning Model (Random Forest trained on training wells)
3. Method C: Hybrid QC (Conventional flags prioritized by ML risk score)

Evaluated across:
- Training Wells: WELL_A, WELL_B, WELL_C
- Validation Well: WELL_D (model selection / threshold calibration)
- Held-Out Unseen Test Well: WELL_E (strictly untouched during training & tuning)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, classification_report
)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from welllogqc.ingestion.las_reader import LASReader
from welllogqc.qc.qc_reporter import run_all_qc
from welllogqc.ml.label_loader import load_labels, labels_to_series
from welllogqc.ml.preprocessor import WellLogPreprocessor
from welllogqc.ml.train import train_qc_model
from welllogqc.ml.inference import predict_well
from welllogqc.ml.explainability import get_feature_importance

OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

reader = LASReader()

print("=" * 75)
print("  AI-ASSISTED WELL-LOG QC: EXPERIMENTAL BENCHMARK STUDY")
print("=" * 75)

# 1. Ingest Data
print("\n[Step 1] Ingesting well data...")
wells = {
    "WELL_A": reader.read("data/raw/training/WELL_A.las", role="training"),
    "WELL_B": reader.read("data/raw/training/WELL_B.las", role="training"),
    "WELL_C": reader.read("data/raw/training/WELL_C.las", role="training"),
    "WELL_D": reader.read("data/raw/validation/WELL_D.las", role="validation"),
    "WELL_E": reader.read("data/raw/test/WELL_E.las", role="test"),
}

# 2. Load Ground-Truth Reference Labels
print("\n[Step 2] Loading independent petrophysical reference labels...")
labels_true = {}
for wname, res in wells.items():
    role_folder = "training" if res.role == "training" else ("validation" if res.role == "validation" else "test")
    lbl_path = ROOT / "data" / "labels" / role_folder / f"{wname}_labels.csv"
    lbl_df = load_labels(lbl_path)
    s = labels_to_series(lbl_df, res.df.index, wname)
    # Binary ground truth: 0 = Acceptable, 1 = Review Required (Questionable/Unreliable)
    labels_true[wname] = (s > 0).astype(int)
    print(f"  {wname} ({res.role:<10}): {len(s)} samples | Positives (review needed): {labels_true[wname].sum():<5} ({labels_true[wname].mean()*100:.1f}%)")

# 3. Method A: Conventional Rule-Based QC Baseline
print("\n[Step 3] Running Conventional Rule-Based QC (Method A)...")
qc_flags = {}
flags_pred_binary = {}

for wname, res in wells.items():
    flags = run_all_qc(res)
    qc_flags[wname] = flags
    
    # Map flags onto depth index: 1 if any depth sample falls inside any flagged interval, else 0
    pred_vec = np.zeros(len(res.df), dtype=int)
    depths = res.df.index.to_numpy()
    for _, row in flags.iterrows():
        # Exclude constant BS on whole well from being a defect flag if needed, or include raw
        mask = (depths >= row.depth_top) & (depths <= row.depth_base)
        pred_vec[mask] = 1
    flags_pred_binary[wname] = pred_vec

# 4. Method B: Train ML Model on Training Wells Only (WELL_A, WELL_B, WELL_C)
print("\n[Step 4] Fitting ML Model strictly on Training Wells (WELL_A, WELL_B, WELL_C)...")
training_pairs = [
    (wells["WELL_A"], labels_true["WELL_A"]),
    (wells["WELL_B"], labels_true["WELL_B"]),
    (wells["WELL_C"], labels_true["WELL_C"]),
]
validation_pairs = [
    (wells["WELL_D"], labels_true["WELL_D"]),
]

train_res = train_qc_model(
    training_data=training_pairs,
    validation_data=validation_pairs,
    model_type="random_forest",
    feature_window=11,
    decision_threshold=0.5,
    save_dir=ROOT / "models",
    random_state=42
)
print(f"  Model trained: {train_res.model_type} (Timestamp: {train_res.timestamp})")
print(f"  Training features: {len(train_res.feature_cols)} features")

# 5. Method B & C: Inference on all wells (including unseen held-out WELL_E)
print("\n[Step 5] Running ML Inference across all wells...")
ml_preds = {}
for wname, res in wells.items():
    p_df = predict_well(
        ingestion_result=res,
        model=train_res.model,
        preprocessor=train_res.preprocessor,
        decision_threshold=0.5,
        save_path=OUTPUTS_DIR / f"{wname}_benchmark_predictions.csv"
    )
    ml_preds[wname] = p_df

# 6. Comparative Evaluation
print("\n" + "=" * 75)
print("  BENCHMARK PERFORMANCE EVALUATION MATRIX")
print("=" * 75)

metrics_list = []

for wname in ["WELL_A", "WELL_B", "WELL_C", "WELL_D", "WELL_E"]:
    y_true = labels_true[wname].to_numpy()
    role = wells[wname].role
    
    # Method A: Conventional QC
    y_pred_qc = flags_pred_binary[wname]
    p_qc = precision_score(y_true, y_pred_qc, zero_division=0)
    r_qc = recall_score(y_true, y_pred_qc, zero_division=0)
    f1_qc = f1_score(y_true, y_pred_qc, zero_division=0)
    cm_qc = confusion_matrix(y_true, y_pred_qc, labels=[0, 1])
    # False Alarm Rate (FP / (FP + TN))
    tn_qc, fp_qc, fn_qc, tp_qc = cm_qc.ravel()
    fpr_qc = fp_qc / (fp_qc + tn_qc) if (fp_qc + tn_qc) > 0 else 0.0

    # Method B: ML Model
    y_pred_ml = ml_preds[wname]["QC_RISK_LABEL"].to_numpy()
    y_prob_ml = ml_preds[wname]["QC_RISK_SCORE"].to_numpy()
    p_ml = precision_score(y_true, y_pred_ml, zero_division=0)
    r_ml = recall_score(y_true, y_pred_ml, zero_division=0)
    f1_ml = f1_score(y_true, y_pred_ml, zero_division=0)
    cm_ml = confusion_matrix(y_true, y_pred_ml, labels=[0, 1])
    tn_ml, fp_ml, fn_ml, tp_ml = cm_ml.ravel()
    fpr_ml = fp_ml / (fp_ml + tn_ml) if (fp_ml + tn_ml) > 0 else 0.0
    roc_auc = roc_auc_score(y_true, y_prob_ml) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_prob_ml) if len(np.unique(y_true)) > 1 else np.nan

    metrics_list.append({
        "Well": wname,
        "Role": role,
        "QC_Precision": p_qc,
        "QC_Recall": r_qc,
        "QC_F1": f1_qc,
        "QC_FPR (False Alarm)": fpr_qc,
        "ML_Precision": p_ml,
        "ML_Recall": r_ml,
        "ML_F1": f1_ml,
        "ML_FPR (False Alarm)": fpr_ml,
        "ML_ROC_AUC": roc_auc,
        "ML_PR_AUC": pr_auc
    })

bench_df = pd.DataFrame(metrics_list)
pd.set_option("display.precision", 3)
pd.set_option("display.width", 1000)
print(bench_df.to_string(index=False))

# 7. Unseen Well In-Depth Analysis (WELL_E)
print("\n" + "=" * 75)
print("  CRITICAL DRILL-DOWN: PREVIOUSLY UNSEEN TEST WELL (WELL_E)")
print("=" * 75)
y_true_e = labels_true["WELL_E"].to_numpy()
y_qc_e = flags_pred_binary["WELL_E"]
y_ml_e = ml_preds["WELL_E"]["QC_RISK_LABEL"].to_numpy()
y_score_e = ml_preds["WELL_E"]["QC_RISK_SCORE"].to_numpy()

print(f"\n[Ground Truth in WELL_E]: {y_true_e.sum()} suspicious samples out of {len(y_true_e)} ({y_true_e.mean()*100:.1f}%)")
print(f"Conventional QC - Flagged: {y_qc_e.sum()} samples | Recall: {recall_score(y_true_e, y_qc_e):.3f} | Precision: {precision_score(y_true_e, y_qc_e):.3f}")
print(f"ML Classifier   - Flagged: {y_ml_e.sum()} samples | Recall: {recall_score(y_true_e, y_ml_e):.3f} | Precision: {precision_score(y_true_e, y_ml_e):.3f}")

# Top Features Influencing Predictions
fi = get_feature_importance(train_res.model, train_res.feature_cols)
print("\nTop 10 Most Influential Features across Training:")
print(fi.head(10).to_string(index=False))

# Save full benchmark table to outputs
bench_csv = OUTPUTS_DIR / "benchmark_comparison_results.csv"
bench_df.to_csv(bench_csv, index=False)
print(f"\nBenchmark table saved to: {bench_csv}")
