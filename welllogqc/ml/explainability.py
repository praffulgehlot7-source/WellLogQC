"""
welllogqc.ml.explainability
===========================
Model interpretation, feature importances, and multi-source evidence extraction.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd


def get_feature_importance(model: Any, feature_cols: List[str]) -> pd.DataFrame:
    """Extract and rank feature importances or coefficient magnitudes from a fitted model.

    For tree-based models (RandomForest, GradientBoosting), extracts Gini/impurity
    importance via .feature_importances_. For linear models (LogisticRegression),
    extracts the absolute value of coefficients.

    Args:
        model: Fitted scikit-learn classifier.
        feature_cols: List of feature names corresponding to model input dimensions.

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'importance'] sorted in
            descending order of importance. The 'importance' column is float64.

    Raises:
        ValueError: If model lacks feature_importances_ and coef_, or if feature count
            does not match model parameters.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            # First row for binary logistic regression or absolute coefficient vector
            importances = np.abs(coef[0])
    else:
        raise ValueError(
            "Model does not expose 'feature_importances_' or 'coef_' attributes."
        )

    if len(importances) != len(feature_cols):
        raise ValueError(
            f"Dimension mismatch: model has {len(importances)} weights but "
            f"feature_cols has length {len(feature_cols)}."
        )

    df_imp = pd.DataFrame({
        'feature': list(feature_cols),
        'importance': np.asarray(importances, dtype=np.float64),
    })
    return df_imp.sort_values(by='importance', ascending=False).reset_index(drop=True)


def get_evidence_for_interval(
    depth_top: float,
    depth_base: float,
    df: pd.DataFrame,
    qc_flags: pd.DataFrame,
    predictions: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Gather multi-source evidence explaining why a specific depth interval was flagged.

    Collects rule-based QC violations, ML risk scores, and raw log anomalies (e.g. data gaps)
    overlapping the interval [depth_top, depth_base].

    Args:
        depth_top: Top depth of interval of interest (ft).
        depth_base: Base depth of interval of interest (ft).
        df: Well log DataFrame indexed by DEPT.
        qc_flags: DataFrame of rule-based QC flags.
        predictions: DataFrame containing 'QC_RISK_SCORE' and 'QC_RISK_LABEL'.

    Returns:
        List[Dict[str, Any]]: List of evidence records, where each record is a dict with:
            - 'type': str category ('QC_RULE', 'ML_PREDICTION', 'DATA_GAP')
            - 'description': str descriptive summary of the finding
            - 'depth_top': float interval top
            - 'depth_base': float interval base
    """
    top = min(float(depth_top), float(depth_base))
    base = max(float(depth_top), float(depth_base))
    evidence: List[Dict[str, Any]] = []

    # 1. Rule-based QC flags
    if qc_flags is not None and not qc_flags.empty:
        required = {'depth_top', 'depth_base'}
        if required.issubset(qc_flags.columns):
            overlap_mask = (
                (qc_flags['depth_top'].astype(float) <= base) &
                (qc_flags['depth_base'].astype(float) >= top)
            )
            overlapping_flags = qc_flags[overlap_mask]
            for row in overlapping_flags.itertuples(index=False):
                curve = getattr(row, 'curve', 'LOG')
                issue = getattr(row, 'issue', 'Flag')
                sev = getattr(row, 'severity', 'WARNING')
                desc = getattr(row, 'description', '')
                f_top = float(getattr(row, 'depth_top', top))
                f_base = float(getattr(row, 'depth_base', base))

                evidence.append({
                    'type': 'QC_RULE',
                    'description': f"{curve} {issue} [{sev}]: {desc}".strip(),
                    'depth_top': f_top,
                    'depth_base': f_base,
                })

    # 2. ML model predictions
    if predictions is not None and not predictions.empty:
        pred_mask = (predictions.index >= top) & (predictions.index <= base)
        sub_preds = predictions.loc[pred_mask]
        if not sub_preds.empty:
            score_col = 'QC_RISK_SCORE'
            label_col = 'QC_RISK_LABEL'
            if score_col in sub_preds.columns:
                mean_score = float(sub_preds[score_col].mean())
                max_score = float(sub_preds[score_col].max())
                flagged_count = int((sub_preds[label_col] > 0).sum()) if label_col in sub_preds.columns else 0
                total_samples = len(sub_preds)
                pct = (flagged_count / total_samples * 100) if total_samples > 0 else 0.0

                evidence.append({
                    'type': 'ML_PREDICTION',
                    'description': (
                        f"ML QC Risk: mean score = {mean_score:.2f}, max = {max_score:.2f} "
                        f"({flagged_count}/{total_samples} samples [{pct:.1f}%] flagged)"
                    ),
                    'depth_top': top,
                    'depth_base': base,
                })

    # 3. Raw log data anomalies / null gaps
    if df is not None and not df.empty:
        df_mask = (df.index >= top) & (df.index <= base)
        sub_df = df.loc[df_mask]
        if not sub_df.empty:
            num_cols = sub_df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                nan_count = int(sub_df[col].isna().sum())
                if nan_count > 0:
                    pct_nan = (nan_count / len(sub_df)) * 100
                    evidence.append({
                        'type': 'DATA_GAP',
                        'description': f"Curve {col} has {nan_count} missing samples ({pct_nan:.1f}% null) in interval.",
                        'depth_top': top,
                        'depth_base': base,
                    })

    return evidence
