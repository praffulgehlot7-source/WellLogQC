"""
welllogqc.ml.evaluate
=====================
Model evaluation, validation metrics, confusion matrix, and PR curve plotting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _default_build_feature_matrix(data: Any, feature_window: int = 11) -> pd.DataFrame:
    """Build rolling and gradient features for a well log DataFrame."""
    try:
        from welllogqc.features import build_feature_matrix
        return build_feature_matrix(data, feature_window=feature_window)
    except (ImportError, AttributeError):
        pass

    df = data.df if hasattr(data, 'df') else data
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected IngestionResult or DataFrame, got {type(data)}.")

    features = pd.DataFrame(index=df.index)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        series = df[col].astype(np.float64)
        features[col] = series
        features[f"{col}_mean_{feature_window}"] = (
            series.rolling(window=feature_window, center=True, min_periods=1).mean()
        )
        features[f"{col}_std_{feature_window}"] = (
            series.rolling(window=feature_window, center=True, min_periods=1)
            .std()
            .fillna(0.0)
        )
        features[f"{col}_diff_{feature_window}"] = series.diff().fillna(0.0)

    if 'CALI' in df.columns and 'BS' in df.columns:
        features['CALI_BS_DIFF'] = (
            df['CALI'].astype(np.float64) - df['BS'].astype(np.float64)
        )

    return features.astype(np.float64)


def evaluate_model(
    model: Any,
    preprocessor: Any,
    test_data: List[Any],
    decision_threshold: float = 0.5,
    binary_mode: bool = True
) -> Dict[str, Any]:
    """Evaluate a trained model on a dataset of well logs and labels.

    Evaluates across all wells in test_data, computing overall precision, recall,
    f1, ROC-AUC, PR-AUC, confusion matrix, per-well breakdowns, and classification report.
    Samples with label == -1 (Unreviewed) are excluded.

    Args:
        model: Fitted scikit-learn classifier.
        preprocessor: Fitted WellLogPreprocessor instance.
        test_data: List of tuples (ingestion_result_or_df, labels_series).
        decision_threshold: Threshold probability for positive class (binary_mode).
            Defaults to 0.5.
        binary_mode: If True, evaluates binary classification (0 vs 1).
            If False, evaluates multiclass (0, 1, 2). Defaults to True.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'precision': float
            - 'recall': float
            - 'f1': float
            - 'roc_auc': float
            - 'pr_auc': float
            - 'confusion_matrix': List[List[int]] (2D list)
            - 'per_well': Dict[str, Dict[str, float]] with precision, recall, f1
            - 'classification_report': str

    Raises:
        ValueError: If test_data is empty or input formats are invalid.
    """
    if not test_data:
        raise ValueError("test_data cannot be empty.")

    per_well: Dict[str, Dict[str, float]] = {}
    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    all_y_proba: List[float] = []

    for idx, item in enumerate(test_data):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                f"Each item in test_data must be a (data, labels) tuple/list. Got item at {idx}."
            )
        data_item, labels_series = item
        well_name = getattr(data_item, 'well_name', f"well_{idx}")

        X_df = _default_build_feature_matrix(data_item)

        for col in preprocessor.feature_cols:
            if col not in X_df.columns:
                X_df[col] = np.nan
        X_df = X_df[preprocessor.feature_cols]

        labels = labels_series.reindex(X_df.index)
        valid_mask = (labels != -1) & (~labels.isna())
        if not valid_mask.any():
            per_well[well_name] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            continue

        X_valid = X_df.loc[valid_mask]
        y_valid = labels.loc[valid_mask].to_numpy(dtype=np.int64)

        X_trans = preprocessor.transform(X_valid)

        if binary_mode:
            y_true_well = (y_valid > 0).astype(int)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_trans)
                if proba.shape[1] == 2:
                    y_prob_well = proba[:, 1]
                elif proba.shape[1] > 2:
                    y_prob_well = 1.0 - proba[:, 0]
                else:
                    y_prob_well = proba[:, 0]
            elif hasattr(model, 'decision_function'):
                dec = model.decision_function(X_trans)
                y_prob_well = 1.0 / (1.0 + np.exp(-dec))
            else:
                y_prob_well = model.predict(X_trans).astype(float)

            y_pred_well = (y_prob_well >= decision_threshold).astype(int)

            p_w = float(precision_score(y_true_well, y_pred_well, zero_division=0))
            r_w = float(recall_score(y_true_well, y_pred_well, zero_division=0))
            f_w = float(f1_score(y_true_well, y_pred_well, zero_division=0))

            all_y_proba.extend(y_prob_well.tolist())
        else:
            y_true_well = y_valid
            y_pred_well = model.predict(X_trans).astype(int)
            if hasattr(model, 'predict_proba'):
                y_prob_well = model.predict_proba(X_trans)[:, -1]
                all_y_proba.extend(y_prob_well.tolist())

            p_w = float(precision_score(y_true_well, y_pred_well, average='macro', zero_division=0))
            r_w = float(recall_score(y_true_well, y_pred_well, average='macro', zero_division=0))
            f_w = float(f1_score(y_true_well, y_pred_well, average='macro', zero_division=0))

        per_well[well_name] = {'precision': p_w, 'recall': r_w, 'f1': f_w}
        all_y_true.extend(y_true_well.tolist())
        all_y_pred.extend(y_pred_well.tolist())

    if not all_y_true:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'roc_auc': 0.0,
            'pr_auc': 0.0,
            'confusion_matrix': [],
            'per_well': per_well,
            'classification_report': "",
        }

    y_true_arr = np.array(all_y_true, dtype=int)
    y_pred_arr = np.array(all_y_pred, dtype=int)

    avg_method = 'binary' if binary_mode else 'macro'
    prec = float(precision_score(y_true_arr, y_pred_arr, average=avg_method, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, average=avg_method, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, average=avg_method, zero_division=0))

    roc_auc = 0.0
    pr_auc = 0.0
    if binary_mode and all_y_proba and len(np.unique(y_true_arr)) > 1:
        y_prob_arr = np.array(all_y_proba, dtype=np.float64)
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))

    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()
    report = classification_report(y_true_arr, y_pred_arr, zero_division=0)

    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'per_well': per_well,
        'classification_report': report,
    }


def plot_confusion_matrix(
    cm: Union[List[List[int]], np.ndarray],
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None
) -> matplotlib.figure.Figure:
    """Plot and optionally save a confusion matrix heatmap.

    Args:
        cm: 2D array or list of lists representing the confusion matrix.
        class_names: List of class display names corresponding to rows/columns.
        save_path: Optional file path to save the generated figure.

    Returns:
        matplotlib.figure.Figure: Generated Figure object.
    """
    cm_arr = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)

    thresh = cm_arr.max() / 2.0 if cm_arr.size > 0 and cm_arr.max() > 0 else 1.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            color = "white" if cm_arr[i, j] > thresh else "black"
            ax.text(j, i, format(cm_arr[i, j], 'd'),
                    ha="center", va="center", color=color, fontweight='bold')

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), bbox_inches='tight', dpi=150)

    return fig


def plot_pr_curve(
    y_true: Union[List[int], np.ndarray],
    y_proba: Union[List[float], np.ndarray],
    save_path: Optional[Union[str, Path]] = None
) -> matplotlib.figure.Figure:
    """Plot and optionally save a Precision-Recall curve.

    Args:
        y_true: True binary labels (0 or 1).
        y_proba: Predicted probabilities for the positive class.
        save_path: Optional file path to save the generated figure.

    Returns:
        matplotlib.figure.Figure: Generated Figure object.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_proba, dtype=np.float64)

    precision, recall, _ = precision_recall_curve(y_t, y_p)
    ap = float(average_precision_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color='#1f77b4', lw=2, label=f'PR curve (AP = {ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), bbox_inches='tight', dpi=150)

    return fig
