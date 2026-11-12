"""
welllogqc.ml.inference
======================
Model inference and QC risk scoring on unreviewed or newly ingested wells.
"""

from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd

from welllogqc.ml.evaluate import _default_build_feature_matrix


def predict_well(
    ingestion_result: Any,
    model: Any,
    preprocessor: Any,
    decision_threshold: float = 0.5,
    save_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Run model inference to compute depth-by-depth QC risk scores on a well.

    Extracts required features, aligns them to the preprocessor's feature schema,
    imputes missing curves, scales features, and predicts risk probabilities.
    Saves predictions to disk at save_path BEFORE any label comparison or inspection,
    preserving held-out integrity.

    Args:
        ingestion_result: IngestionResult instance (with .df attribute) or a raw
            pandas DataFrame indexed by DEPT.
        model: Fitted scikit-learn estimator supporting predict_proba or decision_function.
        preprocessor: Fitted WellLogPreprocessor containing feature_cols schema and scaling parameters.
        decision_threshold: Probability cutoff to flag a sample as positive (1: Review_Needed).
            Defaults to 0.5.
        save_path: Optional path to save the prediction results as a CSV file.

    Returns:
        pd.DataFrame: Prediction DataFrame indexed by DEPT with columns:
            - 'QC_RISK_SCORE' (float64, 0.0 to 1.0)
            - 'QC_RISK_LABEL' (int64, 0 or 1)

    Raises:
        ValueError: If ingestion_result is None or invalid format.
    """
    if ingestion_result is None:
        raise ValueError("ingestion_result cannot be None.")

    df = ingestion_result.df if hasattr(ingestion_result, 'df') else ingestion_result
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"Expected IngestionResult (with .df) or pandas DataFrame, got {type(df)}."
        )

    if df.empty:
        empty_df = pd.DataFrame(
            columns=['QC_RISK_SCORE', 'QC_RISK_LABEL'],
            index=df.index
        )
        empty_df['QC_RISK_SCORE'] = empty_df['QC_RISK_SCORE'].astype(np.float64)
        empty_df['QC_RISK_LABEL'] = empty_df['QC_RISK_LABEL'].astype(np.int64)
        empty_df.index.name = 'DEPT'
        if save_path is not None:
            dest = Path(save_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            empty_df.to_csv(dest)
        return empty_df

    # Extract features using default/custom builder
    X_df = _default_build_feature_matrix(ingestion_result)

    # Align columns to match the preprocessor schema exactly
    for col in preprocessor.feature_cols:
        if col not in X_df.columns:
            X_df[col] = np.nan
    X_df = X_df[preprocessor.feature_cols]

    # Preprocess features
    X_trans = preprocessor.transform(X_df)

    # Compute risk score (probability of defect / review needed)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_trans)
        if proba.shape[1] == 2:
            scores = proba[:, 1]
        elif proba.shape[1] > 2:
            # Probability of any abnormal class (1 or 2) = 1.0 - P(class 0)
            scores = 1.0 - proba[:, 0]
        else:
            scores = proba[:, 0]
    elif hasattr(model, 'decision_function'):
        dec = model.decision_function(X_trans)
        scores = 1.0 / (1.0 + np.exp(-dec))
    else:
        scores = model.predict(X_trans).astype(float)

    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    labels = (scores >= decision_threshold).astype(np.int64)

    pred_df = pd.DataFrame(
        {
            'QC_RISK_SCORE': scores,
            'QC_RISK_LABEL': labels,
        },
        index=df.index
    )
    pred_df.index.name = 'DEPT'

    # Save to disk BEFORE any label loading occurs
    if save_path is not None:
        dest = Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(dest)

    return pred_df
