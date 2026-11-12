"""
welllogqc.ml.train
==================
Training workflow for ML quality-control classification models.
"""

from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from welllogqc.ml.evaluate import _default_build_feature_matrix, evaluate_model
from welllogqc.ml.preprocessor import WellLogPreprocessor


@dataclass
class TrainingResult:
    """Encapsulates model training artifacts, metrics, and metadata.

    Attributes:
        model_type: Identifier of algorithm family ('logistic', 'random_forest', 'gradient_boosting').
        model: Fitted scikit-learn estimator instance.
        preprocessor: Fitted WellLogPreprocessor instance.
        feature_cols: List of feature names used during training.
        training_wells: List of well identifiers included in the training dataset.
        train_metrics: Dictionary of training performance metrics.
        val_metrics: Dictionary of validation performance metrics, or None.
        timestamp: ISO-format or formatted timestamp string of the training run.
        model_path: File system path to the saved model artifact, or None.
        preprocessor_path: File system path to the saved preprocessor artifact, or None.
    """
    model_type: str
    model: Any
    preprocessor: Any
    feature_cols: List[str]
    training_wells: List[str]
    train_metrics: Dict[str, Any]
    val_metrics: Optional[Dict[str, Any]]
    timestamp: str
    model_path: Optional[str]
    preprocessor_path: Optional[str]


def train_qc_model(
    training_data: List[Any],
    validation_data: Optional[List[Any]] = None,
    model_type: str = 'random_forest',
    feature_window: int = 11,
    class_weight: Union[str, Dict[Any, float]] = 'balanced',
    decision_threshold: float = 0.5,
    save_dir: Union[str, Path] = 'models',
    random_state: int = 42
) -> TrainingResult:
    """Train a machine-learning well-log QC classification model.

    Executes the full end-to-end training pipeline:
    1. Extracts engineering and rolling features from each training well via
       build_feature_matrix (or default feature extraction).
    2. Aligns features with reviewed labels and excludes unreviewed samples (-1).
    3. Fits WellLogPreprocessor on combined training data only (preventing leakage).
    4. Fits classifier ('logistic', 'random_forest', or 'gradient_boosting').
    5. Calculates training metrics.
    6. If validation_data is provided, evaluates validation metrics on unseen wells
       using the existing preprocessor (no refit).
    7. Serializes model and preprocessor artifacts with timestamp.
    8. Returns TrainingResult dataclass.

    Args:
        training_data: List of (IngestionResult_or_df, pd.Series_of_labels) tuples.
        validation_data: Optional list of (IngestionResult_or_df, pd.Series_of_labels).
        model_type: Algorithm name: 'logistic', 'random_forest', or 'gradient_boosting'.
        feature_window: Moving window size (samples) for rolling feature calculations.
        class_weight: Class balancing strategy for logistic/random_forest ('balanced' or dict).
        decision_threshold: Decision boundary probability for positive classification (default 0.5).
        save_dir: Directory where model and preprocessor artifacts are saved.
        random_state: Seed for reproducible pseudorandom number generation.

    Returns:
        TrainingResult: Comprehensive dataclass containing fitted artifacts and metrics.

    Raises:
        ValueError: If training_data is empty, model_type is unsupported, or no valid
            labeled samples are found.
    """
    if not training_data:
        raise ValueError("training_data cannot be empty.")

    valid_models = ('logistic', 'random_forest', 'gradient_boosting', 'xgboost', 'xgboost_gpu')
    if model_type not in valid_models:
        raise ValueError(f"Unsupported model_type '{model_type}'. Must be one of {valid_models}.")

    # 1. Build raw feature matrices for training wells
    raw_matrices: List[pd.DataFrame] = []
    training_wells: List[str] = []

    for idx, item in enumerate(training_data):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                f"Each training_data item must be a (data, labels) tuple/list. Got item at index {idx}."
            )
        data_item, _ = item
        well_name = getattr(data_item, 'well_name', f"well_{idx}")
        training_wells.append(well_name)
        feat_df = _default_build_feature_matrix(data_item, feature_window=feature_window)
        raw_matrices.append(feat_df)

    # Determine unified feature columns across all training wells
    ordered_cols: List[str] = []
    for m in raw_matrices:
        for c in m.columns:
            if c not in ordered_cols:
                ordered_cols.append(c)

    # 2. Reindex and collect valid labeled samples
    X_parts: List[pd.DataFrame] = []
    y_parts: List[np.ndarray] = []

    for idx, (_, labels_series) in enumerate(training_data):
        feat_df = raw_matrices[idx].reindex(columns=ordered_cols)
        labels = labels_series.reindex(feat_df.index)

        valid_mask = (labels != -1) & (~labels.isna())
        if not valid_mask.any():
            continue

        X_valid = feat_df.loc[valid_mask]
        # Binary target: 0 = Acceptable, 1 = Review Needed (Questionable or Unreliable)
        y_valid = (labels.loc[valid_mask].to_numpy(dtype=np.int64) > 0).astype(int)

        X_parts.append(X_valid)
        y_parts.append(y_valid)

    if not X_parts:
        raise ValueError("No valid training samples found after excluding label == -1.")

    X_train_df = pd.concat(X_parts, axis=0, ignore_index=True)
    y_train = np.concatenate(y_parts, axis=0)

    # 3. Fit preprocessor on combined training features ONLY
    preprocessor = WellLogPreprocessor(feature_cols=ordered_cols)
    X_train_trans = preprocessor.fit_transform(X_train_df)

    # 4. Instantiate and train model
    if model_type == 'logistic':
        model = LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weight,
            random_state=random_state
        )
    elif model_type in ('xgboost', 'xgboost_gpu'):
        try:
            import xgboost as xgb
            from welllogqc.service.gpu_detector import detect_gpu
            gpu = detect_gpu()
            device_choice = 'cuda' if (gpu.get('gpu_available') and model_type == 'xgboost_gpu') else 'cpu'
            scale_pos = 1.0
            if class_weight == 'balanced':
                num_neg = np.sum(y_train == 0)
                num_pos = np.sum(y_train == 1)
                scale_pos = float(num_neg / max(num_pos, 1))

            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos,
                device=device_choice,
                random_state=random_state,
                eval_metric='logloss'
            )
        except Exception:
            model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=random_state
            )
    else:  # gradient_boosting
        model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state
        )

    # Ensure binary classes are present for models requiring >= 2 classes (e.g. XGBoost)
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        missing_class = 1 if 0 in unique_classes else 0
        pseudo_row = np.zeros((1, X_train_trans.shape[1]))
        X_train_fit = np.vstack([X_train_trans, pseudo_row])
        y_train_fit = np.append(y_train, missing_class)
    else:
        X_train_fit = X_train_trans
        y_train_fit = y_train

    model.fit(X_train_fit, y_train_fit)

    # 5. Training metrics
    train_metrics = evaluate_model(
        model=model,
        preprocessor=preprocessor,
        test_data=training_data,
        decision_threshold=decision_threshold,
        binary_mode=True
    )

    # 6. Validation metrics if provided
    val_metrics = None
    if validation_data:
        val_metrics = evaluate_model(
            model=model,
            preprocessor=preprocessor,
            test_data=validation_data,
            decision_threshold=decision_threshold,
            binary_mode=True
        )

    # 7. Save model and preprocessor
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path_dir = Path(save_dir)
    save_path_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(save_path_dir / f"qc_model_{model_type}_{timestamp}.joblib")
    joblib.dump(model, model_path)

    preprocessor_path = str(save_path_dir / f"preprocessor_{model_type}_{timestamp}.joblib")
    preprocessor.save(preprocessor_path)

    # 8. Return result dataclass
    return TrainingResult(
        model_type=model_type,
        model=model,
        preprocessor=preprocessor,
        feature_cols=ordered_cols,
        training_wells=training_wells,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        timestamp=timestamp,
        model_path=model_path,
        preprocessor_path=preprocessor_path
    )
