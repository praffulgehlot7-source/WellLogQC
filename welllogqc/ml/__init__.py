"""
welllogqc.ml
============
Machine learning pipeline for well-log quality control.
"""

from welllogqc.ml.problem_definition import (
    BINARY_MAP,
    LABEL_MAP,
    QC_RISK_CLASSES,
    SEVERITY_TO_LABEL,
    flags_to_labels,
)
from welllogqc.ml.label_loader import (
    check_label_quality,
    labels_to_series,
    load_labels,
)
from welllogqc.ml.preprocessor import WellLogPreprocessor
from welllogqc.ml.train import (
    TrainingResult,
    train_qc_model,
)
from welllogqc.ml.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_pr_curve,
)
from welllogqc.ml.inference import predict_well
from welllogqc.ml.explainability import (
    get_evidence_for_interval,
    get_feature_importance,
)

__all__ = [
    'BINARY_MAP',
    'LABEL_MAP',
    'QC_RISK_CLASSES',
    'SEVERITY_TO_LABEL',
    'flags_to_labels',
    'check_label_quality',
    'labels_to_series',
    'load_labels',
    'WellLogPreprocessor',
    'TrainingResult',
    'train_qc_model',
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_pr_curve',
    'predict_well',
    'get_evidence_for_interval',
    'get_feature_importance',
]
