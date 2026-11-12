"""
welllogqc.features
==================
Feature engineering and extraction for well-log machine learning models.
"""

from typing import Any
import numpy as np
import pandas as pd


def build_feature_matrix(
    data: Any,
    feature_window: int = 11
) -> pd.DataFrame:
    """Extract rolling statistics and engineering features from well-log curves.

    Computes raw curve values, centered rolling means, rolling standard deviations,
    and vertical gradients (first differences) across all numeric curves. Also computes
    differential caliper (CALI - BS) if bit size and caliper are both present.

    Args:
        data: IngestionResult instance (with .df attribute) or a pandas DataFrame
            indexed by DEPT.
        feature_window: Integer odd window length in samples for rolling statistics.
            Defaults to 11.

    Returns:
        pd.DataFrame: Engineered feature DataFrame indexed by DEPT with all columns
            as float64.

    Raises:
        ValueError: If data cannot be resolved to a pandas DataFrame or is empty.
    """
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

    features.index.name = 'DEPT'
    return features.astype(np.float64)


__all__ = ['build_feature_matrix']
