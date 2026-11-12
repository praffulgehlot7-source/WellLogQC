"""
welllogqc.ml.preprocessor
=========================
Preprocessor pipeline wrapping SimpleImputer and StandardScaler for well-log ML features.
"""

from pathlib import Path
from typing import List, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class WellLogPreprocessor:
    """Wraps sklearn SimpleImputer and StandardScaler for well-log feature normalization.

    MUST be fit on training data only to avoid data leakage across wells or held-out sets.
    Imputes missing values with column medians, then standardizes features to zero mean
    and unit variance.

    Attributes:
        feature_cols: List of column names expected by this preprocessor.
        pipeline: Underlying scikit-learn Pipeline instance.
    """

    def __init__(self, feature_cols: List[str]) -> None:
        """Initialize the preprocessor with expected feature column names.

        Args:
            feature_cols: List of feature column names. Must not be empty.

        Raises:
            ValueError: If feature_cols is empty.
        """
        if not feature_cols:
            raise ValueError("feature_cols must be a non-empty list of column names.")

        self._feature_cols: List[str] = list(feature_cols)
        self.pipeline: Pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        self._is_fitted: bool = False

    @property
    def feature_cols(self) -> List[str]:
        """Return the list of feature column names."""
        return list(self._feature_cols)

    def fit(self, X: pd.DataFrame) -> 'WellLogPreprocessor':
        """Fit the SimpleImputer and StandardScaler on feature DataFrame X.

        Args:
            X: Training DataFrame containing all columns in feature_cols.

        Returns:
            WellLogPreprocessor: Fitted preprocessor instance.

        Raises:
            ValueError: If X is empty, missing required columns, or not a DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        missing = [col for col in self._feature_cols if col not in X.columns]
        if missing:
            raise ValueError(f"X is missing required feature columns: {missing}")

        if X.empty:
            raise ValueError("Cannot fit preprocessor on an empty DataFrame.")

        X_mat = X[self._feature_cols].to_numpy(dtype=np.float64)
        self.pipeline.fit(X_mat)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Impute and scale features in DataFrame X using the fitted pipeline.

        Args:
            X: DataFrame containing all columns in feature_cols.

        Returns:
            np.ndarray: 2D array of transformed features with dtype float64.

        Raises:
            ValueError: If preprocessor is not fitted, X is not a DataFrame, or
                required columns are missing.
        """
        if not self._is_fitted:
            raise ValueError("WellLogPreprocessor has not been fitted. Call fit() first.")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        missing = [col for col in self._feature_cols if col not in X.columns]
        if missing:
            raise ValueError(f"X is missing required feature columns: {missing}")

        if X.empty:
            return np.empty((0, len(self._feature_cols)), dtype=np.float64)

        X_mat = X[self._feature_cols].to_numpy(dtype=np.float64)
        out = self.pipeline.transform(X_mat)
        return np.asarray(out, dtype=np.float64)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit pipeline on X and return transformed features.

        Args:
            X: Training DataFrame containing all columns in feature_cols.

        Returns:
            np.ndarray: 2D array of transformed features with dtype float64.

        Raises:
            ValueError: If X is empty or missing required columns.
        """
        return self.fit(X).transform(X)

    def save(self, path: Union[str, Path]) -> None:
        """Serialize fitted preprocessor to disk via joblib.

        Args:
            path: Destination file path.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(dest))

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'WellLogPreprocessor':
        """Load a serialized WellLogPreprocessor instance from disk.

        Args:
            path: Path to serialized joblib file.

        Returns:
            WellLogPreprocessor: Restored preprocessor instance.

        Raises:
            ValueError: If file does not exist.
            TypeError: If deserialized object is not a WellLogPreprocessor.
        """
        src = Path(path)
        if not src.is_file():
            raise ValueError(f"Preprocessor file not found at: {src}")

        obj = joblib.load(str(src))
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object from {src} is of type {type(obj)}, expected {cls.__name__}."
            )
        return obj
