"""Statistical task embedder: converts a tabular dataset into a 25-dim meta-feature vector."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy as shannon_entropy
from sklearn.feature_selection import f_classif, mutual_info_classif

from .base import TaskEmbedder

_DIM = 25


def _safe(value: float) -> float:
    """Return 0.0 for any non-finite float."""
    if not np.isfinite(value):
        return 0.0
    return float(value)


class StatisticalEmbedder(TaskEmbedder):
    """
    Extracts a fixed 25-dimensional meta-feature vector from a tabular dataset.

    Feature index layout
    --------------------
    Dataset-level (0–5)
        0  log_n_samples
        1  log_n_features
        2  n_classes
        3  class_balance_entropy
        4  missing_ratio
        5  categorical_ratio

    Per-feature statistics aggregated over columns (6–14)
        6  feature_mean_mean
        7  feature_mean_std
        8  feature_std_mean
        9  feature_std_std
       10  feature_skewness_mean
       11  feature_skewness_std
       12  feature_kurtosis_mean
       13  feature_kurtosis_std
       14  feature_missing_mean

    Correlation structure (15–19)
       15  corr_mean_abs
       16  corr_std
       17  corr_max_abs
       18  high_corr_ratio
       19  near_zero_var_ratio

    Class-feature relationships – classification only (20–24)
       20  anova_f_mean
       21  anova_f_std
       22  mutual_info_mean
       23  mutual_info_std
       24  class_separability
    """

    @property
    def embedding_dim(self) -> int:
        return _DIM

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed(self, dataset: pd.DataFrame, labels: pd.Series | None = None) -> np.ndarray:
        vec = np.zeros(_DIM, dtype=np.float64)

        n_rows, n_cols = dataset.shape
        if n_rows == 0 or n_cols == 0:
            return vec
        if labels is not None and len(labels) != n_rows:
            raise ValueError(
                f"labels length ({len(labels)}) must match dataset rows ({n_rows})"
            )

        # ---- 0: log_n_samples ----
        vec[0] = _safe(np.log10(max(n_rows, 1)))

        # ---- 1: log_n_features ----
        vec[1] = _safe(np.log10(max(n_cols, 1)))

        # ---- 2–3: class stats ----
        is_classification = labels is not None and self._is_classification(labels)
        if is_classification and labels is not None:
            classes, counts = np.unique(labels.dropna(), return_counts=True)
            vec[2] = float(len(classes))
            probs = counts / counts.sum()
            vec[3] = _safe(float(shannon_entropy(probs, base=2)))
        else:
            vec[2] = 0.0
            vec[3] = 0.0

        # ---- 4: missing_ratio ----
        vec[4] = _safe(dataset.isna().values.mean())

        # ---- 5: categorical_ratio ----
        _NULLABLE_CATEGORICAL_NAMES = {
            "string", "boolean",
            "Int8", "Int16", "Int32", "Int64",
            "UInt8", "UInt16", "UInt32", "UInt64",
        }
        cat_cols = [
            c for c in dataset.columns
            if (
                isinstance(dataset[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(dataset[c])
                or pd.api.types.is_bool_dtype(dataset[c])
                or (
                    pd.api.types.is_extension_array_dtype(dataset[c])
                    and dataset[c].dtype.name in _NULLABLE_CATEGORICAL_NAMES
                )
            )
        ]
        vec[5] = _safe(len(cat_cols) / max(n_cols, 1))

        # ---- 6–14: per-feature statistics ----
        numeric_df = dataset.select_dtypes(include=[np.number])
        if numeric_df.empty:
            # indices 6–14 remain 0
            pass
        else:
            col_means = numeric_df.mean(skipna=True).values
            col_stds = numeric_df.std(skipna=True, ddof=1).fillna(0).values
            col_skews = numeric_df.apply(lambda s: s.dropna().skew() if s.dropna().shape[0] > 2 else 0.0).values
            col_kurts = numeric_df.apply(lambda s: s.dropna().kurt() if s.dropna().shape[0] > 3 else 0.0).values
            col_missing = numeric_df.isna().mean().values

            # Replace any inf/nan introduced by very small samples
            col_skews = np.nan_to_num(col_skews.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
            col_kurts = np.nan_to_num(col_kurts.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
            col_means_f = np.nan_to_num(col_means.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
            col_stds_f = np.nan_to_num(col_stds.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

            vec[6] = _safe(float(np.mean(col_means_f)))
            vec[7] = _safe(float(np.std(col_means_f, ddof=0)))
            vec[8] = _safe(float(np.mean(col_stds_f)))
            vec[9] = _safe(float(np.std(col_stds_f, ddof=0)))
            vec[10] = _safe(float(np.mean(col_skews)))
            vec[11] = _safe(float(np.std(col_skews, ddof=0)))
            vec[12] = _safe(float(np.mean(col_kurts)))
            vec[13] = _safe(float(np.std(col_kurts, ddof=0)))
            vec[14] = _safe(float(np.mean(col_missing)))

        # ---- 15–19: correlation structure ----
        numeric_df2 = dataset.select_dtypes(include=[np.number])
        if numeric_df2.shape[1] >= 2:
            corr_matrix = numeric_df2.corr().values
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper = corr_matrix[mask]
            finite_upper = upper[np.isfinite(upper)]

            if finite_upper.size > 0:
                abs_upper = np.abs(finite_upper)
                vec[15] = _safe(float(np.mean(abs_upper)))
                vec[16] = _safe(float(np.std(finite_upper, ddof=0)))
                vec[17] = _safe(float(np.max(abs_upper)))
                vec[18] = _safe(float(np.mean(abs_upper > 0.9)))
            # else: remain 0

        col_stds_all = numeric_df2.std(skipna=True, ddof=1).fillna(0).values if not numeric_df2.empty else np.array([])
        if col_stds_all.size > 0:
            vec[19] = _safe(float(np.mean(col_stds_all < 1e-5)))

        # ---- 20–24: class-feature relationships (classification only) ----
        if is_classification and labels is not None:
            numeric_X = dataset.select_dtypes(include=[np.number])
            if numeric_X.shape[1] > 0:
                aligned_labels = labels.reindex(dataset.index)
                valid_mask = aligned_labels.notna()
                X = numeric_X.loc[valid_mask].values
                y = aligned_labels.loc[valid_mask].values
                if X.shape[0] >= 5:
                    # Impute NaN columns with column mean so sklearn doesn't choke
                    col_means_imp = np.nanmean(X, axis=0)
                    nan_cols = np.isnan(col_means_imp)
                    col_means_imp[nan_cols] = 0.0
                    nan_mask = np.isnan(X)
                    X = np.where(nan_mask, np.broadcast_to(col_means_imp, X.shape), X)

                    classes_unique = np.unique(y)
                    if len(classes_unique) >= 2:
                        try:
                            f_vals, _ = f_classif(X, y)
                            f_vals = np.nan_to_num(f_vals.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
                            vec[20] = _safe(float(np.mean(f_vals)))
                            vec[21] = _safe(float(np.std(f_vals, ddof=0)))
                        except Exception:
                            pass

                        try:
                            mi_vals = mutual_info_classif(X, y, random_state=0)
                            mi_vals = np.nan_to_num(mi_vals.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
                            vec[22] = _safe(float(np.mean(mi_vals)))
                            vec[23] = _safe(float(np.std(mi_vals, ddof=0)))
                        except Exception:
                            pass

                        # class_separability: between-class var / within-class var ratio
                        try:
                            grand_mean = np.mean(X, axis=0)
                            between_var = sum(
                                np.sum((np.mean(X[y == c], axis=0) - grand_mean) ** 2) * np.sum(y == c)
                                for c in classes_unique
                            ) / max(X.shape[0], 1)
                            within_var = sum(
                                np.sum((X[y == c] - np.mean(X[y == c], axis=0)) ** 2)
                                for c in classes_unique
                            ) / max(X.shape[0], 1)
                            if within_var > 1e-10:
                                vec[24] = _safe(float(between_var / within_var))
                        except Exception:
                            pass

        return vec

    def embed_batch(self, datasets: list[tuple[pd.DataFrame, pd.Series | None]]) -> np.ndarray:
        results = [self.embed(df, lbl) for df, lbl in datasets]
        if not results:
            return np.empty((0, _DIM), dtype=np.float64)
        return np.stack(results, axis=0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_classification(labels: pd.Series) -> bool:
        """Treat integer, boolean, object, or string label columns as classification tasks."""
        dtype = labels.dtype
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            return True
        if str(dtype).startswith("int") or str(dtype).startswith("uint"):
            return True
        if pd.api.types.is_extension_array_dtype(dtype) and dtype.name in {
            "string", "boolean",
            "Int8", "Int16", "Int32", "Int64",
            "UInt8", "UInt16", "UInt32", "UInt64",
        }:
            return True
        return False
