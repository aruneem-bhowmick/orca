#!/usr/bin/env python3
"""Bootstrap the OrcaMind meta-dataset from OpenML benchmarks.

Downloads CC-18 (classification) and CTR-23 (regression) benchmark suites,
runs 5 baseline models with 5-fold cross-validation on each task, stores
Task/Experiment/Performance records in the OrcaMind registry, generates
25-dim statistical embeddings for every task, and builds a FAISS similarity
index saved to disk.

Usage:
    python scripts/bootstrap_meta_dataset.py [OPTIONS]

Options:
    --max-tasks INT     Max tasks per suite to ingest (default: all)
    --output-dir PATH   Directory for the FAISS index and local cache (default: data/)
    --db-url TEXT       Async PostgreSQL connection URL (default: local dev DB)
    --suites TEXT...    Benchmark suites to download: cc18, ctr23 (default: both)
    --dry-run           Simulate without writing to the registry

Dependencies:
    openml>=0.14.0, scikit-learn>=1.3.0, xgboost>=2.0.0, pandas>=2.0.0,
    orca-shared, orcamind (must be installed in the workspace)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from xgboost import XGBClassifier, XGBRegressor

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suite → OpenML study name mapping
# ---------------------------------------------------------------------------

SUITE_NAMES: dict[str, str] = {
    "cc18": "OpenML-CC18",
    "ctr23": "OpenML-CTR23",
}

# ---------------------------------------------------------------------------
# Baseline model factories (zero-arg lambdas so each CV run gets a fresh instance)
# ---------------------------------------------------------------------------

CLASSIFICATION_MODELS: dict[str, Any] = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=0),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=0),
    "xgboost": lambda: XGBClassifier(
        n_estimators=100, random_state=0, verbosity=0, eval_metric="logloss"
    ),
    "svc_rbf": lambda: SVC(kernel="rbf"),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
}

REGRESSION_MODELS: dict[str, Any] = {
    "ridge": lambda: Ridge(),
    "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=0),
    "xgboost": lambda: XGBRegressor(n_estimators=100, random_state=0, verbosity=0),
    "svr_rbf": lambda: SVR(kernel="rbf"),
    "knn": lambda: KNeighborsRegressor(n_neighbors=5),
}

# SVC/SVR keys that are skipped when the dataset is large
_LARGE_SAMPLE_SKIP = {"svc_rbf", "svr_rbf"}
_LARGE_SAMPLE_THRESHOLD = 10_000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the OrcaMind registry with OpenML benchmark tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Max tasks per suite to ingest (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/",
        help="Output directory for the FAISS index (default: data/)",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default="postgresql+asyncpg://orca:orca_dev_secret@localhost:5432/orca_registry",
        help="Async PostgreSQL connection URL",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["cc18", "ctr23"],
        choices=list(SUITE_NAMES.keys()),
        help="Benchmark suites to download (default: cc18 ctr23)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate without writing to the registry",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# OpenML data acquisition
# ---------------------------------------------------------------------------


def download_suite(suite_name: str, max_tasks: int | None = None) -> list[Any]:
    """Download all tasks from an OpenML benchmark suite.

    Failures on individual task downloads are logged and skipped so one
    unreachable task cannot abort the whole run.
    """
    import openml  # deferred: not needed when the script is imported for testing

    suite_key = SUITE_NAMES[suite_name]
    suite = openml.study.get_suite(suite_key)
    task_ids: list[int] = list(suite.tasks or [])

    tasks: list[Any] = []
    for tid in task_ids:
        try:
            tasks.append(openml.tasks.get_task(tid))
        except Exception as exc:
            log.warning("Skipping OpenML task %s: %s", tid, exc)

    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks


def fetch_dataset(task: Any) -> tuple[pd.DataFrame, pd.Series, str]:
    """Extract (X, y, task_type) from an OpenML task object."""
    X, y = task.get_X_and_y(dataset_format="dataframe")

    # openml.tasks.TaskType.SUPERVISED_CLASSIFICATION == 1
    task_type_id = getattr(task, "task_type_id", None)
    if task_type_id is not None:
        try:
            import openml.tasks

            is_classification = task_type_id == openml.tasks.TaskType.SUPERVISED_CLASSIFICATION
        except Exception:
            is_classification = True
    else:
        # fall back to inspecting the class name
        is_classification = "Classification" in type(task).__name__

    task_type = "classification" if is_classification else "regression"
    return X, y, task_type


def run_baseline_models(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    n_samples: int,
) -> dict[str, dict[str, float]]:
    """Run 5 baseline models with 5-fold cross-validation.

    Returns a mapping from model name to {"mean": float, "std": float}.
    Models that fail (e.g. convergence issues) get NaN scores.
    """
    # --- preprocessing: numeric columns only + median imputation ---
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    X_num = X[numeric_cols] if numeric_cols else pd.DataFrame(index=X.index)

    # Drop rows where y is NaN
    valid_mask = y.notna()
    X_clean = X_num.loc[valid_mask].reset_index(drop=True)
    y_clean = y.loc[valid_mask].reset_index(drop=True)

    if len(y_clean) < 5:
        models_dict = (
            CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
        )
        return {name: {"mean": float("nan"), "std": float("nan")} for name in models_dict}

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_clean) if X_clean.shape[1] > 0 else X_clean.values

    models_dict = (
        CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    )
    scoring = "accuracy" if task_type == "classification" else "r2"

    results: dict[str, dict[str, float]] = {}
    for model_name, factory in models_dict.items():
        if model_name in _LARGE_SAMPLE_SKIP and n_samples > _LARGE_SAMPLE_THRESHOLD:
            results[model_name] = {"mean": float("nan"), "std": float("nan")}
            continue
        try:
            scores = cross_val_score(
                factory(), X_imputed, y_clean.values, cv=5, scoring=scoring
            )
            results[model_name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
        except Exception as exc:
            log.warning("Model %s failed: %s", model_name, exc)
            results[model_name] = {"mean": float("nan"), "std": float("nan")}

    return results


# ---------------------------------------------------------------------------
# Stub async entry point (filled in later commits)
# ---------------------------------------------------------------------------


async def _bootstrap_async(args: argparse.Namespace) -> dict[str, int]:
    return {"tasks": 0, "experiments": 0}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    result = asyncio.run(_bootstrap_async(args))
    print(f"Done. Tasks: {result['tasks']}, Experiments: {result['experiments']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
