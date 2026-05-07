#!/usr/bin/env python3
"""Bootstrap script: download OpenML benchmark tasks and seed the Orca meta-registry.

This is the entry point for populating the Orca ecosystem with an initial set of
meta-learning tasks from the OpenML CC18 benchmark suite (suite_id=271).

Usage:
    python scripts/bootstrap_meta_dataset.py [OPTIONS]

Options:
    --suite-id INT          OpenML suite ID (default: 271, CC18)
    --max-tasks INT         Maximum number of tasks to ingest (default: all)
    --data-dir PATH         Local cache directory for downloaded datasets
    --registry-url TEXT     PostgreSQL connection string for the Orca registry
    --dry-run               Print tasks that would be ingested without writing

Planned implementation (Prompt N — data bootstrap):
    1. Connect to the OpenML API and fetch all tasks in the specified suite.
    2. For each task:
        a. Download the dataset via openml.datasets.get_dataset()
        b. Filter by min/max samples and features (from dataset/openml.yaml)
        c. Run StatisticalEmbedder to produce a 25-dim task embedding
        d. Write task metadata + embedding to the Orca registry (PostgreSQL)
        e. Upload raw dataset parquet to MinIO (s3://orca-datasets/{task_id}/)
        f. Log the ingest event to MLflow
    3. Print a summary table: tasks ingested, skipped, and failed.

Dependencies (not yet installed — will be added in Prompt N):
    openml>=0.14.0
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the Orca meta-registry with OpenML benchmark tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--suite-id", type=int, default=271, help="OpenML suite ID (default: 271, CC18)"
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None, help="Max tasks to ingest (default: all)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data/openml", help="Local dataset cache directory"
    )
    parser.add_argument(
        "--registry-url",
        type=str,
        default="postgresql://orca:orca_dev_secret@localhost:5432/orca_registry",
        help="Registry PostgreSQL connection URL",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List tasks without writing to registry"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("Orca Bootstrap: Meta-Dataset Ingestion (STUB)")
    print("=" * 60)
    print(f"  Suite ID   : {args.suite_id}")
    print(f"  Max tasks  : {args.max_tasks or 'all'}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Registry   : {args.registry_url}")
    print(f"  Dry run    : {args.dry_run}")
    print()
    print("NOTE: This script is a stub. Full implementation arrives in a later prompt.")
    print("      See module docstring for the planned algorithm.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
