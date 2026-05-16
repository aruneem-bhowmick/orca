"""Prefect task: download and return a dataset from storage."""

from __future__ import annotations

import io

import pandas as pd
from prefect import task

from orca_shared.storage.base import StorageBackend


@task(name="prepare_data", retries=2, retry_delay_seconds=30)
async def prepare_data(task_id: str, storage: StorageBackend) -> pd.DataFrame:
    key = f"datasets/{task_id}/data.parquet"
    raw = await storage.download(key)
    return pd.read_parquet(io.BytesIO(raw))
