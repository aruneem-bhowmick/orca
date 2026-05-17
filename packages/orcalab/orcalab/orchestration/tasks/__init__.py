"""Prefect task definitions for OrcaLab."""

from orcalab.orchestration.tasks.evaluate import evaluate
from orcalab.orchestration.tasks.get_priors import get_orcamind_priors
from orcalab.orchestration.tasks.log_results import log_results
from orcalab.orchestration.tasks.prepare_data import prepare_data
from orcalab.orchestration.tasks.train_model import train_model

__all__ = ["evaluate", "get_orcamind_priors", "log_results", "prepare_data", "train_model"]
