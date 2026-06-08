# Changelog

All notable changes to the Orca project are documented in this file,
derived from the merge-request history of the repository.

## Foundation — Monorepo Scaffold and Shared Infrastructure

**PRs #1–3 · May 7–8, 2026**

### Added

- Initialised uv-managed monorepo with ruff, mypy, pytest, and pre-commit
  tooling (#1).
- Scaffolded `orcamind` package with Hydra configuration hierarchy, module
  stubs, CLI entry-point, multi-stage Dockerfile, and docker-compose
  development environment (#1).
- Scaffolded `orca-shared` package with TaskRepository, OrcaMindClient,
  OrcaLabClient, centralised config/logging, and Pydantic domain models (#2).
- Added CODEOWNERS to enforce review gates on `main` (#2).
- Implemented `StatisticalTaskEmbedder` with 25-dimensional meta-feature
  extraction for dataset profiling (#3).
