"""Create Prefect infrastructure needed for OrcaLab sweeps."""

from __future__ import annotations

import subprocess


def create_orcalab_pool() -> None:
    subprocess.run(
        ["prefect", "work-pool", "create", "orcalab-pool", "--type", "process"],
        check=True,
    )


if __name__ == "__main__":
    create_orcalab_pool()
