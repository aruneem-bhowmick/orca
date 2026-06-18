"""Create Prefect infrastructure needed for OrcaLab sweeps."""

from __future__ import annotations

import subprocess


def create_orcalab_pool() -> None:
    """Create the ``orcalab-pool`` Prefect work pool used by sweep deployments.

    Shells out to ``prefect work-pool create orcalab-pool --type process`` with
    ``check=True``, so a non-zero exit from the ``prefect`` CLI propagates as a
    :class:`subprocess.CalledProcessError`. This is intended as a one-time setup
    step per Prefect server instance, run after the server is up and before a
    worker is started.
    """
    subprocess.run(
        ["prefect", "work-pool", "create", "orcalab-pool", "--type", "process"],
        check=True,
    )


if __name__ == "__main__":
    create_orcalab_pool()
