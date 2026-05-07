import pytest
from pathlib import Path


@pytest.fixture
def storage_base_path(tmp_path: Path) -> Path:
    """Temporary directory for local storage backend tests."""
    base = tmp_path / "orca_storage"
    base.mkdir()
    return base
