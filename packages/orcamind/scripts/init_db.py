"""Initialize the database by running all Alembic migrations to head."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config


def main() -> None:
    ini_path = Path(__file__).parent.parent / "alembic.ini"
    alembic_cfg = Config(str(ini_path))

    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    try:
        print("Running database migrations...")
        command.upgrade(alembic_cfg, "head")
        print("Migrations complete.")
    except Exception as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
