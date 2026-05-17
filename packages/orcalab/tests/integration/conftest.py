"""Shared setup for OrcaLab integration tests.

Installs a lightweight Prefect stub so tests that import orchestration
tasks/flows can run without a live Prefect installation.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _make_task_decorator(**kw):
    def decorator(fn):
        fn.fn = fn
        for attr, val in kw.items():
            setattr(fn, attr, val)
        return fn

    return decorator


def _task(fn=None, **kw):
    """Support both @task and @task(...) call patterns."""
    if callable(fn):
        return _make_task_decorator()(fn)
    return _make_task_decorator(**kw)


def _flow(fn=None, name=None, **kw):
    """Support both @flow and @flow(...) call patterns."""
    def decorator(f):
        f.fn = f
        f.flow_name = name
        return f

    if callable(fn):
        return decorator(fn)
    return decorator


def _install_prefect_stub() -> None:
    if "prefect" in sys.modules:
        return
    prefect_mod = types.ModuleType("prefect")
    prefect_mod.task = _task
    prefect_mod.flow = _flow
    prefect_mod.get_run_logger = MagicMock(return_value=MagicMock())
    sys.modules["prefect"] = prefect_mod


_install_prefect_stub()
