"""Stub the prefect package so orchestration tests run without a live Prefect install."""

from __future__ import annotations

import sys
import types
from functools import wraps
from unittest.mock import MagicMock


def _make_task_decorator(**kw):
    """Return a @task-like decorator that adds Prefect attributes to the function."""

    def decorator(fn):
        fn.fn = fn
        for attr, val in kw.items():
            setattr(fn, attr, val)
        return fn

    return decorator


def _task(**kw):
    return _make_task_decorator(**kw)


def _flow(name=None, **kw):
    """Return a @flow-like decorator that exposes .fn on the decorated function."""

    def decorator(fn):
        fn.fn = fn
        fn.flow_name = name
        return fn

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
