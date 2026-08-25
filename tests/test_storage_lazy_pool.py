"""
Regression tests for Launch Checklist 2.12 — running the test suite connects
to PRODUCTION.

`api.storage` is intentionally imported directly here: that's the whole
point (proving import alone no longer opens a connection). This only stays
safe because of the exact fix under test — the pool is lazy, so the import
below performs no real I/O. The conftest.py network guard is redundant
insurance in this particular file (nothing here calls `.connection()`), but
this file existing at all previously would have been the failure mode itself.
"""

import threading

from api import storage
from api.storage import _LazyConnectionPool


def _bare_pool(real_pool):
    pool = _LazyConnectionPool.__new__(_LazyConnectionPool)
    pool._real = real_pool
    pool._setup_callbacks = []
    pool._ready = False
    pool._setup_lock = threading.Lock()
    return pool


def test_importing_storage_does_not_open_the_pool():
    assert storage._pool._real._opened is False, (
        "importing api.storage must not open a real Postgres connection — "
        "pytest imports every test module during collection, so an eager "
        "pool open here means collecting ANY test that imports api.storage "
        "already touches production"
    )


def test_lazy_pool_defers_open_until_first_connection_call():
    calls = []

    class FakeRealPool:
        def __init__(self):
            self.opened = False

        def open(self):
            self.opened = True
            calls.append("open")

        def connection(self):
            calls.append("connection")
            return "fake-connection-cm"

    pool = _bare_pool(FakeRealPool())

    assert calls == []  # constructing/wrapping alone does nothing
    result = pool.connection()
    assert result == "fake-connection-cm"
    assert calls == ["open", "connection"]


def test_on_first_open_callbacks_run_exactly_once():
    run_count = {"n": 0}

    class FakeRealPool:
        def open(self):
            pass

        def connection(self):
            return "conn"

    pool = _bare_pool(FakeRealPool())

    pool.on_first_open(lambda: run_count.__setitem__("n", run_count["n"] + 1))

    pool.connection()
    pool.connection()
    pool.connection()

    assert run_count["n"] == 1, "on_first_open callbacks must run exactly once, not per-connection"


def test_on_first_open_does_not_run_before_a_real_connection_is_requested():
    ran = []

    class FakeRealPool:
        def open(self):
            pass

        def connection(self):
            return "conn"

    pool = _bare_pool(FakeRealPool())

    pool.on_first_open(lambda: ran.append("schema"))
    assert ran == [], "registering a schema callback must not run it immediately"
