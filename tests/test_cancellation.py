"""
Regression tests for Launch Checklist 1.7 — the shared 6-thread executor.

Three things are pinned here:
1. ingestion/llm_client's per-thread cancel flag (set_cancel_event /
   _raise_if_cancelled) — the primitive that lets a timed-out request stop
   making further LLM calls instead of running to completion unbilled.
2. api/concurrency.run_on_executor: busy -> 503 instead of queuing into a
   timeout, and capacity only released once the worker thread actually
   finishes (not when the async wait_for gives up).
3. api/concurrency.track_task: a bare asyncio.create_task() with nothing
   awaiting it must not be garbage-collected mid-flight.
"""

import asyncio
import threading
import time

import pytest
from fastapi import HTTPException

from api import concurrency
from ingestion.llm_client import OperationCancelled, _raise_if_cancelled, set_cancel_event


# ── ingestion/llm_client cancel primitive ────────────────────────────────────

def test_no_raise_when_no_event_set():
    set_cancel_event(None)
    _raise_if_cancelled()  # must not raise


def test_no_raise_when_event_not_set():
    set_cancel_event(threading.Event())
    _raise_if_cancelled()  # not .set() yet — must not raise
    set_cancel_event(None)


def test_raises_once_event_is_set():
    ev = threading.Event()
    set_cancel_event(ev)
    ev.set()
    with pytest.raises(OperationCancelled):
        _raise_if_cancelled()
    set_cancel_event(None)


def test_cancel_flag_is_thread_local():
    """A cancel Event set on one thread must not affect another."""
    seen_other_thread = {}

    def other_thread():
        # This thread never called set_cancel_event — must see no event.
        try:
            _raise_if_cancelled()
            seen_other_thread["raised"] = False
        except OperationCancelled:
            seen_other_thread["raised"] = True

    ev = threading.Event()
    ev.set()
    set_cancel_event(ev)
    try:
        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
        assert seen_other_thread["raised"] is False
    finally:
        set_cancel_event(None)


# ── api/concurrency.run_on_executor ──────────────────────────────────────────

def test_run_on_executor_returns_the_function_result():
    async def _run():
        return await concurrency.run_on_executor(lambda x, y: x + y, 5.0, 2, 3)

    assert asyncio.run(_run()) == 5


def test_run_on_executor_503s_when_every_worker_is_busy(monkeypatch):
    monkeypatch.setattr(concurrency, "_capacity", threading.BoundedSemaphore(1))
    concurrency._capacity.acquire()  # simulate the pool already fully occupied

    async def _run():
        return await concurrency.run_on_executor(lambda: "unreachable", 5.0)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_run())
    assert exc_info.value.status_code == 503


def test_run_on_executor_releases_capacity_only_after_completion(monkeypatch):
    monkeypatch.setattr(concurrency, "_capacity", threading.BoundedSemaphore(1))

    release_order = []

    def slow():
        time.sleep(0.15)
        release_order.append("worker_done")
        return "ok"

    async def _run():
        return await concurrency.run_on_executor(slow, 5.0)

    assert asyncio.run(_run()) == "ok"
    # Capacity must be free again now that the worker has actually finished —
    # if it were released early (e.g. as soon as wait_for returns) this would
    # still pass, but the real regression this guards is the timeout path
    # below, where releasing early would let two jobs occupy one "slot".
    assert concurrency._capacity.acquire(blocking=False)
    concurrency._capacity.release()
    assert release_order == ["worker_done"]


def test_run_on_executor_timeout_keeps_capacity_held_until_worker_finishes(monkeypatch):
    """The bug this guards: releasing capacity when wait_for's timeout fires
    (rather than when the worker thread actually completes) lets a still-running
    cancelled job and a newly admitted one both occupy real threads at once —
    silently doubling the effective pool size under sustained timeouts.

    Uses a persistent event loop (not asyncio.run) because the completion
    signal from the background thread arrives via call_soon_threadsafe well
    after the timed-out coroutine returns — asyncio.run would have already
    closed the loop by then, in this test only (a real server's loop never
    closes mid-request)."""
    monkeypatch.setattr(concurrency, "_capacity", threading.BoundedSemaphore(1))

    finished = threading.Event()

    def slow():
        time.sleep(0.2)
        finished.set()
        return "ok"

    loop = asyncio.new_event_loop()
    try:
        async def _run():
            await concurrency.run_on_executor(slow, 0.05)

        with pytest.raises(asyncio.TimeoutError):
            loop.run_until_complete(_run())

        # Immediately after the timeout, the worker is still running —
        # capacity must still be held, so a new request is correctly
        # refused with 503.
        assert not concurrency._capacity.acquire(blocking=False)

        # Pump the loop so the executor's completion callback (delivered via
        # call_soon_threadsafe from the worker thread) actually runs.
        loop.run_until_complete(asyncio.sleep(0.3))
        assert finished.is_set()
        assert concurrency._capacity.acquire(blocking=False), (
            "capacity was never released after the worker actually finished"
        )
        concurrency._capacity.release()
    finally:
        loop.close()


# ── api/concurrency.track_task ───────────────────────────────────────────────

def test_track_task_holds_a_strong_reference_until_done():
    async def _run():
        done = asyncio.Event()

        async def work():
            await asyncio.sleep(0.05)
            done.set()

        task = concurrency.track_task(work())
        assert task in concurrency._background_tasks
        await done.wait()
        await asyncio.sleep(0.01)  # let the done_callback discard it
        assert task not in concurrency._background_tasks

    asyncio.run(_run())
