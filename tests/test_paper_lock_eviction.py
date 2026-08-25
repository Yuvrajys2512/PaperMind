"""
Regression tests for Launch Checklist 2.11 — `_paper_locks` grows without
bound.

api.concurrency is safe to import directly (no live services at import
time — same reasoning as tests/test_concurrency.py).
"""

import threading

from api import concurrency
from api.concurrency import acquire_paper_lock, paper_locked, release_paper_lock


def test_release_paper_lock_removes_the_entry():
    paper_id = "evict-me"
    acquire_paper_lock(paper_id)
    assert paper_id in concurrency._paper_locks

    release_paper_lock(paper_id)
    assert paper_id not in concurrency._paper_locks


def test_release_paper_lock_is_a_noop_for_an_unknown_paper():
    release_paper_lock("never-locked")  # must not raise


def test_a_fresh_lock_is_created_after_eviction():
    paper_id = "recreate-me"
    lock1 = acquire_paper_lock(paper_id)
    release_paper_lock(paper_id)
    lock2 = acquire_paper_lock(paper_id)
    assert lock1 is not lock2


def test_release_after_the_lock_block_does_not_corrupt_a_waiter():
    """A thread that acquired the lock object BEFORE eviction must still be
    able to use it correctly afterward — eviction only removes the dict
    entry, it must never touch a lock another thread is relying on."""
    paper_id = "waiter-test"
    results = []

    def holder():
        with paper_locked(paper_id):
            results.append("holder-acquired")
            threading.Event().wait(0.05)
        # Evict right after releasing, exactly as the real delete flow does.
        release_paper_lock(paper_id)

    def late_comer():
        threading.Event().wait(0.02)
        with paper_locked(paper_id):
            results.append("late-comer-acquired")

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=late_comer)
    t1.start()
    t2.start()
    t1.join(timeout=2)
    t2.join(timeout=2)

    assert results == ["holder-acquired", "late-comer-acquired"], results
