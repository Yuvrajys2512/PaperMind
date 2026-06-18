"""
Tests for concurrency module: thread safety and per-paper locking.
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.concurrency import get_chroma_client, acquire_paper_lock, paper_locked


def test_chroma_client_singleton():
    """ChromaDB client is a singleton - same instance returned each time."""
    client1 = get_chroma_client()
    client2 = get_chroma_client()
    assert client1 is client2, "get_chroma_client should return same instance"
    print("OK: ChromaDB singleton works")


def test_paper_locks_per_paper():
    """Each paper gets its own lock, different papers don't share locks."""
    lock_a = acquire_paper_lock("paper_a")
    lock_b = acquire_paper_lock("paper_b")
    lock_a2 = acquire_paper_lock("paper_a")

    assert lock_a is lock_a2, "Same paper should get same lock"
    assert lock_a is not lock_b, "Different papers should get different locks"
    print("OK: Per-paper locks work")


def test_paper_locked_context_manager():
    """paper_locked context manager acquires and releases locks correctly."""
    paper_id = "test_paper_123"
    lock = acquire_paper_lock(paper_id)

    # Lock should not be held initially
    acquired = lock.acquire(blocking=False)
    assert acquired, "Lock should be free initially"
    lock.release()

    # Context manager should acquire lock
    with paper_locked(paper_id):
        acquired = lock.acquire(blocking=False)
        # Since it's an RLock (reentrant), same thread can acquire again
        assert acquired, "RLock should allow re-acquisition by same thread"
        lock.release()

    print("OK: paper_locked context manager works")


def test_concurrent_access_blocked():
    """Different threads cannot hold the same paper lock simultaneously."""
    paper_id = "concurrent_test"
    results = []

    def worker(worker_id, duration_sec):
        with paper_locked(paper_id):
            results.append({"worker": worker_id, "acquired_at": time.time()})
            time.sleep(duration_sec)
            results.append({"worker": worker_id, "released_at": time.time()})

    # Start two threads that will try to lock the same paper
    t1 = threading.Thread(target=worker, args=(1, 0.1))
    t2 = threading.Thread(target=worker, args=(2, 0.05))

    t0 = time.time()
    t1.start()
    time.sleep(0.05)  # Let t1 acquire lock first
    t2.start()

    t1.join()
    t2.join()
    elapsed = time.time() - t0

    # Worker 2 should have had to wait for worker 1 to finish
    # Minimum time should be > 0.15 seconds (both durations)
    assert elapsed > 0.15, f"Concurrent access should be serialized, took {elapsed}s"
    print(f"OK: Concurrent access blocked (took {elapsed:.2f}s, min 0.15s)")


def test_different_papers_concurrent():
    """Different papers can be locked concurrently (no cross-paper blocking)."""
    results = []
    t0 = time.time()

    def worker(paper_id):
        with paper_locked(paper_id):
            results.append({"paper": paper_id, "start": time.time() - t0})
            time.sleep(0.1)
            results.append({"paper": paper_id, "end": time.time() - t0})

    # Two threads, two different papers
    t1 = threading.Thread(target=worker, args=("paper_a",))
    t2 = threading.Thread(target=worker, args=("paper_b",))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    total_time = time.time() - t0

    # Should run concurrently, taking ~0.1s, not ~0.2s
    assert total_time < 0.15, f"Different papers should run concurrently, took {total_time:.2f}s"
    print(f"OK: Different papers run concurrently (took {total_time:.2f}s, should be ~0.1s)")


if __name__ == "__main__":
    print("Running concurrency tests...\n")
    test_chroma_client_singleton()
    test_paper_locks_per_paper()
    test_paper_locked_context_manager()
    test_concurrent_access_blocked()
    test_different_papers_concurrent()
    print("\nAll concurrency tests passed!")
