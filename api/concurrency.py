"""
api/concurrency.py — Thread-safe ChromaDB wrapper, per-paper locking, and the
dedicated executor for heavy sync/LLM work.

Prevents concurrent writes to ChromaDB and race conditions on paper operations
(ingest, query, delete, update). Each paper gets its own lock; different papers
don't block each other.

Public API
----------
get_chroma_client() -> chromadb.PersistentClient
  Thread-safe singleton ChromaDB client with locking.

acquire_paper_lock(paper_id: str) -> threading.RLock
  Acquire a lock for a paper. Held during ingest/query/delete to prevent races.
  Use as a context manager or manually acquire/release.

paper_locked(paper_id: str)
  Decorator for functions that need exclusive access to a paper.

release_paper_lock(paper_id: str) -> None
  Forgets a paper's lock entry once it's deleted for good, so _paper_locks
  doesn't grow by one RLock per paper_id ever seen (Launch Checklist 2.11).
  Call after the `with paper_locked(paper_id):` delete block has exited.

run_on_executor(fn, timeout, *args) -> awaitable
  Runs fn(*args) on an explicitly-sized executor with a busy->503 gate and
  cooperative cancellation on timeout. Replaces the old
  `asyncio.wait_for(loop.run_in_executor(None, fn), timeout=...)` pattern in
  api/main.py (Launch Checklist 1.7).

track_task(coro) -> asyncio.Task
  asyncio.create_task() that also keeps a strong reference, so the task can't
  be garbage-collected mid-flight (Launch Checklist 1.7).
"""

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps

import chromadb
from fastapi import HTTPException

from ingestion.llm_client import set_cancel_event

_chroma_lock = threading.Lock()
_chroma_client = None

# Absolute by default so a Chroma client's behavior can't change just because
# the process's CWD does (a launcher, a cron job, a different working
# directory in a container all resolve "data/chroma_db" differently).
# PAPERMIND_CHROMA_PATH lets ops point this at a mounted persistent volume
# (Launch Checklist 1.4(b)) without a code change.
CHROMA_PATH = os.getenv(
    "PAPERMIND_CHROMA_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma_db"),
)

_paper_locks = {}
_paper_locks_lock = threading.Lock()

# Explicitly sized so it's never at the mercy of asyncio's implicit default
# pool (min(32, cpu+4) — 6 threads on a 2-vCPU host, shared with every other
# run_in_executor(None, ...) call in the process). Every heavy sync/LLM
# pipeline call in api/main.py goes through run_on_executor() below, which
# submits onto this pool.
EXECUTOR_WORKERS = int(os.getenv("PAPERMIND_EXECUTOR_WORKERS", "12"))
executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS, thread_name_prefix="papermind-worker")

# Bounded to the same size as `executor`: "acquired" means a worker thread is
# actually free right now, not merely that the job has been queued. Queuing
# into a full pool is exactly the bug this replaces — `asyncio.wait_for`'s
# clock starts immediately even though the job hasn't been picked up yet, so a
# queued request can 504 having done zero work while its quota is still spent.
_capacity = threading.BoundedSemaphore(EXECUTOR_WORKERS)

# Strong references to background tasks fired with asyncio.create_task() and
# nothing awaiting them (the */stream endpoints' worker tasks). CPython only
# holds a weak reference to a bare task, so without this a task can be
# garbage-collected mid-flight.
_background_tasks: set[asyncio.Task] = set()


def track_task(coro) -> asyncio.Task:
    """asyncio.create_task() that keeps a strong reference until completion."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def run_on_executor(fn, timeout: float, *args):
    """Runs fn(*args) on the dedicated `executor`, replacing
    `asyncio.wait_for(loop.run_in_executor(None, fn), timeout=...)`.

    Three protections stacked here (Launch Checklist 1.7):
    1. Busy -> HTTPException(503) immediately if every worker thread is
       already occupied, instead of queuing into a timeout that burns the
       caller's quota for zero work.
    2. A per-call threading.Event cancel flag, set when the timeout fires and
       checked by ingestion/llm_client.chat_completion between provider
       attempts and backoff sleeps — so a timed-out request stops spending
       LLM tokens instead of running to completion, unbilled, in the
       background. Cooperative: it only takes effect at the next checkpoint,
       not mid-network-call.
    3. Capacity is only released once the worker thread actually finishes, not
       when the timeout fires — otherwise a still-running cancelled job and a
       newly admitted one could both occupy real threads at once, defeating
       the busy-503 gate above. This is why the release is attached to the
       raw concurrent.futures.Future (`cf_future`, resolved only by the
       worker thread itself) rather than the asyncio-wrapped one: wait_for
       cancels the *asyncio* future immediately on timeout, which would fire
       a callback attached there right away even though the thread is still
       running.
    """
    if not _capacity.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Server is busy — please try again in a moment.")

    cancel_event = threading.Event()

    def _wrapped():
        set_cancel_event(cancel_event)
        try:
            return fn(*args)
        finally:
            set_cancel_event(None)

    loop = asyncio.get_running_loop()
    cf_future = executor.submit(_wrapped)
    cf_future.add_done_callback(lambda _f: _capacity.release())
    try:
        return await asyncio.wait_for(asyncio.wrap_future(cf_future, loop=loop), timeout=timeout)
    except asyncio.TimeoutError:
        cancel_event.set()
        raise


def get_chroma_client() -> chromadb.PersistentClient:
    """The single ChromaDB client for the whole process.

    Creates on first call, then returns the same instance. This is the ONE
    place a chromadb.PersistentClient is constructed — ingestion/retriever.py,
    ingestion/embedder.py and ingestion/bm25_retriever.py all call this
    instead of instantiating their own (Launch Checklist 2.9). Previously each
    of those four call sites opened its own client against the same on-disk
    path, so the locking this module claims to provide only ever covered the
    delete path — every read bypassed it.

    Locked so two threads racing the first call can't each construct one.
    """
    global _chroma_client
    if _chroma_client is None:
        with _chroma_lock:
            if _chroma_client is None:
                _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


def acquire_paper_lock(paper_id: str) -> threading.RLock:
    """Get or create a per-paper lock for synchronizing operations on that paper.

    Reentrant (RLock) so the same thread can acquire it multiple times
    (e.g., if ingest calls query internally).
    """
    with _paper_locks_lock:
        if paper_id not in _paper_locks:
            _paper_locks[paper_id] = threading.RLock()
        return _paper_locks[paper_id]


@contextmanager
def paper_locked(paper_id: str):
    """Context manager for paper-level locking.

    Usage:
        with paper_locked(paper_id):
            # operations on paper are exclusive
            ...
    """
    lock = acquire_paper_lock(paper_id)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def release_paper_lock(paper_id: str) -> None:
    """Forgets paper_id's lock entry once the paper is gone for good (Launch
    Checklist 2.11) — otherwise _paper_locks grows by one RLock for every
    paper_id ever seen, for the lifetime of the process.

    Call this AFTER the `with paper_locked(paper_id):` block that performs
    the delete has exited (never from inside it — this must not run while
    the lock it's discarding is still held). Safe even if another thread is
    concurrently blocked waiting to acquire the same lock object: that
    thread already holds a reference to it independent of the dict entry, so
    removing the entry here only affects *future* callers of
    acquire_paper_lock(paper_id), who will get a fresh RLock. A future
    caller only exists if something still thinks this paper_id is valid,
    which is a bug elsewhere (the registry row is already gone), not a race
    this function needs to prevent.
    """
    with _paper_locks_lock:
        _paper_locks.pop(paper_id, None)


def paper_lock_decorator(func):
    """Decorator to acquire a paper lock for a function.

    The function must have 'paper_id' as a parameter (positional or kwarg).

    Usage:
        @paper_lock_decorator
        def ingest_paper(paper_id, ...):
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to extract paper_id from kwargs first, then look through args
        paper_id = kwargs.get("paper_id")
        if paper_id is None:
            # Assume first positional arg is paper_id (common pattern)
            # This is fragile but works for the codebase's function signatures
            if args:
                paper_id = args[0]

        if paper_id is None:
            raise ValueError(
                f"{func.__name__} must have 'paper_id' as first arg or kwarg"
            )

        with paper_locked(paper_id):
            return func(*args, **kwargs)

    return wrapper
