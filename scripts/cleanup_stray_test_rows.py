"""
scripts/cleanup_stray_test_rows.py — remove test-created paper rows + R2 blobs.

Why this exists
---------------
`tests/test_storage_tempfiles.py` does `from api import storage` at module
level, and `api/storage.py` opens a Neon pool and an R2 client at import time
using whatever is in `.env`. Pytest imports every test module during collection,
so **running the test suite on a machine with a real `.env` connects to
production.**

On 2026-08-25 that combined with a stubbing bug in `tests/test_fetcher_redirects.py`
(a `sys.modules` guard that silently never fired, because the real `api.storage`
was already loaded by collection) and four rows were written to the live papers
table, each with a matching PDF object in R2:

    user_id = 'user_1', filename = 'Test Paper.pdf', status = 'processing'

The test bug is fixed — the fetcher fixture now patches names on the fetcher
module itself, which is order-independent. This script cleans up what the bug
already wrote.

Usage
-----
    python scripts/cleanup_stray_test_rows.py            # dry run — lists only
    python scripts/cleanup_stray_test_rows.py --confirm  # actually deletes

Only rows matching the exact test signature above are ever touched. Real uploads
have real Clerk user ids (`user_...` with a long suffix), never the literal
`user_1`, so the filter cannot match a genuine paper.
"""

import sys
from pathlib import Path

# Run from anywhere: the repo root has to be importable for `api.storage`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api.storage as st  # noqa: E402

# The exact signature the buggy test wrote. Deliberately narrow.
TEST_USER_ID = "user_1"
TEST_FILENAME = "Test Paper.pdf"


def find_stray_rows():
    with st._pool.connection() as conn:
        return conn.execute(
            "SELECT paper_id, user_id, filename, status, uploaded_at "
            "FROM papers WHERE user_id = %s AND filename = %s "
            "ORDER BY uploaded_at DESC",
            (TEST_USER_ID, TEST_FILENAME),
        ).fetchall()


def main() -> int:
    confirm = "--confirm" in sys.argv

    rows = find_stray_rows()
    if not rows:
        print("No stray test rows found. Nothing to do.")
        return 0

    print(f"Found {len(rows)} stray test row(s):\n")
    for paper_id, user_id, filename, status, uploaded_at in rows:
        try:
            st._s3.head_object(Bucket=st.R2_BUCKET_NAME, Key=f"{paper_id}.pdf")
            blob = "R2 blob present"
        except Exception:
            blob = "no R2 blob"
        print(f"  {paper_id}  {user_id}  {filename!r}  {status}  {uploaded_at}  [{blob}]")

    if not confirm:
        print("\nDry run. Re-run with --confirm to delete these rows and their R2 objects.")
        return 0

    print("\nDeleting...")
    for paper_id, *_ in rows:
        # R2 first, then the row — same ordering as the app's delete path, so a
        # failure leaves a recoverable "row exists, blob gone" state rather than
        # an unreachable orphaned blob.
        try:
            st._s3.delete_object(Bucket=st.R2_BUCKET_NAME, Key=f"{paper_id}.pdf")
            print(f"  R2 object deleted: {paper_id}.pdf")
        except Exception as exc:
            print(f"  R2 delete failed for {paper_id}: {exc}")

        with st._pool.connection() as conn:
            n = conn.execute("DELETE FROM papers WHERE paper_id = %s", (paper_id,)).rowcount
        print(f"  registry row deleted: {paper_id} ({n} row)")

    remaining = find_stray_rows()
    print(f"\nDone. Remaining stray rows: {len(remaining)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
