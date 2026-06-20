import json
import os
import tempfile
import uuid
from datetime import datetime, timezone

import boto3
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured.")

# Preloaded sample papers are owned by one fixed system user. They're shown
# read-only to every signed-in user and are excluded from per-user quotas
# (quota counting only ever looks at a real user's own papers). Seed them with
# scripts/seed_demo_papers.py. The default is namespaced so it can never
# collide with a real Clerk user id.
DEMO_USER_ID = os.getenv("PAPERMIND_DEMO_USER_ID", "__papermind_demo__")

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
    raise RuntimeError(
        "R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET_NAME must all be set."
    )

_pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5, open=True)

_s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
)


def _ensure_schema():
    with _pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                uploaded_at TIMESTAMPTZ NOT NULL,
                completed_at TIMESTAMPTZ,
                error TEXT,
                source_id TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_user_id ON papers (user_id)")


_ensure_schema()


# ── Registry (Neon Postgres) ─────────────────────────────────────────────────

def create_paper_record(original_filename: str, user_id: str, source_id: str = None) -> str:
    """Creates a new paper entry with status 'processing'. Returns the paper_id."""
    paper_id = str(uuid.uuid4())
    with _pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO papers (paper_id, user_id, filename, status, uploaded_at, source_id)
            VALUES (%s, %s, %s, 'processing', %s, %s)
            """,
            (paper_id, user_id, original_filename, datetime.now(timezone.utc), source_id),
        )
    return paper_id


def update_paper_status(paper_id: str, status: str, error: str = None):
    """Updates a paper's status to 'ready' or 'failed'."""
    completed_at = datetime.now(timezone.utc) if status == "ready" else None
    with _pool.connection() as conn:
        conn.execute(
            """
            UPDATE papers
            SET status = %s,
                completed_at = COALESCE(%s, completed_at),
                error = COALESCE(%s, error)
            WHERE paper_id = %s
            """,
            (status, completed_at, error, paper_id),
        )


def get_paper(paper_id: str) -> dict | None:
    """Returns the paper record, or None if not found."""
    with _pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM papers WHERE paper_id = %s", (paper_id,))
            return cur.fetchone()


def get_owned_paper(paper_id: str, user_id: str) -> dict | None:
    """Returns the paper record only if it exists and belongs to user_id."""
    paper = get_paper(paper_id)
    if not paper or paper.get("user_id") != user_id:
        return None
    return paper


def list_papers(user_id: str) -> list:
    """Returns user_id's OWN paper records as a list, newest first. Deliberately
    excludes the shared demo set — quota enforcement counts this, so demo papers
    must never inflate a user's paper count. Use list_demo_papers() / the
    /papers endpoint for the full visible library."""
    with _pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM papers WHERE user_id = %s ORDER BY uploaded_at DESC",
                (user_id,),
            )
            return cur.fetchall()


def list_demo_papers() -> list:
    """Returns the shared, read-only sample papers (owned by DEMO_USER_ID),
    newest first. Shown to every user so a brand-new account has something to
    query immediately; never counted against any user's quota."""
    with _pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM papers WHERE user_id = %s ORDER BY uploaded_at DESC",
                (DEMO_USER_ID,),
            )
            return cur.fetchall()


def get_readable_paper(paper_id: str, user_id: str) -> dict | None:
    """Returns the paper if user_id may READ it — i.e. they own it, or it's a
    shared demo paper. This is the gate for query/status/pdf/glossary/etc.
    Write actions (delete) keep using get_owned_paper, so a user can never
    mutate the shared demo set."""
    paper = get_paper(paper_id)
    if not paper:
        return None
    if paper.get("user_id") in (user_id, DEMO_USER_ID):
        return paper
    return None


def list_ready_paper_ids() -> list[str]:
    """All paper_ids in 'ready' status, across every user. Used by the
    startup routine that rebuilds missing Chroma collections from R2."""
    with _pool.connection() as conn:
        cur = conn.execute("SELECT paper_id FROM papers WHERE status = 'ready'")
        return [row[0] for row in cur.fetchall()]


def delete_paper_record(paper_id: str) -> bool:
    """Removes a paper from the registry. Returns True if it existed."""
    with _pool.connection() as conn:
        cur = conn.execute("DELETE FROM papers WHERE paper_id = %s", (paper_id,))
        return cur.rowcount > 0


# ── PDFs (Cloudflare R2) ──────────────────────────────────────────────────────

def upload_pdf(paper_id: str, local_path: str):
    """Uploads a local PDF file to R2 under the paper's id."""
    _s3.upload_file(local_path, R2_BUCKET_NAME, f"{paper_id}.pdf")


def download_pdf_to_tempfile(paper_id: str) -> str:
    """Downloads a paper's PDF from R2 into a local temp file. Caller must delete it."""
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    _s3.download_file(R2_BUCKET_NAME, f"{paper_id}.pdf", temp_path)
    return temp_path


def get_pdf_stream(paper_id: str):
    """Returns a streaming body for a paper's PDF, for proxying through an HTTP response."""
    return _s3.get_object(Bucket=R2_BUCKET_NAME, Key=f"{paper_id}.pdf")["Body"]


def delete_pdf(paper_id: str):
    """Deletes a paper's PDF from R2."""
    _s3.delete_object(Bucket=R2_BUCKET_NAME, Key=f"{paper_id}.pdf")


# ── Claim-audit reports (Cloudflare R2) ───────────────────────────────────────
# Cached as a JSON blob next to the PDF so re-opening a paper's audit is free
# (an audit is the heaviest LLM operation in the app). Stored in R2 rather than
# Postgres to avoid a schema migration — the report is an opaque document keyed
# only by paper_id, never queried by column.

def _audit_key(paper_id: str) -> str:
    return f"{paper_id}.audit.json"


def upload_audit_report(paper_id: str, report: dict):
    """Persist a claim-audit report as a JSON blob in R2."""
    _s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=_audit_key(paper_id),
        Body=json.dumps(report).encode("utf-8"),
        ContentType="application/json",
    )


def get_audit_report(paper_id: str) -> dict | None:
    """Return the cached audit report for a paper, or None if none exists."""
    try:
        obj = _s3.get_object(Bucket=R2_BUCKET_NAME, Key=_audit_key(paper_id))
    except _s3.exceptions.NoSuchKey:
        return None
    except Exception:
        # Treat any read failure as a cache miss — the caller will recompute.
        return None
    try:
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


def delete_audit_report(paper_id: str):
    """Best-effort delete of a paper's cached audit report."""
    _s3.delete_object(Bucket=R2_BUCKET_NAME, Key=_audit_key(paper_id))


# ── Reviewer / weakness-audit reports (Cloudflare R2) ─────────────────────────
# The methodological-completeness audit (reviewer_auditor) — same caching
# rationale as the claim audit above: an opaque JSON document keyed only by
# paper_id, stored next to the PDF so re-opening a paper's review is free.

def _review_key(paper_id: str) -> str:
    return f"{paper_id}.review.json"


def upload_review_report(paper_id: str, report: dict):
    """Persist a reviewer/weakness-audit report as a JSON blob in R2."""
    _s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=_review_key(paper_id),
        Body=json.dumps(report).encode("utf-8"),
        ContentType="application/json",
    )


def get_review_report(paper_id: str) -> dict | None:
    """Return the cached reviewer-audit report for a paper, or None if none exists."""
    try:
        obj = _s3.get_object(Bucket=R2_BUCKET_NAME, Key=_review_key(paper_id))
    except _s3.exceptions.NoSuchKey:
        return None
    except Exception:
        # Treat any read failure as a cache miss — the caller will recompute.
        return None
    try:
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


def delete_review_report(paper_id: str):
    """Best-effort delete of a paper's cached reviewer-audit report."""
    _s3.delete_object(Bucket=R2_BUCKET_NAME, Key=_review_key(paper_id))
