"""
Regression tests for Launch Checklist 2.7 — CI installs too few backend deps
to even collect the test suite, and never runs the frontend test suite.
"""

from pathlib import Path

CI_YML = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


def _text() -> str:
    return CI_YML.read_text(encoding="utf-8")


def _pip_install_line() -> str:
    """The install step uses a folded YAML scalar (`run: >`) that wraps the
    pip install across several lines, so grab the whole step's block rather
    than a single line containing the literal text "pip install"."""
    text = _text()
    start = text.index("pip install")
    end = text.index("- name: Lint (ruff)")
    return text[start:end].lower()


def test_backend_install_covers_every_package_the_suite_imports():
    """These are exactly the third-party packages needed to *collect*
    tests/*.py — api.auth, api.uploads, api.concurrency, discovery.fetcher,
    ingestion.plagiarism_auditor, ingestion.retriever/chroma_snapshot, etc.
    all get imported at collection time even for AST-based tests that share
    a file with a directly-imported one."""
    line = _pip_install_line()
    required_packages = [
        "fastapi",
        "httpx",
        "slowapi",
        "pyjwt",
        "chromadb",
        "numpy",
        "boto3",
        "psycopg",
        "sentry-sdk",
        "pdfplumber",
    ]
    for pkg in required_packages:
        assert pkg in line, f"CI's pip install line is missing {pkg}, which the test suite needs to collect"


def test_backend_ci_does_not_pull_in_the_gpu_ml_stack():
    """Nothing in tests/ imports torch or sentence-transformers — installing
    them here would turn the CI job into the same multi-GB CUDA download the
    Docker build takes, for no test-coverage benefit."""
    line = _pip_install_line()
    assert "torch" not in line
    assert "sentence-transformers" not in line


def test_frontend_job_runs_npm_test():
    text = _text()
    frontend_start = text.index("frontend:")
    frontend_section = text[frontend_start:]
    assert "npm test" in frontend_section, (
        "the frontend CI job must run `npm test` (Vitest) — it was previously "
        "lint + build only, so the 108 frontend tests never ran in CI"
    )
