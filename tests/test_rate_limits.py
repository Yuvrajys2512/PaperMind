"""
Regression tests for Launch Checklist 2.8 — rate limits uniform across
wildly different costs.

AST-based: `api.main` and `discovery.router` connect to live Postgres/R2 at
import time (see 2.12).
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "api" / "main.py"
DISCOVERY_ROUTER = ROOT / "discovery" / "router.py"
AUTH = ROOT / "api" / "auth.py"


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _decorator_strings(fn) -> list[str]:
    return [ast.unparse(d) for d in fn.decorator_list]


def test_shared_limiter_lives_in_api_auth():
    """The Limiter instance must be constructed once, in api.auth, so
    discovery/router.py can add per-route limits without an import cycle
    back to api.main."""
    tree = _tree(AUTH)
    assigns = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "limiter" in assigns, "expected a module-level `limiter = Limiter(...)` in api/auth.py"


def test_main_imports_limiter_rather_than_constructing_its_own():
    source = MAIN.read_text(encoding="utf-8")
    assert "from api.auth import" in source
    import_line = next(
        line for line in source.splitlines() if line.startswith("from api.auth import")
    )
    assert "limiter" in import_line, "api/main.py must import the shared limiter from api.auth"
    assert "Limiter(" not in source, (
        "api/main.py must not construct its own Limiter instance — that would "
        "give discovery.router's per-route limits a separate bucket store"
    )


UPLOAD_LIKE_ROUTES = [
    ("upload_paper", MAIN),
    ("audit_paper_stream", MAIN),
    ("review_paper_stream", MAIN),
    ("novelty_scan_stream", MAIN),
    ("structure_check_stream", MAIN),
    ("numbers_check_stream", MAIN),
    ("citation_gap_stream", MAIN),
    ("overlap_stream", MAIN),
]


def test_upload_and_audit_stream_routes_have_tight_per_route_limits():
    for fn_name, path in UPLOAD_LIKE_ROUTES:
        fn = _find_function(_tree(path), fn_name)
        assert fn is not None, f"{fn_name} is gone from {path.name}"
        decorators = _decorator_strings(fn)
        assert any("limiter.limit(" in d for d in decorators), (
            f"{fn_name} must have a @limiter.limit(...) decorator tighter than "
            "the 120/minute global default — it's an expensive, LLM-backed route"
        )


def test_query_stream_has_a_per_route_limit():
    fn = _find_function(_tree(MAIN), "query_stream")
    assert fn is not None
    decorators = _decorator_strings(fn)
    assert any("limiter.limit(" in d for d in decorators)


def test_discovery_import_has_a_tight_per_route_limit():
    fn = _find_function(_tree(DISCOVERY_ROUTER), "import_paper")
    assert fn is not None, "import_paper is gone from discovery/router.py"
    decorators = _decorator_strings(fn)
    assert any("limiter.limit(" in d for d in decorators), (
        "/discovery/import runs the same LLM-heavy ingestion pipeline as "
        "/upload and must carry the same tight rate limit"
    )
