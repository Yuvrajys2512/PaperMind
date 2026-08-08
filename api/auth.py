import hashlib
import os
import time

import httpx
import jwt
from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException, Request
from slowapi.util import get_remote_address

load_dotenv()

CLERK_ISSUER = os.getenv("CLERK_ISSUER", "").rstrip("/")

# Comma-separated allow-list of Clerk user ids permitted to hit admin endpoints
# (e.g. the aggregate /admin/usage stats). Empty by default → admin surface is
# locked to *nobody* until the owner opts in, so it can never be left wide open.
_ADMIN_USER_IDS = {
    uid.strip() for uid in os.getenv("PAPERMIND_ADMIN_USER_IDS", "").split(",") if uid.strip()
}

_JWKS_TTL_SECONDS = 3600
_jwks_cache: dict = {"keys": {}, "fetched_at": 0.0}


def _cached_signing_key(kid: str):
    """The signing key for `kid` if it is already in the JWKS cache, else None.

    Never performs network I/O, so it is safe to call from the event loop.
    Callers that must succeed should use `_get_signing_key`.
    """
    jwk = _jwks_cache["keys"].get(kid)
    if jwk is None:
        return None
    try:
        return jwt.algorithms.RSAAlgorithm.from_jwk(jwk)
    except Exception:
        return None


def _get_signing_key(kid: str) -> str:
    now = time.time()
    if not _jwks_cache["keys"] or now - _jwks_cache["fetched_at"] > _JWKS_TTL_SECONDS:
        resp = httpx.get(f"{CLERK_ISSUER}/.well-known/jwks.json", timeout=10.0)
        resp.raise_for_status()
        jwks = resp.json()
        _jwks_cache["keys"] = {key["kid"]: key for key in jwks["keys"]}
        _jwks_cache["fetched_at"] = now

    jwk = _jwks_cache["keys"].get(kid)
    if jwk is None:
        # Key rotated since our last fetch — force a refresh once.
        resp = httpx.get(f"{CLERK_ISSUER}/.well-known/jwks.json", timeout=10.0)
        resp.raise_for_status()
        jwks = resp.json()
        _jwks_cache["keys"] = {key["kid"]: key for key in jwks["keys"]}
        _jwks_cache["fetched_at"] = now
        jwk = _jwks_cache["keys"].get(kid)

    if jwk is None:
        raise HTTPException(status_code=401, detail="Unknown signing key.")

    return jwt.algorithms.RSAAlgorithm.from_jwk(jwk)


def get_current_user_id(authorization: str = Header(None)) -> str:
    if not CLERK_ISSUER:
        raise HTTPException(status_code=500, detail="CLERK_ISSUER is not configured on the server.")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")

    token = authorization[len("Bearer "):]

    try:
        header = jwt.get_unverified_header(token)
        key = _get_signing_key(header["kid"])
        payload = jwt.decode(
            token,
            key=key,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER,
            options={"verify_aud": False},
            leeway=60,  # tolerate clock drift — Clerk's session tokens are only valid 60s
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")

    return payload["sub"]


# ── Rate-limit identity ──────────────────────────────────────────────────────
# Verifying the same token twice per request (once here, once in
# get_current_user_id) would be wasteful, so successful verifications are cached
# by token hash. Clerk session tokens live ~60s and are reused across every
# request in that window, so this costs roughly one extra verification per
# token, not per request.
#
# The token is hashed rather than stored: this dict outlives the request, and a
# bearer token is a credential.
_TOKEN_CACHE_TTL = 60.0
_TOKEN_CACHE_MAX = 1024
_token_cache: dict[str, tuple[float, str]] = {}


def user_id_from_token(token: str) -> str | None:
    """The verified Clerk user id for a bearer token, or None if it isn't valid.

    Deliberately **never raises** — this feeds the rate limiter, which runs
    before routing and must not be able to turn a JWKS hiccup into a 401 on an
    endpoint that never needed auth. Anything unverifiable simply falls back to
    IP-based limiting.

    This is NOT an authentication path. `get_current_user_id` remains the only
    gate for protected endpoints, still performing a full uncached verification
    on every request — a cached rate-limit key can be up to _TOKEN_CACHE_TTL
    stale without consequence (it is only a bucket id), whereas a cached auth
    decision could accept an expired token. Do not merge the two.
    """
    if not CLERK_ISSUER or not token:
        return None

    now = time.time()
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()

    hit = _token_cache.get(digest)
    if hit and now < hit[0]:
        return hit[1]

    try:
        header = jwt.get_unverified_header(token)
        # Deliberately the *non-fetching* lookup. This runs inside the rate
        # limiter's ASGI middleware, on the event loop — the fetching variant
        # does a synchronous httpx.get, which would stall every concurrent
        # request for up to 10s whenever Clerk rotates its keys. Missing the
        # JWKS entry just means this request keys on IP instead; the very next
        # one is fine, because get_current_user_id runs in a threadpool and
        # populates the shared cache.
        key = _cached_signing_key(header["kid"])
        if key is None:
            return None
        payload = jwt.decode(
            token,
            key=key,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER,
            options={"verify_aud": False},
            leeway=60,
        )
        sub = payload["sub"]
    except Exception:
        return None  # unverifiable → caller falls back to IP

    # Crude bound rather than an LRU: entries expire in a minute anyway, so the
    # only job here is to stop unbounded growth under a flood of junk tokens.
    if len(_token_cache) >= _TOKEN_CACHE_MAX:
        _token_cache.clear()
    _token_cache[digest] = (now + _TOKEN_CACHE_TTL, sub)
    return sub


def rate_limit_key(request: Request) -> str:
    """The bucket a request is rate-limited against.

    Keyed on the authenticated Clerk user, falling back to IP.

    The limiter used to key on `get_remote_address` alone, which behind a
    reverse proxy returns the PROXY's address — so on HF Spaces every user
    collapsed into a single bucket and the 120/min limit became a global cap on
    the whole app, where one active user could 429 everybody else. That happens
    because `get_remote_address` reads only `request.client.host`, and uvicorn's
    ProxyHeadersMiddleware only rewrites that from X-Forwarded-For when the
    socket peer is listed in FORWARDED_ALLOW_IPS — which defaults to 127.0.0.1
    and is not set in the Dockerfile.

    The tempting fix, FORWARDED_ALLOW_IPS='*', is worse than the bug: it makes
    X-Forwarded-For (and therefore the bucket id) entirely client-controlled, so
    anyone can rotate the header for unlimited buckets while the limit still
    looks enforced. A verified Clerk `sub` cannot be forged, requires trusting
    no proxy header, and is the same unit the quotas in api/usage.py count in.

    Unauthenticated traffic (health checks, landing page, Stripe webhook) still
    keys on IP and so still shares one bucket behind a proxy. Acceptable: the
    webhook verifies Stripe's signature independently, and everything expensive
    requires auth.
    """
    header = request.headers.get("authorization") or ""
    if header.startswith("Bearer "):
        user_id = user_id_from_token(header[len("Bearer "):])
        if user_id:
            return f"user:{user_id}"
    # Namespaced so a spoofable IP key can never collide with a real user id.
    return f"ip:{get_remote_address(request)}"


def require_admin(user_id: str = Depends(get_current_user_id)) -> str:
    """FastAPI dependency: authenticates the caller, then 403s unless their
    Clerk user id is in PAPERMIND_ADMIN_USER_IDS. Returns the admin's user_id."""
    if user_id not in _ADMIN_USER_IDS:
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user_id
