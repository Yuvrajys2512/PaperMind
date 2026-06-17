import os
import time

import httpx
import jwt
from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

CLERK_ISSUER = os.getenv("CLERK_ISSUER", "").rstrip("/")

_JWKS_TTL_SECONDS = 3600
_jwks_cache: dict = {"keys": {}, "fetched_at": 0.0}


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
