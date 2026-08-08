"""
Unit tests for api.auth.rate_limit_key — the bucket every request is rate
limited against.

This is a security boundary, and it regressed silently once already: keying on
the client address alone meant that behind a reverse proxy (HF Spaces) every
user in the world shared one bucket, turning a 120/min per-user limit into a
120/min cap on the entire application. The tests below pin the two properties
that matter:

  1. Two authenticated users arriving through the SAME proxy address get
     DIFFERENT buckets.
  2. A client cannot choose its own bucket by forging X-Forwarded-For — which
     is exactly what the obvious fix (FORWARDED_ALLOW_IPS='*') would have
     allowed.
"""


from starlette.requests import Request

from api import auth


def _request(*, peer="10.16.0.7", token=None, forwarded=None) -> Request:
    """A request whose socket peer is a proxy, as in production."""
    headers = []
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    if forwarded is not None:
        headers.append((b"x-forwarded-for", forwarded.encode()))
    return Request({
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": (peer, 51234),
        "server": ("testserver", 80),
    })


def _fake_tokens(monkeypatch, mapping):
    """Stub token verification: maps token string -> user id, None otherwise."""
    monkeypatch.setattr(auth, "user_id_from_token", lambda t: mapping.get(t))


# --- the bug this function exists to fix ------------------------------------

def test_two_users_behind_one_proxy_get_different_buckets(monkeypatch):
    _fake_tokens(monkeypatch, {"tok_alice": "user_alice", "tok_bob": "user_bob"})

    alice = auth.rate_limit_key(_request(token="tok_alice"))
    bob   = auth.rate_limit_key(_request(token="tok_bob"))

    assert alice == "user:user_alice"
    assert bob == "user:user_bob"
    assert alice != bob, "same proxy peer must not collapse users into one bucket"


def test_same_user_from_different_addresses_shares_one_bucket(monkeypatch):
    """The flip side: a user on phone + laptop is still one user. Keying on the
    identity rather than the address is what makes the limit meaningful."""
    _fake_tokens(monkeypatch, {"tok_alice": "user_alice"})

    a = auth.rate_limit_key(_request(peer="10.16.0.7", token="tok_alice"))
    b = auth.rate_limit_key(_request(peer="10.16.0.9", token="tok_alice"))
    assert a == b == "user:user_alice"


def test_forged_forwarded_for_cannot_change_an_authenticated_bucket(monkeypatch):
    """The FORWARDED_ALLOW_IPS='*' trap: a client rotating X-Forwarded-For must
    not be able to mint itself fresh buckets."""
    _fake_tokens(monkeypatch, {"tok_alice": "user_alice"})

    keys = {
        auth.rate_limit_key(_request(token="tok_alice", forwarded=f"1.2.3.{i}"))
        for i in range(5)
    }
    assert keys == {"user:user_alice"}


# --- fallback behaviour ------------------------------------------------------

def test_unauthenticated_falls_back_to_ip(monkeypatch):
    _fake_tokens(monkeypatch, {})
    assert auth.rate_limit_key(_request()) == "ip:10.16.0.7"


def test_unverifiable_token_falls_back_to_ip_and_does_not_raise(monkeypatch):
    """A junk or expired token must degrade to IP limiting, never 401 — this
    runs before routing, on endpoints that may not require auth at all."""
    _fake_tokens(monkeypatch, {})
    assert auth.rate_limit_key(_request(token="garbage")) == "ip:10.16.0.7"


def test_malformed_authorization_header_falls_back_to_ip(monkeypatch):
    _fake_tokens(monkeypatch, {"tok": "user_x"})
    req = Request({
        "type": "http", "http_version": "1.1", "method": "GET", "scheme": "http",
        "path": "/", "raw_path": b"/", "query_string": b"", "root_path": "",
        "headers": [(b"authorization", b"Basic tok")],
        "client": ("10.16.0.7", 1), "server": ("testserver", 80),
    })
    assert auth.rate_limit_key(req) == "ip:10.16.0.7"


def test_user_and_ip_keys_cannot_collide(monkeypatch):
    """A user id that looks like an address must not land in an IP bucket."""
    _fake_tokens(monkeypatch, {"tok": "10.16.0.7"})
    assert auth.rate_limit_key(_request(token="tok")) == "user:10.16.0.7"
    assert auth.rate_limit_key(_request()) == "ip:10.16.0.7"


# --- the cached verifier -----------------------------------------------------

def test_unverifiable_tokens_return_none_without_raising(monkeypatch):
    monkeypatch.setattr(auth, "CLERK_ISSUER", "https://clerk.example.com")
    auth._token_cache.clear()
    for junk in ("", "not-a-jwt", "a.b.c"):
        assert auth.user_id_from_token(junk) is None


def test_no_clerk_issuer_configured_returns_none(monkeypatch):
    monkeypatch.setattr(auth, "CLERK_ISSUER", "")
    assert auth.user_id_from_token("anything") is None


def test_key_lookup_never_fetches_jwks(monkeypatch):
    """The key function runs inside the rate limiter's ASGI middleware, on the
    event loop. A synchronous JWKS fetch there would stall every concurrent
    request for up to 10s whenever Clerk rotates keys — so an unknown `kid`
    must degrade to the IP fallback rather than go to the network."""
    monkeypatch.setattr(auth, "CLERK_ISSUER", "https://clerk.example.com")
    auth._token_cache.clear()
    auth._jwks_cache["keys"] = {}          # nothing cached → tempting to fetch

    def boom(*a, **kw):
        raise AssertionError("user_id_from_token performed network I/O")

    monkeypatch.setattr(auth.httpx, "get", boom)
    monkeypatch.setattr(auth.jwt, "get_unverified_header", lambda t: {"kid": "unknown"})

    assert auth.user_id_from_token("tok") is None
    assert auth.rate_limit_key(_request(token="tok")) == "ip:10.16.0.7"


def test_verification_result_is_cached_per_token(monkeypatch):
    """Verifying twice per request (here and in get_current_user_id) would be
    wasteful; the cache should make repeat calls free within a token's life."""
    monkeypatch.setattr(auth, "CLERK_ISSUER", "https://clerk.example.com")
    auth._token_cache.clear()

    calls = {"n": 0}

    def fake_decode(token, **kwargs):
        calls["n"] += 1
        return {"sub": "user_alice"}

    monkeypatch.setattr(auth.jwt, "get_unverified_header", lambda t: {"kid": "k1"})
    monkeypatch.setattr(auth, "_cached_signing_key", lambda kid: "key")
    monkeypatch.setattr(auth.jwt, "decode", fake_decode)

    assert auth.user_id_from_token("tok") == "user_alice"
    assert auth.user_id_from_token("tok") == "user_alice"
    assert calls["n"] == 1, "second lookup should be served from cache"

    # A different token is verified on its own.
    assert auth.user_id_from_token("other") == "user_alice"
    assert calls["n"] == 2


def test_cache_expires(monkeypatch):
    monkeypatch.setattr(auth, "CLERK_ISSUER", "https://clerk.example.com")
    monkeypatch.setattr(auth, "_TOKEN_CACHE_TTL", -1.0)  # already stale
    auth._token_cache.clear()

    calls = {"n": 0}

    def fake_decode(token, **kwargs):
        calls["n"] += 1
        return {"sub": "user_alice"}

    monkeypatch.setattr(auth.jwt, "get_unverified_header", lambda t: {"kid": "k1"})
    monkeypatch.setattr(auth, "_cached_signing_key", lambda kid: "key")
    monkeypatch.setattr(auth.jwt, "decode", fake_decode)

    auth.user_id_from_token("tok")
    auth.user_id_from_token("tok")
    assert calls["n"] == 2, "an expired entry must be re-verified"


def test_raw_tokens_are_not_stored_in_the_cache(monkeypatch):
    """The cache outlives the request and a bearer token is a credential, so
    keys must be hashes, not the tokens themselves."""
    monkeypatch.setattr(auth, "CLERK_ISSUER", "https://clerk.example.com")
    auth._token_cache.clear()
    monkeypatch.setattr(auth.jwt, "get_unverified_header", lambda t: {"kid": "k1"})
    monkeypatch.setattr(auth, "_cached_signing_key", lambda kid: "key")
    monkeypatch.setattr(auth.jwt, "decode", lambda t, **kw: {"sub": "user_alice"})

    auth.user_id_from_token("super-secret-token")
    assert "super-secret-token" not in auth._token_cache
    assert all(len(k) == 64 for k in auth._token_cache), "keys should be sha256 hex"


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(auth, "CLERK_ISSUER", "https://clerk.example.com")
    monkeypatch.setattr(auth, "_TOKEN_CACHE_MAX", 4)
    auth._token_cache.clear()
    monkeypatch.setattr(auth.jwt, "get_unverified_header", lambda t: {"kid": "k1"})
    monkeypatch.setattr(auth, "_cached_signing_key", lambda kid: "key")
    monkeypatch.setattr(auth.jwt, "decode", lambda t, **kw: {"sub": "u"})

    for i in range(50):
        auth.user_id_from_token(f"tok{i}")
    assert len(auth._token_cache) <= 4
