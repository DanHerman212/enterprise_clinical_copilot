"""Identity tokens for calling the private agent service.

Cloud Run refuses anonymous callers (verified: an unauthenticated POST to `/ask`
returns 403), so every evaluation request carries a Google-signed identity token
audienced to the service URL. On a laptop that comes from the user's ADC; inside
Cloud Run it would come from the metadata server — the same two paths the existing
collector uses.
"""

from __future__ import annotations

import subprocess
import threading
import time

_TOKEN_TTL_SECONDS = 2700  # 45 minutes (ID tokens last 60 minutes)
_token_lock = threading.Lock()
_token_cache: dict[str, tuple[str, float]] = {}


def clear_token_cache() -> None:
    """Drop cached identity tokens (used on 401 refresh and in tests)."""
    with _token_lock:
        _token_cache.clear()


def _fetch_identity_token(audience: str) -> str:
    try:  # works with ADC (local user credentials, or a service account)
        import google.auth.transport.requests as google_requests
        from google.oauth2 import id_token

        return id_token.fetch_id_token(google_requests.Request(), audience)
    except Exception:  # noqa: BLE001 — fall back to the CLI the collector uses
        result = subprocess.run(
            ["gcloud", "auth", "print-identity-token"],
            capture_output=True, text=True, check=False,
        )
        token = (result.stdout or "").strip()
        if result.returncode != 0 or not token:
            raise RuntimeError(
                "could not mint an identity token: "
                f"{(result.stderr or '').strip()[:200]}"
            ) from None
        return token


def identity_token(audience: str) -> str:
    """An ID token valid for `audience`, cached in-process across concurrent turns."""
    now = time.monotonic()
    with _token_lock:
        cached = _token_cache.get(audience)
        if cached and now < cached[1]:
            return cached[0]
        token = _fetch_identity_token(audience)
        _token_cache[audience] = (token, time.monotonic() + _TOKEN_TTL_SECONDS)
        return token
