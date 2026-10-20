"""Identity tokens for calling the private agent service.

Cloud Run refuses anonymous callers (verified: an unauthenticated POST to `/ask`
returns 403), so every evaluation request carries a Google-signed identity token
audienced to the service URL. On a laptop that comes from the user's ADC; inside
Cloud Run it would come from the metadata server — the same two paths the existing
collector uses.
"""

from __future__ import annotations

import subprocess


def identity_token(audience: str) -> str:
    """An ID token valid for `audience`, or raise with the reason."""
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
