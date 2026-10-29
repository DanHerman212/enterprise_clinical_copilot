"""The Vertex AI judge client: one rubric-graded call, with a hard ceiling.

Extracted from the retired collector (`evaluation/agent/judge.py`, 2026-10-29) so the
platform harness owns its own LLM transport. Everything here is about *calling* a model
and bounding the call; what the model is asked, and what its answer means, stay in
`judge_call.py` and `evaluators.py`.

Three properties are load-bearing and were each learned from a live failure:

  * The `google-genai` client accepts no `timeout` on `generate_content`, so the ceiling
    is enforced by running the call in a daemon thread and waiting on a queue. A timed-out
    attempt is abandoned rather than killed, and the thread cannot block process exit.
  * A 429 is a quota *window*, not a broken call: the backoff starts at 30 seconds so the
    retry lands past the window's edge, and a plain malformed reply gets a one-second
    beat. A timeout gets one quick retry and then becomes an unjudged answer, because
    waiting out six attempts of a ceiling that will not be met cost 13 minutes per answer.
  * The error string is printed whole. It was once cut to 120 characters, which left a
    truncated `RESOURCE_EXHAUSTED` body with the quotable field removed.

`PROJECT` and `JUDGE_LOCATION` come from `services.mcp.config`, which is where this
deployment's project and region are declared. They are not agent code and carry no
clinical behaviour.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time

from google.genai import types

from evaluation.platform.schemas import PASS_DIMENSIONS
from services.mcp.config import JUDGE_LOCATION, JUDGE_MODEL, PROJECT  # noqa: F401

#: The per-attempt ceiling, read at call time so a retry pass can raise it.
_DEFAULT_JUDGE_TIMEOUT_SECONDS = 150

#: Ceiling on the quota backoff. A 429 is a window at least a minute wide; 60s of waiting
#: meant the retry landed inside the same window and burned an attempt for nothing.
_JUDGE_BACKOFF_CAP_SECONDS = int(os.environ.get("JUDGE_BACKOFF_CAP_SECONDS", "120"))


def judge_timeout_seconds() -> int:
    return int(os.environ.get("JUDGE_TIMEOUT_SECONDS",
                              str(_DEFAULT_JUDGE_TIMEOUT_SECONDS)))


def _is_score(value) -> bool:
    """True for a real dimension score, int or float — not a bool wearing one's type."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def apply_pass_rule(sample: dict) -> dict:
    """Downgrade a PASS the scores do not support. Never upgrades one.

    The model states a verdict and also states the six scores, and the two can disagree —
    a sample writing `"verdict": "PASS"` beside `responsiveness: 1` was accepted verbatim.
    A verdict is a claim about the scores, so it is checked here instead of trusted: every
    dimension in `PASS_DIMENSIONS` must be present and at 2 or above.

    A missing required dimension fails closed, but only when the model claimed PASS: a
    pass nobody can verify from the scores is not a pass, while a FAIL needs no defending.
    `context` is deliberately not required — the rubric omits it for a first-turn answer,
    so requiring it would fail every opening turn.

    The dimension list is `schemas.PASS_DIMENSIONS`, declared once for the judge and for
    the suite gate, so the two cannot disagree about what a pass means.
    """
    if not isinstance(sample, dict) or sample.get("error"):
        return sample
    if sample.get("verdict") != "PASS":
        return sample
    dims = sample.get("dimensions") or {}
    for name in PASS_DIMENSIONS:
        value = dims.get(name)
        if not _is_score(value) or float(value) < 2.0:
            sample["verdict"] = "FAIL"
            flags = list(sample.get("flags") or [])
            flags.append(
                f"pass_rule: {name}={value!r} is below 2, so this cannot be a PASS"
            )
            sample["flags"] = flags
            break
    return sample


def _judge_once(client, system: str, user: str) -> dict:
    """One synchronous judge call (no timeout on this client API)."""
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return apply_pass_rule(json.loads(text))


def judge(client, system: str, user: str) -> dict:
    """Judge one item, with a hard ceiling and a quota-aware retry ladder."""
    attempts = 6
    last = "no attempt made"
    for attempt in range(attempts):
        box: queue.Queue[dict] = queue.Queue(maxsize=1)

        def _call() -> None:
            """Run one attempt and always report something to the queue.

            An exception used to kill this thread before it put anything, so the wait
            below timed out and the real cause was recorded as the word "timeout" — a 429
            and a genuinely stuck call became the same string.
            """
            try:
                box.put(_judge_once(client, system, user))
            except Exception as exc:  # noqa: BLE001
                box.put({"error": f"{type(exc).__name__}: {exc}"})

        threading.Thread(target=_call, daemon=True).start()
        timeout = judge_timeout_seconds()
        try:
            out = box.get(timeout=timeout)
        except queue.Empty:
            out = {"error": f"no response within {timeout}s"}
        if "error" not in out:
            return out
        last = str(out["error"])
        quota = "429" in last or "RESOURCE_EXHAUSTED" in last
        timed_out = last.startswith("no response within")
        if attempt >= attempts - 1 or (quota and attempt >= 4):
            break
        if timed_out and attempt >= 1:
            break
        print(f"  judge retry {attempt + 1}/{attempts} after {last}", flush=True)
        if not timed_out:
            base = 30 if quota else 1
            time.sleep(min(_JUDGE_BACKOFF_CAP_SECONDS, base * 2 ** attempt))
    return {"error": last}
