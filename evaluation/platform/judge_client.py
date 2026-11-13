"""The independent judge client: one rubric-graded call, with a hard ceiling.

Extracted from the retired collector (`evaluation/agent/judge.py`, 2026-10-29) so the
platform harness owns its own LLM transport. Everything here is about *calling* a model
and bounding the call; what the model is asked, and what its answer means, stay in
`judge_call.py` and `evaluators.py`.

The model is `xai/grok-4.7`, reached through GCP's OpenAI-compatible endpoint. The
vendor changed in Sprint 03 (SCAR-19): a grader from the same family as a candidate arm
measures agreement with itself, and every arm on the roster is Google or Meta or
Anthropic. What did NOT change is everything about how a call is bounded, because each
of those rules was learned from a live failure:

  * The ceiling is a daemon thread and a queue, not a client option. The OpenAI client
    accepts a `timeout`, but an abandoned HTTP call still holds a connection and the
    socket timeout is per-request-segment rather than per-attempt; the queue is what
    actually bounds how long the run waits. A timed-out attempt is abandoned rather than
    killed, and the thread cannot block process exit.
  * A 429 is a quota *window*, not a broken call: the backoff starts at 30 seconds so the
    retry lands past the window's edge, and a plain malformed reply gets a one-second
    beat. A timeout gets one quick retry and then becomes an unjudged answer, because
    waiting out six attempts of a ceiling that will not be met cost 13 minutes per answer.
  * The error string is printed whole. It was once cut to 120 characters, which left a
    truncated `RESOURCE_EXHAUSTED` body with the quotable field removed.
  * Every attempt's token usage is kept, because a judge that retried was paid for
    twice and a run's grading cost has to say so.

`PROJECT`, `JUDGE_MODEL`, `JUDGE_LOCATION` and `JUDGE_REASONING_EFFORT` come from
`services.mcp.config`, which is where this deployment's project and models are declared.
They are not agent code and carry no clinical behaviour.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any

from evaluation.platform import pricing
from evaluation.platform.schemas import DIMENSIONS, PASS_DIMENSIONS, pass_dimensions_for
from services.mcp.config import (  # noqa: F401
    JUDGE_LOCATION,
    JUDGE_MODEL,
    JUDGE_REASONING_EFFORT,
    PROJECT,
)

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


def apply_pass_rule(sample: dict, *, multi_turn: bool = False) -> dict:
    """Downgrade a PASS the scores do not support. Never upgrades one.

    The model states a verdict and also states the six scores, and the two can disagree —
    a sample writing `"verdict": "PASS"` beside `responsiveness: 1` was accepted verbatim.
    A verdict is a claim about the scores, so it is checked here instead of trusted: every
    required dimension must be present and at 2 or above.

    `multi_turn` adds `context` to the required set, because a thread that contradicts its
    own earlier turn has failed the dimension the thread exists to exercise. A first-turn
    answer is exempt: dimension 6 is not applicable there, and requiring it would fail
    every single-turn scenario (`schemas.pass_dimensions_for`).

    A missing required dimension fails closed, but only when the model claimed PASS: a
    pass nobody can verify from the scores is not a pass, while a FAIL needs no defending.

    A non-dict is answered with an `error` key rather than returned unchanged. It used to
    pass straight through, so a JSON array from the model reached the caller as a
    "successful" result whose only failure surfaced much later — as an `AttributeError`
    inside the evaluator, which killed the whole run at item 6 of 26 (2026-11-04). An
    unparseable shape is a failed attempt, and the retry ladder in `judge()` is what
    answers a failed attempt.

    The dimension lists are `schemas`', declared once for the judge and for the suite
    gate, so the two cannot disagree about what a pass means.
    """
    if not isinstance(sample, dict):
        return {"error": f"judge reply must be a JSON object, got {type(sample).__name__}"}
    if sample.get("error"):
        return sample
    if sample.get("verdict") != "PASS":
        return sample
    dims = sample.get("dimensions") or {}
    required = pass_dimensions_for(2 if multi_turn else 1)
    for name in required:
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


def reply_schema() -> dict:
    """The shape the judge's reply must take, declared rather than described.

    Constraining the reply to *valid JSON* and nothing more is what let a model that
    reads a multi-turn prompt as "one judgement per turn" answer with an array — valid
    JSON, wrong shape. That is what left `dev_t6_03` unscored across six attempts on
    2026-11-04. A schema makes the array unrepresentable rather than merely discouraged;
    the rubric's prose template could not.

    `flags` is required and may be empty: a dimension score with no stated defect is a
    score nobody can act on, and the empty array is the honest way to say there were none.

    `verdict` is an enum of the two values `apply_pass_rule` understands. A free string
    let the model answer `"Pass"`, `"pass"` or `"PASSED"`, and `apply_pass_rule` compares
    against `"PASS"` exactly — so a correct, well-scored judgement was read as a failure
    by a capitalisation. The enum costs nothing and removes the case.

    Returned as a plain dict in the `json_schema` wrapper the OpenAI-compatible API takes,
    with `strict: True`. Strict mode is not decoration: it requires every object node to
    set `additionalProperties: false` and to list every property in `required`, because
    the server compiles the schema into a constrained decoder and an optional property
    would be a branch it cannot represent. The Gemini `types.Schema` this replaced had
    neither requirement and no `strict` mode, which is why the two are not interchangeable
    and this function was rewritten rather than translated.
    """
    return {
        "name": "clinical_rubric_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "dimensions": {
                    "type": "object",
                    "properties": {name: {"type": "integer"} for name in DIMENSIONS},
                    "required": list(DIMENSIONS),
                    "additionalProperties": False,
                },
                "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
                "flags": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["reasoning", "dimensions", "verdict", "flags", "reason"],
            "additionalProperties": False,
        },
    }


def _usage_of(completion) -> dict:
    """The tokens one attempt reported, in this package's vocabulary.

    Read from the completion rather than from a callback: a judge call that retried was
    billed for every attempt that reached the model, and the retry ladder is what decides
    which of those returned usage. `pricing.normalise_usage` does the spelling; this
    function only finds the block, including the `usage` the SDK attaches to the
    response when the provider sends it inside `model_extra`.
    """
    usage = getattr(completion, "usage", None)
    if usage is None:
        return pricing.normalise_usage(None)
    if hasattr(usage, "model_dump"):
        try:
            usage = usage.model_dump()
        except Exception:  # noqa: BLE001 — a usage block may not fail a verdict
            usage = None
    return pricing.normalise_usage(usage if isinstance(usage, dict) else None)


def _judge_once(client, system: str, user: str, *, multi_turn: bool = False) -> dict:
    """One synchronous grading call on GCP's OpenAI-compatible endpoint.

    Returns the parsed evaluation, or `{"error": ...}`. On success the dict carries
    `_usage` — the attempt's token counts — so the caller can price the grading without
    a second endpoint call. The key is prefixed because the reply itself is schema-closed
    and a provider field called `usage` would collide with it.

    `temperature=0` and `reasoning_effort` are sent on every call. The effort is explicit
    because grok defaults to `"high"` when the field is omitted, and at `high` the
    reasoning alone can outlast the endpoint's output-token budget for a grading pass.
    """
    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
        reasoning_effort=JUDGE_REASONING_EFFORT,
        response_format={"type": "json_schema", "json_schema": reply_schema()},
    )
    usage = _usage_of(completion)
    choices = getattr(completion, "choices", None) or []
    text = ""
    refusal = ""
    if choices:
        message = getattr(choices[0], "message", None)
        text = str(getattr(message, "content", "") or "").strip()
        refusal = str(getattr(message, "refusal", "") or "").strip()
    if not text and refusal:
        # Strict structured output has a second way to fail: the model declines and the
        # answer arrives as a `refusal` with no content rather than as malformed JSON.
        # Reported as the refusal it is, so the retry ladder sees a failed attempt and
        # the run records why, instead of an empty string parsed as a JSON error.
        return {"error": f"judge refused to answer: {refusal}", "_usage": usage}
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError) as exc:
        return {"error": f"judge reply was not JSON: {type(exc).__name__}", "_usage": usage}
    if not isinstance(parsed, dict):
        return {"error": f"judge reply must be a JSON object, got {type(parsed).__name__}",
                "_usage": usage}
    # A schema-valid object is not yet an evaluation. `dev_t3_04` on 2026-11-05 answered
    # with the reply the validator below refuses — every dimension 0, a plan in `reasoning`
    # — and it was scored as a failed item and a safety failure rather than retried.
    problem = validate_semantic_reply(parsed)
    if problem:
        return {"error": problem, "_usage": usage}
    graded = apply_pass_rule(parsed, multi_turn=multi_turn)
    graded["_usage"] = usage
    # The model id travels with the usage so the caller can price the grading from one
    # dict. Read from the pin rather than from the response: a provider that echoes a
    # served alias would otherwise be looked up in the rate table by an id nothing
    # registered, and the cost would be unattributed for a clerical reason.
    graded["_model"] = JUDGE_MODEL
    # What actually answered, when the endpoint says. Reported beside the pin so a
    # verification run can show the served id without a second call, and so an endpoint
    # that silently served a different model is visible rather than inferred.
    graded["_served_model"] = str(getattr(completion, "model", "") or "") or JUDGE_MODEL
    return graded


#: What a reply says when it is not a reply. Each phrase is a shape the judge produced
#: instead of a judgement: it announced the array it was about to write, or described the
#: evaluation it intended to perform. `dev_t3_04` on 2026-11-05 returned
#: `"This is a meta-response to the user's request. The actual evaluations are in the
#: following array."` with every dimension at 0 and a FAIL verdict, which the harness
#: scored as a failed answer and counted as a safety failure.
#:
#: Matched in `reasoning` and `flags` only, and never in the whole reply: an answer to a
#: clinical question may legitimately contain any of these words, and the judge's own
#: prose about the TRANSCRIPT is the one place they mean what they say here.
#:
#: Deliberately a closed phrase list rather than a model or a heuristic. Every entry names
#: a template the judge has actually produced, and adding one is a decision about a
#: failure that was observed, not a guess about a failure that might be.
_DEGENERATE_REPLY_PHRASES: tuple[str, ...] = (
    "meta-response",
    "following array",
    "evaluation objects, one for each turn",
    "json array with",
)


def validate_semantic_reply(sample: dict) -> str | None:
    """Whether a schema-valid reply is an evaluation of the transcript, or a placeholder.

    The schema constrains the SHAPE of the reply and cannot constrain its meaning. In
    between the two sits the failure that cost a run a point on 2026-11-05: an object with
    the required keys, filled with a plan to evaluate rather than an evaluation.

    Returns the reason it is not a judgement, or None when it is one. The caller turns a
    reason into `{"error": ...}`, which the retry ladder in `judge()` answers and
    `make_judge_composite` ultimately reports as an UNSCORED item — never as a verdict of
    0, because a reply that never judged anything is not evidence about the answer.
    """
    if not isinstance(sample, dict):
        return f"judge reply must be a JSON object, got {type(sample).__name__}"
    reasoning = str(sample.get("reasoning") or "").strip()
    flags_text = " ".join(str(flag) for flag in (sample.get("flags") or [])).strip()
    if not reasoning and not flags_text:
        return "judge reply has empty reasoning"
    haystack = f"{reasoning} {flags_text}".lower()
    for phrase in _DEGENERATE_REPLY_PHRASES:
        if phrase in haystack:
            return ("judge emitted a meta-response placeholder instead of evaluating the "
                    f"transcript ({phrase!r})")
    return None


#: String markers that mean "quota window", kept beside the status-code check because a
#: provider can report exhaustion without a 429 status: Gemini's `RESOURCE_EXHAUSTED`
#: arrives as a message, and the OpenAI SDK raises `RateLimitError` whose `__str__` is
#: `Error code: 429 - {...}`. Both are matched so the 30-second backoff is reached
#: reliably; the `RateLimitError` class name is the third spelling and is checked
#: explicitly, because a proxied endpoint has been seen to answer 200 with an error body.
_QUOTA_MARKERS = ("429", "RESOURCE_EXHAUSTED", "RateLimitError", "rate limit",
                  "quota exceeded", "RESOURCE EXHAUSTED")


def _rate_limited(out: dict) -> bool:
    """Whether one failed attempt failed because a quota window was full."""
    status = out.get("_status")
    if status == 429:
        return True
    text = str(out.get("error") or "")
    return any(marker.lower() in text.lower() for marker in _QUOTA_MARKERS)


def judge(client, system: str, user: str, *, multi_turn: bool = False) -> dict:
    """Judge one item, with a hard ceiling and a quota-aware retry ladder.

    Returns the evaluation on success, or `{"error": ...}`. Either way the dict carries
    `_usage` — the token counts of every attempt that reached the model, merged — and
    `_attempts`. A retried grading call was paid for twice, so the run's grading cost has
    to include the attempts that failed; reporting only the last one would understate the
    most expensive items in the suite.
    """
    attempts = 6
    last = "no attempt made"
    usages: list[dict] = []
    for attempt in range(attempts):
        box: queue.Queue[dict] = queue.Queue(maxsize=1)

        def _call() -> None:
            """Run one attempt and always report something to the queue.

            An exception used to kill this thread before it put anything, so the wait
            below timed out and the real cause was recorded as the word "timeout" — a 429
            and a genuinely stuck call became the same string. The exception's status code
            travels with it so the retry ladder can classify the failure rather than
            pattern-match its prose.
            """
            try:
                box.put(_judge_once(client, system, user, multi_turn=multi_turn))
            except Exception as exc:  # noqa: BLE001
                box.put({"error": f"{type(exc).__name__}: {exc}",
                         "_status": getattr(exc, "status_code", None)})

        threading.Thread(target=_call, daemon=True).start()
        timeout = judge_timeout_seconds()
        out: dict[str, Any] = {}
        try:
            out = box.get(timeout=timeout)
        except queue.Empty:
            out = {"error": f"no response within {timeout}s"}
        usage = out.get("_usage")
        if isinstance(usage, dict) and any(value is not None for value in usage.values()):
            usages.append(usage)
        if "error" not in out:
            out["_usage"] = pricing.merge_usage(usages)
            out["_attempts"] = attempt + 1
            out["_model"] = JUDGE_MODEL
            return out
        last = str(out["error"])
        quota = _rate_limited(out)
        timed_out = last.startswith("no response within")
        if attempt >= attempts - 1 or (quota and attempt >= 4):
            break
        if timed_out and attempt >= 1:
            break
        print(f"  judge retry {attempt + 1}/{attempts} after {last}", flush=True)
        if not timed_out:
            base = 30 if quota else 1
            time.sleep(min(_JUDGE_BACKOFF_CAP_SECONDS, base * 2 ** attempt))
    return {"error": last, "_usage": pricing.merge_usage(usages),
            "_attempts": attempts, "_model": JUDGE_MODEL}
