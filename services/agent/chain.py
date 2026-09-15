"""The chain's identity: which model, which code, and what each run did.

Google's requirement is that the prompt template, the chain definition, the tool
wiring and the model pin are versioned *together* as one artifact, and that every
execution records which combination served it. This module is where that identity
lives, so "which prompt and which model produced this answer?" has one place to
be read from rather than four.

  - `MODEL_ID` is the chain's model. Pinned in code (`services/mcp/config.py`),
    never read from the environment, because an environment default means a
    deploy can change the model with no commit anywhere.
  - `CODE_REVISION` is the identity of the running code, taken from the
    deployment rather than maintained by hand.
  - `record_execution()` emits one structured line per execution, success or
    failure.

The identity is deliberately *not* a version number typed by hand. An earlier
version of this module carried a `CHAIN_REVISION = "1"` that whoever edited the
prompt was expected to bump — a claim about the code rather than a fact derived
from it, and one that fails silently: edit the prompt, forget the bump, and two
different behaviours report the same revision for ever, so a bad answer can no
longer be traced to the prompt that produced it. The deployment already produces
an exact identifier for the prompt, the tool wiring and the model together, so
that is what is reported here (`resolve_code_revision`).
"""

import json
import logging
import os
from collections.abc import Mapping

from services.mcp.config import GEMINI_MODEL

logger = logging.getLogger(__name__)

# The chain's model. Imported rather than redefined so there is exactly one
# place the string lives; this name is what the rest of the chain reads.
MODEL_ID = GEMINI_MODEL


def resolve_code_revision(env: Mapping[str, str] | None = None) -> str:
    """The identifier of the code that is running.

    Taken from the deployment rather than typed by hand, so it cannot go stale:

      1. `CODE_REVISION` — set by whoever deploys, when they can supply the
         commit they built from.
      2. `K_REVISION` — Cloud Run's own revision name, which every Cloud Run
         container carries, so a deploy that sets nothing is still identified.
         A revision maps to an image digest, and the digest to a build.
      3. `local` — an honest label for a run that is not a deployment.

    An empty or unexpanded value counts as absent: a `$COMMIT_SHA` that never got
    substituted is not an identity, and reporting it as one would be the same
    class of mistake this replaced.
    """
    values = os.environ if env is None else env
    for name in ("CODE_REVISION", "K_REVISION"):
        value = (values.get(name) or "").strip()
        if value and not value.startswith("$"):
            return value
    return "local"


CODE_REVISION = resolve_code_revision()

# The fields the record carries. Named here so a test can assert the shape
# without duplicating the list.
RECORD_FIELDS = (
    "event",
    "trace",
    "code_revision",
    "model",
    "outcome",
    "question_chars",
    "duration_ms",
    "stages",
    "tool_calls",
    "guardrail_flags",
)


def record_execution(
    *,
    trace: str,
    question: str,
    stages: list[dict],
    duration_ms: int,
    outcome: str,
    tool_calls=(),
    guardrail_flags: int = 0,
    error: str | None = None,
    model: str = MODEL_ID,
    code_revision: str = CODE_REVISION,
) -> dict:
    """Log one execution as a single structured line, and return the record.

    Emitted for failures as well as successes: "every execution logs its inputs,
    its outputs, the intermediate state of each step, and the chain configuration
    used" is the requirement, and a record that only exists for answers that
    worked cannot explain the ones that did not.

    The question and answer *text* are deliberately absent. Their shape is
    recorded instead — length, which tools ran, what the guardrails flagged —
    because the text is patient-derived and a log store has a different privacy
    regime from the repository. If the answer text is ever needed for an incident
    review, that is a decision to take deliberately, with retention, not a
    side effect of adding a log line.

    JSON on one line so it is queryable by whatever the observability layer
    eventually picks; today it lands in Cloud Logging with the rest of stdout.
    """
    record = {
        "event": "agent_execution",
        "trace": trace,
        "code_revision": code_revision,
        "model": model,
        "outcome": outcome,
        "question_chars": len(question or ""),
        "duration_ms": duration_ms,
        "stages": list(stages),
        "tool_calls": [name for name in tool_calls],
        "guardrail_flags": guardrail_flags,
    }
    if error:
        record["error"] = error

    logger.info(json.dumps(record, separators=(",", ":"), default=str))
    return record
