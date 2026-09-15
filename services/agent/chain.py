"""The chain's identity: which model, which revision, and what each run did.

Google's requirement is that the prompt template, the chain definition, the tool
wiring and the model pin are versioned *together* as one artifact, and that every
execution records which combination served it. This module is where that identity
lives, so "which prompt and which model produced this answer?" has one place to
be read from rather than four.

  - `MODEL_ID` is the chain's model. Pinned in code (`services/mcp/config.py`),
    never read from the environment, because an environment default means a
    deploy can change the model with no commit anywhere.
  - `CHAIN_REVISION` is the human-readable handle for one combination of prompt,
    question templates, tool wiring and model.
  - `record_execution()` emits one structured line per execution, success or
    failure.

`CHAIN_REVISION` is maintained by hand, which is a real weakness: nothing stops
someone editing the prompt and forgetting to bump it. It is what the layer 3
document specifies, and the workaround is discipline plus review. If a bad answer
is ever traced to a stale revision, derive the revision from a digest of the four
inputs instead and the class of mistake disappears.
"""

import json
import logging

from services.mcp.config import GEMINI_MODEL

logger = logging.getLogger(__name__)

# The chain's model. Imported rather than redefined so there is exactly one
# place the string lives; this name is what the rest of the chain reads.
MODEL_ID = GEMINI_MODEL

# Bump when the prompt (`prompts.py`), the question templates (`questions.py`),
# the tool wiring (`graph.py`) or the model changes. One value, one combination.
CHAIN_REVISION = "1"

# The fields the record carries. Named here so a test can assert the shape
# without duplicating the list.
RECORD_FIELDS = (
    "event",
    "trace",
    "chain_revision",
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
    revision: str = CHAIN_REVISION,
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
        "chain_revision": revision,
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
