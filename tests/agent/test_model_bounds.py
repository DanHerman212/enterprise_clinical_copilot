"""Gap 2: one model call is bounded.

Two things were absent from the door, and neither shows up as a wrong answer:

  - no timeout on the client, which a stalled call turns into a slow answer that
    spends the whole question's deadline — and which makes the SDK's
    retry-on-timeout inert, because there is no timeout for it to fire on;
  - no thinking cap, so a reasoning model may spend the answer's allowance on
    thoughts and return HTTP 200 with empty text and no exception.

These are the acceptance for both, and for the bound nesting inside the ones that
already existed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.agent.engine import graph  # noqa: E402
from services.agent.engine import model as model_gateway  # noqa: E402
from services.mcp.config import (  # noqa: E402
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_REASONING_EFFORT,
)
from services.mcp.runtime import timeout_chain  # noqa: E402

#: The question's own wall clock, which the retry ladder has to fit inside. Read from the
#: same chain the route reads, so a changed deadline moves both rather than one.
ASK_TIMEOUT_SECONDS = timeout_chain().ask


def test_the_model_call_has_a_timeout():
    llm = graph._build_llm(graph.MODEL_ID)
    assert llm.timeout == graph.MODEL_TIMEOUT_SECONDS
    assert graph.MODEL_TIMEOUT_SECONDS > 0


def test_the_model_bound_nests_inside_the_others():
    # One model call < one tool call < one question < the site's proxy. The
    # innermost bound has to fire first, or the outer one cuts the request off
    # with nothing structured to report.
    chain = timeout_chain()
    assert chain.model < chain.tool < chain.ask < 120
    assert graph.MODEL_TIMEOUT_SECONDS == chain.model


def test_the_thinking_parameter_matches_the_pinned_model_family():
    # The two families take different parameters and each rejects the other's: the 2.5
    # family answers a level with 400 INVALID_ARGUMENT, and the 3.x family deprecates
    # the budget in favour of the level. So the pin and the parameter are one decision,
    # and this is what stops them drifting apart — swapping the model without swapping
    # this is a broken call, not a wrong answer.
    llm = graph._build_llm(graph.MODEL_ID)
    if "gemini-3" in graph.MODEL_ID:
        assert llm.reasoning_effort == GEMINI_REASONING_EFFORT
        assert llm.thinking_budget is None
    else:
        assert llm.thinking_budget
        assert llm.reasoning_effort is None


def test_the_allowance_covers_thinking_it_has_not_done_yet():
    # Thinking and the answer share one allowance, and running out of it returns HTTP
    # 200 with empty text and finish_reason=MAX_TOKENS instead of raising. A level
    # bounds effort rather than tokens, so unlike the budget this replaced the
    # allowance cannot be sized by subtraction — it has to cover thinking that has not
    # happened yet. 64 tokens was enough to reproduce the empty answer on both
    # candidates, and a probe at this pin's level spent 269 thinking tokens on one
    # question.
    assert GEMINI_MAX_OUTPUT_TOKENS >= 2048


# --- retries: stated, bounded, and read back off the client -------------------
#
# `max_retries` was a bare keyword. Nothing asserted it, so nothing failed if it was
# removed — and the failure that costs is the one it exists for: a Vertex 429 burst on
# `dev_t2_03` reaching the caller as HTTP 502 because the attempts were never made.
#
# The policy belongs to the SDK (google-api-core backs off on 429 and 503 and not on 4xx);
# what this file pins is that the bound is a NAMED constant, that it fits inside the ask
# deadline, and that the constructed client actually carries it.
#
# It pinned "at least three" until 2026-11-24, and that was the wrong half of the number.
# Three retries at a 60 s model bound is up to four minutes of ladder, and the ask deadline
# is 110 s: on 2026-11-24 the provider answered `504 DEADLINE_EXCEEDED`, the SDK retried,
# and three questions (`dev_d4_risk_01` turns 3-4, `dev_d5_fup_01` turn 4) were served an
# HTTP 504 by their own retry rather than by the failure. A bound that is only ever read in
# isolation can grow past the deadline it lives inside, so the assertion is now the
# RELATIONSHIP: worst case (`(retries + 1) * model bound`) has to leave the deadline room.


def test_the_retry_ladder_fits_inside_the_ask_deadline():
    ladder = (model_gateway.MODEL_MAX_RETRIES + 1) * model_gateway.MODEL_TIMEOUT_SECONDS
    assert model_gateway.MODEL_MAX_RETRIES >= 2, (
        "one retry is not enough to survive a transient 429 or a single 504")
    assert ladder <= ASK_TIMEOUT_SECONDS - 10, (
        f"a {ladder:.0f}s retry ladder has to finish with time to spare inside the "
        f"{ASK_TIMEOUT_SECONDS:.0f}s ask deadline, or the retry is what times the question out"
    )


def test_the_client_is_built_with_the_retry_bound():
    """Read back off the constructed client, not off the source.

    A source check passes on a keyword that is written and then shadowed; the client is
    what the SDK actually uses.
    """
    llm = model_gateway.build_llm(graph.MODEL_ID)

    assert llm.max_retries == model_gateway.MODEL_MAX_RETRIES
    assert llm.max_retries >= 2


def test_the_extraction_client_carries_the_same_bound():
    """One construction, so the two thresholds cannot diverge in the retry policy either."""
    llm = model_gateway.build_extraction_llm(graph.MODEL_ID)

    assert llm.max_retries == model_gateway.MODEL_MAX_RETRIES
