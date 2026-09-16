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

from services.agent import graph  # noqa: E402
from services.mcp.config import (  # noqa: E402
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_REASONING_EFFORT,
)
from services.mcp.runtime import timeout_chain  # noqa: E402


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
