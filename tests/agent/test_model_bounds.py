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
    GEMINI_THINKING_BUDGET,
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


def test_thinking_is_capped_rather_than_left_to_the_allowance():
    llm = graph._build_llm(graph.MODEL_ID)
    assert llm.thinking_budget == GEMINI_THINKING_BUDGET
    assert 0 < GEMINI_THINKING_BUDGET


def test_the_cap_still_leaves_the_answer_room():
    # The allowance covers thinking and the answer together, so a cap that took
    # all of it would guarantee the empty-answer failure this exists to prevent.
    assert GEMINI_MAX_OUTPUT_TOKENS - GEMINI_THINKING_BUDGET >= 512
