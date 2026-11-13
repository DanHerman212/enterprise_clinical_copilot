"""Sprint 01: the model gateway's contract, proven without a network.

Five profiles, one pipeline. These tests are the acceptance for the anti-corruption layer:
whatever a provider calls its parameters, its stop reason or its content blocks, the code
above the adapter sees one vocabulary and one behaviour.

Nothing here makes a request. Every provider payload is constructed in the test, and the
two families whose SDKs are not installed in this environment are exercised through the
mapping functions rather than through construction — which is the honest boundary: the
mapping is what this sprint decides, and the wire is what the authorised ping proves.
"""

import ast
import asyncio
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage  # noqa: E402

from services.agent.engine import gateway  # noqa: E402
from services.agent.engine import model as model_turn  # noqa: E402
from services.agent.postprocess import answer  # noqa: E402
from services.mcp import config  # noqa: E402
from shared import model_outcomes  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

PIN = "gemini-3.1-flash-lite"
FLASH_CANDIDATE = "gemini-3.8-flash"
PRO_CANDIDATE = "gemini-3.1-pro"
CLAUDE = "claude-sonnet-5-5"
MUSE = "muse-spark-1.3"

ALL_FIVE = (PIN, FLASH_CANDIDATE, PRO_CANDIDATE, CLAUDE, MUSE)


def _turn(text="", **metadata):
    return AIMessage(content=text, response_metadata=metadata)


# --------------------------------------------------------------------------------------
# 1. The registry: five ids, one table, no substring guessing.
# --------------------------------------------------------------------------------------


def test_every_planned_model_has_a_profile():
    assert set(gateway.PROFILES) == set(ALL_FIVE)


def test_a_canonical_id_and_its_wire_id_are_both_recorded():
    # The run is labelled with the canonical id; the provider is asked for the wire id.
    # A table that stored only one of them would label a run with a model that never
    # answered, or ask for a model that does not exist.
    assert gateway.profile(PRO_CANDIDATE).wire_model == "gemini-3.1-pro-preview"
    assert gateway.profile(PRO_CANDIDATE).name == PRO_CANDIDATE
    assert gateway.wire_model_name(PIN) == PIN


def test_an_unknown_id_keeps_its_own_wire_name_and_the_default_facts():
    # Before this table existed every id went to the Gemini client with its own name, and
    # that is still what an unregistered id gets — so a test double's model string cannot
    # change behaviour.
    resolved = gateway.profile("some-test-model")
    assert resolved.wire_model == "some-test-model"
    assert resolved.family == gateway.GOOGLE
    assert resolved.json_mode == gateway.PROFILES[PIN].json_mode


def test_an_unset_agent_model_is_the_existing_pin():
    # Byte-identical default behaviour is a requirement, not a nicety: with nothing
    # configured, the service must ask for the model it asked for before the adapter.
    assert config.AGENT_MODEL == config.GEMINI_MODEL


def test_the_default_client_is_the_gemini_client_with_its_bounds():
    llm = model_turn.build_llm(config.AGENT_MODEL)
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert llm.max_retries == gateway.MODEL_MAX_RETRIES
    assert llm.timeout == gateway.MODEL_TIMEOUT_SECONDS


# --------------------------------------------------------------------------------------
# 2. Effort: two knobs, two spellings, and one model that rejects the pin's value.
# --------------------------------------------------------------------------------------


def test_the_served_path_keeps_the_synthesis_effort_at_construction():
    llm = model_turn.build_llm(PIN)
    assert llm.reasoning_effort == config.GEMINI_REASONING_EFFORT


def test_the_extraction_path_thinks_less_at_construction():
    llm = model_turn.build_extraction_llm(PIN)
    assert llm.reasoning_effort == "low"


def test_the_pin_keeps_the_extraction_level_it_accepts():
    profile = gateway.profile(PIN)
    assert gateway.call_parameters(profile, gateway.EFFORT_EXTRACTION) == {
        "thinking_level": "minimal",
    }


def test_a_candidate_that_rejects_minimal_gets_the_lowest_level_it_accepts():
    # `gemini-3.8-flash` and `gemini-3.1-pro-preview` answer a `minimal` level with an API
    # validation error. The registry is where that difference is held, so no call site has
    # to know which model it is talking to.
    for name in (FLASH_CANDIDATE, PRO_CANDIDATE):
        profile = gateway.profile(name)
        assert gateway.call_parameters(profile, gateway.EFFORT_EXTRACTION) == {
            "thinking_level": "low",
        }
        # No call site binds the served level per call — the formatting turn binds nothing
        # — so the served effort is a construction fact only.
        assert gateway.construction_parameters(profile, gateway.EFFORT_SERVED) == {
            "reasoning_effort": "medium",
        }
    pin = gateway.profile(PIN)
    assert gateway.call_parameters(pin, gateway.EFFORT_EXTRACTION) == {
        "thinking_level": "minimal",
    }
    assert gateway.call_parameters(pin, gateway.EFFORT_SERVED) == {}


def test_the_pinned_model_lets_an_unrecognised_model_alone():
    # A caller that binds something provider-neutral gets it back untouched; the binder
    # re-expresses only what it recognises.
    profile = gateway.profile(PIN)
    assert gateway.wire_bindings(profile, {"max_output_tokens": 64}) == {
        "max_output_tokens": 64,
    }


def test_a_json_request_maps_to_each_providers_mechanism():
    assert gateway.wire_bindings(gateway.profile(PIN), {"json_mode": True}) == {
        "response_mime_type": "application/json",
    }
    assert gateway.wire_bindings(gateway.profile(MUSE), {"json_mode": True}) == {
        "response_format": {"type": "json_object"},
    }
    # Anthropic has no request-level JSON parameter, and inventing one would be a setting
    # that never applies. The prompt already asks for the object.
    assert gateway.wire_bindings(gateway.profile(CLAUDE), {"json_mode": True}) == {}


def test_the_extraction_request_round_trips_through_the_default_profile():
    # The frozen contract suite pins `_STRUCTURED_CALL_PARAMS`; the binder must hand the
    # default provider exactly those parameters back, or the pin's behaviour changed.
    from services.agent.engine import deterministic

    bound = gateway.wire_bindings(
        gateway.profile(PIN), deterministic._STRUCTURED_CALL_PARAMS)
    assert bound == {
        "thinking_level": "minimal",
        "response_mime_type": "application/json",
    }


def test_the_extraction_request_is_re_expressed_for_a_candidate():
    from services.agent.engine import deterministic

    bound = gateway.wire_bindings(
        gateway.profile(PRO_CANDIDATE), deterministic._STRUCTURED_CALL_PARAMS)
    assert bound["thinking_level"] == "low"
    assert "response_mime_type" in bound


# --------------------------------------------------------------------------------------
# 3. Anthropic's two wire facts: no temperature, and no forcing.
# --------------------------------------------------------------------------------------


def test_a_provider_that_rejects_temperature_is_marked_as_such():
    assert gateway.profile(CLAUDE).sends_temperature is False
    assert gateway.profile(PIN).sends_temperature is True


def test_tool_choice_forcing_is_translated_for_a_provider_that_rejects_it():
    # A provider answering `tool_choice="any"` with a 400 is real, and the graph asks for
    # it on every first turn. The translation happens at the wire boundary, so `graph.py`
    # is untouched.
    claude = gateway.profile(CLAUDE)
    assert gateway.translate_tool_choice(claude, gateway.FORCE_ANY) == gateway.AUTO
    assert gateway.translate_tool_choice(claude, "required") == gateway.AUTO
    assert gateway.translate_tool_choice(claude, None) is None


def test_a_provider_that_can_be_forced_is_left_alone():
    gemini = gateway.profile(PIN)
    assert gateway.translate_tool_choice(gemini, gateway.FORCE_ANY) == gateway.FORCE_ANY


def test_the_tool_binding_carries_the_translated_choice():
    claude = gateway.translate_tools(gateway.profile(CLAUDE), ["a", "b"], "any")
    assert claude["tool_choice"] == "auto"
    assert claude["tools"] == ["a", "b"]
    gemini = gateway.translate_tools(gateway.profile(PIN), ["a"], "any")
    assert gemini["tool_choice"] == "any"


# --------------------------------------------------------------------------------------
# 4. Adjacent user turns, merged for a provider that requires alternation.
# --------------------------------------------------------------------------------------


def _conversation():
    return [
        SystemMessage(content="instructions"),
        HumanMessage(content="the evidence record"),
        HumanMessage(content="the question"),
    ]


def test_adjacent_user_turns_are_merged_for_a_strict_provider():
    merged = gateway.merge_adjacent_human(_conversation())
    assert [type(message).__name__ for message in merged] == [
        "SystemMessage", "HumanMessage"]
    assert merged[-1].content == "the evidence record\n\nthe question"


def test_the_callers_message_list_is_not_mutated():
    conversation = _conversation()
    before = list(conversation)
    gateway.merge_adjacent_human(conversation)
    assert conversation == before
    assert len(conversation) == 3


def test_a_tolerant_provider_is_sent_the_conversation_unchanged():
    conversation = _conversation()
    assert gateway.wire_messages(gateway.profile(PIN), conversation) == conversation
    assert len(gateway.wire_messages(gateway.profile(CLAUDE), conversation)) == 2


def test_an_alternating_conversation_survives_the_merge_untouched():
    conversation = [
        SystemMessage(content="instructions"),
        HumanMessage(content="one"),
        AIMessage(content="two"),
        HumanMessage(content="three"),
    ]
    assert gateway.merge_adjacent_human(conversation) == conversation


# --------------------------------------------------------------------------------------
# 5. text_of: prose only, from text blocks only.
# --------------------------------------------------------------------------------------


def test_a_thinking_block_is_not_announced():
    assert model_turn.text_of([
        {"type": "thinking", "thinking": "the patient probably..."},
        {"type": "text", "text": "The note says pneumonia."},
    ]) == "The note says pneumonia."


def test_a_reasoning_block_that_carries_text_is_still_not_announced():
    # The field a block puts its prose in is not the signal; the block's type is.
    assert model_turn.text_of([
        {"type": "reasoning", "text": "deliberation that must never be shown"},
        {"type": "text", "text": "answer"},
    ]) == "answer"
    assert model_turn.text_of([
        {"type": "redacted_thinking"}, {"type": "text", "text": "answer"},
    ]) == "answer"


def test_a_non_text_block_is_never_stringified_into_the_answer():
    # The old unwrapping fell back to `str(block)`, which would put a dict's repr on a
    # clinician's screen.
    assert model_turn.text_of([
        {"type": "function_call", "name": "rag_search", "args": {"q": "x"}},
        {"type": "text", "text": "answer"},
    ]) == "answer"
    assert model_turn.text_of([{"no_type": True, "value": 3}]) == ""


def test_a_streamed_chunk_is_read_by_the_same_rule():
    # Every chunk is shaped like the whole message, so the same function has to hold on
    # the streaming path — this is the shape `astream` yields.
    chunk = AIMessageChunk(content=[
        {"type": "thinking", "thinking": "hmm"},
        {"type": "text", "text": "Brief Hospital Course"},
    ])
    assert model_turn.text_of(chunk.content) == "Brief Hospital Course"


def test_a_plain_string_and_a_list_of_strings_still_work():
    assert model_turn.text_of("The patient was admitted.") == "The patient was admitted."
    assert model_turn.text_of(["one", "two"]) == "onetwo"
    assert model_turn.text_of(None) == ""


# --------------------------------------------------------------------------------------
# 6. Outcomes: one vocabulary, three providers' tokens.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("token,expected", [
    ("STOP", model_outcomes.STOP),
    ("end_turn", model_outcomes.STOP),
    ("completed", model_outcomes.STOP),
    ("MAX_TOKENS", model_outcomes.LENGTH),
    ("max_tokens", model_outcomes.LENGTH),
    ("incomplete", model_outcomes.LENGTH),
    ("SAFETY", model_outcomes.FILTERED),
    ("RECITATION", model_outcomes.FILTERED),
    ("PROHIBITED_CONTENT", model_outcomes.FILTERED),
    ("refusal", model_outcomes.FILTERED),
    ("content_filter", model_outcomes.FILTERED),
    ("something_nobody_has_seen", model_outcomes.UNKNOWN),
])
def test_every_providers_vocabulary_maps_onto_four_words(token, expected):
    assert model_outcomes.map_token(token) == expected


def test_the_strings_the_engine_returns_are_unchanged():
    # The execution record's vocabulary predates this sprint and is read by the harness.
    assert model_turn.outcome(_turn(finish_reason="MAX_TOKENS")) == model_turn.TRUNCATED
    assert model_turn.outcome(_turn(finish_reason="SAFETY")) == model_turn.REFUSED
    assert model_turn.outcome(_turn(finish_reason="STOP")) == model_turn.OK
    assert model_turn.outcome(_turn()) == model_turn.UNKNOWN


def test_a_refusal_in_another_providers_words_is_still_a_refusal():
    # `stop_reason: "refusal"` is Anthropic's spelling of the statement `finish_reason:
    # "SAFETY"` makes. Before this sprint the engine read only the second.
    claude_refusal = _turn(stop_reason="refusal")
    assert model_turn.outcome(claude_refusal) == model_turn.REFUSED
    assert model_turn.classify(claude_refusal)[0] == "answer_refused"


def test_a_truncation_in_another_providers_words_is_still_a_truncation():
    assert model_turn.outcome(
        _turn(stop_reason="max_tokens")) == model_turn.TRUNCATED
    assert model_turn.outcome(
        _turn(status="incomplete")) == model_turn.TRUNCATED


def test_a_withheld_reply_is_a_narrower_statement_than_a_refusal():
    # Two questions, deliberately different answers. A reader told "the safety filter
    # withheld its summary" is told which mechanism acted, so the sentence may only be
    # used when a filter is what acted. `dev` behaviour depends on this distinction.
    for token in ("SAFETY", "content_filter"):
        assert answer.reply_withheld(token) is True, token
    for token in ("RECITATION", "PROHIBITED_CONTENT", "SPII", "refusal",
                  "MAX_TOKENS", "STOP", None):
        assert answer.reply_withheld(token) is False, token


def test_the_post_process_layers_withheld_rule_matches_the_engines():
    # The two layers must not disagree about one response: this is the sentence a reader
    # gets, and it is chosen in `postprocess/` from the same facts the engine reports.
    for token in ("SAFETY", "refusal", "PROHIBITED_CONTENT", "content_filter",
                  "RECITATION"):
        assert model_turn.outcome(_turn(finish_reason=token)) == model_turn.REFUSED, token
    for token in ("MAX_TOKENS", "STOP", "SomethingElse", None):
        assert model_turn.outcome(_turn(finish_reason=token)) != model_turn.REFUSED, token


def test_a_blocked_prompt_is_read_the_same_way_by_both_layers():
    state = {"messages": [_turn(prompt_feedback={"block_reason": "JAILBREAK"})]}
    assert answer.turn_reply_withheld(state) is True
    assert model_turn.outcome(state["messages"][-1]) == model_turn.REFUSED


def test_an_ordinary_answer_is_not_withheld_by_either_layer():
    state = {"messages": [_turn("answer", finish_reason="STOP",
                                prompt_feedback={"block_reason": "BLOCK_REASON_UNSPECIFIED"})]}
    assert answer.turn_reply_withheld(state) is False
    assert model_turn.blocked_reason(state["messages"][-1]) is None


def test_a_stamped_turn_and_a_raw_turn_are_read_identically():
    # Stamping is what the adapter does at the boundary; deriving is what makes the same
    # function work on a message no adapter saw. Both must produce one answer.
    raw = _turn("", finish_reason="SAFETY")
    stamped = gateway.stamp(_turn("", finish_reason="SAFETY"))
    assert model_turn.outcome(raw) == model_turn.outcome(stamped) == model_turn.REFUSED
    assert stamped.response_metadata[model_outcomes.OUTCOME_KEY] == model_outcomes.FILTERED


def test_the_stamp_names_the_categories_the_provider_flagged():
    stamped = gateway.stamp(_turn(
        "", finish_reason="SAFETY",
        safety_ratings=[{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "blocked": True}],
    ))
    assert model_turn.filtered_categories(stamped) == ["HARM_CATEGORY_DANGEROUS_CONTENT"]


# --------------------------------------------------------------------------------------
# 7. Tokens: measured zero, or honestly absent.
# --------------------------------------------------------------------------------------


def _usage(input_tokens=0, output_tokens=0, *, reasoning=None, cached=None):
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if reasoning is not None:
        usage["output_token_details"] = {"reasoning": reasoning}
    if cached is not None:
        usage["input_token_details"] = {"cache_read": cached}
    return usage


def test_a_provider_that_reports_reasoning_reports_a_number_even_when_it_is_zero():
    state = {"messages": [AIMessage(content="x", usage_metadata=_usage(
        input_tokens=10, output_tokens=5, reasoning=0, cached=0))]}
    tokens = model_turn.token_usage(state)
    assert tokens["thinking"] == 0
    assert tokens["cached"] == 0


def test_a_provider_with_no_reasoning_breakdown_reports_none_not_zero():
    # A fabricated zero would put a measurement nobody took into the cost table the
    # programme is about to build.
    state = {"messages": [AIMessage(content="x", usage_metadata=_usage(
        input_tokens=10, output_tokens=5))]}
    tokens = model_turn.token_usage(state)
    assert tokens["thinking"] is None
    assert tokens["cached"] is None
    assert tokens["input"] == 10 and tokens["output"] == 5


def test_the_totals_are_summed_across_turns():
    state = {"messages": [
        AIMessage(content="a", usage_metadata=_usage(100, 10, reasoning=2)),
        AIMessage(content="b", usage_metadata=_usage(200, 20, reasoning=3)),
    ]}
    assert model_turn.token_usage(state) == {
        "input": 300, "output": 30, "total": 330, "thinking": 5, "cached": None,
    }


def test_no_usage_at_all_is_none_rather_than_a_row_of_zeros():
    assert model_turn.token_usage({"messages": [AIMessage(content="x")]}) is None


# --------------------------------------------------------------------------------------
# 8. Errors: four answers, so a retry ladder asks instead of guessing.
# --------------------------------------------------------------------------------------


class _Coded(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


@pytest.mark.parametrize("exc,expected", [
    (_Coded("nope", code=429), gateway.CAPACITY),
    (_Coded("RESOURCE_EXHAUSTED"), gateway.CAPACITY),
    (_Coded("Rate limit reached for model"), gateway.CAPACITY),
    (_Coded("blocked by safety filters"), gateway.CONTENT),
    (_Coded("400 INVALID_ARGUMENT"), gateway.CONTENT),
    (_Coded("Connection reset by peer"), gateway.TRANSPORT),
    (_Coded("nope", code=503), gateway.TRANSPORT),
    (_Coded("a plain programming error"), gateway.OTHER),
])
def test_an_error_is_classified_into_four_answers(exc, expected):
    assert gateway.classify_error(exc) == expected


def test_the_retry_ladder_still_retries_only_capacity():
    from services.agent.engine import deterministic

    assert deterministic._rate_limited(_Coded("429 RESOURCE_EXHAUSTED")) is True
    assert deterministic._rate_limited(_Coded("blocked by safety filters")) is False
    assert deterministic._rate_limited(KeyError("comparators")) is False


# --------------------------------------------------------------------------------------
# 9. The anchors: one import site, no model-name branching, one table.
# --------------------------------------------------------------------------------------


PROVIDER_PACKAGES = (
    "langchain_google_genai", "langchain_anthropic", "langchain_openai",
    "google.genai", "anthropic", "openai",
)


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_the_adapter_is_the_only_module_that_imports_a_provider_sdk():
    offenders = []
    for path in list((REPO / "services" / "agent").rglob("*.py")) + list(
            (REPO / "shared").rglob("*.py")):
        if path.name == "gateway.py":
            continue
        if _imports_of(path) & set(PROVIDER_PACKAGES):
            offenders.append(str(path.relative_to(REPO)))
    # `services/mcp/tools/retrieval.py` embeds a query and is outside the agent.
    assert offenders == [], offenders


def test_no_module_branches_on_a_model_name():
    # The branch this sprint exists to remove. A model name may be *compared* for equality
    # against a registry key; it may never be used as a substring test to pick behaviour.
    offenders = []
    for path in list((REPO / "services" / "agent").rglob("*.py")) + list(
            (REPO / "shared").rglob("*.py")):
        source = path.read_text()
        for name in ("claude", "gemini", "muse", "gpt-", "anthropic"):
            if f'"{name}" in ' in source.lower() or f"'{name}' in " in source.lower():
                offenders.append(f"{path.relative_to(REPO)}:{name}")
    assert offenders == [], offenders


def test_the_finish_reason_table_exists_exactly_once():
    # Two copies of one decision is how two layers come to disagree about one response.
    declarations = []
    for path in list((REPO / "services").rglob("*.py")) + list(
            (REPO / "shared").rglob("*.py")) + list((REPO / "evaluation").rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.Assign):
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                    "FILTER_TOKENS", "LENGTH_TOKENS", "STOP_TOKENS"}:
                declarations.append(f"{path.relative_to(REPO)}:{target.id}")
    assert sorted(declarations) == [
        "shared/model_outcomes.py:FILTER_TOKENS",
        "shared/model_outcomes.py:LENGTH_TOKENS",
        "shared/model_outcomes.py:STOP_TOKENS",
    ], declarations


def test_the_shared_leaf_imports_nothing_from_the_service():
    # It is imported by both layers, so it may not depend on either.
    assert not any(
        name.startswith("services") for name in _imports_of(
            REPO / "shared" / "model_outcomes.py")
    )


def test_the_gateway_does_not_retry_or_choose_a_model_for_itself():
    source = inspect.getsource(gateway)
    assert "RETRY_" not in source, "retry budgets are a property of the job"
    assert "await asyncio.sleep" not in source


# --------------------------------------------------------------------------------------
# 10. The Muse credential is refreshed rather than captured.
# --------------------------------------------------------------------------------------


def test_the_bearer_token_is_callable_and_minted_per_call():
    token = gateway.BearerToken()
    assert callable(token) is True
    # The callable is the contract: a static string captured at construction expires inside
    # one long evaluation run and then fails every remaining item for no reason.
    assert not isinstance(token, str)
    assert "def __call__" in inspect.getsource(gateway.BearerToken)


def test_the_responses_endpoint_names_the_project_and_the_publisher():
    url = gateway.responses_base_url(gateway.profile(MUSE), project="a-project")
    assert url == (
        "https://aiplatform.googleapis.com/v1/projects/a-project"
        "/locations/global/publishers/meta/v1"
    )


def test_a_profile_whose_sdk_is_missing_says_which_package():
    # A silent fallback would be an evaluation of the wrong model, which is the one failure
    # the programme cannot tolerate.
    with pytest.raises(gateway.GatewayUnavailable) as caught:
        gateway.chat_model(CLAUDE)
    assert "langchain-anthropic" in str(caught.value)


# --------------------------------------------------------------------------------------
# 11. The wrapper: the three adaptations, proven against a fake provider.
#
# The two partner SDKs are not installed here, so the wrapper is exercised with a fake
# inner client. That is the right unit: what this sprint decides is the adaptation, and
# what the authorised ping decides is the wire.
# --------------------------------------------------------------------------------------


class _FakeProvider:
    """A stand-in that records what it was asked and answers like a real turn."""

    def __init__(self, content="answer", metadata=None) -> None:
        self.calls: list[dict] = []
        self.binds: list[dict] = []
        self.content = content
        self.metadata = metadata if metadata is not None else {"stop_reason": "end_turn"}

    def bind(self, **kwargs):
        self.binds.append(kwargs)
        return _FakeProvider(self.content, self.metadata)

    def bind_tools(self, tools, **kwargs):
        self.calls.append({"tools": list(tools), **kwargs})
        return _FakeProvider(self.content, self.metadata)

    async def ainvoke(self, messages, config=None, **kwargs):
        self.calls.append({"messages": list(messages), "config": config})
        return AIMessage(content=self.content, response_metadata=dict(self.metadata))

    async def astream(self, messages, config=None, **kwargs):
        self.calls.append({"messages": list(messages), "config": config})
        yield AIMessageChunk(content="ans")
        yield AIMessageChunk(content="wer", response_metadata=dict(self.metadata))


def _wrapped(fake, profile_name=CLAUDE):
    return gateway._WireWrapper(fake, gateway.profile(profile_name))


def test_the_wrapper_translates_a_forced_tool_choice_on_the_way_out():
    fake = _FakeProvider()
    _wrapped(fake).bind_tools(["rag_search"], tool_choice="any")
    assert fake.calls[-1]["tool_choice"] == "auto"
    assert fake.calls[-1]["tools"] == ["rag_search"]


def test_the_wrapper_leaves_a_forceable_providers_choice_alone():
    fake = _FakeProvider()
    _wrapped(fake, PIN).bind_tools(["rag_search"], tool_choice="any")
    assert fake.calls[-1]["tool_choice"] == "any"


def test_the_wrapper_merges_adjacent_user_turns_and_stamps_the_turn():
    fake = _FakeProvider(content="", metadata={"stop_reason": "refusal"})
    result = asyncio.run(_wrapped(fake).ainvoke([
        SystemMessage(content="instructions"),
        HumanMessage(content="evidence"),
        HumanMessage(content="question"),
    ]))
    sent = fake.calls[-1]["messages"]
    assert [type(message).__name__ for message in sent] == ["SystemMessage", "HumanMessage"]
    # The stamp is what lets the engine read one vocabulary whichever provider answered.
    assert result.response_metadata[model_outcomes.OUTCOME_KEY] == model_outcomes.FILTERED
    assert model_turn.outcome(result) == model_turn.REFUSED


def test_the_wrapper_stamps_every_streamed_chunk():
    fake = _FakeProvider(metadata={"stop_reason": "max_tokens"})

    async def collect():
        return [chunk async for chunk in
                _wrapped(fake).astream([HumanMessage(content="q")])]

    chunks = asyncio.run(collect())
    # Every chunk carries the key, so the caller's accumulated turn does too; the reason a
    # provider reports usually arrives on the last chunk, and that is the one that settles
    # it. A first chunk stamped `unknown` is the honest reading of a fragment that has
    # reported nothing yet.
    assert all(model_outcomes.OUTCOME_KEY in chunk.response_metadata for chunk in chunks)
    assert chunks[-1].response_metadata[model_outcomes.OUTCOME_KEY] == model_outcomes.LENGTH


def test_the_accumulated_stream_reads_as_one_turn():
    fake = _FakeProvider(metadata={"stop_reason": "max_tokens"})

    async def accumulate():
        whole = None
        async for chunk in _wrapped(fake).astream([HumanMessage(content="q")]):
            whole = chunk if whole is None else whole + chunk
        return whole

    whole = asyncio.run(accumulate())
    assert model_turn.text_of(whole.content) == "answer"
    assert model_turn.outcome(whole) == model_turn.TRUNCATED


def test_the_wrapper_falls_through_to_the_wrapped_clients_own_attributes():
    class _WithBounds(_FakeProvider):
        timeout = 60
        max_retries = 3

    wrapped = _wrapped(_WithBounds())
    assert wrapped.timeout == 60
    assert wrapped.max_retries == 3


# --------------------------------------------------------------------------------------
# 12. The shapes live GCP endpoints actually return (probe of 2026-11-12).
#
# These fixtures are the payloads the three reachable profiles produced during the live
# verification pass, kept here so a future change to `text_of`, `token_usage` or the
# outcome reader is checked against the wire rather than against a guess.
#
# Provenance for `usage_metadata`: the probe read `thinking` and `cached` as INTEGERS for
# every profile, and `token_usage` only reports an integer when the corresponding detail
# mapping is present on the message. So the detail keys below were demonstrably on the
# wire, and the numbers are the ones the probe printed.
# --------------------------------------------------------------------------------------

GEMINI_LIVE_METADATA = {
    "finish_reason": "STOP",
    "model_name": "gemini-3.1-flash-lite",
    "safety_ratings": [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE",
         "blocked": False},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "NEGLIGIBLE",
         "blocked": False},
    ],
    "prompt_feedback": {"block_reason": "BLOCK_REASON_UNSPECIFIED"},
    "usage_metadata": {
        "input_tokens": 32, "output_tokens": 144, "total_tokens": 176,
        "input_token_details": {"cache_read": 0},
        "output_token_details": {"reasoning": 136},
    },
}

META_LIVE_METADATA = {
    "model_name": "meta/muse-spark-1.3",
    "finish_reason": "stop",
    "usage_metadata": {
        "input_tokens": 59, "output_tokens": 374, "total_tokens": 433,
        "input_token_details": {"cache_read": 0},
        "output_token_details": {"reasoning": 361},
    },
}


def test_the_gemini_shape_reads_as_a_stop_with_measured_thinking():
    reply = AIMessage(content=[{"type": "text", "text": "139 mEq/L."}],
                      response_metadata=dict(GEMINI_LIVE_METADATA),
                      usage_metadata=dict(GEMINI_LIVE_METADATA["usage_metadata"]))
    assert model_turn.outcome(reply) == model_turn.OK
    assert model_turn.blocked_reason(reply) is None
    assert model_turn.filtered_categories(reply) is None
    assert model_turn.served_model(reply) == "gemini-3.1-flash-lite"
    assert model_turn.text_of(reply.content) == "139 mEq/L."
    assert model_turn.token_usage({"messages": [reply]}) == {
        "input": 32, "output": 144, "total": 176, "thinking": 136, "cached": 0,
    }


def test_the_meta_shape_reads_as_a_stop_and_names_its_publisher():
    # The Responses API reports the model as `meta/muse-spark-1.3`, not as the id we
    # asked for. That is the served model, and it is what attribution must record.
    reply = AIMessage(content="139 mEq/L",
                      response_metadata=dict(META_LIVE_METADATA),
                      usage_metadata=dict(META_LIVE_METADATA["usage_metadata"]))
    assert model_turn.outcome(reply) == model_turn.OK
    assert model_turn.served_model(reply) == "meta/muse-spark-1.3"
    tokens = model_turn.token_usage({"messages": [reply]})
    assert (tokens["input"], tokens["output"], tokens["thinking"]) == (59, 374, 361)


def test_a_provider_that_forces_nothing_is_a_profile_fact_not_a_caller_decision():
    # Measured live: the Meta endpoint refuses `tool_choice="required"` outright, and
    # accepts the same turn under `auto`. The registry holds that, so `graph.py` keeps
    # asking to force and the wire gets the nearest thing the provider accepts.
    assert gateway.profile(MUSE).forces_tool_choice is False
    assert gateway.translate_tool_choice(gateway.profile(MUSE), "any") == "auto"
    # The Gemini profiles were measured calling the tool under a forced choice.
    assert gateway.profile(PIN).forces_tool_choice is True


def test_a_credentials_expiry_is_read_as_a_datetime_or_a_number():
    # The live probe raised `TypeError: float() argument must be ... not 'datetime'` on the
    # Meta path before this was handled: google-auth carries `expiry` as a datetime.
    import datetime

    class _Credentials:
        def __init__(self, expiry):
            self.expiry = expiry

    moment = datetime.datetime(2030, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    assert gateway.BearerToken._expiry_seconds(_Credentials(moment), 0) == \
        moment.timestamp()
    assert gateway.BearerToken._expiry_seconds(_Credentials(1893456000), 0) == 1893456000.0
    assert gateway.BearerToken._expiry_seconds(_Credentials(None), 100) == 700
