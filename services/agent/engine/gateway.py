"""The one place a provider's name, spelling or SDK is allowed to appear.

This module is the anti-corruption boundary between the agent and whoever serves the
model. Everything above it — the graph, the command path, the prompts, the routing table,
the post-processing layer, the presentation layer — asks for behaviour: *think hard*, *think
less*, *answer in JSON*, *these are your tools*, *may I force a tool call*. This module
turns that into whatever the provider under it accepts.

What lives here, and nowhere else:

* **Which model ids exist.** `PROFILES` names the canonical ids the programme evaluates and
  the wire id each one is actually addressed by. A canonical id and a wire id are not the
  same thing — `gemini-3.1-pro` is served as `gemini-3.1-pro-preview` — and conflating them
  is how a run ends up labelled with a model that never answered.
* **Request construction.** Endpoint, location, credentials, and every generation parameter,
  expressed once per family.
* **Semantic-to-wire mapping.** `effort` and `json_mode` are the two things the call sites
  ask for; each family spells them differently, and two of them reject the spellings the
  others require.
* **Wire-level message normalisation.** A provider that requires strict role alternation
  gets its messages alternated here, so no caller has to know.
* **Error classification.** `capacity`, `content`, `transport`, `other` — so a retry ladder
  asks a question instead of matching strings.
* **Outcome stamping.** Every turn this module produces carries the neutral outcome from
  `shared.model_outcomes`, so the layers above read one vocabulary.

Three deliberate non-goals. It does not branch on which model *answered* — a profile is
chosen by the caller and never inferred from a response. It does not retry: retry counts are
a property of the job, and they stay where they were (`deterministic.py` for the formatting
and extraction steps, `model.py` for the SDK's own bound). And it does not edit prompts.

Why a registry rather than a class hierarchy
--------------------------------------------
The pipeline is the same for every model: build a client, bind tools, stream or block,
read the outcome. What differs between families is a handful of facts — does it take a
temperature, which spellings does it accept for "think less", can it be forced to call a
tool. Those facts are data. A registry of profiles keeps them in one table a reader can
compare row by row, which is the whole point of the exercise: the models must differ in
*nothing* the evaluation measures except the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from services.mcp.runtime import timeout_chain
from shared import model_outcomes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The two bounds a model call carries, declared here because construction is here.
#
# `model.py` re-exports both, because eleven tests read them from there and because the
# execution record quotes them. They are stated in this module so that the client and its
# bounds come from one place rather than two.
# ---------------------------------------------------------------------------

#: One model call, bounded. The bound is part of the chain `timeout_chain` enforces at
#: startup — model < tool < ask < the proxy's own limit — and it is set on the client
#: rather than left absent for two reasons: a call that stalls should fail as a model
#: failure rather than spend the whole question's deadline looking like a slow answer, and
#: with no timeout at all the SDK's retry-on-timeout can never fire.
MODEL_TIMEOUT_SECONDS = timeout_chain().model

#: How many times the SDK retries one model call before the failure is ours. Three, which
#: is what the client was already given, and it is not larger because each attempt is
#: bounded by `MODEL_TIMEOUT_SECONDS`.
MODEL_MAX_RETRIES = 3

#: Every configurable content filter off, for the extraction turns only.
#:
#: The served path keeps the product's thresholds and should: that is a control on what
#: the assistant says to a clinician. Extraction is not that job. It reads a discharge note
#: and returns a condition and a few phrases; its reply is never shown to anyone, and its
#: input is the note itself — which routinely describes overdoses, toxic exposures,
#: withdrawal, trauma and psychiatric illness, because a real hospital record does.
CLINICAL_SAFETY_SETTINGS = {
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

#: The reasoning level the structured extraction calls run at on the default pin.
EXTRACTION_REASONING_EFFORT = "low"

#: What a caller means by "think hard" and "think less". Never a provider parameter.
EFFORT_SERVED = "served"
EFFORT_EXTRACTION = "extraction"

#: What a caller means by "this is a clinician's answer" and "this is an internal
#: classification". The distinction is a product control, not a wire detail.
SAFETY_SERVED = "served"
SAFETY_EXTRACTION = "extraction"

#: The three families this deployment speaks. Declared once, compared by equality.
GOOGLE = "google"
ANTHROPIC = "anthropic"
META = "meta"


class GatewayUnavailable(RuntimeError):
    """A profile was requested whose SDK is not installed in this environment.

    Raised rather than degraded: a run that silently fell back to another model would be
    an evaluation of the wrong thing, which is the one failure this programme cannot
    tolerate. The message names the package to install.
    """


class UnsupportedLevel(ValueError):
    """A thinking level was bound that no profile in this table can name.

    A configuration error, not a transport failure, so it is raised here with the value and
    the model in it rather than sent to an endpoint that will answer 400. The distinction
    matters downstream: a 400 is classified as a content refusal and is not retried, which
    is the correct treatment for a request the provider will not accept, but it arrives
    attached to whatever call happened to be in flight and reads in the record as though
    the provider disliked something about the patient text.
    """

    def __init__(self, key: str, value: Any, profile_: "ModelProfile") -> None:
        accepted = ", ".join(sorted(_GEMINI_TO_SEMANTIC.get(key, {}))) or "none"
        super().__init__(
            f"{key}={value!r} is not a level this table can name for {profile_.name}; "
            f"the levels it translates are: {accepted}. A caller states the effort "
            f"semantically (`effort`) and the profile supplies the spelling."
        )
        self.key = key
        self.value = value
        self.profile = profile_


#: Ids that used to be served and are not any more. A retired id is a configuration error
#: rather than an unknown one: `profile()` still serves a string it has never heard of with
#: the default facts, because that is what a test double needs, but an id that WAS in this
#: table and was removed on purpose must stop the deployment rather than be measured.
#:
#: `gemini-3.1-flash-lite` retired on 2026-11-13 (Sprint 04b). No traffic is served for it
#: and no run may be labelled with it, so both the deployment environment and a mistyped
#: `AGENT_MODEL` fail here, before a request is made.
RETIRED_MODELS: dict[str, str] = {
    "gemini-3.1-flash-lite": (
        "retired on 2026-11-13 (Sprint 04b): it was the only profile that accepted a "
        "minimal thinking level and it is no longer in the registry. Use "
        "gemini-3.8-flash, which is the default pin."
    ),
}


class RetiredModel(GatewayUnavailable):
    """A retired model id was asked for. Fails at construction, not at the first call."""

    def __init__(self, model_id: object) -> None:
        super().__init__(
            f"model {model_id!r} is {RETIRED_MODELS[str(model_id).strip()]}"
        )
        self.model_id = model_id


# ---------------------------------------------------------------------------
# Error classification.
#
# The retry ladders in `deterministic.py` and the judge client ask this instead of
# matching provider strings. Four answers, because four things can be done about a
# failure: wait and try again, do not try again, check the network, or let it raise.
# ---------------------------------------------------------------------------

CAPACITY = "capacity"
CONTENT = "content"
TRANSPORT = "transport"
OTHER = "other"

#: Phrases that mean "the endpoint refused for throughput, not for something you said".
#: `RESOURCE_EXHAUSTED` is Google's, `rate_limit_exceeded` is OpenAI's, and the rest are
#: what the same refusal looks like when it arrives as prose from a transport.
_CAPACITY_MARKERS = (
    "resource_exhausted", "resource exhausted", "rate limit", "ratelimit",
    "rate_limit", "too many requests", "quota", "429", "overloaded",
    "capacity",
)

#: Phrases that mean the request was refused for what it contained. Retrying these is
#: pointless at temperature 0 and, for a clinical product, actively misleading.
_CONTENT_MARKERS = (
    "safety", "blocked", "content_filter", "content filter", "prohibited",
    "recitation", "refus", "invalid_argument", "invalid argument", "policy",
)

#: Failures a second attempt can plausibly fix without the request changing.
_TRANSPORT_MARKERS = (
    "timeout", "timed out", "deadline", "unavailable", "connection", "connect",
    "reset", "broken pipe", "502", "503", "504", "stream",
)


def _status_code(exc: BaseException) -> int | None:
    """An HTTP status off an exception, however the SDK chose to expose it."""
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            code = code()
        except Exception:  # noqa: BLE001 — a code that cannot be read is not a code
            code = None
    if isinstance(code, int):
        return code
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    candidate = getattr(response, "status_code", None)
    return candidate if isinstance(candidate, int) else None


def classify_error(exc: BaseException) -> str:
    """What kind of failure this is, in four terms.

    The status code is read first because it is unambiguous, and the message text only
    afterwards. That order matters: an SDK that reports 429 as an exception class with no
    discriminative name still classifies correctly, and a message that happens to contain
    the word "quota" inside a patient's note text cannot, because exception text never
    carries note text.
    """
    status = _status_code(exc)
    if status == 429:
        return CAPACITY
    if status in (500, 502, 503, 504):
        return TRANSPORT
    if status in (400, 403, 422):
        return CONTENT
    text = f"{type(exc).__name__} {exc}".lower()
    if any(marker in text for marker in _CAPACITY_MARKERS):
        return CAPACITY
    if any(marker in text for marker in _CONTENT_MARKERS):
        return CONTENT
    if any(marker in text for marker in _TRANSPORT_MARKERS):
        return TRANSPORT
    return OTHER


# ---------------------------------------------------------------------------
# Profiles.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelProfile:
    """One model, as the facts the pipeline needs about it.

    Two mappings rather than one, because the two are different knobs and conflating them
    is how a request gets a parameter that belongs on the other call. `construction` is what
    the client is built with — a Gemini client takes its thinking level as
    `reasoning_effort` at construction. `call` is what one call takes — a Gemini client
    takes a *per-call* level as `thinking_level`, and the two spellings are not
    interchangeable.
    """

    name: str
    family: str
    wire_model: str
    #: The endpoint this family is addressed on. Read from config so a deployment can pin
    #: a region without a code change.
    location: str
    #: What the client is constructed with, per semantic effort level.
    construction: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: What one call is bound with, per semantic effort level.
    call: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: Whether the request may carry a sampling temperature. One candidate rejects it
    #: outright with HTTP 400 rather than ignoring it, so this is a wire fact.
    sends_temperature: bool = True
    #: Whether this provider can be told "you must call a tool". False means the caller's
    #: forcing is translated to the nearest thing the provider accepts.
    forces_tool_choice: bool = True
    #: Whether the provider takes a per-category safety threshold dictionary.
    takes_safety_settings: bool = True
    #: How this provider is asked for a JSON object. `mime` is a response mime type,
    #: `response_format` is an OpenAI-compatible structured-output request, and `prompt`
    #: means there is no wire parameter and the instruction in the prompt is the whole
    #: mechanism — which is the honest state rather than a guess.
    json_mode: str = "mime"
    #: Whether the provider accepts strict user/assistant alternation. True means adjacent
    #: user turns must be merged by this module before the request leaves.
    strict_role_alternation: bool = False


def _google_thinking(served: str, extraction: str, *, call: str | None) -> tuple[
        dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """A Gemini profile's construction and per-call parameters.

    `call` is the level bound on one call, and it is not always `extraction`: the pin
    accepts `minimal` there and the two newer models reject it, which is the single fact
    this table exists to hold.
    """
    construction = {
        EFFORT_SERVED: {"reasoning_effort": served},
        EFFORT_EXTRACTION: {"reasoning_effort": extraction},
    }
    per_call: dict[str, Mapping[str, Any]] = {}
    if call is not None:
        per_call[EFFORT_EXTRACTION] = {"thinking_level": call}
    return construction, per_call


#: The canonical ids the programme evaluates. A profile is looked up by exact id and never
#: by a substring test on the model string: that test is the branch this registry exists to
#: remove.
PROFILES: dict[str, ModelProfile] = {}


def _build_registry(google_location: str, anthropic_location: str,
                    meta_location: str, served_effort: str,
                    extraction_effort: str) -> dict[str, ModelProfile]:
    """The profile table. A function rather than a literal so config is read at import."""
    table: dict[str, ModelProfile] = {}

    def add(profile: ModelProfile) -> None:
        table[profile.name] = profile

    # Google. Every profile here maps the extraction effort to `low`, which is the lowest
    # level these models accept. `gemini-3.1-flash-lite` was retired on 2026-11-13: it was
    # the one profile whose per-call level was `minimal`, and it is deliberately absent from
    # this table, so asking for it raises the unknown-model error below rather than being
    # served by a profile that no longer has a maintenance window.
    for name in ("gemini-3.8-flash", "gemini-3.1-pro"):
        construction, per_call = _google_thinking(
            served_effort, extraction_effort, call="low")
        add(ModelProfile(
            name=name, family=GOOGLE,
            # A canonical id and a wire id are not the same thing: `gemini-3.1-pro` is
            # served as `gemini-3.1-pro-preview`, and a run labelled with the canonical id
            # that asked for the wrong wire model would be a run of something else.
            wire_model=("gemini-3.1-pro-preview" if name == "gemini-3.1-pro" else name),
            location=google_location,
            construction=construction, call=per_call, json_mode="mime",
        ))

    # Anthropic on Vertex. Three wire facts, each reported as an HTTP 400 when got wrong:
    # no sampling temperature, thinking expressed as a mode rather than an effort level,
    # and tool choice limited to auto/none.
    claude_thinking = {
        EFFORT_SERVED: {"thinking": {"type": "adaptive"}},
        EFFORT_EXTRACTION: {"thinking": {"type": "between_tools"}},
    }
    add(ModelProfile(
        name="claude-sonnet-5-5", family=ANTHROPIC,
        wire_model="claude-sonnet-5-5", location=anthropic_location,
        sends_temperature=False,
        construction=claude_thinking, call=claude_thinking,
        forces_tool_choice=False,
        takes_safety_settings=False,
        # No wire parameter is used: Anthropic has no `response_format`-style request, and
        # the extraction prompt already asks for the object. Declared rather than invented,
        # so the gap is visible instead of being a parameter nobody verified.
        json_mode="prompt",
        strict_role_alternation=True,
    ))

    # Meta on the GCP OpenAI-compatible Responses API. Bearer credentials are refreshed per
    # request (see `BearerToken`), because a token captured at construction expires inside
    # one long evaluation run.
    meta_thinking = {
        EFFORT_SERVED: {"reasoning": {"effort": "medium"}},
        EFFORT_EXTRACTION: {"reasoning": {"effort": "low"}},
    }
    add(ModelProfile(
        name="muse-spark-1.3", family=META,
        wire_model="muse-spark-1.3", location=meta_location,
        construction=meta_thinking, call=meta_thinking,
        # Measured on the live endpoint, 2026-11-12: forcing a tool call is refused with
        # `400 invalid_request — tool_choice 'required' is not supported yet: forcing a
        # tool call needs a grammar without the terminal user branch`. So the caller's
        # `tool_choice="any"` is translated to `auto` for this profile, which means the
        # first retrieval round is a request rather than a guarantee — a real behavioural
        # difference between arms, recorded in the comparison rather than papered over.
        forces_tool_choice=False,
        takes_safety_settings=False,
        json_mode="response_format",
    ))
    return table


#: The profile an unrecognised id is served by. Not a fallback in the "degrade quietly"
#: sense: before this table existed, every id went to the Gemini client with its own wire
#: name, and an id this table does not know still does exactly that. What changed is that
#: a *known* id now gets the parameters it actually accepts.
#:
#: Moved from `gemini-3.1-flash-lite` to `gemini-3.8-flash` on 2026-11-13, when the
#: retirement directive promoted the baseline to the pin. A retired id must not be the
#: default, because a default is what an unconfigured deployment silently gets.
DEFAULT_PROFILE_NAME = "gemini-3.8-flash"


def _load_registry() -> None:
    """Populate `PROFILES` from config, once, at import."""
    if PROFILES:
        return
    from services.mcp import config

    PROFILES.update(_build_registry(
        google_location=config.GEMINI_LOCATION,
        anthropic_location=config.PROVIDER_LOCATIONS["anthropic"],
        meta_location=config.PROVIDER_LOCATIONS["meta"],
        served_effort=config.GEMINI_REASONING_EFFORT,
        extraction_effort="low",
    ))


def profile(model_id: str | None) -> ModelProfile:
    """The profile for a canonical id, or the default profile carrying that id.

    An unregistered id keeps its own wire name and is served by the default profile's
    facts. That is what the service did for every id before the registry existed, so a
    caller passing a test name is unaffected — and a run labelled with a model the
    registry does not know is visible as such in the execution record.
    """
    _load_registry()
    known = PROFILES.get(str(model_id or ""))
    if known is not None:
        return known
    base = PROFILES[DEFAULT_PROFILE_NAME]
    if not model_id or model_id == base.name:
        return base
    return ModelProfile(
        name=str(model_id), family=base.family, wire_model=str(model_id),
        location=base.location, sends_temperature=base.sends_temperature,
        construction=base.construction, call=base.call,
        forces_tool_choice=base.forces_tool_choice,
        takes_safety_settings=base.takes_safety_settings, json_mode=base.json_mode,
        strict_role_alternation=base.strict_role_alternation,
    )


#: The profile a constructed client was built from, keyed by the client's identity.
#:
#: Held alongside the client itself so the id cannot be reused by a later object while an
#: entry still refers to it. A client is built once per process per arm, so this is a
#: handful of entries, and it is the only way to hang a fact off a pydantic model this
#: module does not own.
_CONSTRUCTED: dict[int, tuple[Any, ModelProfile]] = {}


def _attach(model: Any, profile_: ModelProfile) -> Any:
    _CONSTRUCTED[id(model)] = (model, profile_)
    return model


def _declared_model(model: Any, depth: int = 0) -> str | None:
    """The model id a client — or a runnable derived from one — was built for.

    Needed because binding loses the identity this module hangs a profile on. LangChain
    returns a `_ChatModelBinding` from `bind` and `bind_tools`, and that object is not the
    client, so neither the `gateway_profile` attribute nor the `_CONSTRUCTED` entry
    applies to it. The binding does still carry the client it was made from, and the
    client still names its model, so the name is what survives a bind.

    Read in that order — the runnable itself, then its kwargs, then what it wraps — because
    `RunnableBinding` proxies attribute access to the runnable it wraps, and a `model`
    string found there is the same answer as one found directly.

    `depth` stops a binding that wraps itself, which would otherwise recurse forever.
    """
    if depth > 4:
        return None
    for attribute in ("model", "model_name"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    kwargs = getattr(model, "kwargs", None)
    if isinstance(kwargs, Mapping):
        for key in ("model", "model_name"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                return value
    inner = getattr(model, "bound", None)
    if inner is not None and inner is not model:
        return _declared_model(inner, depth + 1)
    return None


def _profile_named(name: str, table: Mapping[str, ModelProfile]) -> ModelProfile | None:
    """The profile for a canonical id or a wire id, or None when neither names one.

    Both are looked up because the two are not always the same string — a run is labelled
    `gemini-3.1-pro` and the provider is asked for `gemini-3.1-pro-preview` — and a caller
    can legitimately hold either.
    """
    if name in table:
        return table[name]
    for candidate in table.values():
        if candidate.wire_model == name:
            return candidate
    return None


def profile_for(model: Any) -> ModelProfile:
    """The profile a client was built from, or the default profile for a double.

    A test double has no profile, and pretending otherwise would make the mapping
    untestable: it gets the default profile's facts, which is precisely the behaviour the
    production default path must have.

    A client derived by binding is a different case and the reason this function reads a
    model name at all. Resolution by object identity misses every `bind` and `bind_tools`
    result, and the fallback is silent — a binding made from `gemini-3.8-flash` resolved
    to the pin's profile and was then given the pin's per-call level, `minimal`, which
    `gemini-3.8-flash` refuses with a 400 INVALID_ARGUMENT. Measured 2026-11-13 on
    `agent-00286` (Sprint 04 pre-flight, `sel_t2_01`): every structured or ceiling-bound
    call made on a tool-bound client failed this way. So a client that names a model the
    table knows is resolved by that name before the default is reached.
    """
    _load_registry()
    owned = getattr(model, "gateway_profile", None)
    if isinstance(owned, ModelProfile):
        return owned
    entry = _CONSTRUCTED.get(id(model))
    if entry is not None and entry[0] is model:
        return entry[1]
    declared = _declared_model(model)
    if declared is not None:
        named = _profile_named(declared, PROFILES)
        if named is not None:
            return named
    return PROFILES[DEFAULT_PROFILE_NAME]


# ---------------------------------------------------------------------------
# Semantic parameters.
# ---------------------------------------------------------------------------

#: What the default provider's spelling of a request means, so a request written in that
#: spelling can be re-expressed for a provider that spells it differently.
_GEMINI_TO_SEMANTIC: dict[str, dict[str, dict[str, Any]]] = {
    "thinking_level": {
        "minimal": {"effort": EFFORT_EXTRACTION},
        "low": {"effort": EFFORT_EXTRACTION},
        "medium": {"effort": EFFORT_SERVED},
        "high": {"effort": EFFORT_SERVED},
    },
    "response_mime_type": {
        "application/json": {"json_mode": True},
    },
}


def _semantic(params: Mapping[str, Any]) -> dict[str, Any]:
    """A request with the default provider's spellings read back into semantics."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        translation = _GEMINI_TO_SEMANTIC.get(key, {}).get(str(value))
        if translation is None:
            out[key] = value
        else:
            out.update(translation)
    return out


def construction_parameters(profile_: ModelProfile, effort: str) -> dict[str, Any]:
    """This provider's spelling of "think at this effort", for a construction call."""
    return dict(
        profile_.construction.get(effort)
        or profile_.construction.get(EFFORT_SERVED)
        or {}
    )


def call_parameters(profile_: ModelProfile, effort: str) -> dict[str, Any]:
    """This provider's spelling of it for one call, or nothing when it has no such knob."""
    return dict(profile_.call.get(effort) or {})


def wire_bindings(profile_: ModelProfile, request: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a semantic request into this provider's per-call parameters.

    The request may be written either way round — semantic (`effort`, `json_mode`) or in
    the default provider's spelling (`thinking_level`, `response_mime_type`) — because the
    two call sites that use it were written before this module existed. A parameter this
    module does not recognise passes through untouched, so a caller can still bind
    something provider-neutral such as `max_output_tokens`.

    One spelling is refused rather than passed through: a `thinking_level` this table
    cannot name. Every level the Gemini client accepts is in `_GEMINI_TO_SEMANTIC`, so a
    surviving `thinking_level` is a value no profile here agrees on, and sending it would
    be asking a provider to interpret a word this module never verified. Raising names the
    value instead of letting the endpoint answer with a 400 that is indistinguishable from
    a transport failure — which is exactly what `THINKING_LEVEL_MINIMAL` on
    `gemini-3.8-flash` looked like on 2026-11-13.
    """
    semantic = _semantic(request)
    out: dict[str, Any] = {}
    for key, value in semantic.items():
        if key == "effort":
            out.update(call_parameters(profile_, str(value)))
        elif key == "json_mode":
            if not value:
                continue
            if profile_.json_mode == "mime":
                out["response_mime_type"] = "application/json"
            elif profile_.json_mode == "response_format":
                out["response_format"] = {"type": "json_object"}
            # `prompt` adds nothing: the instruction already in the prompt is the whole
            # mechanism, and claiming a wire parameter this provider does not have would
            # be a setting that never applies.
        elif key == "thinking_level":
            raise UnsupportedLevel(key, value, profile_)
        else:
            out[key] = value
    return out


def bind(model: Any, request: Mapping[str, Any]) -> Any:
    """The client with one call's parameters bound, when it can carry them.

    Falls back to the client unchanged when it has no `bind`, which is what keeps a test
    double usable here without implementing LangChain's binding surface. That fallback is
    the reason this is a function and not an inline `bind` call: the rule lives in one
    place for every caller.
    """
    binder = getattr(model, "bind", None)
    if binder is None:
        return model
    resolved = wire_bindings(profile_for(model), request)
    if not resolved:
        return model
    return binder(**resolved)


# ---------------------------------------------------------------------------
# Tool choice.
# ---------------------------------------------------------------------------

#: What `graph.py` asks for when it needs the first turn to retrieve. Spelled as LangChain
#: spells it, because that is the caller's vocabulary and `graph.py` is frozen.
FORCE_ANY = "any"
AUTO = "auto"


def translate_tool_choice(profile_: ModelProfile, choice: Any) -> Any:
    """The tool choice this provider will accept, given the one the caller asked for.

    A provider that cannot be forced gets `auto`: it still receives the tool schemas and
    the prompt still instructs it to retrieve, so the difference is a guarantee becoming a
    request. That is a real behavioural difference between arms, it is a property of the
    provider rather than of our integration, and it belongs in the comparison report
    rather than being papered over.
    """
    if profile_.forces_tool_choice:
        return choice
    if choice is None:
        return None
    if isinstance(choice, str):
        return AUTO if choice.strip().lower() in {FORCE_ANY, "required", "tool"} else choice
    return AUTO


def translate_tools(profile_: ModelProfile, tools: Sequence[Any],
                    tool_choice: Any = None) -> dict[str, Any]:
    """The `bind_tools` keyword arguments for this provider."""
    kwargs: dict[str, Any] = {"tools": list(tools)}
    translated = translate_tool_choice(profile_, tool_choice)
    if translated is not None:
        kwargs["tool_choice"] = translated
    return kwargs


# ---------------------------------------------------------------------------
# Wire-level message normalisation.
# ---------------------------------------------------------------------------


def merge_adjacent_human(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Collapse adjacent user turns into one, leaving the caller's list alone.

    A provider that requires strict alternation rejects two user turns in a row, and the
    evidence overlay is inserted exactly that way: `graph._with_overlay` places a
    `HumanMessage` immediately before the user's own `HumanMessage`, which is correct for
    a provider that tolerates it and a 400 for one that does not. Merging at the wire
    boundary keeps the overlay's content and position while satisfying the requirement,
    and it is done here rather than in `graph.py` because it is a property of the
    transport, not of the conversation.

    The caller's sequence is never mutated: a new list of new messages is returned, and
    the merged message joins the texts in order with a blank line between them.
    """
    merged: list[BaseMessage] = []
    for message in messages:
        previous = merged[-1] if merged else None
        if (isinstance(message, HumanMessage) and isinstance(previous, HumanMessage)):
            joined = "\n\n".join(
                part for part in (_text(previous.content), _text(message.content)) if part
            )
            merged[-1] = HumanMessage(
                content=joined,
                additional_kwargs=dict(getattr(previous, "additional_kwargs", {})),
                id=getattr(previous, "id", None),
            )
            continue
        merged.append(message)
    return merged


def _text(content: Any) -> str:
    """A message's content as text, for the merge only."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n\n".join(parts)
    return ""


def wire_messages(profile_: ModelProfile, messages: Sequence[BaseMessage],
                  callable_: Callable[[Sequence[BaseMessage]], Any] | None = None
                  ) -> list[BaseMessage]:
    """The messages as this provider will accept them."""
    prepared = list(messages)
    if profile_.strict_role_alternation:
        prepared = merge_adjacent_human(prepared)
    return prepared


# ---------------------------------------------------------------------------
# Outcomes.
# ---------------------------------------------------------------------------


def stamp(message: Any) -> Any:
    """Write the neutral outcome onto a finished turn."""
    return model_outcomes.stamp(message)


# ---------------------------------------------------------------------------
# Clients.
# ---------------------------------------------------------------------------


def _google_client(profile_: ModelProfile, *, safety_profile: str, effort: str) -> Any:
    """The Gemini client, built exactly as this service has always built it.

    Byte-identical parameters are a requirement, not a convenience: the default pin must
    behave as it did before this module existed, or every number in the programme would be
    measured against a moved baseline.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    from services.mcp import config

    safety = (config.GEMINI_SAFETY_THRESHOLDS if safety_profile == SAFETY_SERVED
              else CLINICAL_SAFETY_SETTINGS)
    return ChatGoogleGenerativeAI(
        model=profile_.wire_model,
        project=config.PROJECT,
        location=profile_.location,
        vertexai=True,
        temperature=0,
        max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
        **construction_parameters(profile_, effort),
        max_retries=MODEL_MAX_RETRIES,
        timeout=MODEL_TIMEOUT_SECONDS,
        safety_settings=dict(safety),
    )


class BearerToken:
    """A bearer token that is minted per request rather than captured once.

    A token read at construction expires inside one long evaluation run — Google's are
    valid for about an hour — and a run that then sends a dead token fails every remaining
    item for a reason that has nothing to do with the model. Calling this object returns a
    current token every time, so an SDK that accepts a callable as its API key refreshes
    automatically.
    """

    #: Refresh this many seconds before expiry, so a token cannot die mid-request.
    _LEEWAY_SECONDS = 120

    def __init__(self, scopes: Sequence[str] = ("https://www.googleapis.com/auth/cloud-platform",)):
        self._scopes = tuple(scopes)
        self._cached: tuple[str, float] | None = None

    def __call__(self) -> str:
        import time

        now = time.time()
        if self._cached is not None and self._cached[1] - self._LEEWAY_SECONDS > now:
            return self._cached[0]
        from google.auth import default as google_auth_default
        from google.auth.transport.requests import Request

        credentials, _project = google_auth_default(scopes=list(self._scopes))
        if not getattr(credentials, "valid", False) or not getattr(
                credentials, "token", None):
            credentials.refresh(Request())
        token = str(credentials.token or "")
        self._cached = (token, self._expiry_seconds(credentials, now))
        return token

    @staticmethod
    def _expiry_seconds(credentials: Any, now: float) -> float:
        """When this token stops being usable, as a Unix timestamp.

        Google's credentials carry `expiry` as a `datetime`, and one shape carries it as a
        number already. `float(datetime)` raises `TypeError`, which is what a live probe
        produced on 2026-11-12 before this branch existed — so the two shapes are handled
        rather than assumed. A credential that reports no expiry gets ten minutes, which is
        shorter than any Google token and therefore safe.
        """
        import datetime

        expiry = getattr(credentials, "expiry", None)
        if isinstance(expiry, datetime.datetime):
            return expiry.timestamp()
        if isinstance(expiry, (int, float)) and expiry:
            return float(expiry)
        return now + 600

    def __str__(self) -> str:  # pragma: no cover - convenience for a caller printing it
        return self()


def _anthropic_client(profile_: ModelProfile, *, effort: str) -> Any:
    """The Anthropic client, on Vertex, with this profile's wire facts applied."""
    ChatAnthropicVertex = _load_anthropic_class()
    from services.mcp import config

    kwargs: dict[str, Any] = {
        "model": profile_.wire_model,
        "project": config.PROJECT,
        "location": profile_.location,
    }
    kwargs.update(construction_parameters(profile_, effort))
    # No temperature, no top_p, no top_k: this profile rejects them with a 400 rather than
    # ignoring them, which is why `sends_temperature` is a profile fact.
    client = ChatAnthropicVertex(**kwargs)
    return _WireWrapper(client, profile_)


def _load_anthropic_class() -> Any:
    """Anthropic's Vertex LangChain class, or a refusal that names what is missing.

    Deliberately does not fall back to `ChatAnthropic`. That class speaks the first-party
    Anthropic API, so accepting it here would send a patient's notes to a different
    endpoint, under different credentials, on a different invoice — and it would do so
    silently, because a class that answers correctly is indistinguishable from the right
    one until somebody reads the network log. The installed `langchain-anthropic` (1.7.0,
    the newest release compatible with the pinned `langchain-core==1.6.0`) publishes
    `ChatAnthropic` and `AnthropicLLM` and no Vertex class at all, so this raises.

    What would satisfy it: a `langchain-anthropic` release that carries
    `ChatAnthropicVertex`, or the `anthropic.AnthropicVertex` client wired in by hand.
    Either is a deliberate change with a live probe behind it, not a fallback.
    """
    for module_name, class_name in (
        ("langchain_anthropic", "ChatAnthropicVertex"),
        ("langchain_google_vertexai", "ChatAnthropicVertex"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
        except ImportError:
            continue
        found = getattr(module, class_name, None)
        if found is not None:
            return found
    raise GatewayUnavailable(
        "no Vertex-capable Anthropic chat model is installed: `langchain-anthropic` "
        "exposes ChatAnthropic (first-party API) and no ChatAnthropicVertex, and "
        "`anthropic[vertex]` is present but unwired. Refusing rather than falling back to "
        "the first-party endpoint."
    )


def _meta_client(profile_: ModelProfile, *, effort: str) -> Any:
    """The Meta client, on the GCP OpenAI-compatible Responses API."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - depends on the deployment image
        raise GatewayUnavailable(
            "the Meta profile needs `langchain-openai` installed in this environment"
        ) from exc

    from services.mcp import config

    kwargs: dict[str, Any] = {
        "model": profile_.wire_model,
        "base_url": responses_base_url(profile_, project=config.PROJECT),
        "api_key": BearerToken(),
        "use_responses_api": True,
        "stream_usage": True,
    }
    kwargs.update(construction_parameters(profile_, effort))
    return _WireWrapper(ChatOpenAI(**kwargs), profile_)


def responses_base_url(profile_: ModelProfile, *, project: str) -> str:
    """The OpenAI-compatible endpoint for one publisher on GCP.

    The trailing `/v1` is part of the base: an OpenAI-compatible client appends
    `/responses` to whatever it is given, so a base without it addresses
    `.../publishers/meta/responses`, which does not exist. The path below plus the
    client's suffix is `.../publishers/meta/v1/responses`, the shape GCP documents.
    """
    return (
        "https://aiplatform.googleapis.com/v1/projects/"
        f"{project}/locations/{profile_.location}/publishers/meta/v1"
    )


class _WireWrapper:
    """A provider client that gets this module's wire normalisation applied.

    Wraps rather than subclasses because the SDK classes are pydantic models whose
    validation runs at construction; a wrapper is a place to put the three adaptations
    that must happen on every call and in one place:

    * adjacent user turns merged, for a provider that requires alternation;
    * tool choice translated, for a provider that cannot be forced;
    * the neutral outcome stamped, so the layers above read one vocabulary.

    Attribute access falls through to the wrapped client, so a caller that inspects
    `.timeout` or `.max_retries` still sees the SDK's own answer.
    """

    def __init__(self, model: Any, profile_: ModelProfile) -> None:
        self._model = model
        self.gateway_profile = profile_

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    # --- binding ---------------------------------------------------------

    def bind(self, **kwargs: Any) -> "_WireWrapper":
        return _WireWrapper(self._model.bind(**kwargs), self.gateway_profile)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_WireWrapper":
        requested = kwargs.pop("tool_choice", None)
        merged = translate_tools(self.gateway_profile, tools, requested)
        merged.update(kwargs)
        return _WireWrapper(
            self._model.bind_tools(merged.pop("tools"), **merged),
            self.gateway_profile,
        )

    # --- invocation ------------------------------------------------------

    async def ainvoke(self, messages: Sequence[BaseMessage], config: Any = None,
                      **kwargs: Any) -> Any:
        wire = wire_messages(self.gateway_profile, messages)
        if config is None:
            result = await self._model.ainvoke(wire, **kwargs)
        else:
            result = await self._model.ainvoke(wire, config=config, **kwargs)
        return stamp(result)

    async def astream(self, messages: Sequence[BaseMessage], config: Any = None,
                      **kwargs: Any):
        """Stream the turn, stamping each chunk so the accumulated one is stamped too.

        Stamping per chunk rather than at the end is not bookkeeping for its own sake: the
        caller accumulates chunks, LangChain merges their metadata, and the reason a
        provider reports usually arrives on the last one. Stamping as they pass through is
        what makes the caller's accumulated turn carry the neutral outcome without this
        wrapper having to hold on to anything or the caller having to ask for it.
        """
        wire = wire_messages(self.gateway_profile, messages)
        if config is None:
            async for chunk in self._model.astream(wire, **kwargs):
                yield stamp(chunk)
        else:
            async for chunk in self._model.astream(wire, config=config, **kwargs):
                yield stamp(chunk)

    # --- convenience -----------------------------------------------------

    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.with_structured_output(*args, **kwargs)


def chat_model(model_id: str | None, *, safety_profile: str = SAFETY_SERVED,
               effort: str = EFFORT_SERVED) -> Any:
    """The client for a canonical model id.

    The only place in the service that names a provider package.
    """
    retired = RETIRED_MODELS.get(str(model_id or "").strip())
    if retired is not None:
        raise RetiredModel(model_id)
    profile_ = profile(model_id)
    if profile_.family == GOOGLE:
        return _attach(_google_client(profile_, safety_profile=safety_profile,
                                      effort=effort), profile_)
    if profile_.family == ANTHROPIC:
        return _attach(_anthropic_client(profile_, effort=effort), profile_)
    if profile_.family == META:
        return _attach(_meta_client(profile_, effort=effort), profile_)
    raise GatewayUnavailable(f"no client is registered for the {profile_.family!r} family")


def wire_model_name(model_id: str | None) -> str:
    """What the provider is actually asked for, as opposed to what the run is called."""
    return profile(model_id).wire_model


def answered_model(message: Any, requested: str | None = None) -> str | None:
    """Which model answered, from the response when it says, else what was asked.

    `requested` is a weak claim — it is what we wanted — so it is only used when the
    response is silent, and the caller can tell the two apart by reading the response
    field first.
    """
    metadata = getattr(message, "response_metadata", None)
    if isinstance(metadata, Mapping):
        served = metadata.get("model_name") or metadata.get("model")
        if served:
            return str(served)
    return wire_model_name(requested) if requested else None


#: Re-exported so a caller can build the same message a provider would accept without
#: importing `langchain_core` itself.
HUMAN = HumanMessage
AI = AIMessage


# The registry is filled at import rather than on first use. It is a table of constants read
# from config, so building it eagerly costs one module import and removes the possibility
# that two callers see a different table — `profile()` still calls `_load_registry()` as a
# guard for a module imported in an unusual order.
_load_registry()
