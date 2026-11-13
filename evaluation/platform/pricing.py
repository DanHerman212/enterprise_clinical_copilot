"""Vertex AI token prices, and the arithmetic that turns usage into dollars.

One registry, one formula, no defaults. A model this file does not know raises rather
than costing `0.00`: a run that silently priced an unknown model at zero would report a
cheaper arm than it measured, and the whole point of the Sprint 02 census is to compare
arms on cost as well as on quality.

Two properties of the provider numbers are load-bearing, and both were measured live
rather than inferred from documentation:

* **The cached read is inside `input`.** The Sprint 01b traces show
  `total == input + output` with a non-zero cached sub-count inside `input`, so the
  uncached portion is a subtraction and never an addition.
* **Reasoning tokens are accounted for differently by different providers.** Gemini and
  Meta, read back through the harness's Langfuse blocks, report
  `total == input + output` with the reasoning count inside `output` (Gemini: output 160,
  thinking 153; Meta: output 374, thinking 361). `xai/grok-4.7`, read back from the
  OpenAI-compatible endpoint, reported output 390 with thinking 4347 on 2026-11-13 —
  more reasoning than completion tokens, so the reasoning cannot be inside `output`.
  `billable_output` decides between the two from the numbers in the block rather than
  assuming either, which is what stops a grader's reasoning being billed twice on one
  provider and not at all on another.

Rates are USD per 1,000,000 tokens on the pay-as-you-go tier, at the ≤200K prompt-context
band. They are transcribed, not computed, which is why each entry names its role: a rate
that disagrees with the console is a data edit here and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Rate:
    """One model's prices, per 1,000,000 tokens."""

    model_id: str
    role: str
    input: float
    cached_read: float
    cache_write: float
    output: float


class UnknownModelError(LookupError):
    """A model id this registry has no rate for.

    Raised rather than returning a number, because every alternative is a lie: zero
    would under-report the arm, and an average would invent a price. A caller that
    cannot price a generation reports its cost as unattributed (`None`), which is the
    same thing it does when the trace is missing.
    """


#: The registry. Every canonical id and every wire alias the harness can see, mapping to
#: one `Rate`. Aliases are listed explicitly rather than derived by stripping a provider
#: prefix: `meta/muse-spark-1.3` and `grok-4.7` are served names, and a stripping rule
#: would also "resolve" a typo into a neighbouring model's price.
_RATES: dict[str, Rate] = {}


def _register(rate: Rate, *aliases: str) -> None:
    for name in (rate.model_id, *aliases):
        _RATES[name.lower()] = rate


_DEFAULT_PIN = Rate("gemini-3.1-flash-lite", "default pin",
                    input=0.25, cached_read=0.025, cache_write=0.0, output=1.50)
_register(_DEFAULT_PIN)

_BASELINE = Rate("gemini-3.8-flash", "active baseline",
                 input=0.75, cached_read=0.075, cache_write=0.0, output=3.75)
_register(_BASELINE)

_PRO = Rate("gemini-3.1-pro", "candidate",
            input=2.00, cached_read=0.20, cache_write=0.0, output=12.00)
_register(_PRO, "gemini-3.1-pro-preview")

_MUSE = Rate("muse-spark-1.3", "candidate",
             input=1.25, cached_read=0.15, cache_write=0.0, output=4.25)
_register(_MUSE, "meta/muse-spark-1.3")

_CLAUDE = Rate("claude-sonnet-5-5", "dormant candidate",
               input=2.00, cached_read=0.20, cache_write=2.50, output=10.00)
_register(_CLAUDE)

_JUDGE = Rate("xai/grok-4.7", "independent judge",
              input=2.00, cached_read=0.50, cache_write=0.0, output=6.00)
_register(_JUDGE, "grok-4.7")

_LEGACY_JUDGE = Rate("gemini-2.5-pro", "legacy judge",
                     input=1.25, cached_read=0.125, cache_write=0.0, output=10.00)
_register(_LEGACY_JUDGE)

#: The order a report lists the roster in. Named rather than taken from the dict, so a
#: table in a report does not reorder itself when an alias is added.
ROSTER: tuple[str, ...] = (
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-pro",
    "muse-spark-1.3",
    "claude-sonnet-5-5",
    "xai/grok-4.7",
)


def rate_for(model_id: object) -> Rate:
    """The rate for a canonical id or a served alias, case- and space-insensitive."""
    key = str(model_id or "").strip().lower()
    try:
        return _RATES[key]
    except KeyError:
        raise UnknownModelError(
            f"no Vertex AI rate for model {model_id!r}; add it to "
            "evaluation/platform/pricing.py rather than reporting 0.00"
        ) from None


def known(model_id: object) -> bool:
    """Whether this id can be priced. For a caller that must report, not raise."""
    return str(model_id or "").strip().lower() in _RATES


# --- Usage normalisation -----------------------------------------------------------

#: The spellings a usage block arrives in. Langfuse stores `usage_details` as
#: `{"input": n, "output": n, "total": n}` (plus `input_cached_tokens` on recent SDKs),
#: OpenAI-shaped providers answer `prompt_tokens`/`completion_tokens`, and a reasoning
#: model nests its sub-counts one level down. One table, read left to right.
_INPUT_KEYS = ("input", "input_tokens", "prompt_tokens", "promptTokens")
_OUTPUT_KEYS = ("output", "output_tokens", "completion_tokens", "completionTokens")
_CACHED_KEYS = ("cached", "input_cached_tokens", "cached_tokens",
                "cache_read_input_tokens", "prompt_tokens_cached")
_CACHE_WRITE_KEYS = ("cache_write", "cache_write_tokens", "cache_creation_input_tokens")
_THINKING_KEYS = ("thinking", "reasoning_tokens", "thinking_tokens",
                  "output_reasoning_tokens", "reasoning")
_TOTAL_KEYS = ("total", "total_tokens", "totalTokens")
_NESTED = ("prompt_tokens_details", "completion_tokens_details",
           "output_tokens_details", "input_tokens_details")


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _find(raw: Mapping[str, Any], keys: Iterable[str]) -> int | None:
    for key in keys:
        found = _as_int(raw.get(key))
        if found is not None:
            return found
    return None


def _find_nested(raw: Mapping[str, Any], keys: Iterable[str]) -> int | None:
    for parent in _NESTED:
        child = raw.get(parent)
        if isinstance(child, Mapping):
            found = _find(child, keys)
            if found is not None:
                return found
    return None


def normalise_usage(raw: Mapping[str, Any] | None) -> dict[str, int | None]:
    """One usage block as `input`, `cached`, `output`, `thinking`, `cache_write`.

    Nothing is added together. Each field is a count the provider reported, or `None`
    when it did not report one — which is the honest value for a provider that omits
    a field, and the reason `thinking` is allowed to be `None` rather than `0`.
    """
    block = raw if isinstance(raw, Mapping) else {}
    return {
        "input": _find(block, _INPUT_KEYS),
        "cached": _find(block, _CACHED_KEYS) or _find_nested(block, _CACHED_KEYS),
        "output": _find(block, _OUTPUT_KEYS),
        "cache_write": _find(block, _CACHE_WRITE_KEYS),
        "thinking": _find(block, _THINKING_KEYS) or _find_nested(block, _THINKING_KEYS),
        "total": _find(block, _TOTAL_KEYS),
    }


def merge_usage(blocks: Iterable[Mapping[str, Any] | None]) -> dict[str, int | None]:
    """Sum several usage blocks field by field, keeping "unreported" as `None`.

    Used for two things: a judge that retried, and a generation's split usage. A field
    counts as `0` only when every contributor reported it as zero or unreported-nothing;
    if any contributor reported a number, the sum is a number.
    """
    totals: dict[str, int | None] = {"input": None, "cached": None, "output": None,
                                     "cache_write": None, "thinking": None,
                                     "total": None}
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        for key, value in block.items():
            if key not in totals or value is None:
                continue
            totals[key] = (totals[key] or 0) + int(value)
    return totals


def billable_input(usage: Mapping[str, Any] | None) -> int:
    """Input tokens that are not cached reads.

    The subtraction is clamped at zero: a provider that reported a cached count larger
    than its input count is reporting something this formula cannot represent, and a
    negative billable count would silently rewrite the price in the model's favour.
    """
    block = usage if isinstance(usage, Mapping) else {}
    total = _as_int(block.get("input")) or 0
    cached = _as_int(block.get("cached")) or 0
    return max(0, total - cached)


def billable_output(usage: Mapping[str, Any] | None) -> int:
    """The output tokens that are actually billed, reasoning included exactly once.

    The rule depends on how the provider accounts for reasoning, and this function reads
    that from the numbers instead of assuming it:

    * When the reported `total` is no larger than `input + output`, the reasoning count
      is already inside `output` — that is the identity Sprint 01b measured on the
      Langfuse blocks for Gemini (`input 32, output 160, total 192, thinking 153`) and
      Meta (`input 59, output 374, total 433, thinking 361`). Adding it again would
      bill reasoning twice.
    * When the provider reports more reasoning tokens than completion tokens, they
      cannot be inside `output`. Measured live on `xai/grok-4.7` on 2026-11-13:
      `output 390` with `thinking 4347`. Those reasoning tokens are billed as output
      and are added.
    * With no `total` to check against, `thinking <= output` is the only evidence
      available, and it is read the conservative way: inside `output`, not added.

    Both behaviours are real and neither generalises, which is why this is a rule with
    evidence behind each branch rather than one line that adds or one line that does not.
    """
    block = usage if isinstance(usage, Mapping) else {}
    output = _as_int(block.get("output")) or 0
    thinking = _as_int(block.get("thinking")) or 0
    if thinking <= 0:
        return output
    total = _as_int(block.get("total"))
    input_total = _as_int(block.get("input")) or 0
    if total is not None:
        inside = total <= input_total + output
    else:
        inside = thinking <= output
    return output if inside else output + thinking


def cost_usd(model_id: object, usage: Mapping[str, Any] | None) -> float:
    """Deterministic USD cost of one model's reported usage.

        (uncached_input x input + cached_read x cached + cache_write x write
         + billable_output x output) / 1e6

    `billable_output` is the provider's output count plus any reasoning tokens the
    provider reports outside it — decided per usage block, never assumed. `thinking` is
    therefore never billed twice and never dropped.
    """
    rate = rate_for(model_id)
    block = usage if isinstance(usage, Mapping) else {}
    cached = _as_int(block.get("cached")) or 0
    cache_write = _as_int(block.get("cache_write")) or 0
    total = (
        billable_input(block) * rate.input
        + cached * rate.cached_read
        + cache_write * rate.cache_write
        + billable_output(block) * rate.output
    )
    return total / 1_000_000


def describe(model_id: object) -> dict[str, Any]:
    """One roster row, for a report or a diagnostics record."""
    rate = rate_for(model_id)
    return {
        "model_id": rate.model_id,
        "role": rate.role,
        "input_per_million": rate.input,
        "cached_read_per_million": rate.cached_read,
        "cache_write_per_million": rate.cache_write,
        "output_per_million": rate.output,
    }
