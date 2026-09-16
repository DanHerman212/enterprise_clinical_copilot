"""Measure whether implicit caching actually hits for this application.

Gap 9's question was whether the 90% discount on cached input applies to us. The answer is
read from the cached-token count every response already reports — not inferred from prompt
size, which is what the document did before.

Two prefix sizes are measured, not one. A prefix below the family minimum cannot cache at
all, so a measurement that only tested our real prompt could not tell "below the threshold"
from "not happening here", and those are different findings with different decisions behind
them.

What it found, 2026-09-16, 31 calls across two endpoints and two models:

  - the real prefix, our system prompt, is 3,088 input tokens — below the documented minimum
    of 4,096 for the Gemini 3 family (2,048 for Gemini 2, which is the figure the document
    carried and which stopped being ours when the model pin moved). So nothing we currently
    send can be cached, and that part is settled;
  - above the minimum it is *not* settled. Prefixes of 5,176 and 8,247 tokens produced no hit
    in 28 calls, one hit was seen (4,066 cached tokens, global, at a 5,176-token prefix) and
    did not reproduce across eight repeats, and the platform documents implicit caching as
    enabled by default from Gemini 2.5 onwards — so the model is supported and the zeros above
    the minimum are unexplained.

Do not settle that second half on a sample this size. An earlier pass did, on three calls per
endpoint, and concluded that the global endpoint caches while `us` does not; eight repeats
disproved it.

So nothing here is written to depend on the discount, and the decision that follows is not to
pad the prompt to reach the minimum. The count stays on the execution record, which means the
position is measurable without re-deploying anything.

Run this again after a model change, or after any change to the size of the system prompt.
It bills for what it sends: `--calls 8` with the real system prompt is about 25,000 input
tokens, and the padded variant is larger.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from services.agent import graph  # noqa: E402
from services.agent.prompts import SYSTEM_PROMPT  # noqa: E402

# Identical reference text appended to the real prompt, to carry the prefix past the family
# minimum without touching the instructions themselves. It is the same on every call, which is
# the only property that matters for a cache.
PADDING = (
    "\n\nREFERENCE — discharge summary conventions used by the source notes.\n"
    "Medication lines are recorded as name, dose, route and frequency; a stopped medication "
    "appears with its stop date; allergy entries are prefixed with the reaction. Vital signs "
    "are recorded per shift. Laboratory values carry a reference range in parentheses. "
    "Procedure notes name the operator role rather than the individual. Follow-up is written "
    "as an interval in weeks rather than a date.\n"
) * 60


def measure(label: str, system: str, calls: int) -> bool:
    """Send the same prefix `calls` times and report what the responses said."""
    llm = graph._build_llm(graph.MODEL_ID)
    inputs, cached = [], []
    for i in range(calls):
        reply = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Reply with the single word: {i}."),
        ])
        usage = reply.usage_metadata or {}
        inputs.append(usage.get("input_tokens"))
        cached.append((usage.get("input_token_details") or {}).get("cache_read") or 0)

    hits = [c for c in cached if c]
    print(f"  {label}")
    print(f"    input_tokens: {inputs}")
    print(f"    cache_read  : {cached}")
    print(f"    hits        : {len(hits)}/{calls}"
          + (f"  cached tokens: {hits}" if hits else ""))
    return bool(hits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=5,
                        help="calls per prefix size; they bill, so keep it small")
    parser.add_argument("--padding", action="store_true",
                        help="also measure a prefix padded past the family minimum")
    args = parser.parse_args()

    print(f"model {graph.MODEL_ID}, {args.calls} calls per prefix size")
    print("")
    hit_small = measure("real system prompt", SYSTEM_PROMPT, args.calls)
    hit_large = False
    if args.padding:
        hit_large = measure("padded past the minimum", SYSTEM_PROMPT + PADDING, args.calls)

    print("")
    if hit_small or hit_large:
        print("A hit was observed. Record the size it happened at before relying on it —")
        print("a single hit in a small sample is what an earlier attempt mistook for a rule.")
    else:
        print("No hits. Treat the discount as unavailable rather than as unmeasured, and do")
        print("not pad the prompt to chase it: that adds tokens to every request to buy a")
        print("discount that has not been demonstrated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
