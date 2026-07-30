"""Verify Gemini is reachable and actually answers, before any graph exists.

Why this is a script and not a note in the guide: a model-availability error
surfacing inside a LangGraph trace is much harder to read than the same error
on its own. Run this first; if it fails, the problem is Vertex, not the agent.

It checks more than reachability. On 2026-07-30 `gemini-2.5-pro` and
`gemini-2.5-flash` both returned HTTP 200 with **empty text** — the 2.5 models
are thinking models, `max_output_tokens` budgets thinking *and* the answer, and
a tight cap is spent entirely on thoughts. finish_reason=MAX_TOKENS, no
exception raised. So an empty answer is treated here as a failure, loudly.
"""

import argparse
import sys
from pathlib import Path

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.config import (  # noqa: E402
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    LOCATION,
    PROJECT,
)

PROMPT = "Reply with exactly the word: ready"
EXPECTED = "ready"


def check(model: str, location: str, max_output_tokens: int) -> bool:
    client = genai.Client(vertexai=True, project=PROJECT, location=location)
    print(f"project={PROJECT} location={location} model={model} "
          f"max_output_tokens={max_output_tokens}")

    try:
        response = client.models.generate_content(
            model=model,
            contents=PROMPT,
            config=types.GenerateContentConfig(
                temperature=0, max_output_tokens=max_output_tokens
            ),
        )
    except Exception as exc:  # availability, quota, permission
        print(f"FAIL  generate_content raised: {exc}")
        return False

    finish = response.candidates[0].finish_reason if response.candidates else None
    usage = response.usage_metadata
    text = (response.text or "").strip()

    print(f"finish_reason={finish} thoughts={usage.thoughts_token_count} "
          f"output={usage.candidates_token_count}")
    print(f"text={text!r}")

    # A 200 with no text is the failure this script exists to catch.
    if not text:
        print("FAIL  empty response. If finish_reason=MAX_TOKENS the thinking "
              "budget consumed the whole allowance — raise GEMINI_MAX_OUTPUT_TOKENS.")
        return False

    if EXPECTED not in text.lower():
        print(f"WARN  model answered but not as instructed (wanted {EXPECTED!r})")

    print("OK    Gemini is reachable and answering.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--location", default=LOCATION,
                        help="use 'global' only if the region cannot serve the model")
    parser.add_argument("--max-output-tokens", type=int, default=GEMINI_MAX_OUTPUT_TOKENS)
    args = parser.parse_args()

    return 0 if check(args.model, args.location, args.max_output_tokens) else 1


if __name__ == "__main__":
    sys.exit(main())
