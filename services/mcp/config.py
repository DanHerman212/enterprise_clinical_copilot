"""Shared constants for the MCP server and its feature sources."""

import os
import re
from .runtime import positive_int_env, resolve_project_id


def _validated_table_ref(ref: str, name: str, parts: tuple[int, ...]) -> str:
    """Fail at import if a table reference is not a plain dotted identifier.

    ECC-29: FEATURE_TABLE / DISCHARGE_TABLE come from env vars and are
    interpolated into SQL as identifiers (identifiers cannot be bound as
    query parameters). A strict shape check — 2 or 3 dot-separated segments
    of [A-Za-z0-9_-] only — makes an env var carrying SQL (quotes, spaces,
    semicolons, backticks) a loud boot failure instead of an injection
    surface, and catches obvious repointing typos early.
    """
    segments = ref.split(".")
    if len(segments) not in parts or not all(
        re.fullmatch(r"[A-Za-z0-9_-]+", seg) for seg in segments
    ):
        raise RuntimeError(
            f"{name} is not a valid BigQuery table reference: {ref!r}. "
            f"Expected {' or '.join(str(p) for p in parts)} dot-separated "
            "segments of letters/digits/underscore/hyphen only."
        )
    return ref


PROJECT = resolve_project_id()
LOCATION = os.environ.get("LOCATION", "us-east1")

# Vertex serving
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "readmission-endpoint")
FINAL_MODEL_PREFIX = "readmission-final-"
BUNDLE_URI_OVERRIDE = os.environ.get("BUNDLE_URI")

# BigQuery feature source
DATASET = "readmission"
# Defaults to the HYBRID features table — the eval/demo cohort is the hybrid
# admissions (90000001+), whose feature rows live in readmission.hybrid_features.
# The real MIMIC-derived analytics_dataset_encoded table is out of scope for
# the demo and never carries the synthetic/hybrid admissions.
TABLE = _validated_table_ref(
    os.environ.get("FEATURE_TABLE", f"{DATASET}.hybrid_features"),
    "FEATURE_TABLE", parts=(2,),
)
TABLE_FQN = f"{PROJECT}.{TABLE}"
ENTITY_ID_COLUMN = "hadm_id"

# The hand-picked demo cohort. Built by scripts/build_demo_cohort.py.
#
# Authorization boundary (ECC-22): the demo site enforces cohort membership —
# it rejects any hadm_id not in its DemoPatient allowlist BEFORE calling the
# agent (S1-09). The tables the MCP tools read (hybrid_features/hybrid_notes)
# hold ONLY the synthetic hybrid cohort (hadm_id 90000001+, MT-* notes) by
# construction, so even a request that bypassed the site can only ever reach
# synthetic demo rows — never real MIMIC data.
COHORT_TABLE = f"{DATASET}.demo_cohort"
COHORT_TABLE_FQN = f"{PROJECT}.{COHORT_TABLE}"

# RAG / Vector Search serving.
# The index endpoint is resolved by display name (like ENDPOINT_NAME) so the
# deployed id never needs to be hardcoded, and so teardown/stand-up just works.
INDEX_ENDPOINT_NAME = os.environ.get("INDEX_ENDPOINT_NAME", "readmission-rag-index")
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "rag_tree_ah")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = positive_int_env("EMBEDDING_DIM", 768)
RESTRICT_NAMESPACE = "hadm_id"
# The discharge notes table (note_id -> hadm_id mapping, text by note_id).
# Defaults to the HYBRID notes table — the deployed RAG index is built from
# these MT-* notes, so serving must read passage text from the same place.
# The real MIMIC-derived table is out of scope for the demo and must never
# resolve passage text from real patient notes.
DISCHARGE_TABLE = _validated_table_ref(
    os.environ.get("DISCHARGE_TABLE", f"{PROJECT}.readmission.hybrid_notes"),
    "DISCHARGE_TABLE", parts=(3,),
)
DEFAULT_TOP_K = positive_int_env("RAG_TOP_K", 5)

# Gemini. Verified reachable from us-east1 on 2026-07-30, so the agent stays
# co-located with the prediction endpoint; "global" is the fallback, not the default.
#
# PINNED, not read from the environment: an environment default lets a deploy
# change the model the chain uses with no commit anywhere, which is exactly what
# makes an answer unreproducible. Changing the model is now a reviewed change,
# and nothing has to be remembered to make it visible: the execution record
# carries the code revision the deployment set, so a change here moves the
# identity that an answer is attributed to.
#
# `gemini-2.5-flash` is a versioned GA model rather than a moving alias —
# released 2025-06-17. Per Google's model lifecycle table it RETIRES 2026-10-20,
# with Gemini 3.5 Flash-Lite or Gemini 3.1 Flash-Lite named as the replacements.
# The pin therefore has an expiry: the migration is scheduled work, not a
# surprise when calls start failing.
GEMINI_MODEL = "gemini-2.5-flash"

# 2.5 models are thinking models and this budget covers thinking AND the answer.
# Too small and the whole budget is spent on thoughts: the call returns 200 with
# empty text and finish_reason=MAX_TOKENS, no exception. 16 tokens was enough to
# reproduce that. Keep this generous.
GEMINI_MAX_OUTPUT_TOKENS = positive_int_env("GEMINI_MAX_OUTPUT_TOKENS", 2048)

# How much of that allowance thinking may spend. Pinned in code rather than read
# from the environment, for the same reason as the model name: an environment
# default is a change that nobody reviews. The answer uses what is left.
#
# The number is a starting point, not a tuned value. What would justify moving it
# is the evaluation layer's evidence, and the per-call thinking token count that
# the response already reports — neither of which exists yet, which is why this is
# recorded as a declared choice rather than a measured one.
GEMINI_THINKING_BUDGET = 1024

# Why this model, on the record rather than in someone's memory.
#
# The requirement is to start on the cheapest model that passes evaluation and
# escalate from there, and neither half holds yet: this pin is the mid tier,
# nothing has been compared against the cheaper model named below, and no request
# is routed to a smaller one. What makes that a decision rather than an oversight is
# that the entry below has to move whenever the model does — a test refuses a pin
# that disagrees with it — so the next change cannot happen quietly, and whoever
# makes it has to fill in the evidence that justified it.
#
# The evidence route is the evaluation layer's harness, which does not exist yet.
# When it does, the comparison has to be judged by something other than the model
# under test, which is what `evaluation/agent/judge.py` is today.
MODEL_CHOICE = {
    "model": GEMINI_MODEL,
    "decided": "2026-09-16",
    "tier": "mid",
    "cheaper_alternative": "gemini-2.5-flash-lite",
    "retires": "2026-10-20",
    "replacements": ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"),
    "evidence": None,
}

# How long before a retirement date the suite starts failing. A date nobody acts on
# is the same failure as no date at all, and acting on it the day it arrives is too
# late — the calls stop working that day. Two weeks is enough to run the comparison
# the migration needs and to schedule the swap.
MIGRATION_LEAD_DAYS = 14

# Content filtering, configured here rather than inherited from the model.
#
# Four categories are configurable, and these are all of them; the CSAM and
# personal-data filters are not configurable and always apply, so there is nothing
# to set for them. Setting a threshold makes the behaviour ours: Google's page says
# an unset threshold falls back to the model's default, and the default is not the
# same on every model — `OFF` is the default for `gemini-3.5-flash` and later, which
# is where the replacements for this pin live. Without this, the migration would
# remove the filtering that applies today, silently.
#
# BLOCK_MEDIUM_AND_ABOVE is the middle of three usable thresholds, and it is a
# trade rather than a preference: stricter and legitimate clinical content gets
# refused — an overdose, a sexual history, a self-harm assessment — looser and the
# control is nominal. Moving a category on its own needs screening evidence, which
# means an evaluation of what this product actually gets asked; the first candidate
# for review is dangerous content, because a discharge note describes dangerous
# things.
GEMINI_SAFETY_THRESHOLDS = {
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
}

API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
