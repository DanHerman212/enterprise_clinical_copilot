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

# Where the project's own resources are, pinned rather than read from the environment,
# for the same reason as the model below: this is not a preference, it is where the
# things are. The vector index, the prediction endpoint, the feature bundle and the
# BigQuery dataset all live here, and the deployment scripts import this constant rather
# than repeating it, so the app and the resources it reads cannot be pointed at
# different regions. Until 2026-09-16 it was an environment read, which let a deploy
# move the app to a region its resources were not in — a 404 at request time rather than
# a failed deployment — and the region carries a residency question, which belongs in a
# commit either way.
#
# Residue: the Vertex job scripts under `scripts/agent/` that run inside their own
# containers still hard-code `"us-east1"` instead of importing it. They agree today, and
# the duplication is the reason the pin is worth having here rather than nowhere.
LOCATION = "us-east1"

# Where the chat model is served, which is deliberately not `LOCATION`.
#
# `LOCATION` names the region of the project's own resources — the vector index, the
# prediction endpoint, the bundle, and the BigQuery dataset — and the chat model used
# to share it. It cannot any more: the pinned model has no regional endpoint at all.
# `gemini-3.1-flash-lite` returns 404 on us-east1, us-central1, us-east5, us-east4,
# us-west1, us-south1 and europe-west4, all checked on 2026-09-16, while the old pin
# answers on us-east1. Sharing one value means the next change to it moves retrieval
# and the prediction endpoint too, silently, which is what makes this a constant of
# its own rather than a second reading of the environment.
#
# `us` is Google's multi-region endpoint, the one its locations page describes as
# keeping ML processing inside a jurisdictional boundary. The alternative is `global`,
# where the same page says you "can't control or know which region your ML processing
# requests are sent to" and that it doesn't support data residency requirements — not a
# thing to select by omitting a value, for a service that reads clinical text.
GEMINI_LOCATION = "us"

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

# Gemini. PINNED, not read from the environment: an environment default lets a
# deploy change the model the chain uses with no commit anywhere, which is exactly
# what makes an answer unreproducible. Changing the model is now a reviewed change,
# and nothing has to be remembered to make it visible: the execution record
# carries the code revision the deployment set, so a change here moves the
# identity that an answer is attributed to.
#
# `gemini-2.5-flash` held this line from 2025-06-17 until 2026-09-16, when it was
# replaced rather than retired: Google's lifecycle table dates it 2026-10-20, and
# swapping while the old model still answers is what leaves room to put the two side
# by side, and to go back. The replacements Google names for it are Gemini 3.5
# Flash-Lite and Gemini 3.1 Flash-Lite.
#
# Why 3.1 rather than the newer 3.5 — `gemini-3.5-flash-lite` is on the SDK's list of
# models using fixed sampling defaults, so `temperature=0` is dropped from the request
# with a `UserWarning` that nothing would surface in a Cloud Run log. Both the refusal
# sentence in `model_turn` and the empty-text guard in `http.py` reason from
# determinism at temperature 0, and on that model the reasoning would be quietly
# false. Measured 2026-09-16, eight repeats of one question each: 3.1 at temperature 0
# returned identical reasoning-token counts every time and scattered ones at
# temperature 2; 3.5 scattered at temperature 0 exactly as it did at temperature 2.
# 3.5 buys three more months of runway (2027-07-21 against 2027-05-07) and costs that.
GEMINI_MODEL = "gemini-3.1-flash-lite"

# The endpoint the model is reached on is `GEMINI_LOCATION` above, which is not
# `LOCATION` any more — see the note there. The swap is what forced the split: the
# pinned model has no regional endpoint in any region checked, and the constant it used
# to be reached through also names the vector index and the prediction endpoint, which
# do live in us-east1.

# Thinking and the answer share this allowance, and the model stops when it runs out
# without raising: the call returns 200 with empty text and finish_reason=MAX_TOKENS.
# A level bounds effort rather than tokens, so what thinking will spend is known only
# afterwards. Measured 2026-09-16: at 64 tokens both 3.x candidates returned empty text
# this way, and a synthesis probe spent 269 thinking tokens at the level set below.
#
# Pinned rather than read from the environment, like the model and the region: a deploy
# that lowered it would reintroduce that silent empty answer, which reaches the caller as
# a wrong-looking result rather than a failed deployment. Keep it generous.
GEMINI_MAX_OUTPUT_TOKENS = 2048

# How much thinking the model may do, as a level rather than a token budget. The 3.x
# family replaces `thinking_budget` with this and the 2.5 family rejects it outright
# (400 INVALID_ARGUMENT), so the pin and this line move together — there is no value
# both families accept. Pinned in code rather than read from the environment, for the
# same reason as the model name: an environment default is a change nobody reviews.
#
# The level is a declared choice rather than a tuned one, and the measurements are
# recorded so the next person can argue with a number instead of a feeling. On a probe
# shaped like the chain's last turn — a question over retrieved passages — the old pin
# at `thinking_budget=1024` spent 231 thinking tokens; this one spends 269 at `medium`,
# 117 at `low`, and none at all at `minimal`, which is a different regime rather than a
# cheaper one. `medium` is closest to what the chain does today, which keeps the model
# the only thing that changed; `low` is the lever to pull once the evaluation layer can
# show quality holds.
GEMINI_REASONING_EFFORT = "medium"

# Why this model, on the record rather than in someone's memory.
#
# The requirement is to start on the cheapest model that passes evaluation and escalate
# from there. The pin is now the smallest tier of its family, so the first half is
# closer than it was — but "passes evaluation" is still unmeasured, and nothing routes
# up from it. What makes this a decision rather than an oversight is that the entry
# below has to move whenever the model does — a test refuses a pin that disagrees with
# it — so the next change cannot happen quietly, and whoever makes it has to fill in the
# evidence that justified it.
#
# `cheaper_alternative` is empty deliberately. The model named there before
# (`gemini-2.5-flash-lite`) retires on the same date the old pin did, and nothing below
# this one has been checked: Google's table points at Gemma 4 for that slot, a different
# family, unverified here. Empty with the reason in this comment beats a name nobody
# checked.
#
# `evidence` records a debt rather than a result, and it is the one entry here that is
# not good news. The evidence route is the evaluation layer's harness, which does not
# exist yet; when it does, the comparison has to be judged by something other than the
# model under test, which is what `evaluation/agent/judge.py` is today.
MODEL_CHOICE = {
    "model": GEMINI_MODEL,
    "decided": "2026-09-16",
    "tier": "entry",
    "cheaper_alternative": None,
    "retires": "2027-05-07",
    "replacements": (),
    "previous": "gemini-2.5-flash",
    "previous_retired": "2026-10-20",
    "evidence": {
        "comparison": None,
        "why": (
            "The pin changed under a retirement date, not under evidence. Decision 6.4 "
            "requires a comparison through the evaluation layer's harness, and neither "
            "the harness nor the retrieval endpoint it would need was available — the "
            "endpoint was torn down after the demo. What was checked before the swap is "
            "in section 7 of the layer 4 document. What was not is the one thing that "
            "matters: that the two models answer the same questions equally well."
        ),
        "partial": (
            "Four questions asked of both pins on 2026-09-16, one run each, so this is a "
            "signal and not a rate. Three answered on both. On 'A patient took 20 "
            "paracetamol tablets two hours ago. What is the antidote' the old pin "
            "answered and the new one was refused by the dangerous-content filter; on "
            "the suicidal-ideation question it was the reverse. So the two pins do not "
            "draw the same line, in either direction, and the screening thresholds are "
            "now a measured question rather than a theoretical one. The old pin also "
            "hit MAX_TOKENS on a long answer at the same allowance the new pin "
            "answered within."
        ),
        "owed": (
            "the old pin against the new one over the question set, judged by "
            "evaluation/agent/judge.py. It needs two things that did not exist on "
            "2026-09-16: that harness, and the retrieval endpoint, because the questions "
            "are about the patient's own notes and a comparison run without tools would "
            "measure the wrong thing."
        ),
    },
}

# How long before a retirement date the suite starts failing. A date nobody acts on
# is the same failure as no date at all, and acting on it the day it arrives is too
# late — the calls stop working that day. Two weeks is enough to run the comparison
# the migration needs and to schedule the swap.
MIGRATION_LEAD_DAYS = 14

# How long before a retirement date a pin change must be backed by a comparison, which
# is deliberately longer than `MIGRATION_LEAD_DAYS`. The last swap was made under the
# deadline with nothing behind it, and the guard that forced it would have been
# satisfied by the swap alone — which is how the same thing happens twice. Making the
# evidence due first is what gives the comparison a date of its own.
COMPARISON_DUE_DAYS = 90

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
