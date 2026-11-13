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

# Which store serves the model's feature rows. Reported in every prediction payload and
# on `/health` so the provenance of a score is visible without reading the reader.
#
# Declared here rather than beside `dependencies/features/`: it is a deployment fact
# about which store this service was built against, the same kind of fact as `PROJECT`,
# and three tools plus the server's health endpoint report it. Keeping it in the
# dependency package made the server's top level import a subpackage to print a
# constant, which the package's charter forbids — the boundary test could see that edge
# and named it.
FEATURE_SOURCE = "bigquery"

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
#
# 2026-11-08: `us` stopped answering. Every request came back `429 RESOURCE_EXHAUSTED`
# ("Resource has been exhausted (e.g. check quota)") while the project was making a
# handful of requests per hour — not a burst, and not another caller: neither
# `eval-clinical` nor `eval-adversarial` has run since 2026-10-18/20, and the deployed
# services are only the agent, the site, Langfuse and the MCP server. The `us`
# multi-region endpoint for this model was saturated, and `/ehr` failed for a clinician
# with no defect anywhere in the request path.
#
# `global` served the same `gemini-3.1-flash-lite`, at `temperature=0`, in the same
# second — measured by addressing one 1-token request to each endpoint. That is the whole
# reason for the move: the model, the sampling configuration and every determinism
# argument resting on it are unchanged, and only the endpoint the request is addressed to
# moves. `us-east1` is not an option: the model is not published there (404).
#
# What this gives up is the jurisdictional boundary above. That is a real loss and it is
# recorded rather than dropped: requests are now routed to a region Google selects and
# does not disclose, which is acceptable only while this service reads the demo cohort's
# synthetic note text and its published literature. A deployment of real PHI has to
# revisit this line — either obtain `us` capacity or put the model behind a boundary that
# answers.
GEMINI_LOCATION = "global"

# Vertex serving
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "readmission-endpoint")
FINAL_MODEL_PREFIX = "readmission-final-"
# The stage label the training pipeline writes with that name. Selection requires
# both: a record with the name and not the label was not written by a pipeline run,
# and the manual registration path (readmission-manual-*) cannot take the position.
PIPELINE_STAGE = "final"
BUNDLE_URI_OVERRIDE = os.environ.get("BUNDLE_URI")

# BigQuery feature source
DATASET = "readmission"
# Defaults to the HYBRID features table — the eval/demo cohort is the hybrid
# admissions (90000001+), whose feature rows live in readmission.hybrid_features.
# The real MIMIC-derived analytics_dataset_encoded table is out of scope for
# the demo and never carries the hybrid admissions.
TABLE = _validated_table_ref(
    os.environ.get("FEATURE_TABLE", f"{DATASET}.hybrid_features"),
    "FEATURE_TABLE", parts=(2,),
)
TABLE_FQN = f"{PROJECT}.{TABLE}"
ENTITY_ID_COLUMN = "hadm_id"

# The authorisation boundary for the demonstration.
#
# Written by scripts/agent/authorise_demo_cohort.py, which derives this table
# from the corpus the tools actually serve rather than selecting a cohort
# beside it: two selections of the same thing drift, and this one had. It held
# 24 admissions while the served corpus held 89, so 69 admissions were
# reachable by anything that reached the tools without passing the site's
# allowlist below (fixed 2026-09-18; the served corpus is authoritative).
#
# Authorization boundary (ECC-22): the demo site enforces cohort membership —
# it rejects any hadm_id not in its DemoPatient allowlist BEFORE calling the
# agent (S1-09). The tables the MCP tools read (hybrid_features/hybrid_notes)
# hold ONLY the demonstration cohort by construction — public MTSamples
# transcriptions re-keyed to demonstration admission ids (90000001+, MT-*
# notes), with the features derived from the note and the label derived from
# the model — so even a request that bypassed the site can only ever reach
# that cohort and never real MIMIC data. That argument covers the PHI risk and
# does not cover cohort membership, which is why the boundary has to be the
# same set as the data rather than a subset of it.
COHORT_TABLE = f"{DATASET}.demo_cohort"
COHORT_TABLE_FQN = f"{PROJECT}.{COHORT_TABLE}"

# RAG / Vector Search serving.
# The index endpoint is resolved by display name (like ENDPOINT_NAME) so the
# deployed id never needs to be hardcoded, and so teardown/stand-up just works.
INDEX_ENDPOINT_NAME = os.environ.get("INDEX_ENDPOINT_NAME", "readmission-rag-index")
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "rag_tree_ah")
# The embedding parameters are deliberately NOT here, and not settable. They decide
# which vector space the index was built in, so they are defined once in
# `retrieval/embed.py` and the serving path imports them from there: a value that can
# change silently and still return plausible neighbours is not a setting (gap 5 — the
# same argument that pins GEMINI_MODEL above).
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

# PubMed literature search (NCBI E-utilities).
#
# Read from the environment rather than pinned, unlike the model and the region
# above, and for the opposite reason: these are not properties of the world that must
# not drift, they are facts about who is calling. NCBI's usage policy asks every
# caller to identify itself, so the contact address belongs to the deployment.
#
# `NCBI_API_KEY` is recommended rather than required, and this comment has now been
# wrong in both directions. It first said the key "buys headroom rather than
# capability"; measurement on 2026-10-08 was that without a key the first search
# answered and the second was refused with 429. It then said the key was "NOT optional
# in practice", which the retry added the same day disproved — a bounded retry carried
# an unpaced burst of three searches.
#
# What is true: NCBI allows 3 requests per second per source address without a key and
# 10 with one, and a search is two requests. The limit is per ADDRESS, and Cloud Run
# presents one egress address for every instance, so the budget is shared across
# clinicians rather than granted to each. The retry covers an isolated burst; it does
# not cover sustained or concurrent load, where its pause becomes a visible wait.
# Register a key and set it here: it is free and takes minutes.
#
# An absent address must still not stop the service booting: the same process serves
# prediction and retrieval, and those must keep working on a deployment where nobody
# configured a literature contact. So it fails only this tool, at call time, with a
# structured error the caller can read.
NCBI_EUTILS_BASE = os.environ.get(
    "NCBI_EUTILS_BASE", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
)
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
# Per request, and there are two of them. Both finish well inside
# MCP_TOOL_TIMEOUT_SECONDS, which is the outermost of the three timeouts.
NCBI_TIMEOUT_SECONDS = positive_int_env("NCBI_TIMEOUT_SECONDS", 10)

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
#
# Moved to `gemini-3.8-flash` on 2026-11-13 (Sprint 04b). `gemini-3.1-flash-lite` was
# retired from the runtime stack rather than deprecated alongside: it was the only profile
# that accepted a `minimal` thinking level, its output allowance was being spent on
# deliberation rather than on answers, and keeping two pins alive meant every measurement
# carried an arm nobody was going to ship. Lifecycle date and replacement checked against
# Google Cloud's model-versions table the same day; the retirement date is in `MODEL_CHOICE`.
GEMINI_MODEL = "gemini-3.8-flash"

# Which model the agent asks for, which is the pin above unless a deployment names
# another. It is read from the environment rather than pinned because the whole point of
# the multi-model programme is to run the SAME code against several models and compare;
# what is pinned is the default, so an unconfigured deployment is exactly the deployment
# that shipped before this constant existed.
#
# A configured value must be a canonical id the adapter knows (see
# `services/agent/engine/gateway.py`); an id it does not know is served by the default
# provider profile, which is what happened to every id before the adapter existed. The
# execution record and `/health` report whichever one answered, so a run can never be
# attributed to the pin when it was not the pin.
AGENT_MODEL = (os.environ.get("AGENT_MODEL") or "").strip() or GEMINI_MODEL

# Where each provider's models are reached. The Google family has its own constant above
# for a measured reason; the partner families are addressed on the global endpoint unless
# a deployment pins them, because the regional endpoints carry per-region quota that the
# evaluation harness would then be measuring instead of the model.
PROVIDER_LOCATIONS = {
    "google": GEMINI_LOCATION,
    "anthropic": (os.environ.get("VERTEX_ANTHROPIC_LOCATION") or "").strip() or "global",
    "meta": (os.environ.get("VERTEX_META_LOCATION") or "").strip() or "global",
}

# The endpoint the model is reached on is `GEMINI_LOCATION` above, which is not
# `LOCATION` any more — see the note there. The swap is what forced the split: the
# pinned model has no regional endpoint in any region checked, and the constant it used
# to be reached through also names the vector index and the prediction endpoint, which
# do live in us-east1.

#: The JUDGE. A different model from the one under test, reached on a different
#: endpoint, and pinned here for the same reason everything else on this page is pinned.
#:
#: Why it must differ: a judge that loads the agent's own model cannot be relied on to
#: detect a failure mode the two share, so its verdicts would rest on the agent's blind
#: spots and measure agreement with itself. Until Sprint 03 the judge was
#: `gemini-2.5-pro` — a different generation and a higher tier than the pinned
#: `gemini-3.1-flash-lite`, which is as much separation as one vendor provides, and not
#: family independence. That limitation is now closed: the judge is `xai/grok-4.7`, which
#: shares no vendor or model family with any arm of the roster (`gemini-3.8-flash`,
#: `gemini-3.1-pro`, `muse-spark-1.3`, `claude-sonnet-5-5`), so a Grading candidate
#: cannot be graded by its own family (SCAR-19).
#:
#: Availability was measured rather than assumed, twice. On 2026-09-30 `gemini-2.5-pro`
#: answered on the regional endpoint and was NOT_FOUND on the global one; re-measured
#: 2026-10-28, both answered, and the default moved to `global` because the judge is the
#: one caller that issues several requests in the same second and a global endpoint pools
#: the per-region quota that produced the `429 RESOURCE_EXHAUSTED` rows blocking Gate 2.
#: `xai/grok-4.7` is reached through GCP's OpenAI-compatible MaaS endpoint, which is why
#: the location matters to the URL as well as to the quota: `global` and a region build
#: different hostnames (`judge_call.endpoint_url`).
#:
#: `VERTEX_JUDGE_LOCATION` overrides the location without a code change. The client reads
#: both at call time, so a run can be re-pinned to a region for one experiment.
JUDGE_MODEL = "xai/grok-4.7"
JUDGE_LOCATION = os.environ.get("VERTEX_JUDGE_LOCATION") or "global"

#: How hard the judge thinks before answering. Grok on GCP defaults to `"high"` when the
#: field is omitted, and the endpoint's output-token ceiling (21,000 OTPM at `global`) is
#: what bounds a grading pass, not the 150-second per-attempt timer: at `high` the
#: reasoning alone can hold a call open past the ceiling, and a retry then lands inside
#: the same window. `medium` is enough for a rubric that asks for six integers and two
#: short prose fields, and it keeps the OTPM budget spread across the run's concurrency.
#: A level, not a token budget, exactly like `GEMINI_REASONING_EFFORT` above: what
#: thinking spends is known only afterwards.
JUDGE_REASONING_EFFORT = (os.environ.get("JUDGE_REASONING_EFFORT") or "").strip() or "medium"

# Thinking and the answer share this allowance, and the model stops when it runs out
# without raising: the call returns 200 with empty text and finish_reason=MAX_TOKENS.
# A level bounds effort rather than tokens, so what thinking will spend is known only
# afterwards. Measured 2026-09-16: at 64 tokens both 3.x candidates returned empty text
# this way, and a synthesis probe spent 269 thinking tokens at the level set below.
#
# Pinned rather than read from the environment, like the model and the region: a deploy
# that lowered it would reintroduce that silent empty answer, which reaches the caller as
# a wrong-looking result rather than a failed deployment. Keep it generous.
#
# Raised 2048 -> 4096 on 2026-10-28, from a measurement rather than a margin of comfort.
# The `/meds` extraction answers with one JSON row per medication, and a sixteen-drug
# discharge list with multi-step tapers writes rows long enough that the reply ran out of
# allowance mid-string: on Gate 3 (`00d2`) the extraction for one record ended inside its
# fifteenth `dosage` at 1,729 characters, with no closing braces, and the truncated reply
# was refused as unparseable — which surfaced to the reader as "the section was retrieved,
# but no medication could be read from it". The same record on the previous revision
# cleared the old ceiling by about a hundred characters, so the margin was an accident of
# wording. Truncation is silent, costs a whole turn, and reads as a retrieval failure, so
# the allowance is set where a discharge list of that size has room for thought as well as
# rows. It is a ceiling, not a spend: a turn that needs 200 tokens still uses 200.
#
# Raised 4096 -> 16384 on 2026-11-13, for a reason the 2026-10-28 note does not cover:
# this allowance is shared with THINKING, and a model that thinks harder at the same effort
# level therefore has less room for the answer, not more. On Sprint 04 pre-flight
# (`agent-00286`, `AGENT_MODEL=gemini-3.8-flash`, item `sel_t1_01`, 90000010 `/meds`) the
# extraction spent 3,929 tokens deliberating and had 163 left, finished MAX_TOKENS, and the
# JSON never closed — the same reader-facing sentence as the 2026-10-28 incident, from the
# opposite cause. The ceiling has to clear deliberation before it can bound an answer, so it
# is sized for a reasoning model rather than for a discharge list: 16,384 is comfortably
# inside every Gemini 3.x per-request output limit, and it remains a ceiling rather than a
# spend.
GEMINI_MAX_OUTPUT_TOKENS = 16384

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
# model under test, which is what `evaluation/platform/judge_client.py` is today.
MODEL_CHOICE = {
    "model": GEMINI_MODEL,
    "decided": "2026-11-13",
    "tier": "entry",
    "cheaper_alternative": None,
    "retires": "2027-09-02",
    "replacements": (),
    "previous": "gemini-3.1-flash-lite",
    "previous_retired": "2027-05-07",
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
            "evaluation/platform/judge_client.py. It needs two things that did not exist on "
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
#
# Dangerous content moved to BLOCK_ONLY_HIGH on 2026-11-11, and the measurement is
# the reason. `dev_t3_02` turn 2 asked which discharge medication has a taper and how
# it is dosed; the reply reported an opioid step-down of 20 mg to 10 mg every twelve
# hours and was withheld with `finish_reason=SAFETY`, `HARM_CATEGORY_DANGEROUS_CONTENT`
# blocked at `probability: MEDIUM` with `severity: HARM_SEVERITY_LOW` — a dosing
# schedule, rated low severity and blocked for the category it was written in. Under
# this threshold set the same rating is reported rather than withheld, which is the
# trade this product makes: a median-severity dangerous-content rating on a discharge
# note is not the failure mode the filter exists to stop, and withholding the answer
# serves the reader a sentence about a safety filter instead of a dose. The other
# three categories stay where they were; nothing about this moves them.
GEMINI_SAFETY_THRESHOLDS = {
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
}
