# Layer 7 — Data & indexes

Status: audited 2026-09-17, and independently reviewed the same day. Eight gaps recorded in
section 5, with the decision that closes each; none is closed yet. A gap is described once —
defect, decision, outcome — rather than restated per section.

---

## 1. What the layer is

Everything this system says about a patient is read from here: the discharge note a
citation points at, the vectors that rank candidate passages, the feature row the model
scores, and the artifacts every offline step leaves behind. The layer is fed by pipelines a
person submits, and it is read-only from the request path — no live request writes a row, a
vector or an object.

It is not the tool boundary, which layer 5 owns: that layer decides what `rag_search`
accepts and may return, and re-checks every resolved note against the requested admission.
It is not the model (layer 6) or the prompt (layer 3). And it is not the data-protection
regime: the MIMIC-IV agreement governs what this layer may hold, and layer 11 owns that
discipline.

```
  ┌───────────────────────────────────────────────────────────────┐
  │  LAYER 5 — the tool boundary                                  │   owns the call, the
  │  rag_search · rag_search_sections, tools/retrieval.py 92      │   filter it passes, and
  │  the feature row, dependencies/features/bigquery_source.py    │   the isolation re-check
  └──────────────────────┬────────────────────────────────────────┘
     IN │ one admission id, and a query phrase
        v
  ┌───────────────────────────────────────────────────────────────┐
  │  LAYER 7 — the rows, the vectors and the artifacts            │   owns what exists,
  │  notes tables (BigQuery) · Vector Search index                │   what built it, and
  │  454 demo vectors / 555,770 MIMIC · feature tables            │   when it is rebuilt
  │  GCS ingest artifacts, gs://…-mlops/rag/ingest/<ts>/          │
  └──────────────────────┬────────────────────────────────────────┘
     OUT │ chunk ids with distances · note text · one feature row
        v
  ┌───────────────────────────────────────────────────────────────┐
  │  THE OFFLINE LOOP — writes only, never called by a request    │   chunk → embed →
  │  rag_ingest_pipeline.py 46 · deploy_rag.py                    │   gate → index →
  │  Dataform definitions/ · scripts/agent/*.py                   │   deploy
  └───────────────────────────────────────────────────────────────┘
```

What crosses each boundary, and what does not:

| Boundary | Crosses | Does not cross | Who decides |
|---|---|---|---|
| BigQuery → ingest | The notes of one split, read once per run — the chunk step joins the split table and binds `split_name` as a parameter (`chunk_notes.py` 62–68) | The other splits: the index is built from one split only, so a training row cannot reach a served citation | The pipeline's parameters decide the scope |
| Ingest → index | One JSONL artifact of datapoint ids, vectors and admission restricts (`embed.py` 46) | Note text of any kind: the index stores vectors and ids, and the text is re-fetched and re-chunked at serve time | `embed_chunks` writes the artifact; `build_index` reads it |
| Index → serving | Chunk ids with distances, scoped by the admission restrict in the query (`retrieval.py` 311–320) | Another patient's chunks: the restrict bounds the ranking, and every resolved note is re-checked in BigQuery (`retrieval.py` 197–228) | The tool passes the filter; layer 5 refuses a mismatch |
| Tables → serving | One feature row per admission, one note text per admission | Any write from the request path: the serving code contains no insert, update or delete | The MCP service reads; the pipelines write |
| Offline loop → the rest | New indexes, new ingest artifacts, tables rebuilt in place | Anything already serving: a rebuild does not touch the deployed index | A person submits the pipeline; nothing triggers it |

Terms used in this document:

| Term | Meaning |
|---|---|
| Corpus | One notes table plus the split that scopes it. Two are configured: `mimic` (real MIMIC-IV discharge notes, under the data use agreement) and `demo` (89 MTSamples transcriptions — public, de-identified sample notes — re-keyed to demo admission ids). |
| Chunk | One section of one note, packed to a character budget. The id is deterministic: `{note_id}:{section}:{ordinal}` (`retrieval/chunking.py` 166). |
| Datapoint id | A chunk id folded to Vector Search's allowed alphabet (`embed.py` 24). It is the only join key between the index and the notes table. |
| Restrict | The admission filter attached to each vector at index time and applied in the query, so ranking happens inside one admission. |
| Ingest artifact | The JSONL of ids, vectors and restricts that `embed_chunks` writes and `build_index` indexes. Versioned by run: `rag/ingest/<timestamp>/`. |
| Data fingerprint | A hash of the source tables' modified time and row count, folded into the chunk step's inputs so KFP caching invalidates when the data changes (`rag_ingest_pipeline.py` 132–150). |
| Eval gate | Two different gates: the integrity gate inside the ingest pipeline, and the recall gate on the deploy step. |

**Layer 7 is: the documents, the vectors and the tables the system reasons from — built
offline, read only at serve time.**

---

## 2. What Google requires

| # | Requirement | Level |
|---|---|---|
| B1 | Three separate subsystems: data ingestion, serving, quality evaluation. They share the database layer; they do not share code paths. | MUST |
| B2 | Ingestion is event-driven and offline: upload → Pub/Sub → function → parse → chunk → embed → vector store. Live requests never mutate the index. | MUST |
| B3 | Serving uses the same embedding model and parameters as ingestion. | MUST |
| B6 | Index maintenance (rebuild/refresh) is a scheduled operational task, versioned alongside the dataset (BigQuery / AlloyDB / Cloud Storage). | MUST |
| E1 | Version control for prompt templates, chain code, external datasets, adapter weights and container images. Only the data half is audited here: the BigQuery tables and the Cloud Storage objects. | MUST |

B3 is claimed by layer 5 as well, which audits the call path; this layer audits the identity
of the model at both ends of it. Two requirements in the same group belong elsewhere and are
not audited here: B4 (serving screens responses through safety filters) is layer 4's plane,
which audits it, and B5 (serving logs requests and loads responses to BigQuery) is layer
10's.

---

## 3. How this application implements it

### The corpus is configuration, and the embedding parameters deliberately are not

Two corpora are declared side by side in one committed file: `mimic`, reading
`mimiciv_note.discharge` with 555,770 expected vectors on an `e2-standard-16`, and `demo`,
reading `readmission.hybrid_notes` with 454 expected vectors on an `e2-standard-2`
(`services/mcp/rag_config.yaml` 18–37). `RAG_CORPUS` selects one, so switching corpora is
one value rather than a scatter of environment variables (`retrieval/config.py` 95–99).

The embedding parameters are the deliberate exception, and the reason is written where a
reader would look for them: they decide which vector space the index was built in, so a
copy in the YAML could build an index the serving path then queries in a different space
and still return plausible neighbours. They live once, as five constants —
`gemini-embedding-001`, 768 dimensions, `RETRIEVAL_DOCUMENT` at index time and
`RETRIEVAL_QUERY` at serve time, and `hadm_id` as the restrict namespace
(`retrieval/embed.py` 17–21) — and the config loader raises if an `embedding:` block
reappears in the YAML (`retrieval/config.py` 86–93).

### The ingest pipeline

`services/mcp/pipelines/rag_ingest_pipeline.py` compiles four steps whose every knob comes
from that config (`rag_ingest_pipeline.py` 46–122, `submit` 152–212). Data moves only
through BigQuery and Cloud Storage.

- *Chunk.* `chunk_notes` reads the notes joined to the split, keeps only whitelisted
  narrative sections, and emits chunks whose ids are deterministic — a duplicate id is a
  hard failure rather than a silent overwrite (`chunk_notes.py` 42–93). The chunk manifest
  records counts per section and the packing budget (100–107).
- *Embed.* `embed_chunks` imports the same four constants the serving path uses, embeds in
  batches with retries, checks each record's dimension, and refuses to write an artifact
  whose length disagrees with what it embedded (`embed_chunks.py` 66–79, 179–210). A
  previous run's artifact can be passed to reuse embeddings, so a chunking fix re-embeds
  the changes rather than the corpus (`embed_chunks.py` 143–155).
- *Gate.* `eval_ingest` validates the artifact — vector count against the corpus's expected
  count, one dimension set, no duplicate datapoint ids — and fails the run before an index
  can be built from it; its report is a typed HTML artifact so the Vertex UI renders it
  (`eval_ingest.py` 25, 58–70, 84–94, 96–123).
- *Index.* `build_index` uploads the artifact, builds a brute-force ground-truth index from
  a sample and a `tree-ah` serving index from the whole artifact, and asserts both built
  vector counts against what was expected — a shortfall fails the run instead of shipping a
  quietly incomplete index (`build_index.py` 24, 45–70, 72–90). It writes a manifest naming
  both indexes and the ingest directory (92–103).

Step caching is enabled on this pipeline, which is only safe because of one input: the data
fingerprint. KFP caches on code and inputs and not on table contents, so without it a
re-submission after the notes changed would reuse stale chunks and build an index that does
not match the data (`rag_ingest_pipeline.py` 132–150, 203).

### What a run leaves behind, measured

The demo ingest run `rag-ingest-20260902141148` produced 454 vectors at 768 dimensions, and
its index manifest names `rag-tree-ah-20260903001405` (`…/indexes/5889384500600766464`)
alongside the ingest directory `gs://…-mlops/rag/ingest/20260903001405/full/`. Two details
about that run belong in the record rather than out of it: its artifact store holds only
the `build-index` task, so the chunk, embed and gate steps were served from cache on that
run — the same code and the same inputs, so the same artifact the gate had passed, but the
run is not evidence that the gate executed that day; and the count is reproducible without
any run at all, because `count_hybrid_chunks.py` 1–12 chunks the same notes with the same
section whitelist and the same `pack_to` and prints the number the pipeline's
`expected_vectors` is set from. The MIMIC corpus's counterpart is
`rag-tree-ah-20260806221043` (`…/indexes/2371299135438454784`) at 555,770 vectors, whose
embedding manifest — written 2026-08-06 — records `gemini-embedding-001`, 768 dimensions
and `RETRIEVAL_DOCUMENT` beside the batch counts. Both ingest artifacts sit under a
timestamped prefix, and the index display name carries the same timestamp, so an index can
be traced back to the artifact that produced it.

The indexes are created by the pipeline and **deployed by a separate script**:
`scripts/agent/deploy_rag.py` takes the newest `rag-tree-ah-*` index, refuses anything above
the synthetic-scale limit so the real corpus can never land on a public endpoint
(`_deploy_guard.py` 10–18), deploys it under a staging id while the live id keeps serving,
gates the promotion on a recall report when one is supplied, and promotes or rolls back
(`deploy_rag.py` 1–12, 70–86, 124–149, 154–209).

### The serve-time read path

`rag_search` resolves the index endpoint by display name and caches it for the process
(`retrieval.py` 92–106), then queries with the admission restrict passed into the vector
query so ranking happens inside one admission (`retrieval.py` 311–320). What comes back is
ids and distances; the *text* is fetched from the notes table by note id and re-checked
against the requested admission, where a foreign row raises `IsolationViolation` instead of
being served (`retrieval.py` 197–228). There is no chunk table: passage text is recomputed
from the note by the same deterministic chunker that built the index (`retrieval.py` 250,
555), which is also why `rag_search_sections` needs no index at all — it re-parses the note
and returns one passage per section, marked `retrieval: deterministic` rather than carrying
a score it does not have (`retrieval.py` 529–571).

The feature side is the same shape: the served path reads one row from a BigQuery table
whose name comes from config (`services/mcp/config.py` 74–84), restricted to the model's 49
feature names (`dependencies/features/bigquery_source.py` 54–60), and the training table is a different table in the same dataset
(`readmission.analytics_dataset_encoded`, 219,744 rows, rebuilt in place by Dataform).

### The tables are built by Dataform, the demo tables by scripts

`definitions/` holds the graph that produces the training data — sources, staging,
features, marts, and assertions — and `workflow_settings.yaml` pins its project and
dataset. The served demo tables are produced by scripts instead: `hybrid_notes` (89 rows)
is the demo corpus the index is built from and the table the retrieval text is read from,
and `hybrid_features` (89 rows) is the feature table the predict path reads by default. Both
are regenerated by `scripts/agent/` scripts rather than by the Dataform graph (layer 6
recorded the feature-side consequences of that in its gap 8).

### The quality evaluation

Two evaluations exist and they are different things. The integrity gate above runs inside
the pipeline and can stop an index from being built. The *recall* evaluation runs as a
separate job: 100 queries, seed 42, top_k 10, against a deployed index, folding into
recall@1/5/10 (`rag_config.yaml` 53–58). The recorded reports are recall@10 0.992 for the
demo corpus (2026-09-04) and 0.828 for the MIMIC corpus (2026-08-07) — the second is below
the 0.90 threshold the config sets, and the deploy script is where that threshold would be
enforced, because the report is an input to `deploy_rag.py` rather than a step in the
pipeline (`gate.py` 18, `deploy_rag.py` 124–149).

### The data boundary

The MIMIC-IV corpus stays where the agreement allows it: the notes table is real
(`mimiciv_note.discharge`, 331,793 rows), it is never copied into the repository, and the
`data/` tree is gitignored so no note text, fixture or artifact is committed (`.gitignore`
33, the eval-results directory at 73, and the notes artifact itself at 78 — the
features-only cohort file is tracked, the notes file deliberately is not). What is committed is the *configuration* of both corpora, the scripts that build both
of them, and the code that reads them.

The demo corpus is not MIMIC data and not generated text either: `readmission.hybrid_notes`
holds 89 MTSamples transcriptions — a public, de-identified corpus of sample procedure
notes — re-keyed to demo admission ids (90000001 onwards), each paired with a 49-feature
row derived from the note's own story and scored by the served model. The provenance is
worth spelling out because three scripts built it: the first 24 admissions were hand-picked
and scored (`build_hybrid_cohort.py` 1–12), 84 more were added from the MTSamples manifest
(`build_hybrid_cohort_108.py` 1–11), and 19 were then removed as cohort-inclusion
violations — neonatal and under-18 notes in a corpus whose model was trained on adults
(`prune_inclusion_violations.py` 1–11). 108 minus 19 is the 89 that serve, and
`load_hybrid_notes.py` 1–13, 61 is where they land in BigQuery. A second, fully
fictional corpus exists beside it (`readmission.synthetic_notes`, 24 rows, rendered by
`generate_synthetic_notes.py`), and is not what the demo serves. The consequence for this
layer is that the demo index is built from public data by construction, and the scale
guard exists to keep the corpus that is *not* public off the same endpoint.

---

## 4. Current state against the requirement

| Requirement | State | Why |
|---|---|---|
| B1 — three subsystems, separate code paths | **Partly** | Three separately invoked things exist: ingestion is a KFP pipeline (chunk → embed → gate → index), serving is the MCP service (read-only), and quality evaluation is a recall job that queries a deployed index. Two deviations from the requirement as written are real and deliberate, and neither is claimed as met. First, ingestion and serving import the same library — `services/mcp/retrieval/` carries the chunker and the five embedding constants into both — which is a shared code path in the plain sense of the requirement, chosen because a forked chunker or a forked embedding constant is the failure this layer is most exposed to. Second, one half of the evaluation subsystem is not separate at all: the integrity gate is a step inside the ingestion pipeline (`rag_ingest_pipeline.py` 104–116), and only the recall evaluation runs as its own job. What the requirement is really protecting — that a change in one subsystem cannot silently alter another's behaviour — holds, because the shared code is pure and deterministic and both sides read the same BigQuery and Cloud Storage. |
| B2 — event-driven and offline, never mutated by a live request | **Not met** | The offline half holds and is evidenced below. The event-driven half does not exist: ingestion is a hand-submitted pipeline over a BigQuery table. No Pub/Sub topic or subscription exists in the project, no function is deployed (`gcloud functions list` returns nothing), and the Eventarc API is not enabled, so no trigger can exist; nothing in the repository reacts to a note arriving. A new note is invisible to the index until a person submits the pipeline. |
| B3 — same embedding model and parameters both sides | **Met** | One definition in `retrieval/embed.py`, imported by the serving path, by the pipeline's embed step, by the config loader, and by the tests; the YAML is refused if it carries an `embedding:` block; the recorded ingest manifest states the model, the dimension and the task type. What the artifacts do not do is name the model on the *pipeline* path — see gap 5, which is about the record rather than the identity. |
| B6 — scheduled refresh, versioned alongside the dataset | **Not met** | There is no schedule of any kind: the Cloud Scheduler API is enabled by the environment script (`scripts/setup_environment.sh` 24) and no job exists, and nothing in the repository schedules an ingest, a rebuild or a deploy. Refresh is manual end to end. The versioning half is partly there — ingest artifacts and index names carry the run's timestamp — but the tables themselves are rebuilt in place and the object store has no versioning. See gap 3. |
| E1 — datasets and objects under version control | **Partly** | What is under version control: the code that builds every dataset, the corpus configuration, and the Dataform graph including its split assertion. What is not: the tables (materialised in place; no snapshot is taken and no dataset version is written to any artifact), the objects (the bucket has no object versioning enabled, and the embedding cache is a single fixed path that each run overwrites), and the index (its name carries a timestamp, but no artifact states which dataset *version* it was built from beyond the ingest directory it points at). The platform offers the cheap version of all three — BigQuery table snapshots and time travel, GCS object versioning — and none is in use, so the honest statement is that the data is not versioned, not that it cannot be. What the layer does have is the reproducibility the requirement is really after: the SQL, the corpus configuration and the code are committed, and an index can be traced to the artifact that produced it. |

**Evidence for the half of B2 that holds:** the serving code contains no insert, update,
delete or create statement, and the request path reads rows, vectors and objects. What it
does not do is never touch an index at all: a rebuild creates indexes rather than mutating
one (`build_index.py` 58–61 creates a brute-force ground-truth index and 63–90 the serving
index), and indexes are also deleted outside the repository — this project's teardown
removes them, and a brute-force index the demo run created no longer exists. So the claim
that survives is the one the requirement makes: no *live request* mutates the index.

---

## 5. Gaps

Each entry states the defect, the evidence for it, and the decision that closes it. Ordered
by how much of the requirement each one breaks.

**1 — The index that serves is not deployed, and nothing in the system says so.**
Verified on 2026-09-17 against the live project: the demo index exists
(`rag-tree-ah-20260903001405`, 454 vectors) and **no index endpoint exists at all** in
`us-east1` — `MatchingEngineIndexEndpoint.list()` returns empty and
`gcloud ai index-endpoints list` agrees. Resolving it the way the service does raises: the
repo's own code and config produce `RuntimeError: No index endpoint with
display_name='readmission-rag-index'`, and the deployed `mcp-server` carries no override
that could change that (its environment is `PROJECT_ID`, `LOCATION`, `FEATURE_SOURCE`).
The consequence, measured through the tool: `rag_search` returns
`{"error": "search_failed", "message": "The retrieval index could not be reached."}` while
`rag_search_sections` still answers, because the section path re-parses the note and does
not touch the index. So free-text note questions are degraded right now and section
summaries are not, and the difference is invisible from outside — both look like a model
that chose not to quote.

Why this is a defect rather than a state: nothing *before a request* observes it. Two facts
have to be held together to see why that matters. The service's `/health` route is
deliberately shallow and says so — a deep check "would call Vertex and BigQuery, turning
every probe into billed API calls on a service that scales to zero" (`server.py` 43–52) —
which is the right call for a liveness probe and is not the same thing as a readiness check
anyone can run. And teardown removes the endpoint as a matter of routine, so its absence is
an expected state; there is simply no way to tell an expected absence from a failed deploy
except by making a request and reading the sentence that comes back. The first demo request
of the day is therefore the test.

The evidence above is a point-in-time fact and should be read as one: within the hour, both
this endpoint and the predict endpoint were redeployed deliberately (see the note at the
end of this section), and a probe run after that deploy found the endpoint present and
answering with a 503 because the index was still provisioning. The reproducible check is
the command, not the result: `PYTHONPATH=<repo root> python -c "from services.mcp.tools
import retrieval; retrieval._index_endpoint()"`, which raises when nothing is deployed and
returns the resource name when something is.

Fix. Decide the standing state and make the system able to report it, rather than adding a
probe that contradicts a cost decision already made. Three things, in order of cost: state
in the runbook that teardown removes an endpoint retrieval needs, so "no citations" has an
expected cause before it has an investigation; have the agent distinguish the two cases in
what it tells the user — a retrieval error that says *note search is unavailable* rather
than a sentence that reads like a retrieval gap; and only if the demo is public for long
enough to matter, add a readiness check that is not on the request path and not on a probe
(e.g. run at deploy time, and recorded), so the cost argument in the health route stays
intact.

**2 — Ingestion has no trigger: a new note is invisible until someone submits the pipeline.**
The pipeline reads the notes table and reads it well, but nothing runs it. Measured on
2026-09-17: no Pub/Sub topic or subscription exists in the project, `gcloud functions list`
returns nothing, and the Eventarc API is not enabled, so no trigger or channel can exist.
The repository contains no function deployment and no trigger, and the runbook's build
order is a list of commands a person runs (`docs/operations.md` 18–25). This is the first
clause of B2 and it is absent in substance, not in detail: the requirement's path begins
with an upload and this system's begins with a table that Dataform or a script has already
written. (The Cloud Functions API was enabled in this project on 2026-09-17 while auditing
this; it changes nothing here — no function is deployed — and it is recorded so the
finding is not read as an artefact of a disabled API.)

Fix. Decide which it is, and write it down. If the product needs freshness, the honest
shape is an object landing path — notes written to a bucket, a bucket event, a function that
writes the table and submits the ingest — because a BigQuery table does not emit change
events. If it does not need freshness, say so in the operations document and in section 4 of this
document, which is where the state verdicts live, so the requirement is recorded as an
accepted deviation rather than left reading as met. Building an event path to satisfy a MUST is the first option; declaring the manual
step is the second, and it should be declared rather than implied.

**3 — Nothing refreshes the index on a schedule, the dataset has no version, and two
artifacts that a refresh would depend on are themselves overwritten.**
Cloud Scheduler is enabled and empty (`gcloud scheduler jobs list` returns nothing in
`us-east1`, `us-central1` or `europe-west1`, and no script creates a job). The ingest
pipeline, the recall job and the index deploy are all hand-submitted, so an index is
refreshed when a person remembers. Two of B6's three clauses are therefore unmet together
with E1's data half: the *rebuild* is neither scheduled nor attached to the dataset, and
the *dataset* has no version to be refreshed against — the Dataform graph rewrites its
marts in place, and the demo tables are regenerated by scripts that overwrite the tables
the serving path reads while it reads them.

Two smaller things belong to the same defect because they are what a refresh would use.
The embedding reuse source is a single fixed path —
`rag/embeddings/ingest/embed_ingest.jsonl.gz` is both the default `previous_ingest_uri` of
the pipeline and the object `scripts/agent/embed_chunks.py` writes — so every run
overwrites the artifact an earlier index was partly built from, and "reuse the previous
embeddings" names whatever the last run left rather than the embeddings of the index being
rebuilt. And the artifact the gate validates is not a refresh signal: the cache can and did
serve `chunk-notes`, `embed-chunks` and the gate from an earlier run (see section 3), which
is correct for identical inputs and means a "successful" ingest is not by itself evidence
that today's data passed today's gate.

The mechanism that makes this cheap is already built: the ingest pipeline carries a data
fingerprint that invalidates KFP's cache when the source tables change
(`rag_ingest_pipeline.py` 132–150), so a scheduled submission would do the right thing
without a second mechanism. What is missing is a schedule, a decision about who pays for
one, and a version on the dataset for the schedule to compare against.

Fix. Pick the cadence and make it a job: a Cloud Scheduler trigger that submits the ingest
pipeline with the corpus's parameters, and — because the MIMIC corpus is billable and
large — a rule about which corpora it may refresh. Record the dataset version the index was
built from where the index can be read back, so "refreshed" means "against a version"
rather than "at some time". If the answer is that a public demo does not need one, the
operations document should say the refresh is manual, and section 2 of this document should
carry B6 as an accepted deviation rather than a met requirement.

**4 — The recall gate is optional and its evidence is discarded.**
`deploy_rag.py` promotes a new index only if the recall report passes the configured
thresholds — and the report is an optional argument: with no `--recall-report`, `_gate_passes`
returns True and the deploy proceeds (`deploy_rag.py` 124–129). The docstring says the
caller is responsible for running the recall job first, which makes the gate a convention
rather than a control, and the script the runbook and the endpoint launcher actually call
(gap 6) does not run it at all. Separately, when the gate does run it writes its report to
`/tmp/rag_eval` (`deploy_rag.py` 147), so the verdict that allowed a promotion is not kept
anywhere; the numbers in this document's section 3 come from a report the recall job left in
Cloud Storage, not from any deploy.

Fix. Require the report on whichever script is the deploy, or require an explicit flag that
says the gate was waived and prints why; make the launcher and the runbook call the gated
script; write the gate's artifacts where they can be found afterwards, beside the ingest
artifact the index was built from, with the corpus, the index and the data fingerprint in
the filename; and record the promoted index's recall in the deployment's own record so the
serving side can state which numbers were true when it was promoted.

**5 — The ingest manifest does not name the embedding model.**
The pipeline's embed step writes a manifest of counts, dimensions and timing
(`services/mcp/pipelines/components/embed_chunks.py` 143–155, and the artifact this
document's numbers come from) — but not
the model name or the task type. The standalone script's version of the same manifest does
record them, model and task type together with dimensions
(`scripts/agent/embed_chunks.py` 203–214), which
is how the 2026-08-06 MIMIC manifest came to carry them. So the identity of the embedding
model behind an index is knowable from the code that produced it and from one older
manifest, and not from the artifact the pipeline writes.

B3 itself is met — there is one definition and the config refuses a second — so this gap is
about the record, and it costs one line in a dictionary. It matters because the index is a
deployment fact: a reviewer asking "which model built the index that answered this query"
deserves an answer from the artifact, not from a `git blame` of the file that reads it.

Fix. Put `model`, `task_type` and `dimensions` in the pipeline's embed manifest, the same
fields the standalone path already writes, and assert in a test that the manifest the
pipeline writes names the model the serving path imports.

**6 — The runbook and the endpoint launcher name two different deploy scripts, neither of which is the gated one.**
`docs/operations.md` 19–20 tells a reader to run `deploy_synthetic_rag.py` to deploy the
Vector Search index; the same document's script table lists `scripts/agent/deploy_rag.py`
as the deployment script (32); `scripts/launch_endpoints.sh` 13–16 runs `deploy_synthetic_rag.py`;
and `deploy_rag.py`'s own docstring says it replaces the manual `deploy_synthetic_rag` /
`wait_rag_deploy` / `chain_rag_deploy` sequence with one scripted, gated step (1–12). So
there are two documented deploy paths and the document does not say which is current.

To be exact about what the older path does and does not risk: it applies the same
synthetic-scale guard — `deploy_synthetic_rag.py` 31 imports `_deploy_guard` and calls it at
54, and `deploy_rag_endpoint.py`, which it delegates to, asserts it again at 79 — so the
real corpus cannot reach the public endpoint by this route either. What it does not do is
the blue-green deploy or the recall gate, which are only in `deploy_rag.py` (gap 4). The
defect is therefore a documentation defect with a gate-skipping consequence, not a
DUA-exposure one.

Fix. Pick one deploy path, say so in the runbook and the launcher, and delete or mark the
other. A runbook that names a superseded script is a defect in the runbook; here it also
means the documented demo path bypasses the recall gate.

**7 — The cohort that authorizes the demo is not the cohort the tools serve.**
Measured on 2026-09-17: `readmission.demo_cohort` holds 24 rows (admissions
90000001–90000024) while `readmission.hybrid_notes` and `readmission.hybrid_features` hold
89 (90000001–90000108). The configuration names the first as the authorization boundary
and says what enforces it: "the demo site enforces cohort membership — it rejects any
hadm_id not in its DemoPatient allowlist BEFORE calling the agent"
(`services/mcp/config.py` 87–96). So the allowlist the site is described as enforcing covers
24 admissions and the corpus the tools will answer for covers 89, which leaves 65
admissions inside the served data and outside the named boundary — each of them a public
MTSamples note rather than MIMIC data, which is why this is a consistency defect rather
than an exposure, but the two numbers cannot both be the cohort. The tree also holds two
further answers: `data/agent/demo_cohort.json` (89) and
`data/agent/demo_fixtures/demo_cohort.json` (108). And the script the comment names as the
builder of the boundary does not exist — `scripts/build_demo_cohort.py` — while the table is
actually written by `scripts/agent/seed_demo_cohort.py` 267.

Layer 11 owns whether the site's enforcement is correct; what this layer owns is that its
own tables state three different cohort sizes and that the comment explaining the smallest
of them points at a file nobody can run.

Fix. Decide which cohort is authoritative — the authorization list, the served corpus, or
both with the served corpus as a documented superset — write the table from that decision,
say which population a refusal is defined against, and point the comment at the script that
writes the table. Then the statement "a request can only reach demo rows" is checkable
against one number instead of three.

**8 — The artifacts that guard the data describe its provenance wrongly.**
Three places say the demo corpus is synthetic, and the guard that keeps the real corpus off
the public endpoint reasons from that description: `services/mcp/config.py` 87–94 ("the
tables the MCP tools read … hold ONLY the synthetic hybrid cohort … never real MIMIC
data"), `docs/data-and-model.md` ("The public demonstration uses synthetic notes and
synthetic structured features"), and the deploy guard's own refusal message ("the real
MIMIC-derived corpus, not the synthetic demo cohort", `_deploy_guard.py` 13–18). The notes
are neither synthetic nor MIMIC: they are real MTSamples transcriptions, a public and
de-identified corpus — as the RAG configuration itself says ("89 MTSamples notes",
`services/mcp/rag_config.yaml` 33) and the loader states in its first line ("hybrid (real
MTSamples) notes", `load_hybrid_notes.py` 1). The intent of the claim is right and is the
important part — no MIMIC-derived note text reaches the demo path — but a reader takes
"synthetic" to mean the text was invented, which is a different and weaker statement than
"public sample data", and it is the kind of description that ends up in a compliance
answer. Two smaller instances of the same class sit beside it: `retrieval/embed.py` 8–12
still describes a BigQuery chunk store and an id-to-chunk-row mapping that the serving path
no longer performs (there is no chunk table; the id is parsed instead), and
`scripts/launch_endpoints.sh` 15 still calls the demo index "~560 vectors" when it is 454.

Fix. Say what the data is, once, in the artifacts that make claims about it — public
MTSamples transcriptions, re-keyed to demo admission ids, with features derived from the
note and labels derived from the model (layer 6's gap 11 owns the label half) — and correct
the stale chunk-store docstring and the stale vector count while the files are open. A
provenance claim that is wrong in the code that enforces the boundary is worth fixing even
when the behaviour is correct, because the next person to change the boundary will believe
it.

---

## 6. Interview questions this layer answers

**Where does the text the model cites actually come from?**
From the patient's discharge note in BigQuery, not from the index. The index returns
datapoint ids and distances; the id is mapped back to a note, the note text is fetched and
re-checked against the requested admission, and the passage is recomputed by re-running the
same deterministic chunker that built the index. That is why there is no chunk table to go
stale: the only stored artefact of chunking is the vector artifact, and the text is
reproducible from the note.

**How does a note get into the index?**
A person submits the ingest pipeline: it reads the notes of one split, chunks them
deterministically, embeds them in batches with `gemini-embedding-001`, validates the
resulting artifact — count, dimensions, duplicate ids, empty embeddings — and only then
builds the index, asserting that the vector count it got is the count it expected. The
whole run is recorded: the chunk manifest, the embedding manifest, the eval report as a
rendered artifact, and an index manifest naming the indexes and the ingest directory.

**Is the served index built with the same embedding model the query uses?**
Yes, and it cannot drift by code: the model, the dimension and both task types are four
constants in one module, imported by the serving path, by the pipeline, by the config
loader and by the tests, and the configuration file is rejected if it grows an `embedding:`
block. What the pipeline's own manifest does not do is name the model — the dimension only
— so the record is weaker than the code; that is gap 5.

**How is the index refreshed?**
Today: by hand, and rarely — a pipeline submission followed by a deploy script. Which deploy
script depends on which document you read (gap 6): `deploy_rag.py` picks the newest index,
refuses anything at real-corpus scale so the public endpoint can only ever serve the demo
corpus, deploys under a staging id and promotes only if the recall report passes; the script
the runbook and the endpoint launcher actually name, `deploy_synthetic_rag.py`, deploys
straight to the live id and never sees a recall report (gap 4). Nothing is scheduled, and
the dataset has no version to refresh against, which is gap 3.

**Which patients may the demo answer for?**
As recorded today, that is ambiguous, and gap 7 is the defect: the table the configuration
names as the authorization boundary holds 24 admissions, the corpus the tools read holds 89,
and two fixtures in the tree hold 89 and 108. Nothing that reads a note is bounded by the
smallest of those numbers, and no single number is authoritative yet.

**What keeps the real corpus off the public endpoint?**
A scale check on every script that can put an index on it: the guard refuses an index above
100,000 vectors (`_deploy_guard.py` 10–18), and all three deploy paths import it — the real
corpus is 555,770 vectors and the demo corpus is 454. It is a coarse check, chosen because
PSC or VPC would be the real control and the public demo does not have one; layer 11 owns
the network half. Teardown also touches the endpoint (`scripts/agent/teardown.py` 99–142)
and imports no guard, which is the right asymmetry: it removes.

**What stops one patient's note reaching another patient's answer?**
Two independent things, both audited in layer 5: the admission restrict is passed into the
vector query so ranking is scoped before it happens, and every note the index returns is
re-fetched from BigQuery and re-checked against the requested admission, where a foreign
row raises and the tool refuses to serve the passage.

---

**Recorded, not owned here:** what a tool accepts and returns is layer 5's; the model and
its registry entry are layer 6's; the prompt and the citation contract are layer 3's; the
DUA, retention, PII and network isolation of the real corpus are layer 11's; the logs,
traces and drift signals over the serving path are layer 10's; the evaluation harness that
judges answers against a golden set is layer 9's; and the CI that would run these ingest
and deploy steps is layer 12's.
