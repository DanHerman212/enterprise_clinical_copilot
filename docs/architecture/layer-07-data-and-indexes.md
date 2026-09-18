# Layer 7 — Data and indexes

Status: audited 2026-09-17, independently reviewed the same day, and restructured on
2026-09-18 into the four-part form used from here on: an architectural overview, the
requirements the layer is held to, a brief account of the current implementation, and the
gaps with their remediations. No gap remains open; one entry is withdrawn and seven are
closed, each kept in place so that the record of what was decided is not lost.

---

## 1. Overview

### 1.1 The role of the layer

The data and index layer is the substrate that the reasoning layers of the application read
from. It holds the source documents, the derived representations that make those documents
searchable, the structured feature inputs that machine learning models consume, and the
artifacts produced by the processes that build all of it.

Two properties define the layer as a design concern. First, it is written offline: every
transformation that produces it runs as a batch or pipeline job rather than as part of
serving a request. Second, it is read-only while the application is running, so no code on
the request path is permitted to insert, update or delete a row, a vector or an object. The
first property is what allows a corpus to be rebuilt without stopping the application. The
second is what prevents a live request from altering the evidence that a previous answer
cited, and it is the reason this layer can be audited at all: the state that produced an
answer remains the state that is inspected afterwards.

The layer answers two questions at serving time. The first is what a patient's record
contains, which is answered from the documents and the retrieval index built over them. The
second is what the model should be told about an admission, which is answered from a feature
table. Every component described below exists to produce, validate or replace the inputs to
one of those two answers.

The illustration below shows the layer in the middle band, with the processes that write it
above and the processes that read it below. The direction of the arrows is the design: state
flows downward into the layer and is only ever read out of it.

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  SOURCES                                                                 │
  │                                                                          │
  │  a real corpus          MIMIC-IV discharge notes, under the data use     │
  │                         agreement                                        │
  │  a demonstration corpus 89 public MTSamples transcriptions               │
  │                                                                          │
  │  A corpus is one document table together with the split identifier that  │
  │  scopes it, so admissions used to fit a model are never indexed.         │
  └──────────────────────────────────────────────────────────────────────────┘
                      │  the offline path reads one split, never the whole table
                      v
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  OFFLINE PRODUCERS  (the only writers)                                   │
  │                                                                          │
  │  ingest pipeline    chunk → embed → validate → build index               │
  │                     a fingerprint of the source tables joins the first   │
  │                     step, so changed data invalidates the cached result  │
  │  data build         materialises the training tables                     │
  │  corpus scripts     assemble and load the demonstration tables           │
  │  deploy script      attaches a finished index to a serving endpoint      │
  └──────────────────────────────────────────────────────────────────────────┘
                      │  produces the state, and records what produced it
                      v
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  THE LAYER  (the state a request may read)                               │
  │                                                                          │
  │  documents         one row per admission, scoped by split                │
  │  derived form      section chunks → vectors → similarity index           │
  │                    one vector per chunk, each carrying the admission it  │
  │                    belongs to as a restriction attribute                 │
  │  feature tables    served features (49 values) · training features       │
  │  artifacts         ingest file · manifests · validation report           │
  └──────────────────────────────────────────────────────────────────────────┘
                      │  read-only at serving time
                      v
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  SERVING READERS                                                         │
  │                                                                          │
  │  tool boundary    admission + query → chunk ids and distances            │
  │                   → passage text re-derived from the document            │
  │                   → re-checked against the admission requested           │
  │  model layer      one feature row per admission → the model              │
  └──────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Components

**Source documents.** The input to the layer is a corpus of clinical documents, one row per
admission, held in a relational store. A corpus in this design is not simply a table: it is a
table combined with a split identifier, because the admissions used to fit a model must not
also appear in the documents that a served answer may cite. The split therefore bounds what
the index is built from, and it is applied at build time rather than at query time. This
system configures two corpora: a real corpus held under the MIMIC-IV data use agreement, and
a demonstration corpus assembled from public MTSamples transcriptions.

**Chunking.** Documents are too long to embed as whole units, and their internal headings
carry meaning that a single vector would blur. Each document is therefore divided into chunks
along its section boundaries, and only narrative sections are retained: history, examination,
hospital course, diagnoses, medications, discharge condition, disposition, instructions and
summary. Metadata headings and laboratory result lines are excluded because they add
embedding cost without improving retrieval. Chunk identifiers are derived deterministically
from the document identifier, the section name and an ordinal, which means the same document
always yields the same set of identifiers. That determinism is what later allows a citation
to recover the exact text of the passage it names without the layer having to store a second
copy of that text.

**The embedding space.** Each chunk is converted into a fixed-length numeric vector by an
embedding model. The model, the output dimension and the task type together define the vector
space, and two vectors are comparable only if all three agree. A mismatch does not raise an
error; it produces neighbours that are plausible and meaningless, which makes this the most
dangerous kind of configuration defect in a retrieval system. In this application the
parameters are therefore defined once in code and treated as part of the identity of the
index rather than as deployment settings.

**The vector index.** The vectors are loaded into an approximate nearest-neighbour index,
which supports similarity search across the corpus at sub-linear query cost. Each vector
carries a restriction attribute, the admission identifier, that scopes a query to a single
patient's documents. Applying the restriction inside the query rather than filtering results
afterwards means that ranking itself is confined to one admission; the alternative is both
slower and unsafe when the corpus is shared, because a global ranking can surface another
patient's text before any filter is applied.

**The feature table.** Structured model inputs are materialised in tables rather than
assembled during a request. This application serves a numeric vector of forty-nine features
per admission. The table read at serving time is deliberately not the table the model was
fitted from, which is a normal separation in a layered system, but it means the two must
agree on the feature contract. If they diverge, the model is asked a question it was not
trained to answer, and the answer will still be returned in a confident form.

**Artifacts.** The offline processes record what they produced and what they read: the ingest
artifact from which an index is built, manifests describing counts and parameters, checksums
for integrity, and the reports of the quality gates. These artifacts are the layer's evidence
trail. Without them, an index cannot be tied to the version of the data it was built from,
and a rebuild cannot be shown to have reproduced the same corpus.

**The producing processes.** Three kinds of offline process create this layer: a pipeline that
performs chunking, embedding, validation and index construction; a data build that
materialises the analytical tables; and a set of scripts that assemble and load the
demonstration corpus. A fourth process attaches a finished index to a serving endpoint. None
of them executes on the request path.

### 1.3 Integration with the stack

Upward, the layer is read by the tool boundary, which passes an admission identifier and a
query and receives chunk identifiers with distances, re-derived passage text, or a feature
row; and by the model layer, which consumes the feature row as its input. Downward, it is
written only by the offline processes, which reach it through the analytical store and object
storage and share no runtime with serving code.

The boundaries with the adjacent layers are stated once here rather than tabulated. The tool
layer owns the request contract and the validation of what a tool returns, including the
isolation of a resolved document; this layer supplies the data and does not decide what may
be asked for. The model layer owns the model, its acceptance gates and its registry entry;
this layer supplies the row that is scored, and the quality of the model is not this layer's
judgement. The evaluation plane owns the quality of an answer and the delivery plane owns
deployment automation. The security plane owns the data protection regime that determines
what may be held here at all.

**Terms used throughout this document.**

| Term | Definition |
|---|---|
| Corpus | A source document table together with the split identifier that scopes it. The split keeps admissions used for model fitting out of the indexed set. |
| Chunk | One section of one document, packed to a fixed character budget. Its identifier is derived from the document, the section and an ordinal, so it is reproducible. |
| Datapoint identifier | A chunk identifier translated into the character set the vector index accepts. It is the only key joining a search result to its source document. |
| Restriction attribute | A per-vector attribute, the admission identifier in this system, which confines a similarity query to one patient's documents. |
| Ingest artifact | The file of datapoint identifiers, vectors and restrictions from which an index is built, written once per ingest run. |
| Data fingerprint | A digest of the modification times and row counts of the source tables, used to invalidate cached pipeline steps when the underlying data changes. |

---

## 2. What Google requires

| # | Requirement | Level |
|---|---|---|
| B1 | Three separate subsystems: data ingestion, serving, quality evaluation. They share the database layer; they do not share code paths. | MUST |
| B2 | Ingestion is event-driven and offline: upload → Pub/Sub → function → parse → chunk → embed → vector store. Live requests never mutate the index. | MUST |
| B3 | Serving uses the same embedding model and parameters as ingestion. | MUST |
| B6 | Index maintenance (rebuild/refresh) is a scheduled operational task, versioned alongside the dataset (BigQuery / AlloyDB / Cloud Storage). | MUST |
| E1 | Version control for prompt templates, chain code, external datasets, adapter weights and container images. Only the data half is audited here: the analytical tables and the stored objects. | MUST |

Two requirements in the same group belong to other layers and are not audited here. B4, which
requires that served responses pass through safety screening, is audited in the model runtime
layer. B5, which requires that served requests and responses are logged for offline analysis,
belongs to the observability plane.

---

## 3. Implementation in this repository

Both corpora and every operational parameter are declared in one committed configuration file,
`services/mcp/rag_config.yaml`, which the ingest pipeline, the evaluation job and the deploy
step all read. The embedding parameters are deliberately excluded from that file and defined
in `services/mcp/retrieval/embed.py`; the configuration loader raises an error if an embedding
block reappears in the configuration, so the vector space cannot be changed by configuration
alone (`retrieval/config.py` 86–93).

The ingest pipeline is defined in `services/mcp/pipelines/rag_ingest_pipeline.py` and consists
of four steps: chunking, embedding, integrity validation and index construction. The
validation step runs before the index is built and fails the run on a vector count, dimension
or identifier defect (`eval_ingest.py` 58–62). Step caching is enabled, and the pipeline
computes a digest of the source tables, covering each table's last modification time and row
count, once per submission and uses it twice: as the chunking step's cache key, so a change in
the data invalidates the cached result, and as the key of the embedding reuse path, so that
"reuse the previous embeddings" names one specific artifact rather than a shared object that
every run overwrites (`rag_ingest_pipeline.py` 180–210). The same digest is written into the
chunk, embedding and index manifests, so each artifact records the data state it was built
against. Before the embed step reuses a single record it reads the description beside the
reuse artifact and stops the run if the model, the output dimensionality or the task type
differ from the constants the serving path imports, because reusing across a model change
would place two incompatible vector spaces in one index while every count and dimension check
still passed (`embed_chunks.py` 66–138). The digest is computed in
`source_fingerprint.py` (21–37), a module shared by the pipeline and the standalone embedding
driver so that both name the same version of the data. Two indexes exist at present: the
demonstration index of 454 vectors and the MIMIC-derived index of 555,770 vectors.

An index is attached to a serving endpoint by `scripts/agent/deploy_rag.py`, which refuses any
index above 100,000 vectors so that the real corpus cannot be exposed on a public endpoint
(`_deploy_guard.py` 10–18). At serving time the admission restriction is passed into the
similarity query (`retrieval.py` 311–320); the returned identifiers are then resolved to note
text in the analytical store, re-chunked with the same parameters used at build time
(`retrieval.py` 250–267) and re-checked against the admission that was requested. Section
retrieval does not consult the index at all. Measured recall at rank ten is 0.992 for the
demonstration corpus and 0.828 for the MIMIC corpus, against a configured threshold of 0.90.

The training table is materialised by Dataform from the committed definitions in
`definitions/`, while the tables that the demonstration serves are built by scripts under
`scripts/agent/`. The serving path reads one row per admission and restricts it to the
forty-nine feature names the model expects (`bigquery_source.py` 54–60).

---

## 4. Gaps, remediation and record of change

Against the requirements in section 2: B3 is met. B1 and E1 are partly met. B2 and B6 are not
met, and in both cases the shortfall is a decision recorded below rather than an unfinished
implementation. No gap remains open.

| # | Gap | Remediation | State |
|---|---|---|---|
| 1 | The two hourly-billed serving endpoints are absent outside a live test, so free-text retrieval fails closed while section retrieval continues to answer. | None required. The endpoints are deployed for testing and for a release; the documented behaviour between tests is the record. | Withdrawn 2026-09-18. This is the project's deployment practice, not a defect. |
| 2 | Ingestion is manual: nothing reacts to a change in the notes table, so a corpus change is invisible until the pipeline is submitted (B2, first clause). | Accepted for this environment. A demonstration has no event stream, and embedding cost should be incurred by decision. The production mechanism is recorded: an object landing area, an event, and a function that writes the table and submits the ingest. | Closed 2026-09-18 as an accepted deviation. |
| 3 | The dataset is not versioned, the embedding reuse source is a single shared object that each run overwrites, and a cached validation step is not evidence of freshness (B6 second clause, E1). | The state of the source tables is recorded in every artifact an ingest writes; the reuse path is specific to a corpus and a data version; and reuse is refused unless the vector space recorded beside the reuse source matches the one this run embeds in. | Closed 2026-09-18. |
| 4 | The recall gate is optional, its evidence is written to a temporary directory, and the deploy path named by the runbook does not run it. | The deploy measures the index it is about to promote, refuses to promote without that measurement, writes the evidence to the artifact bucket, and records the achieved recall on the deployment. | Closed 2026-09-18. Examining the gate to close it also showed that it could never have passed: the configured ceiling on empty results had no measurement anywhere, so a report satisfying it did not exist. |
| 5 | The pipeline's embedding manifest records counts and dimensions but not the model or the task type (B3, evidence rather than behaviour). | Add the model and task type to the manifest, and assert in a test that it names the model the serving path imports. | Closed 2026-09-18 with entry 3, which added the description to both of the manifest's write paths and a test that resolves those values to the constants the serving module defines. |
| 6 | The runbook and the endpoint launcher name different deploy scripts, and the one they name is not the gated path. | Settle on one deploy path and let the launcher and the runbook name it, with the others removed. | Closed 2026-09-18. `deploy_rag.py` is the only path that attaches an index to the endpoint; the four scripts it superseded are deleted and the launcher calls it. |
| 7 | The cohort that authorises the demonstration holds 24 admissions while the corpus the tools serve holds 89, and two fixtures hold 89 and 108 respectively. | Settle which cohort is authoritative, derive the boundary and the fixtures from it, and point the configuration comment at the script that actually writes the table. | Closed 2026-09-18. The served corpus is authoritative: the boundary table is derived from it and the fixtures are trimmed to it. |
| 8 | Two artifacts describe the corpus as synthetic when the text is public MTSamples material, and one docstring describes a chunk table that does not exist. | Correct each description where the claim is made, in the configuration comment, the operations summary and the module docstring. | Closed 2026-09-18. Four descriptions now say what the data is and keep the claim that matters; the chunk-store docstring describes what serving actually does. |

**Record of change.** Entry 1 was withdrawn on 2026-09-18: it had been recorded as a defect
because no index endpoint existed and free-text retrieval failed, but the absence is the
project's deployment practice, and the entry now records the behaviour a request sees between
live tests. Entry 2 was closed the same day as an accepted deviation after the owner
confirmed that this is a demonstration environment without an event stream; the reasoning is
recorded in the entry, the production mechanism is noted for release planning, and the
requirement verdict remains "not met" so that a reviewer sees the deviation rather than an
argument that the clause is irrelevant.

Entry 3 was closed on 2026-09-18. The state of the source tables, reduced to a short
fingerprint over each table's last modification time and row count, is now computed once per
submission and written into the chunk manifest, the embedding manifest, the index manifest and
the standalone embedding driver's ingest manifest, so an artifact can be traced to the data
state it was built against. The reuse
artifact default is now keyed by corpus and by that fingerprint rather than by a single
shared path, and the embed step reads the manifest beside the reuse artifact before it
streams a single record: if the model, the output dimensionality or the task type recorded
there differs from the constants this repository embeds with, the run stops with the two
identities printed instead of mixing two vector spaces in one index. A reuse artifact with no
manifest beside it is likewise refused, because absent provenance is not good provenance. The
embedding parameters the check compares against are read from the module the serving path
imports, so the two sides of the index cannot drift onto different models. Fifteen tests in
`tests/agent/test_ingest_provenance.py` pin the behaviour, using a stub object store, and one
of them holds the driver and the pipeline to the same conventions for the manifest contents
and for the upload path, which are stated in two files that must agree. The requirement
verdict for B6 remains "not met", since the second clause asks for a scheduled and versioned
refresh and this environment has neither; what changed is that the versioning evidence now
exists.

Entry 5 was closed on 2026-09-18 by the same change. The embedding manifest now records the
model, the output dimensionality and the task type in both of its write paths, the copy path
and the full embed, and a test resolves those recorded values to the constants defined in
`services/mcp/retrieval/embed.py`, so a written-out model name would fail rather than pass.
The requirement verdict for B3 does not change: it was already met, and what was missing was
the evidence an artifact carries about itself.

Entry 6 was closed on 2026-09-18 by settling on one deploy path for the retrieval index.
`scripts/agent/deploy_rag.py` picks the newest demonstration index, refuses anything above
the synthetic-scale limit, deploys under a staging identifier while the live identifier
keeps serving, and promotes only if the recall gate passes; the machine size comes from the
corpus the run is for. Four scripts that attached an index to the endpoint with fewer
safeguards were deleted, a wrapper, the primitive beneath it and two pollers, and
`scripts/launch_endpoints.sh` now calls the gated script with the demonstration corpus
selected, because that selection is what sizes the machine at the small type instead of the
real corpus's large one. No requirement in section 2 named a deploy path, so no verdict
changes.

Entry 4 was closed on 2026-09-18 by moving the measurement inside the deploy. The deploy
now finds the ingest run that built the index it is about to promote, by reading each recent
run's index manifest and comparing the index that manifest records against the index under
consideration, and it stops if no run records building it. That check is what makes the
measurement describe the candidate: reading the newest successful run would have produced a
report about the previous index, which authorises the next one just as convincingly and for
no reason. The deploy then brings the candidate up under a staging identifier while the live
identifier keeps serving, runs the recall job against that staging identifier, applies the
thresholds in the corpus configuration, and promotes only if they pass. A failure undeploys
the staging index and leaves the live one answering. The report, the rendered report, the
tabulated results and the failure list are written to the artifact bucket under
`rag/recall/<corpus>/<index>/<timestamp>/`, and the promoted deployment carries the recall it
achieved in its display name, so the number that authorised it is visible on the endpoint
rather than only in a directory that has since been emptied. The on-demand measurement script
was rewritten to resolve the endpoint, the deployed identifier and the artifacts it measures,
because the constants it carried named an endpoint that no longer existed, a pipeline run
from an earlier month, and the live deployed identifier regardless of which index was being
examined. One further finding came out of testing the gate: the configured ceiling on the
rate of empty results had no measurement anywhere in the repository, and a threshold with no
measurement fails every gate that applies it, so the recall job now reports that rate, having
been asking the questions all along.

Entry 7 was closed on 2026-09-18 by making the served corpus the single authority for the
cohort. Three sets of admissions existed under one name. The table the configuration calls
the authorisation boundary held 24, of which 20 were served. The corpus the tools and the
index serve holds 89, so 69 admissions were outside that boundary and reachable by anything
that reached the tools without passing the site's allowlist. The fixtures held 108,
including six captured answers for admissions that never existed in the served corpus, four
of them carrying real MTSamples identifiers from an earlier era of the demonstration. The
shortfall was invisible because the argument beside it held: the tables cannot reach real
MIMIC data, which answers the disclosure question and says nothing about authorisation. The
resolution is a derivation rather than a second selection. `scripts/agent/authorise_demo_cohort.py`
writes the boundary table from the served features table, refuses to run against anything but
the demonstration corpus, and refuses to finish unless the boundary and the served corpus are
the same set by identifier; it took the table from 24 admissions to 89. The fixtures were
trimmed from 108 to the 89 served admissions and the six files describing unserved admissions
were removed. The builders already derive the fixtures from the cohort source, so a rebuild
cannot reintroduce them, and five tests hold the invariant that the smaller artifacts are
subsets of the authoritative list rather than selections beside it. The configuration
comment, which named a script that no longer exists — its bytecode is all that survives —
now names the script that writes the table and records that the boundary is derived. Two
consequences are recorded rather than fixed here: the site's own database is seeded from the
cohort artifact and needs reseeding before the boundary takes effect there, and
`seed_demo_cohort.py` still refers to a feature store resynchronisation script that is not in
this repository.

Entry 8 was closed on 2026-09-18 by correcting four descriptions rather than one. The
demonstration corpus was called synthetic in the configuration's boundary comment, in the
data and model summary, and in both the docstring and the refusal message of the deploy
guard — the guard being the component that decides whether the real corpus may reach a public
endpoint, which is what made the description worth fixing: the claim is read by whoever next
changes the boundary. The text is not synthetic. It is public MTSamples transcriptions,
re-keyed to demonstration admission identifiers, with the features derived from the note and
the outcome label derived from the model. Each of the four now says that, and each keeps the
claim that was load-bearing all along, that no MIMIC-derived note text reaches the
demonstration path. A fifth description was stale in a different way: the embedding module's
docstring had serving resolving a returned identifier to a row in a BigQuery chunk store, and
no such store exists. Serving parses the note and the section back out of the identifier and
recomputes the passage by re-running the same deterministic chunker that built the index, and
the docstring now says so. The sixth item, the endpoint launcher describing the demonstration
index as about 560 vectors when it holds 454, was corrected when the launcher was repointed at
the gated deploy path under entry 6. The guard's constant and function names still use the
word synthetic; that is an identifier rather than a statement, and renaming it is a decision
that has been left to the owner.

No entry remains open. The layer was audited on 2026-09-17, reviewed the same day and again on
2026-09-18, and every entry is now either closed or withdrawn, each with the reasoning that
settled it recorded beside it.