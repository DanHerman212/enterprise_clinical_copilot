# Agentic RAG — Requirements (Draft)

_Date: 2026-07-24 · Status: discussion notes, not yet a design_

Requirements for the RAG system, captured from the architecture discussion. This is
the **WHAT/WHY** — the design (embedding model, chunking, index type, tool contract)
comes later. See [NEXT_STEPS.md](NEXT_STEPS.md) for overall sequencing.

## Purpose

The readmission model produces a **quantitative signal** (risk score + SHAP factors
over 49 tabular features). RAG provides the **qualitative evidence** the model is blind
to — the clinical narrative in unstructured discharge notes. Together they support
clinician review; neither alone is sufficient.

RAG exists to let the agent **ground its reasoning in real chart evidence** rather than
hallucinate a plausible-sounding explanation.

## What the notes contribute (that the 49 features cannot)

- Social determinants (e.g. lives alone, unreliable transportation)
- Medication adherence concerns (e.g. missed prior appointments, cost barriers)
- Clinician worry / subjective assessment (e.g. "borderline stable at discharge")
- The actual discharge and follow-up plan

## Core mechanics (shared understanding)

- The **embedding is only a lookup key**, never the thing the agent reasons over.
- Ingest: note text → chunk → embed once → store vector **alongside the original text
  chunk** and metadata.
- Query time: embed the query → nearest-neighbor search → return the **original text
  chunks** (plain English), not vectors.
- The LLM/agent always reads raw note passages; embeddings only make *semantic* search
  (find by meaning, not keyword) possible.

## Agent workflow (target flow)

For a given `hadm_id`:

1. Agent calls `predict_readmission(hadm_id)` → risk score, decision, top factors.
2. Agent forms a retrieval query from that context (e.g. "discharge barriers,
   medication adherence, follow-up plan for this admission").
3. `rag_search(query, filter=hadm_id)` → returns top-k original note chunks for that
   admission.
4. Agent combines the structured prediction **and** the retrieved passages into a
   single grounded, evidence-cited answer.

## Requirements

### R1 — Scoped / filtered retrieval (highest priority)
Retrieval must support metadata filtering so queries are scoped to the **current
patient's admission** (`hadm_id` / `subject_id`) by default. A patient summary must
never be contaminated by another patient's notes. This scoping is the core correctness
requirement — the embedding + vector store itself is the easy part.

### R2 — Return original text, not embeddings
The tool must return the source note passage(s) so the agent can read and cite them.
Retrieved chunks should carry enough provenance (which note, which admission) to cite.

### R3 — Grounded, citable answers
The agent's output must be traceable to retrieved evidence — quote/point to the actual
note that supports each qualitative claim, rather than inventing narrative.

### R4 — Complements, does not replace, the model
RAG is additive context alongside the prediction. The quantitative risk signal remains
the model's job; RAG supplies the "why/context" the tabular features miss.

### R5 — Exposed to the agent as a tool
RAG is consumed as the agent's **second tool** (`rag_search`), registered the same way
as `predict_readmission` — not as a standalone system.

## Out of scope (for now — later use cases)

- **Cross-patient retrieval** ("find similar past patients") — a distinct, later
  feature that deliberately does NOT scope to a single admission.
- Embedding model / chunking strategy / index type selection — design phase.
- Corpus sourcing details (which MIMIC-IV note tables, availability) — open decision,
  the single biggest unknown before design can start.

## Open questions to resolve before design

- ~~What unstructured corpus is actually available?~~ **RESOLVED:**
  `physionet-data.mimiciv_note.discharge` (+ `radiology`) in BigQuery, keyed by `hadm_id`.
- Chunking granularity (whole note vs sections vs sliding window).
- Embedding model choice and where it runs.
- Vector store / index (e.g. Vertex AI Vector Search) and its metadata-filter support.
