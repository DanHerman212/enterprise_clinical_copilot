"""Config-as-code loader for the RAG pipeline.

Single source of truth for corpus, chunking, index, eval, and deploy settings.
Reads ``rag_config.yaml`` (committed) and exposes typed values, so the ingest
pipeline, the eval gate, and the deploy step never hold a second copy of a
value that could drift.

The embedding parameters are the deliberate exception: they are NOT in the YAML
and not settable here, because they decide which vector space the index was
built in. They are imported from :mod:`services.mcp.retrieval.embed`, the same
definition the serving path uses, so the build and the query cannot disagree
(gap 5). A YAML that carries an ``embedding`` block again is an error rather
than an override.

Env overrides:
  * ``PROJECT_ID`` — override ``corpus.project`` (one-off / local runs).
  * ``RAG_CORPUS`` — override ``corpus.active`` (``mimic`` | ``demo``).

The section whitelist is intentionally NOT in the YAML: it stays single-sourced
in :mod:`rag.chunking.INDEX_SECTIONS` and validated against
:mod:`rag.sections.KNOWN_HEADINGS` (ECC-33 / S7-02 consolidation target).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from services.mcp.retrieval.embed import (
    EMBEDDING_MODEL,
    OUTPUT_DIMENSIONALITY,
    QUERY_TASK_TYPE,
)

HARNESS = Path(__file__).resolve().parents[1]
CONFIG_PATH = HARNESS / "rag_config.yaml"


@dataclass(frozen=True)
class Corpus:
    name: str
    notes_table_ref: str
    split_table_ref: str
    split_name: str
    expected_vectors: int
    shard_size: str
    deploy_machine_type: str


@dataclass(frozen=True)
class RAGConfig:
    project: str
    location: str
    corpus: Corpus
    pack_to: int
    max_chars: int
    embedding_model: str
    dimensions: int
    query_task_type: str
    approximate_neighbors: int
    brute_sample: int
    shard_size: str
    chunk_cpu: str
    chunk_mem: str
    embed_cpu: str
    embed_mem: str
    index_cpu: str
    index_mem: str
    eval_num_queries: int
    eval_top_k: int
    eval_seed: int
    recall_at_10_min: float
    empty_result_rate_max: float
    endpoint_name: str
    machine_type: str


def load(path: Path | None = None) -> RAGConfig:
    """Load the committed config and apply env overrides (typed)."""
    path = path or CONFIG_PATH
    doc = yaml.safe_load(path.read_text())

    if "embedding" in doc:
        raise ValueError(
            f"{path.name} carries an embedding block again. Those parameters are "
            "defined once in services/mcp/retrieval/embed.py, because a copy here "
            "can build an index in a different space than the serving path queries "
            "(gap 5). Delete the block rather than editing it."
        )

    project = os.environ.get("PROJECT_ID", doc["corpus"]["project"])
    active = os.environ.get("RAG_CORPUS", doc["corpus"]["active"])
    if active not in doc["corpus"]:
        raise ValueError(
            f"RAG_CORPUS={active!r} is not a corpus in {path.name}"
        )

    c = doc["corpus"][active]
    return RAGConfig(
        project=project,
        location=doc["corpus"]["location"],
        corpus=Corpus(
            name=active,
            notes_table_ref=c["notes_table_ref"],
            split_table_ref=c["split_table_ref"],
            split_name=c["split_name"],
            expected_vectors=int(c["expected_vectors"]),
            shard_size=c["shard_size"],
            deploy_machine_type=c["deploy_machine_type"],
        ),
        pack_to=int(doc["chunking"]["pack_to"]),
        max_chars=int(doc["chunking"]["max_chars"]),
        embedding_model=EMBEDDING_MODEL,
        dimensions=OUTPUT_DIMENSIONALITY,
        query_task_type=QUERY_TASK_TYPE,
        approximate_neighbors=int(doc["index"]["approximate_neighbors"]),
        brute_sample=int(doc["index"]["brute_sample"]),
        shard_size=c["shard_size"],
        chunk_cpu=doc["index"]["chunk_cpu"],
        chunk_mem=doc["index"]["chunk_mem"],
        embed_cpu=doc["index"]["embed_cpu"],
        embed_mem=doc["index"]["embed_mem"],
        index_cpu=doc["index"]["index_cpu"],
        index_mem=doc["index"]["index_mem"],
        eval_num_queries=int(doc["eval"]["num_queries"]),
        eval_top_k=int(doc["eval"]["top_k"]),
        eval_seed=int(doc["eval"]["seed"]),
        recall_at_10_min=float(doc["eval"]["recall_at_10_min"]),
        empty_result_rate_max=float(doc["eval"]["empty_result_rate_max"]),
        endpoint_name=doc["deploy"]["endpoint_name"],
        machine_type=c["deploy_machine_type"],
    )
