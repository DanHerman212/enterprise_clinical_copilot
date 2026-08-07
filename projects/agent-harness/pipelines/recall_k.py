"""recall@k job — exact vs tree-AH neighbor agreement over the full corpus.

Runs inside a Vertex AI custom job (cloud-only; no PHI leaves the cloud).
Plan (guide §12):

  1. Sample N real query texts from the chunk corpus artifact (gzip JSONL).
  2. Embed them with gemini-embedding-001 (RETRIEVAL_QUERY, 768 dims).
  3. Load the full ingest (9 GB JSONL) into a float32 matrix + id list.
  4. Exact top-k per query via dot-product over all 555,770 vectors.
  5. Query the deployed tree-AH index with the same query embeddings.
  6. recall@k = |exact_top_k ∩ approx_top_k| / k, reported at k=1,5,10.
  7. Write a report JSON + plain-text summary to GCS.

Usage (from inside the image):
    python /app/scripts/recall_k.py \
        --chunks gs://.../chunks --ingest gs://.../ingest \
        --endpoint projects/.../indexEndpoints/... --deployed-id rag_tree_ah \
        --out-dir gs://.../recall/ \
        [--num-queries 100] [--top-k 10] [--seed 42]
"""

import argparse
import gzip
import json
import time

import numpy as np
from google import genai
from google.cloud import aiplatform, storage
from google.genai import types

from rag.embed import (
    EMBEDDING_MODEL,
    OUTPUT_DIMENSIONALITY,
    QUERY_TASK_TYPE,
)

BATCH = 100  # embed_content max contents per call


def _blob(uri: str) -> storage.Blob:
    bucket, obj = uri.replace("gs://", "", 1).split("/", 1)
    return storage.Client().bucket(bucket).blob(obj)


def load_chunk_texts(chunks_uri: str, n: int, seed: int) -> list[str]:
    """Sample n real query texts from the gzip chunk artifact (deterministic)."""
    texts: list[str] = []
    blob = _blob(chunks_uri)
    with blob.open("rb") as src, gzip.open(src, "rt", encoding="utf-8") as fin:
        for line in fin:
            texts.append(json.loads(line)["text"])
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    return [texts[i] for i in pick]


def embed_texts(client: genai.Client, texts: list[str]) -> np.ndarray:
    """Embed query texts -> float32 matrix (len(texts), 768)."""
    rows = []
    for i in range(0, len(texts), BATCH):
        resp = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts[i:i + BATCH],
            config=types.EmbedContentConfig(
                output_dimensionality=OUTPUT_DIMENSIONALITY,
                task_type=QUERY_TASK_TYPE,
            ),
        )
        rows.extend([list(e.values) for e in resp.embeddings])
    return np.asarray(rows, dtype=np.float32)


def load_ingest(ingest_uri: str) -> tuple[list[str], np.ndarray]:
    """Load the full ingest -> (ids, float32 matrix (N, 768)).

    Each embedding is converted to a compact float32 array as it is read so we
    never hold the list-of-Python-floats form (that is ~12 GB; the float32
    matrix is ~1.7 GB).
    """
    ids: list[str] = []
    cols: list[np.ndarray] = []
    blob = _blob(ingest_uri)
    with blob.open("rb") as src:
        for line in src:
            d = json.loads(line)
            ids.append(d["id"])
            cols.append(np.asarray(d["embedding"], dtype=np.float32))
    return ids, np.vstack(cols)


def exact_topk(ids: list[str], X: np.ndarray, queries: np.ndarray,
               k: int) -> list[list[str]]:
    """Exact top-k by dot product (DOT_PRODUCT: higher = closer)."""
    out = []
    for q in queries:
        scores = X @ q  # (N,)
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        out.append([ids[i] for i in top])
    return out


def approx_topk(ep, deployed_id: str, queries: np.ndarray,
                k: int) -> list[list[str]]:
    """Query the deployed tree-AH index for the same embeddings."""
    out = []
    for q in queries:
        res = ep.find_neighbors(
            deployed_index_id=deployed_id,
            queries=[q.tolist()],
            num_neighbors=k,
        )
        out.append([nb.id for nb in (res[0] if res else [])])
    return out


def recall_k(exact: list[str], approx: list[str], k: int) -> float:
    exact_set = set(exact[:k])
    hits = sum(1 for a in approx[:k] if a in exact_set)
    return hits / k


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--ingest", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--deployed-id", default="rag_tree_ah")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.monotonic()
    print(f"[recall] sampling {args.num_queries} query texts from chunks…",
          flush=True)
    texts = load_chunk_texts(args.chunks, args.num_queries, args.seed)
    print(f"[recall] got {len(texts)} query texts", flush=True)

    client = genai.Client(
        vertexai=True,
        project=args.endpoint.split("/")[1],
        location=args.endpoint.split("/")[3],
    )
    queries = embed_texts(client, texts)
    print(f"[recall] embedded queries -> {queries.shape}", flush=True)

    print("[recall] loading full ingest…", flush=True)
    ids, X = load_ingest(args.ingest)
    print(f"[recall] ingest: {len(ids)} vectors, {X.shape}", flush=True)

    print(f"[recall] exact top-{args.top_k} over {len(ids)} vectors…", flush=True)
    exact = exact_topk(ids, X, queries, args.top_k)

    aiplatform.init(project=args.endpoint.split("/")[1],
                    location=args.endpoint.split("/")[3])
    ep = aiplatform.MatchingEngineIndexEndpoint(args.endpoint)
    print("[recall] querying tree-AH endpoint…", flush=True)
    approx = approx_topk(ep, args.deployed_id, queries, args.top_k)

    ks = sorted({1, 5, args.top_k})
    report = {"num_queries": len(texts), "top_k": args.top_k, "seed": args.seed,
              "recall": {f"@{k}": round(float(np.mean(
                  [recall_k(e, a, k) for e, a in zip(exact, approx)])), 4)
                  for k in ks},
              "per_query": [
                  {"q": i, **{f"@{k}": recall_k(e, a, k)
                              for k in ks}}
                  for i, (e, a) in enumerate(zip(exact, approx))
              ]}
    elapsed = round(time.monotonic() - t0, 1)
    report["elapsed_seconds"] = elapsed

    lines = ["=== recall@k (exact vs tree-AH) ===",
             f"queries: {len(texts)}  top_k: {args.top_k}  seed: {args.seed}",
             f"elapsed: {elapsed}s"]
    for k, v in report["recall"].items():
        lines.append(f"  mean {k} = {v}")
    summary = "\n".join(lines)
    print(summary, flush=True)

    out_uri = f"{args.out_dir.rstrip('/')}/recall_report.json"
    _blob(out_uri).upload_from_string(
        json.dumps(report, indent=2), content_type="application/json")
    _blob(f"{args.out_dir.rstrip('/')}/recall_summary.txt").upload_from_string(
        summary, content_type="text/plain")
    print(f"[recall] wrote {out_uri}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
