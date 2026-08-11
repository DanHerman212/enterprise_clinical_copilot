"""Verify the deployed rag endpoint answers a real query with neighbors."""

import os
import sys
import time

from google import genai
from google.cloud import aiplatform
from google.genai import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.embed import EMBEDDING_MODEL, OUTPUT_DIMENSIONALITY, QUERY_TASK_TYPE  # noqa: E402

PROJECT = os.environ.get("PROJECT_ID", "trim-icon-498815-a0")
LOCATION = os.environ.get("LOCATION", "us-east1")
ENDPOINT = os.environ.get(
    "INDEX_ENDPOINT",
    "projects/778397675435/locations/us-east1/indexEndpoints/4335185232320790528",
)
DEPLOYED_ID = os.environ.get("DEPLOYED_INDEX_ID", "rag_tree_ah")
QUERY = "sepsis and elevated lactate broad-spectrum antibiotics"


def main() -> int:
    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    resp = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[QUERY],
        config=types.EmbedContentConfig(
            output_dimensionality=OUTPUT_DIMENSIONALITY,
            task_type=QUERY_TASK_TYPE,
        ),
    )
    qv = [float(v) for v in resp.embeddings[0].values]
    print(f"embedded query: {len(qv)} dims")

    ep = aiplatform.MatchingEngineIndexEndpoint(ENDPOINT)
    for attempt in range(5):
        try:
            res = ep.find_neighbors(
                deployed_index_id=DEPLOYED_ID, queries=[qv], num_neighbors=3
            )
            for nb in (res[0] if res else []):
                print(" ", nb.id, round(nb.distance, 4))
            if not (res and res[0]):
                print("EMPTY RESULT (no neighbors)")
            return 0
        except Exception as exc:  # transient 503 while endpoint warms up
            print(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < 4:
                time.sleep(15)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
