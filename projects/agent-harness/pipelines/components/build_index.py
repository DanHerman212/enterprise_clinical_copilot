"""build_index — create the Vector Search indexes from the ingest artifact.

Cloud home of scripts/deploy_index.py. Uploads the ingest artifact to a GCS
directory, builds the small BRUTE_FORCE ground-truth index (exact neighbors at
pennies) and the full TREE_AH serving index, and fails loudly if the built
vector count does not match expectations (silent shortfall = dropped chunks).

Note: a single ingest directory cannot hold both files, so the sample lives in
its own directory for the brute-force build.
"""

import json
from datetime import datetime, timezone

from google.cloud import aiplatform, storage
from google.cloud.aiplatform.matching_engine.matching_engine_index_config import (
    DistanceMeasureType,
)
from kfp import dsl

from ._image import RAG_IMAGE, component


def run_build_index(
    *,
    project_id: str,
    location: str,
    ingest_path: str,
    dimensions: int,
    brute_sample: int,
    approximate_neighbors: int,
    expected: int,
    manifest_path: str,
) -> None:
    client_s = storage.Client(project=project_id)
    bucket = client_s.bucket(f"{project_id}-mlops")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    prefix = f"rag/ingest/{ts}/"

    full_dir = f"{prefix}full/"
    brute_dir = f"{prefix}brute/"
    bucket.blob(f"{full_dir}ingest.json").upload_from_filename(ingest_path)

    sample_path = "/tmp/sample.json"
    with open(ingest_path, encoding="utf-8") as fin, \
            open(sample_path, "w", encoding="utf-8") as fout:
        for index, line in enumerate(fin):
            if index >= brute_sample:
                break
            fout.write(line)
    bucket.blob(f"{brute_dir}sample.json").upload_from_filename(sample_path)

    aiplatform.init(project=project_id, location=location)
    dist = DistanceMeasureType.DOT_PRODUCT_DISTANCE

    brute = aiplatform.MatchingEngineIndex.create_brute_force_index(
        display_name=f"rag-brute-force-{ts}",
        contents_delta_uri=f"gs://{bucket.name}/{brute_dir}",
        dimensions=dimensions,
        distance_measure_type=dist,
    )
    brute.wait()
    brute_vectors = brute.gca_resource.index_stats.vectors_count
    print(f"brute-force: {brute_vectors} vectors → {brute.resource_name}")
    if brute_vectors != brute_sample:
        raise SystemExit(f"brute-force expected {brute_sample} vectors, got {brute_vectors}")

    tree = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=f"rag-tree-ah-{ts}",
        contents_delta_uri=f"gs://{bucket.name}/{full_dir}",
        dimensions=dimensions,
        approximate_neighbors_count=approximate_neighbors,
        # SDK defaults are None; without these the tree-AH algorithmConfig is
        # empty and Vertex rejects the build ("algorithmConfig required").
        leaf_node_embedding_count=1000,
        leaf_nodes_to_search_percent=10,
        distance_measure_type=dist,
    )
    tree.wait()
    tree_vectors = tree.gca_resource.index_stats.vectors_count
    print(f"tree-ah: {tree_vectors} vectors → {tree.resource_name}")
    if tree_vectors != expected:
        raise SystemExit(f"tree-ah expected {expected} vectors, got {tree_vectors}")

    with open(manifest_path, "w") as handle:
        json.dump(
            {
                "brute_force_index": brute.resource_name,
                "brute_force_vectors": brute_vectors,
                "tree_ah_index": tree.resource_name,
                "tree_ah_vectors": tree_vectors,
                "ingest_dir": f"gs://{bucket.name}/{full_dir}",
            },
            handle,
            indent=2,
        )


@component(
    base_image=RAG_IMAGE,
    packages_to_install=["google-cloud-aiplatform", "google-cloud-storage"],
)
def build_index(
    project_id: str,
    location: str,
    ingest: dsl.Input[dsl.Artifact],
    dimensions: int,
    brute_sample: int,
    approximate_neighbors: int,
    expected: int,
    manifest: dsl.Output[dsl.Artifact],
) -> None:
    """KFP component: build BRUTE_FORCE + TREE_AH Vector Search indexes."""
    from pipelines.components.build_index import run_build_index

    run_build_index(
        project_id=project_id,
        location=location,
        ingest_path=ingest.path,
        dimensions=dimensions,
        brute_sample=brute_sample,
        approximate_neighbors=approximate_neighbors,
        expected=expected,
        manifest_path=manifest.path,
    )
