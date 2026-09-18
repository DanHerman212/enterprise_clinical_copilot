"""A digest of the state of a set of source tables, used as a data version.

Three places need to name the same version of the same data: the ingest
pipeline, which uses the digest as a cache key so changed data invalidates
cached steps; the pipeline again, which uses it to key the embedding reuse path
so "reuse the previous embeddings" names one artifact rather than a shared
object every run overwrites; and the standalone embedding driver, which uploads
its artifact under the same key so an ingest can find and verify it.

Because they must agree, the function lives here rather than in the pipeline
module: importing that module pulls in the pipeline framework, and a second copy
of this logic is exactly the kind of duplicate that drifts. The digest covers
each table's modification time and row count, which is enough to detect a
rebuild, an append or a rewrite of the source data.
"""

from __future__ import annotations

import hashlib


def source_fingerprint(project_id: str, refs: tuple[str, ...]) -> str:
    """A stable short digest of the current state of the given tables.

    Each reference is ``dataset.table`` or ``project.dataset.table``. A table
    that cannot be read raises, because a fingerprint that silently omits a
    source is worse than no fingerprint at all.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    parts = []
    for ref in refs:
        table = client.get_table(ref)
        modified = table.modified.timestamp() if table.modified else 0
        parts.append(f"{ref}:{int(modified)}:{table.num_rows}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
