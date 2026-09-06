"""Retrieval-augmented generation over MIMIC-IV discharge notes.

Build order (see docs/RAG_BUILD_GUIDE.md):

    sections.py   parse notes into named sections          <- A0
    concepts.py   tag clinical concepts, negation-aware     <- A3
    chunking      split sections into embeddable passages   <- guide sec 4
    index         Vertex AI Vector Search build + query     <- guide sec 6

Nothing in this package talks to Vector Search yet. The evidence gate (A0-A5)
runs entirely locally and decides whether the index is worth building at all.
"""
