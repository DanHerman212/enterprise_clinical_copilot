"""MCP tools. Each module exposes plain functions; server.py registers them."""

from .literature import search_literature
from .prediction import predict_readmission
from .retrieval import rag_search, rag_search_sections

__all__ = [
    "predict_readmission",
    "rag_search",
    "rag_search_sections",
    "search_literature",
]
