"""MCP tools. Each module exposes plain functions; server.py registers them."""

from .labs import get_lab_trends
from .literature import search_literature
from .prediction import predict_readmission
from .retrieval import discharge_medications, rag_search, rag_search_sections

__all__ = [
    "predict_readmission",
    "rag_search",
    "rag_search_sections",
    "discharge_medications",
    "search_literature",
    "get_lab_trends",
]
