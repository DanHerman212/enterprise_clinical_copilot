"""MCP tools. Each module exposes plain functions; server.py registers them."""

from .predict import predict_readmission
from .rag_search import rag_search, rag_search_sections

__all__ = ["predict_readmission", "rag_search", "rag_search_sections"]
