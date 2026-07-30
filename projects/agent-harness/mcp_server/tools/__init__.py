"""MCP tools. Each module exposes plain functions; server.py registers them."""

from .predict import predict_readmission

__all__ = ["predict_readmission"]
