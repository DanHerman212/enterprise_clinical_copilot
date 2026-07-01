#!/usr/bin/env python3
"""Quick syntax check for all pipeline files."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
files = sorted((ROOT / "components").glob("*.py")) + [ROOT / "training_pipeline.py"]

for f in files:
    try:
        ast.parse(f.read_text())
        print(f"OK  {f.name}")
    except SyntaxError as e:
        print(f"FAIL {f.name}: {e}")
