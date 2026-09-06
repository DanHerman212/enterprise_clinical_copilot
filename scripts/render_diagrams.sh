#!/usr/bin/env bash
# Render docs/diagrams/*.mmd (Mermaid source of truth) -> assets/diagrams/*.png
# Usage: scripts/render_diagrams.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/docs/diagrams"
OUT="$ROOT/assets/diagrams"
WIDTH="${1:-1600}"

if ! command -v mmdc >/dev/null 2>&1; then
  echo "mmdc not found. Install it with:" >&2
  echo "  npm install -g @mermaid-js/mermaid-cli" >&2
  exit 1
fi

mkdir -p "$OUT"
for f in "$SRC"/*.mmd; do
  name="$(basename "$f" .mmd)"
  mmdc -i "$f" -o "$OUT/$name.png" -b white -w "$WIDTH"
  echo "rendered $name.png"
done
