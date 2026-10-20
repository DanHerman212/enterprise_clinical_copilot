"""A2UI component names, from an envelope.

The surface is A2UI *messages*, not a flat list: components live under
`updateComponents.components`, with the root a `Column` whose children carry the card.
Walking generically means a surface gaining a wrapper does not silently report zero
components to an assertion looking for one.

This mirrors `evaluation/agent/collect_http._components` deliberately and with the same
semantics — the legacy collector is being retired, and importing it here would drag the
HTTP client and the service packages into every task call.
"""

from __future__ import annotations

from typing import Any


def components(node: Any, out: set[str] | None = None) -> list[str]:
    if out is None:
        out = set()
    if isinstance(node, dict):
        name = node.get("component")
        if isinstance(name, str):
            out.add(name)
        for value in node.values():
            components(value, out)
    elif isinstance(node, list):
        for item in node:
            components(item, out)
    return sorted(out)
