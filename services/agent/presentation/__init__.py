"""`services.agent.presentation` — the A2UI surface and the clinical tables.

The envelope names live here rather than in `a2ui.py` because both the composer
and the canvas need them, and the composer imports the canvas: putting them in
either one makes the pair circular.
"""

CATALOG_ID = "https://example.com/catalogs/readmission-risk-v1.json"

SURFACE_ID = "risk-canvas"
A2UI_VERSION = "v0.9"

# R7: rendered payloads are for the user, not for the model to re-read on a
# later turn. Hiding them cuts tokens and, more importantly, stops the model
# treating its own UI output as evidence — a faithfulness risk under Tier 2.
AUDIENCE = ["user"]
