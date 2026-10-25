"""The A2UI v0.9 envelope's fixed names, in one place.

Three modules compose parts of the same surface — the citation records, the canvas,
and the composer that assembles them — and every one of them needs the catalog id,
the surface id and the spec version that the site's renderer resolves the custom
components against. They live here rather than in `a2ui.py` so that the modules
that compose for `a2ui` do not have to import it back.

A mismatch in either id draws nothing: the vendored renderer logs "Component
implementation not found for type: [object Object]" and the answer arrives as an
empty box.
"""

# Read off the site's registered catalog: the vendored renderer resolves the
# custom components against this exact id, and a mismatch means nothing draws.
CATALOG_ID = "https://example.com/catalogs/readmission-risk-v1.json"

SURFACE_ID = "risk-canvas"
A2UI_VERSION = "v0.9"

# R7: rendered payloads are for the user, not for the model to re-read on a
# later turn. Hiding them cuts tokens and, more importantly, stops the model
# treating its own UI output as evidence — a faithfulness risk under Tier 2.
AUDIENCE = ["user"]
