"""Layers 5 and 6 stand apart: no `services` module imports `mlops`.

Phase 4, 2026-10-25. The layer document says online serving must not import the
training package. That is true today — nothing under `services/` imports `mlops` —
and nothing tested it, so it was true by discipline rather than by rule.

The check walks the tree rather than naming modules, because the failure it exists
for is a NEW file reaching across, not an existing one. A single import of
`mlops.data.encoding` from, say, `services/mcp/dependencies/features/` would make the
serving image depend on the training package: the container is built from
`services/mcp`, and that import would either break the build or drag pandas and the
training tree into it.

`tests/` is allowed to reach across — `test_feature_contract.py` imports
`mlops.data.encoding` deliberately, to compare the generators against the canonical
order. The rule is about shipped code.
"""

import ast
import pathlib

SERVICES = pathlib.Path(__file__).resolve().parents[2] / "services"

#: The package that must never be reachable from shipped code.
TRAINING_ROOT = "mlops"


def _imports(path):
    """(module, lineno) for every import in one file, absolute or relative."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import cannot leave its own package, so it cannot
                # reach `mlops` — but resolving it would mean tracking the
                # package path, and saying so is cheaper than doing it.
                continue
            if node.module:
                yield node.module, node.lineno


def test_the_services_tree_was_actually_walked():
    """A guard on the guard: a wrong path would make the assertion below vacuous."""
    files = list(SERVICES.rglob("*.py"))

    assert len(files) > 20, f"only {len(files)} files found under {SERVICES}"


def test_no_shipped_module_imports_the_training_package():
    offenders = [
        f"{path.relative_to(SERVICES)}:{lineno} imports {module}"
        for path in SERVICES.rglob("*.py")
        for module, lineno in _imports(path)
        if module == TRAINING_ROOT or module.startswith(TRAINING_ROOT + ".")
    ]

    assert not offenders, (
        "online serving must not depend on the training package: "
        + "; ".join(offenders)
    )


def test_the_layer4_runtime_does_not_import_the_training_package_either():
    """The model client is the one place that could plausibly be tempted to.

    `engine/model.py` reads the model name from `services.mcp.config`, which the
    deployment stamps; a change that reached for the training registry instead
    would be the same mistake in the file most likely to make it.
    """
    runtime = pathlib.Path(SERVICES, "agent", "engine", "model.py")

    assert runtime.is_file()
    assert not [
        module for module, _ in _imports(runtime)
        if module == TRAINING_ROOT or module.startswith(TRAINING_ROOT + ".")
    ]


# --- the Phase 7 manifest ---------------------------------------------------
#
# Pinned as an exact frozenset of relative paths rather than a count: a count says
# something changed, the set says what. The tree is 26 files — 7 `__init__.py`,
# `turn.py`, and 18 subpackage modules — with no compatibility shims at the root.
AGENT_MODULES = frozenset({
    '__init__.py',
    'turn.py',
    'engine/__init__.py',
    'engine/deterministic.py',
    'engine/graph.py',
    'engine/mcp_client.py',
    'engine/model.py',
    'engine/prompts.py',
    'ingress/__init__.py',
    'ingress/contracts.py',
    'ingress/http.py',
    'ingress/run.py',
    'postprocess/__init__.py',
    'postprocess/citations.py',
    'postprocess/followups.py',
    'postprocess/guardrail.py',
    'presentation/__init__.py',
    'presentation/a2ui.py',
    'presentation/canvas.py',
    'presentation/domain.py',
    'routing/__init__.py',
    'routing/admission_context.py',
    'routing/routing.py',
    'telemetry/__init__.py',
    'telemetry/observability.py',
    'telemetry/stages.py',
})


def test_agent_module_manifest_is_exact():
    """Every file under services/agent/ is one this tree names, and no more.

    The failure this prevents is a half-finished move: a shim left at the root so
    one import keeps working, or a module that was renamed and copied instead of
    moved. Both leave the layout describing something other than the code.
    """
    root = pathlib.Path(SERVICES, "agent")
    actual = frozenset(
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*.py")
        if '__pycache__' not in str(path)
    )

    assert actual - AGENT_MODULES == set(), "unexpected files under services/agent/"
    assert AGENT_MODULES - actual == set(), "modules named by the tree are missing"
