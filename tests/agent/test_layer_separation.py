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
    'postprocess/answer.py',
    'postprocess/citations.py',
    'postprocess/followups.py',
    'postprocess/guardrail.py',
    'presentation/__init__.py',
    'presentation/a2ui.py',
    'presentation/canvas.py',
    'presentation/domain.py',
    'routing/__init__.py',
    'routing/admission_context.py',
    'routing/encounter.py',
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


# --- the one-way subpackage DAG ---------------------------------------------

LEAF_PACKAGES = {"postprocess", "presentation", "telemetry", "routing"}


def _agent_imports(node):
    """Module names under services.agent that this node imports."""
    names = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.ImportFrom) and sub.module:
            if sub.module.startswith("services.agent"):
                names.append(sub.module)
        elif isinstance(sub, ast.Import):
            names.extend(a.name for a in sub.names if a.name.startswith("services.agent"))
    return names


def _package_of(path):
    parts = path.parts
    index = parts.index("agent")
    return parts[index + 1] if index + 1 < len(parts) - 1 else "root"


def test_subpackage_import_dag_is_acyclic():
    """The six subpackages only ever import downwards, and nothing late-imports.

    Two properties, because they fail for the same reason and cannot be fixed
    independently: a cycle shows up first as a function-local import, which is how
    `deterministic` used to reach back into `graph`. A guard that allowed the local
    import would let the next cycle in through the same door.
    """
    root = pathlib.Path(SERVICES, "agent")
    violations, late = [], []

    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        source = path.read_text()
        tree = ast.parse(source)
        package = _package_of(path)

        for module in _agent_imports(tree):
            target = module.split("services.agent.")[-1].split(".")[0]
            if package in LEAF_PACKAGES and target in {"ingress", "engine"}:
                violations.append(f"{path.relative_to(root)}: {package} imports {target}")
            if package == "engine" and target == "ingress":
                violations.append(f"{path.relative_to(root)}: engine imports ingress")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for module in _agent_imports(node):
                    late.append(f"{path.relative_to(root)}:{node.lineno} in {node.name}()")

    assert violations == [], "the subpackage DAG is not one-way"
    assert late == [], "function-local services.agent imports hide a cycle"


# --- every patch still lands on something -----------------------------------

def _module_aliases(tree):
    """alias -> dotted module path, for aliases that name a real module."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def test_no_patch_target_names_a_module_that_no_longer_binds_it():
    """A patch naming an attribute that is no longer bound passes silently.

    That is the failure this phase kept producing: `graph._build_llm` stayed a valid
    name on the module while the call site had moved to another binding, so the
    suite substituted nothing and called the real model. Importing the module is the
    only way to ask the question the patch will ask.
    """
    import importlib.util

    broken = []
    for path in sorted(pathlib.Path("tests").rglob("*.py")):
        tree = ast.parse(path.read_text())
        aliases = _module_aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in {"object", "setattr"}:
                continue
            owner, attr = node.args[0], node.args[1] if len(node.args) > 1 else None
            if not isinstance(owner, ast.Name) or not isinstance(attr, ast.Constant):
                continue
            module = aliases.get(owner.id)
            if module is None or not isinstance(attr.value, str):
                continue
            try:
                spec = importlib.util.find_spec(module)
            except (ImportError, ValueError):
                continue
            if spec is None:
                continue
            loaded = importlib.import_module(module)
            if not hasattr(loaded, attr.value):
                broken.append(f"{path.name}:{node.lineno} patches {module}.{attr.value}")

    assert broken == [], "patch targets that would silently substitute nothing"


# --- every module binds the globals it uses ---------------------------------

def _bound_names(tree):
    """Names bound anywhere in the module: imports, defs, assignments, params."""
    import builtins

    bound = set(dir(builtins))
    bound.add("__file__")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.comprehension):
            for inner in ast.walk(node.target):
                if isinstance(inner, ast.Name):
                    bound.add(inner.id)
    return bound


def test_every_module_binds_the_globals_it_uses():
    """No module may reference a global it never binds.

    This is the guard for the one defect the whole Phase 7 gate could not see.
    A module-level `from ... import run_command` was deleted together with the
    function-local import it had replaced, so `graph.ask()` raised `NameError` on
    every `/slash` command — and 1,474 tests passed, because the suites that drive
    `ask()` with a command are live-marked and CI deselects them. A NameError on a
    path no hermetic test walks is invisible until a patient presses the button;
    an unbound global is visible to `ast` on every module at once.
    """
    root = pathlib.Path(SERVICES, "agent")
    unbound = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text())
        bound = _bound_names(tree)
        missing = sorted({node.id for node in ast.walk(tree)
                          if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                          and node.id not in bound})
        if missing:
            unbound[str(path.relative_to(root))] = missing

    assert unbound == {}, "modules reference globals they never bind"
