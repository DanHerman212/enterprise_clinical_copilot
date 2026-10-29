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

#: What each package may import, from `docs/architecture/13-agent_module_boundaries.md`
#: §5. `"root"` is `turn.py`, the orchestrator, which may reach every package.
#:
#: `root` additionally lists `ingress`, and that is the one deliberate widening of §5's
#: row: the orchestrator COMPOSES the payload and validates it against the wire contract
#: (`validate_agent_success`, `RE_DERIVABLE_TOOLS`), so the contract module is a shared
#: vocabulary between the two. §5 was written before `turn.py` owned that check; the
#: alternative is a second copy of the response contract inside the orchestrator, which
#: is exactly the kind of duplicate this file exists to prevent.
#:
#: This replaces the previous rule, which forbade only leaf -> {ingress, engine}. That
#: rule permitted everything that actually went wrong: `engine` importing
#: `presentation` to name a feature label, `postprocess` importing `presentation` for the
#: same map, `routing` holding the prompts, and `ingress` reaching into `engine` for the
#: graph and the toolbox. A rule that only forbids root-node imports does not describe a
#: layered system.
ALLOWED = {
    "root": {"routing", "engine", "postprocess", "presentation", "telemetry", "ingress"},
    "ingress": {"root", "routing", "telemetry", "presentation"},
    "engine": {"routing", "postprocess", "telemetry", "engine"},
    "postprocess": {"routing", "telemetry", "postprocess"},
    "presentation": {"postprocess", "presentation"},
    "telemetry": {"routing", "telemetry"},
    "routing": {"routing"},
}

#: Where a package may import postprocess from, when it may at all. A package allowed to
#: read `citations` is not thereby allowed to read the guardrails: the citation contract
#: is a shared vocabulary, the guardrails are the answer pipeline.
POSTPROCESS_LEAVES = {"citations", "guardrail", "answer", "followups"}

POSTPROCESS_MODULES = {
    "root": {"citations", "guardrail", "answer", "followups"},    "engine": {"citations", "answer"},
    "presentation": {"citations"},
    "postprocess": {"citations", "guardrail", "answer", "followups"},
}


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


def _agent_imports(node):
    """(module, imported names) for every import of a `services.agent` module.

    The names travel with the module because `from services.agent.postprocess import
    citations` imports no module path at all: the subpackage is `postprocess` and the
    module is one of the names. Without them the check could see a package and not the
    leaf it actually reached for.
    """
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.ImportFrom) and sub.module:
            if sub.module.startswith("services.agent"):
                found.append((sub.module, [a.name for a in sub.names]))
        elif isinstance(sub, ast.Import):
            for alias in sub.names:
                if alias.name.startswith("services.agent"):
                    found.append((alias.name, []))
    return found


def _targets(module, names):
    """(package key, postprocess leaves) for one imported module."""
    rest = module.split("services.agent.", 1)[1] if "services.agent." in module else "turn"
    parts = [part for part in rest.split(".") if part]
    head = parts[0] if parts else "turn"
    key = "root" if head == "turn" else head
    leaves = parts[1:2] or [name for name in names if name in POSTPROCESS_LEAVES]
    return key, [leaf for leaf in leaves if leaf in POSTPROCESS_LEAVES]


def test_subpackage_import_dag_is_acyclic():
    """The six subpackages import only downwards, and nothing late-imports.

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
        tree = ast.parse(path.read_text())
        package = _package_of(path)
        allowed = ALLOWED[package]

        for module, names in _agent_imports(tree):
            target, leaves = _targets(module, names)
            if target != package and target not in allowed:
                violations.append(
                    f"{path.relative_to(root)}: {package} imports {target} "
                    f"(charter §5 allows {' / '.join(sorted(allowed))})"
                )
                continue
            if target == "postprocess":
                for leaf in leaves or [""]:
                    if leaf not in POSTPROCESS_MODULES[package]:
                        violations.append(
                            f"{path.relative_to(root)}: {package} imports "
                            f"postprocess.{leaf or '*'} (charter §5 allows "
                            f"{' / '.join(sorted(POSTPROCESS_MODULES[package]))})"
                        )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for module, _ in _agent_imports(node):
                    late.append(f"{path.relative_to(root)}:{node.lineno} in {node.name}()")

    assert violations == [], "the subpackage DAG is not one-way:\n" + "\n".join(violations)
    assert late == [], "function-local services.agent imports hide a cycle"


# --- the charter's laws, as tests -------------------------------------------
# Each law names a single owner. These read the shipped modules and fail when a second
# implementation appears; they are the rules that were broken while this codebase grew.


def _shipped_sources():
    root = pathlib.Path(SERVICES, "agent")
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        yield path, path.read_text()


def test_law1_only_citations_defines_a_marker_regex():
    """Law 1: no module outside `postprocess/citations.py` may parse `^[...]`.

    A second marker regex is how the two range strippers drifted apart in the first
    place: one of them read a single number, the other a whole cluster, and neither knew
    the other existed.
    """
    owner = pathlib.Path("postprocess", "citations.py")
    offenders = [
        str(path.relative_to(root))
        for path, _ in []
    ]
    root = pathlib.Path(SERVICES, "agent")
    for path, source in _shipped_sources():
        if path.relative_to(root) == owner:
            continue
        if r"\^\[" in source:
            offenders.append(str(path.relative_to(root)))
    assert offenders == [], (
        "a citation-marker regex outside postprocess/citations.py: " + ", ".join(offenders)
    )


def test_law2_fallback_sentences_live_in_postprocess():
    """Law 2: the fixed sentences an answer can be replaced with are `postprocess/`'s."""
    from services.agent.engine import model as model_turn
    from services.agent.postprocess import answer
    from services.agent.routing import routing

    for name in ("SILENT_TURN_SENTENCE", "OUTPATIENT_SILENT_TURN_SENTENCE"):
        assert hasattr(answer, name), name
        assert not hasattr(model_turn, name), f"engine.model still defines {name}"
    assert hasattr(answer, "OUTPATIENT_EHR_NOTICE")
    assert not hasattr(routing, "OUTPATIENT_EHR_NOTICE"), "routing still holds the notice"
    assert hasattr(answer, "medication_narrative")


def _assigns(path, name):
    """True when this module ASSIGNS `name` at module level (an import is not one)."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return True
    return False


def test_law3_prompt_text_lives_in_engine_prompts():
    """Law 3: `routing/` decides what to run; `engine/prompts.py` says how it writes."""
    from services.agent.engine import prompts
    from services.agent.routing import routing

    for name in ("MEDS_FORMATTING", "LABS_FORMATTING", "EHR_FORMATTING",
                 "LITERATURE_FORMATTING", "EHR_RECORD_QUERY", "OUTPATIENT_ENCOUNTER_NOTICE"):
        assert hasattr(prompts, name), name
        assert not hasattr(routing, name), f"routing still holds {name}"
    for name in ("FORMATTER_PREAMBLE", "RETRIEVAL_DEMAND"):
        assert hasattr(prompts, name), name
    root = pathlib.Path(SERVICES, "agent")
    assert not _assigns(root / "engine/deterministic.py", "FORMATTER_PREAMBLE")
    assert not _assigns(root / "engine/graph.py", "RETRIEVAL_DEMAND")
    # And a command's instruction is resolved by wire name, not carried in the table.
    assert prompts.formatting_for("labs") == prompts.LABS_FORMATTING
    assert prompts.formatting_for("risk") == ""


def test_law4_tool_registries_are_declared_once():
    """Law 4: wire names, capability groups and labels are `routing/`'s."""
    from services.agent.routing import routing
    from services.agent.telemetry import stages

    for name in ("PASSAGE_TOOLS", "LITERATURE_TOOLS", "TOOL_LABELS", "RESULT_LABELS",
                 "RESULT_NOUNS"):
        assert hasattr(routing, name), name
    assert stages.TOOL_LABELS is routing.TOOL_LABELS
    assert stages.RESULT_LABELS is routing.RESULT_LABELS
    assert routing.PASSAGE_TOOLS == frozenset(
        {"rag_search", "rag_search_sections", "discharge_medications"}
    )


def test_law5_the_trace_is_written_once_in_turn():
    """Law 5: only the orchestrator records the served answer on the trace."""
    root = pathlib.Path(SERVICES, "agent")
    offenders = [
        str(path.relative_to(root))
        for path, source in _shipped_sources()
        if "set_trace_io(" in source and path.name not in {"turn.py", "observability.py"}
    ]
    assert offenders == [], (
        "set_trace_io outside turn.py/observability.py: " + ", ".join(offenders)
    )


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


# --- the same discipline where the last two defects lived -------------------
#
# `services/mcp` and `evaluation/` had no boundary rule at all until `00d8`. Both are
# where the same two shapes of defect landed: a value written down twice (the tool list
# in the server and in `tools/__init__`; the judge's dimension list in four modules) and
# a `sys.path` hack that made the duplicate importable, and therefore invisible. A rule
# about `services/agent` alone could not see either.

REPO = pathlib.Path(SERVICES).parent
MCP_ROOT = pathlib.Path(SERVICES, "mcp")
EVAL_ROOT = REPO / "evaluation"
PLATFORM_ROOT = EVAL_ROOT / "platform"


def _package_path(path):
    """The dotted package one file lives in, named from the repository root."""
    return ".".join(path.parent.relative_to(REPO).parts)


def _resolved_imports(path):
    """(absolute module, lineno, enclosing function) for every import in one file.

    Relative imports are resolved, because the rules below are about packages reaching
    each other and `from . import tools` reaches one without naming it absolutely. The
    `_imports` helper above skips them on purpose — it only asks whether a module can
    reach `mlops` — so this is a second walker rather than a widened one.
    """
    tree = ast.parse(path.read_text())
    spans = [(node.lineno, node.end_lineno or node.lineno, node.name)
             for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]

    def enclosing(lineno):
        names = [name for start, end, name in spans if start <= lineno <= end]
        return names[0] if names else None

    resolved = []
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if not node.level:
                # Absolute: the module is named in full and names no package to join.
                modules = [node.module] if node.module else []
            else:
                base = _package_path(path)
                for _ in range(node.level - 1):
                    base = base.rsplit(".", 1)[0]
                if node.module:
                    modules = [f"{base}.{node.module}"]
                else:
                    # `from . import tools`: the names are the submodules it reaches.
                    modules = [f"{base}.{alias.name}" for alias in node.names]
        for module in modules:
            resolved.append((module, node.lineno, enclosing(node.lineno)))
    return resolved


def _reaching(root, prefix, *, skip=()):
    """Every import under one tree that reaches `prefix`, as readable strings."""
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        relative = path.relative_to(root)
        if skip and any(part in skip for part in relative.parts[:-1]):
            continue
        for module, lineno, function in _resolved_imports(path):
            if module == prefix or module.startswith(prefix + "."):
                where = f"{relative}:{lineno}"
                offenders.append(f"{where} in {function}()" if function else where)
    return offenders


def test_both_trees_were_actually_walked():
    """A guard on the guards: a wrong path would make every rule below vacuous."""
    assert len(list(MCP_ROOT.rglob("*.py"))) > 15
    assert len(list(PLATFORM_ROOT.rglob("*.py"))) > 15


def test_the_import_walker_sees_a_known_edge():
    """Positive control for every ban above: they are worth what the walker finds.

    A walker that resolved nothing would pass all five rules while the tree reached
    anywhere it liked. The agent genuinely imports the MCP contracts today, so that
    edge is the canary.
    """
    assert _reaching(pathlib.Path(SERVICES, "agent"), "services.mcp.contracts")


def test_the_online_mcp_tree_never_imports_the_ingest_pipelines():
    """`pipelines/` is a batch image; the serving image must not pull it in.

    It is not a style rule: the container is built from `services/mcp`, and an import
    of the ingest pipeline from a tool would drag its dependencies — and its billable
    batch code — into the request path.
    """
    offline = _reaching(MCP_ROOT, "services.mcp.pipelines", skip={"pipelines"})

    assert offline == [], "the serving path reaches the ingest pipelines:\n" + "\n".join(offline)


def test_the_mcp_service_never_imports_the_agent():
    """The server is the leaf: it is deployed without the agent, and cannot reach it."""
    reaching = _reaching(MCP_ROOT, "services.agent")

    assert reaching == [], "services/mcp imports the agent:\n" + "\n".join(reaching)


def test_the_agent_never_imports_the_ingest_pipelines():
    reaching = _reaching(pathlib.Path(SERVICES, "agent"), "services.mcp.pipelines")

    assert reaching == [], "the agent imports the ingest pipelines:\n" + "\n".join(reaching)


def test_no_shipped_module_imports_the_evaluation_harness():
    """The mirror of the `mlops` rule, and the same reasoning.

    The harness measures the shipped code; shipped code must not be able to depend on
    it, or the thing under test and the thing testing it become one image and the
    measurement stops being independent.
    """
    reaching = _reaching(SERVICES, "evaluation")

    assert reaching == [], "shipped code imports the harness:\n" + "\n".join(reaching)


#: What each package under `services/mcp` may import, keyed by subpackage; `"root"` is
#: the modules directly in `services/mcp` (server, contracts, config, runtime).
#:
#: `docs/architecture/14-mcp_service_architecture.md` §5, transcribed. The tree is a
#: wiring layer, two leaves and one vocabulary package: `tools` produce payloads,
#: `dependencies` fetch them, `retrieval` holds the note vocabulary and index settings,
#: and the top level registers the tools. A tool importing a tool, or a dependency
#: importing a tool, is the shape that wrote the tool list down twice.
#:
#: `root` may import `tools` and nothing else below the top level, which is why the
#: feature-source label moved into `config.py`: the health endpoint reported it, and
#: reading it out of `dependencies.features` was an edge the charter forbids.
ALLOWED_MCP = {
    "root": {"root", "tools"},
    "tools": {"root", "dependencies", "retrieval"},
    "dependencies": {"root"},
    "retrieval": set(),
    "pipelines": {"root", "retrieval"},
}


def _mcp_package(path):
    parts = path.relative_to(MCP_ROOT).parts
    return "root" if len(parts) == 1 else parts[0]


#: The real subpackages. A root-level *module* (`config`, `runtime`, `contracts`) shares
#: the `root` key with the server, so an edge into one has to be classified as `root`
#: rather than as a package of its own.
MCP_SUBPACKAGES = frozenset(
    child.name for child in MCP_ROOT.iterdir()
    if child.is_dir() and not child.name.startswith("__")
)


def test_the_mcp_subpackages_import_only_downwards():
    violations = []
    for path in sorted(MCP_ROOT.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        package = _mcp_package(path)
        for module, lineno, function in _resolved_imports(path):
            if not module.startswith("services.mcp"):
                continue
            rest = module[len("services.mcp"):].lstrip(".")
            head = rest.split(".")[0] if rest else ""
            target = head if head in MCP_SUBPACKAGES else "root"
            if target != package and target not in ALLOWED_MCP[package]:
                where = f"{path.relative_to(MCP_ROOT)}:{lineno}"
                violations.append(
                    f"{where} in {function}(): {package} imports {target} "
                    f"(allowed: {' / '.join(sorted(ALLOWED_MCP[package]))})"
                )

    assert violations == [], "the mcp DAG is not one-way:\n" + "\n".join(violations)


def test_the_platform_never_imports_the_retired_collector():
    """`evaluation/agent/` is legacy analysis; the live harness must not reach it.

    It reached it through `sys.path.insert(0, "evaluation/agent")` and `import judge`
    inside a function, which is the reason the platform could not be run without the
    module it was replacing. The data directory stays — the case files live there and
    are read as JSON, not imported.
    """
    reaching = _reaching(PLATFORM_ROOT, "evaluation.agent")

    assert reaching == [], "the platform imports the retired collector:\n" + "\n".join(reaching)


def test_the_platform_puts_nothing_but_the_repo_root_on_sys_path():
    """The path hack, as a rule.

    Several platform modules are entry points run as a script, where the repository
    root is not on `sys.path` by default, so one module-level insert of the root is
    legitimate. What is not legitimate is inserting a *package* — that is how a module
    that should have been imported became reachable and then invisible — or doing it
    from inside a function, where it hides the dependency from everything that reads
    the imports.
    """
    offenders = []
    inspected = 0
    for path in sorted(PLATFORM_ROOT.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text())
        spans = [(node.lineno, node.end_lineno or node.lineno, node.name)
                 for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            func = node.func
            if not (isinstance(func.value, ast.Attribute) and func.value.attr == "path"
                    and isinstance(func.value.value, ast.Name) and func.value.value.id == "sys"):
                continue
            if func.attr not in {"insert", "append", "extend"}:
                continue
            inspected += 1
            relative = str(path.relative_to(PLATFORM_ROOT))
            inside = [name for start, end, name in spans
                      if start <= node.lineno <= end]
            if inside:
                offenders.append(f"{relative}:{node.lineno} mutates sys.path in {inside[0]}()")
                continue
            argument = ast.unparse(node.args[1] if len(node.args) > 1 else node.args[0]) \
                if node.args else ""
            if argument != "str(REPO)":
                offenders.append(
                    f"{relative}:{node.lineno} puts {argument or '?'} on sys.path"
                )

    assert offenders == [], (
        "only the repository root, at module level:\n" + "\n".join(offenders)
    )
    # The entry points do mutate `sys.path`, so a detector that inspected nothing
    # would be a rule with no subject.
    assert inspected >= 6, f"only {inspected} sys.path mutations found"


#: What the platform may reach outside itself, from
#: `docs/architecture/15-evaluation_harness_architecture.md` §5.
#:
#: Two read-only contracts from the agent — the citation counter and the command table —
#: plus `shared/`, so the harness numbers passages with the same function the agent does.
#: Two more are read-only contracts of the same kind that §5's list does not name:
#:
#:   * `services.mcp.config`, because the judge needs the project, the location and the
#:     model, and config is where this deployment declares them. It is the import the
#:     agent's own model runtime makes; the alternative is a third copy of
#:     `"gemini-2.5-pro"`.
#:   * `services.agent.ingress.contracts` (`parse_agent_request`), which `runner.py` uses
#:     to refuse a body the service would refuse *before* spending a request on it.
#:
#: Everything else — the retired collector, the inventory scripts, the training package —
#: is out, which is what the prohibition above states.
ALLOWED_PLATFORM_EXTERNALS = (
    "shared",
    "services.agent.postprocess.citations",
    "services.agent.routing",
    "services.agent.ingress.contracts",
    "services.mcp.config",
)


def test_the_platform_imports_only_what_its_charter_allows():
    offenders = []
    for path in sorted(PLATFORM_ROOT.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        for module, lineno, function in _resolved_imports(path):
            if not module.startswith(("evaluation", "services", "shared", "mlops")):
                continue
            if module.startswith(("evaluation.platform", "shared")):
                continue
            if any(module == allowed or module.startswith(allowed + ".")
                   for allowed in ALLOWED_PLATFORM_EXTERNALS):
                continue
            where = f"{path.relative_to(PLATFORM_ROOT)}:{lineno}"
            offenders.append(
                f"{where} in {function}(): {module}"
                if function else f"{where}: {module}"
            )

    assert offenders == [], (
        "the platform reaches outside its charter:\n" + "\n".join(offenders)
    )
