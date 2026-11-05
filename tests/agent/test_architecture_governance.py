"""The production services may not encode a scenario, and the layers may not reach past each other.

Two classes of decay this checks for, both of which the clinical copilot has already paid for
once:

  * **A scenario written into production code.** The suite names six admissions and the
    notes name specific drugs and procedures. A module that refers to any of them has
    stopped being a general reader of a record and become a lookup table for the fixture it
    was debugged against — and it will be right about that fixture and silent about the
    next one. The exemptions are two files that legitimately enumerate records or address
    the model: `ingress/contracts.py` (the cohort whitelist the deployment serves) and
    `engine/prompts.py` (instructions to the model, which name drugs as examples).

  * **A layer importing the layer that runs after it.** `postprocess` and `presentation`
    are called BY the engine and must not call it back; `ingress` and `routing` decide what
    a turn is before anything is rendered, so they may not see a renderer or a settler. The
    DAG those rules describe is enforced at runtime elsewhere
    (`tests/agent/test_layer_separation.py`); this is the source-level half, and it also
    holds for modules nothing currently imports.

  * **`services/` reading the harness.** The product must not be able to see how it is
    being graded. `evaluation/` imports `services/`; the reverse would make a deployable
    image depend on a directory that is locked read-only and excluded from it.

Scope is executable string constants, read through the AST. A comment or a docstring that
records the incident that motivated a rule — `(90000023, 2026-10-26)` — is a record of what
was fixed, not a decision the code makes, and a test that failed on those would force the
history out of the source. Comments never reach the AST at all; docstring constants are
excluded explicitly.
"""

from __future__ import annotations

import ast
from pathlib import Path

SERVICES = Path(__file__).resolve().parents[2] / "services"

#: Admissions the Tier 2 suite is written against. A production module may not name one.
BANNED_RECORD_IDS: frozenset[str] = frozenset({
    "90000015", "90000106", "90000023", "90000108", "90000105", "90000051",
})

#: Drugs and procedures that belong to particular notes in the cohort. A drug-class
#: vocabulary is a different thing and is judged by its own test; these are the literals
#: that only ever appear because one fixture was being debugged.
BANNED_CLINICAL_LITERALS: frozenset[str] = frozenset({
    "lovenox", "plavix", "zithromax", "xopenex", "toprol", "oxycodone", "carvedilol",
    "ercp",
})

#: Files allowed to name a record or a drug: the whitelist of admissions this deployment
#: serves, and the instructions given to a model.
#:
#: Two more are exempt from the LITERAL check only, and each for a reason that does not
#: generalise to any other file:
#:
#:   * `agent/postprocess/followups.py` holds `DRUG_CLASSES` — a vocabulary of anticoagulant
#:     and beta-blocker names used to ask a chip about a class of drug. `lovenox`,
#:     `toprol` and `carvedilol` there are members of a class, not one note's drugs; the
#:     module reads a discharge summary it has never seen.
#:   * `agent/ingress/run.py` holds `DEFAULT_QUESTION`, the question a local run asks when
#:     the caller supplies none. It names an admission because a default has to name
#:     something, and it never reaches a served answer.
#:
#: Neither exemption widens the import-boundary tests below: those walk every module under
#: the subpackage, exempt or not.
EXEMPT_FILES: frozenset[str] = frozenset({
    "agent/ingress/contracts.py",
    "agent/engine/prompts.py",
    "agent/ingress/run.py",
    "agent/postprocess/followups.py",
})


def _modules() -> list[Path]:
    return sorted(path for path in SERVICES.rglob("*.py")
                  if "__pycache__" not in path.parts)


def _relative(path: Path) -> str:
    return path.relative_to(SERVICES).as_posix()


def _code_strings(tree: ast.AST) -> list[tuple[int, str]]:
    """Every string constant the module EXECUTES, with its line.

    Docstrings are excluded by identity: the first `Expr(Constant(str))` of a module, a
    class or a function is documentation, and the incident dates in those docstrings are
    the reason a rule exists rather than something the rule reads.
    """
    documented: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None) or []
        first = body[0] if body else None
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            documented.add(id(first.value))

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in documented):
            found.append((node.lineno, node.value))
    return found


def test_no_scenario_record_or_drug_is_hardcoded_in_production_code():
    """A cohort literal in `services/` is a fixture the product cannot generalise past."""
    failures: list[str] = []
    for path in _modules():
        relative = _relative(path)
        if relative in EXEMPT_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, text in _code_strings(tree):
            lowered = text.lower()
            for record in BANNED_RECORD_IDS:
                if record in lowered:
                    failures.append(
                        f"{relative}:{line}: record id {record} is hardcoded in a string "
                        "constant")
            for literal in BANNED_CLINICAL_LITERALS:
                if literal in lowered:
                    failures.append(
                        f"{relative}:{line}: clinical literal {literal!r} is hardcoded in "
                        "a string constant")

    assert not failures, "\n".join(failures)


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    """Every module this file imports, by dotted name, with the line it is imported on."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # A relative import is inside the package the file already belongs to, and
            # `from . import engine` is the same statement as `from services.agent import
            # engine` written shorter. Both are resolved through the caller's subpackage.
            found.append((node.lineno, module))
            for alias in node.names:
                found.append((node.lineno, f"{module}.{alias.name}" if module
                              else alias.name))
    return found


def _importers(subpackage: str) -> list[tuple[str, int, str]]:
    """Every import statement written in one `services/agent/<subpackage>/` directory."""
    root = SERVICES / "agent" / subpackage
    rows: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for line, module in _imported_modules(path):
            rows.append((_relative(path), line, module))
    return rows


def _reaches(module: str, target: str) -> bool:
    """Whether one import names `target`, absolutely or relative to its own package.

    A relative import inside `services/agent/postprocess/` of `from ..engine import x`
    arrives here as `engine`, so the comparison is against the last component too. That is
    deliberately loose: an import of anything named `engine` from a layer that may not see
    the engine is the defect, whatever it was spelled.
    """
    parts = [part for part in module.split(".") if part]
    return target in parts


def test_postprocess_and_presentation_never_import_the_engine():
    """The settlement and the projection are called BY the engine, never the reverse."""
    failures: list[str] = []
    for subpackage in ("postprocess", "presentation"):
        for relative, line, module in _importers(subpackage):
            if _reaches(module, "engine"):
                failures.append(f"{relative}:{line}: imports {module!r}; {subpackage} "
                                "may not depend on engine")
    assert not failures, "\n".join(failures)


def test_ingress_and_routing_never_import_a_renderer_or_a_settler():
    """What a turn IS is decided before anything is drawn or settled."""
    failures: list[str] = []
    for subpackage in ("ingress", "routing"):
        for relative, line, module in _importers(subpackage):
            for forbidden in ("presentation", "postprocess"):
                if _reaches(module, forbidden):
                    failures.append(f"{relative}:{line}: imports {module!r}; {subpackage} "
                                    f"may not depend on {forbidden}")
    assert not failures, "\n".join(failures)


def test_services_never_imports_the_evaluation_harness():
    """The product may not be able to see how it is graded."""
    failures: list[str] = []
    for path in _modules():
        for line, module in _imported_modules(path):
            if _reaches(module, "evaluation"):
                failures.append(f"{_relative(path)}:{line}: imports {module!r}; services "
                                "may not depend on evaluation")
    assert not failures, "\n".join(failures)
