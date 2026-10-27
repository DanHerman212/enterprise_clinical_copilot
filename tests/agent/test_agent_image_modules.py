"""The agent image must contain every module the agent imports.

Why this is a test and not a habit. The agent's image copies `services/agent/`
and a named handful of `services/mcp/` modules — deliberately, because the rest
of that package pulls in BigQuery and Vertex and the agent reaches all of it
through the MCP server instead. So an agent module may import only what is on
that list, and nothing local can tell the difference: in the repository, every
module is present.

When the rule is broken the failure is quiet from the developer's side and loud
only in production. `routing.py` imported a constant from the chunker
(2026-10-13); the image built, pushed and deployed, and the container died on
`ModuleNotFoundError` before it could listen. Cloud Run kept the previous
revision, so the deploy reported failure while production kept serving a build
from days earlier — three pushes in a row. The only way to see it locally is to
ask the Dockerfile what the image actually contains, which is what this does.

Read the file list from the Dockerfile rather than restating it here: a test that
carried its own copy of the list would pass while the Dockerfile drifted.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "services" / "agent" / "Dockerfile"


def _image_python_files() -> set[str]:
    """Repo-relative paths of the .py files the image's COPY lines put in /app.

    Line continuations are joined first: the retrieval modules are spread over two
    physical lines, and reading them line by line reported the modules on the
    second line as absent from the image. A parser that lies about the image
    fails the build for a module that is right there.
    """
    copied: set[str] = set()
    text = DOCKERFILE.read_text().replace("\\\n", " ")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("COPY "):
            continue
        for token in stripped.split()[1:]:
            if not token or token.startswith("./") or token.startswith("/"):
                continue  # the destination, or a path outside the build context
            source = ROOT / token
            if source.is_dir():
                copied.update(
                    str(path.relative_to(ROOT)) for path in source.rglob("*.py")
                )
            elif source.suffix == ".py":
                copied.add(str(source.relative_to(ROOT)))
    return copied


def _imported_packages(path: Path) -> set[str]:
    """Every `services.*` module path this file imports, as module names."""
    modules: set[str] = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.name for alias in node.names if alias.name.startswith("services.")
            )
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "services."
        ):
            modules.add(node.module)
            # `from services.mcp import config` names a module in the imported
            # package, not an attribute of it.
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def _candidate_files(parts: list[str]) -> set[str]:
    """The two shapes an import of `parts` can name: a module, or a package."""
    base = "/".join(parts)
    return {c for c in (f"{base}.py", f"{base}/__init__.py") if (ROOT / c).is_file()}


def _resolutions(module: str) -> tuple[str, set[str]] | None:
    """The importable prefix of `module`, and the files it could be.

    `from services.agent.presentation.a2ui import CitedNarrative` names a symbol at the end, not
    a module, so the prefix is trimmed until it resolves. A name that resolves to
    nothing at all is a standard-library or third-party import, or a symbol this
    check has no business judging, and is reported as None rather than as missing.

    The earlier version of this dropped the candidate file whenever it existed —
    "a package, not a module" — which left every real module resolving to an empty
    set, so every missing import was skipped and the guard could not fail. The
    probe that reinstated the production import is what found it.
    """
    parts = module.split(".")
    while parts:
        files = _candidate_files(parts)
        if files:
            return ".".join(parts), files
        parts.pop()
    return None


def _unreachable_modules() -> dict[str, set[str]]:
    """Imports made from inside the image that the image cannot satisfy.

    Walked transitively: a module that is copied can itself import a module that
    is not, which is the same failure one layer further in.
    """
    in_image = _image_python_files()
    missing: dict[str, set[str]] = {}
    seen: set[str] = set()
    queue = sorted(in_image)
    while queue:
        relative = queue.pop()
        if relative in seen:
            continue
        seen.add(relative)
        for module in _imported_packages(ROOT / relative):
            resolved = _resolutions(module)
            if resolved is None:
                continue  # the standard library, a dependency, or a symbol
            name, files = resolved
            if not (files & in_image):
                missing.setdefault(relative, set()).add(name)
            queue.extend(sorted(files & in_image))
    return missing


def test_the_agent_image_copies_every_module_the_agent_imports():
    missing = _unreachable_modules()

    assert missing == {}, (
        "These modules are imported inside the image but are not copied into it, so "
        "the container dies at import and Cloud Run keeps serving the previous "
        f"revision: { {k: sorted(v) for k, v in sorted(missing.items())} }"
    )


def test_the_check_can_actually_fail():
    """The control. A module reached only through the chunker is still absent.

    Without this, a parser that returned no imports at all would report success
    and this file would be decoration.
    """
    in_image = _image_python_files()

    assert "services/agent/http.py" in in_image
    assert "services/mcp/retrieval/sections.py" in in_image
    assert "services/mcp/retrieval/chunking.py" not in in_image
    assert "services/mcp/eval_report.py" not in in_image


def test_the_copied_retrieval_module_can_be_imported_without_the_rest():
    """`sections.py` is the one retrieval module the agent gets, so it has to stand
    on its own: pure standard library, no package siblings."""
    imports = _imported_packages(ROOT / "services" / "mcp" / "retrieval" / "sections.py")

    assert not {m for m in imports if m.startswith("services.")}, sorted(imports)
