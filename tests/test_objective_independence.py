"""The wall between reward and measurement, enforced by reading the code.

An agent in this framework writes rewards and is scored by an evaluator it
cannot touch. That separation is only real if the *measurement path* cannot
reach reward code at all -- not directly, not through four hops of helper
module. So this test parses sources with `ast` and walks the first-party
import graph from both ends of the measurement path:

* `autobranch.objective`, which turns a checkpoint into a score, and
* the environment's own evaluator (`examples/gridworld/grid_env.py`).

A `sys.modules` check would be weaker in exactly the way that matters: it only
sees what some earlier import happened to pull in, so it passes by luck. The
graph walk fails the moment someone adds the import, before it is ever run.

The predecessor project this framework generalises had the same test, and it
caught the leak. It is the backbone of the framework's credibility: without
it, "the objective is immutable" is a comment.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "examples" / "gridworld"

#: Reward composition and the trainer that applies it. Anything on the
#: measurement path that can import these can, in principle, be influenced by
#: the thing being measured.
FORBIDDEN = ("autobranch.reward", "autobranch.trainer")

#: Where first-party modules may live: the repo root (for `autobranch.*`) and
#: the bundle root (for an environment's own modules, e.g. `grid_env`).
SEARCH_ROOTS = (REPO, BUNDLE)


def _resolve(module: str) -> Path | None:
    """Map a dotted module name to a file, or None if it is not first-party."""
    parts = module.split(".")
    for root in SEARCH_ROOTS:
        base = root.joinpath(*parts)
        init = base / "__init__.py"
        if init.exists():
            return init
        leaf = base.with_suffix(".py")
        if leaf.exists():
            return leaf
    return None


def _imports_in(path: Path, module_name: str) -> list[str]:
    """Every module name `path` imports, including inside function bodies."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    package = module_name.rsplit(".", 1)[0] if "." in module_name else module_name
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: rebuild the absolute name
                prefix = package.split(".")[: max(0, len(package.split(".")) - (node.level - 1))]
                base = ".".join(prefix)
                found.append(f"{base}.{node.module}" if node.module else base)
            elif node.module:
                found.append(node.module)
                # `from autobranch import reward` imports a submodule, not a name.
                found.extend(f"{node.module}.{a.name}" for a in node.names)
    return found


def import_closure(entry_module: str) -> dict[str, Path]:
    """Transitively walk first-party imports from `entry_module`.

    Package `__init__` files are included: importing `autobranch.env_api`
    executes `autobranch/__init__.py`, so an import added there would be a
    real leak on the measurement path.
    """
    start = _resolve(entry_module)
    if start is None:
        raise AssertionError(f"cannot find source for {entry_module!r} under {SEARCH_ROOTS}")
    seen: dict[str, Path] = {entry_module: start}
    queue = [(entry_module, start)]
    while queue:
        name, path = queue.pop()
        for imported in _imports_in(path, name):
            if imported in seen:
                continue
            resolved = _resolve(imported)
            if resolved is None:
                continue  # third-party or stdlib: not ours to police
            seen[imported] = resolved
            queue.append((imported, resolved))
            # Walking a submodule means its packages run too.
            parts = imported.split(".")
            for i in range(1, len(parts)):
                parent = ".".join(parts[:i])
                if parent not in seen:
                    parent_path = _resolve(parent)
                    if parent_path is not None:
                        seen[parent] = parent_path
                        queue.append((parent, parent_path))
    return seen


@pytest.mark.parametrize("entry", ["autobranch.objective", "grid_env"])
def test_measurement_path_cannot_reach_reward_code(entry: str) -> None:
    closure = import_closure(entry)
    leaked = sorted(set(closure) & set(FORBIDDEN))
    assert not leaked, (
        f"{entry} transitively imports {leaked}; the objective must not be able to see "
        f"reward composition. Reached modules: {sorted(closure)}"
    )


@pytest.mark.parametrize("entry", ["autobranch.objective", "grid_env"])
def test_no_dynamic_import_of_reward_code(entry: str) -> None:
    """Catch the dynamic dodge: `importlib.import_module("autobranch.reward")`.

    The import graph above only sees `import` statements, so a module name
    hidden in a string would slip past it. Checking string *constants* for
    equality (not prose for substrings) keeps this precise -- the docstrings
    on the measurement path discuss these module names constantly, and they
    should be allowed to.
    """
    for name, path in import_closure(entry).items():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for forbidden in FORBIDDEN:
                    assert not (
                        node.value == forbidden or node.value.startswith(forbidden + ".")
                    ), f"{name} ({path}) line {node.lineno} names {forbidden!r} as a string"


def test_the_walker_actually_finds_a_dependency() -> None:
    """Positive control.

    A test that cannot fail is decoration. The trainer *does* import the
    reward compiler, so the walk must see it from there -- otherwise the two
    tests above would pass on a broken walker.
    """
    closure = import_closure("autobranch.trainer")
    assert "autobranch.reward" in closure
    assert "autobranch.spec" in closure
    assert "autobranch.env_api" in closure


def test_objective_runs_without_importing_reward_code() -> None:
    """Belt and braces: the AST test is the gate, this confirms it at runtime."""
    code = (
        "import sys;"
        "sys.path.insert(0, %r);"
        "sys.path.insert(0, %r);"
        "import autobranch.objective, grid_env;"
        "leaked=[m for m in %r if m in sys.modules];"
        "print(leaked)" % (str(BUNDLE), str(REPO), FORBIDDEN)
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO), check=True
    )
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
