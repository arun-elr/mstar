"""Fail when code still uses a CUDA-graph name that #262 renamed.

The engine and the graph runner call only the new hook names
(``get_accelerator_graph_configs``, ``get_piecewise_accelerator_graph_configs``,
``can_use_accelerator_graphs``, ``build_accelerator_graph_buffers``). A
submodule or resource that still overrides an old name inherits the base
default instead, so its graphs are never captured and it runs eager with no
error; the CPU suite cannot notice because it never captures. An import of a
removed module behind ``pytest.importorskip`` or a GPU-only skip is just as
quiet. This scan reads the source with ``ast`` and imports nothing.
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_DIRS = ("mstar", "test", "benchmark", "examples")

RENAMED_MODULES = {
    "mstar.engine.cuda_graph_config": "mstar.engine.accelerator_graph_config",
    "mstar.engine.cuda_graph_runner": "mstar.engine.accelerator_graph_runner",
}
# The engine looks these up on submodules and resources by name.
RENAMED_HOOKS = {
    "get_cuda_graph_configs": "get_accelerator_graph_configs",
    "get_piecewise_cuda_graph_configs": "get_piecewise_accelerator_graph_configs",
    "can_use_cuda_graphs": "can_use_accelerator_graphs",
    "build_cuda_graph_buffers": "build_accelerator_graph_buffers",
}
RENAMED_NAMES = {
    **RENAMED_HOOKS,
    "sample_cuda_graphable_gpu": "sample_accelerator_graphable_gpu",
    "CudaGraphConfig": "AcceleratorGraphConfig",
    "BatchedCudaGraphConfig": "BatchedAcceleratorGraphConfig",
    "PackedCudaGraphConfig": "PackedAcceleratorGraphConfig",
    "PiecewiseCudaGraphConfig": "PiecewiseAcceleratorGraphConfig",
    "CudaGraphConfigType": "AcceleratorGraphConfigType",
    "CudaGraphSlot": "AcceleratorGraphSlot",
    "CudaGraphBucket": "AcceleratorGraphBucket",
    "CudaGraphRunner": "AcceleratorGraphRunner",
    "PiecewiseCudaGraphRunner": "PiecewiseAcceleratorGraphRunner",
    "CudaGraphableSampler": "AcceleratorGraphableSampler",
}
# Read with os.environ.get, so the old spelling is ignored without an error.
RENAMED_ENV_VARS = {
    "MSTAR_REQUIRE_CUDA_GRAPHS": "MSTAR_REQUIRE_ACCELERATOR_GRAPHS",
    "MSTAR_VIT_CUDA_GRAPH": "MSTAR_VIT_ACCELERATOR_GRAPH",
    "COSMOS3_DISABLE_CUDA_GRAPH": "COSMOS3_DISABLE_ACCELERATOR_GRAPH",
    "COSMOS3_DISABLE_PREFILL_CUDA_GRAPH": "COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH",
}
_MODULE_BASENAMES = {old.rsplit(".", 1)[1]: new for old, new in RENAMED_MODULES.items()}


def _renamed_module(dotted: str) -> str | None:
    for old, new in RENAMED_MODULES.items():
        if dotted == old or dotted.startswith(old + "."):
            return new + dotted[len(old):]
    return None


def _package_of(rel_path: str) -> list[str]:
    parts = rel_path[: -len(".py")].split("/")
    return parts[:-1]


def find_renamed_uses(source: str, rel_path: str) -> list[str]:
    """Return ``path:line: message`` for every use of a name #262 renamed."""
    tree = ast.parse(source, filename=rel_path)
    found: dict[tuple[int, str], str] = {}

    def report(node: ast.AST, old: str, new: str, what: str) -> None:
        key = (node.lineno, old)
        if key in found:
            return
        message = f"{rel_path}:{node.lineno}: {what} `{old}`; #262 renamed it to `{new}`"
        if old in RENAMED_HOOKS and what == "defines":
            message += " and the engine calls only the new name, so this override is skipped"
        found[key] = message

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in RENAMED_NAMES:
                report(node, node.name, RENAMED_NAMES[node.name], "defines")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                new = _renamed_module(alias.name)
                if new:
                    report(node, alias.name, new, "imports")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                package = _package_of(rel_path)
                base = package[: len(package) - (node.level - 1)]
                module = ".".join(base + ([module] if module else []))
            new = _renamed_module(module)
            if new:
                report(node, module, new, "imports")
            for alias in node.names:
                if module == "mstar.engine" and alias.name in _MODULE_BASENAMES:
                    report(node, f"mstar.engine.{alias.name}", _MODULE_BASENAMES[alias.name], "imports")
                elif alias.name in RENAMED_NAMES:
                    report(node, alias.name, RENAMED_NAMES[alias.name], "imports")
        elif isinstance(node, ast.Name) and node.id in RENAMED_NAMES:
            report(node, node.id, RENAMED_NAMES[node.id], "uses")
        elif isinstance(node, ast.Attribute) and node.attr in RENAMED_NAMES:
            report(node, node.attr, RENAMED_NAMES[node.attr], "uses")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value in RENAMED_NAMES:
                report(node, value, RENAMED_NAMES[value], "names")
            elif value in RENAMED_ENV_VARS:
                report(node, value, RENAMED_ENV_VARS[value], "reads the env var")
            elif _renamed_module(value):
                report(node, value, _renamed_module(value), "names the module")
    return [found[key] for key in sorted(found)]


def _python_files() -> list[Path]:
    this_file = Path(__file__).resolve()
    files = []
    for directory in SCANNED_DIRS:
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            if path.resolve() != this_file:
                files.append(path)
    return files


def test_no_code_uses_a_cuda_graph_name_that_was_renamed():
    files = _python_files()
    assert files, "found no Python files to scan"
    problems = []
    for path in files:
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        problems.extend(find_renamed_uses(path.read_text(encoding="utf-8"), rel_path))
    assert not problems, "Rename these CUDA-graph names (#262):\n" + "\n".join(problems)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "class Sub(NodeSubmodule):\n    def get_piecewise_cuda_graph_configs(self, d, a):\n        return {}\n",
            "m.py:2: defines `get_piecewise_cuda_graph_configs`; #262 renamed it to "
            "`get_piecewise_accelerator_graph_configs` and the engine calls only the new name",
        ),
        (
            "from mstar.engine import cuda_graph_runner as cgr\n",
            "m.py:1: imports `mstar.engine.cuda_graph_runner`; "
            "#262 renamed it to `mstar.engine.accelerator_graph_runner`",
        ),
        (
            "from mstar.engine.cuda_graph_config import AcceleratorGraphConfig\n",
            "m.py:1: imports `mstar.engine.cuda_graph_config`",
        ),
        ("import mstar.engine.cuda_graph_runner\n", "m.py:1: imports `mstar.engine.cuda_graph_runner`"),
        (
            "x = BatchedCudaGraphConfig(1)\n",
            "m.py:1: uses `BatchedCudaGraphConfig`; #262 renamed it to `BatchedAcceleratorGraphConfig`",
        ),
        ("ok = super().can_use_cuda_graphs(b, i)\n", "m.py:1: uses `can_use_cuda_graphs`"),
        ("f = getattr(sub, 'get_cuda_graph_configs', None)\n", "m.py:1: names `get_cuda_graph_configs`"),
        ("pytest.importorskip('mstar.engine.cuda_graph_runner')\n", "m.py:1: names the module"),
        (
            "on = os.environ.get('COSMOS3_DISABLE_CUDA_GRAPH')\n",
            "m.py:1: reads the env var `COSMOS3_DISABLE_CUDA_GRAPH`",
        ),
    ],
)
def test_scan_reports_each_kind_of_old_name(source, expected):
    (problem,) = find_renamed_uses(source, "m.py")
    assert problem.startswith(expected)


def test_scan_resolves_relative_imports():
    (problem,) = find_renamed_uses("from .cuda_graph_runner import run\n", "mstar/engine/x.py")
    assert problem.startswith("mstar/engine/x.py:1: imports `mstar.engine.cuda_graph_runner`")
    (problem,) = find_renamed_uses("from . import cuda_graph_config\n", "mstar/engine/x.py")
    assert "`mstar.engine.cuda_graph_config`" in problem


def test_scan_ignores_new_names_comments_and_docstrings():
    source = (
        '"""Replaces get_cuda_graph_configs and CudaGraphConfig."""\n'
        "# CudaGraphConfig was renamed\n"
        "from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig\n"
        "class Sub:\n"
        "    def get_accelerator_graph_configs(self, d):\n"
        "        return [AcceleratorGraphConfig()]\n"
        "use = dict(use_cuda_graph=True)\n"
    )
    assert find_renamed_uses(source, "m.py") == []
