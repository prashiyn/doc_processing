"""Boundary checks to keep doc-processing and llm-service separated."""

from __future__ import annotations

import ast
from pathlib import Path


def _iter_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_doc_entrypoint_does_not_mount_llm_router() -> None:
    main_py = Path("src/doc_processing/main.py")
    content = main_py.read_text(encoding="utf-8")
    assert "include_router(llm.router" not in content


def test_ffp_and_documents_do_not_import_local_llm_stack() -> None:
    """
    Doc pipeline layers must call llm-service through llm_runtime only.
    """
    roots = [
        Path("src/doc_processing/ffp"),
        Path("src/doc_processing/routers/documents.py"),
    ]
    forbidden_prefixes = (
        "doc_processing.llms",
        "doc_processing.routers.llm",
    )
    offenders: list[str] = []

    for root in roots:
        files = [root] if root.is_file() else _iter_py_files(root)
        for file_path in files:
            imported = _imported_modules(file_path)
            for module in imported:
                if module.startswith(forbidden_prefixes):
                    offenders.append(f"{file_path}:{module}")

    assert not offenders, f"Found forbidden imports: {offenders}"
