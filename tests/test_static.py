import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_EXTS = {".py", ".yaml", ".yml", ".json", ".sh", ".sbatch", ".ipynb"}
SCANNED_DIRS = ("src/mbs", "configs", "scripts", "data_prep")


def iter_source_py_files():
    for path in (REPO_ROOT / "src" / "mbs").rglob("*.py"):
        if path.is_file():
            yield path


def iter_text_files():
    for relative in SCANNED_DIRS:
        base = REPO_ROOT / relative
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in CONFIG_EXTS:
                yield path


def iter_data_prep_notebooks():
    base = REPO_ROOT / "data_prep"
    if not base.exists():
        return
    yield from base.rglob("*.ipynb")


def test_no_old_src_imports():
    """No module should import from the pre-reorg ``src`` package layout.

    Uses AST so we don't false-positive on docstrings or unrelated identifiers
    (and so this file itself doesn't need to dodge its own keywords).
    """
    offenders = []
    for path in iter_source_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".", 1)[0] == "src":
                    offenders.append((path.relative_to(REPO_ROOT), node.lineno))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] == "src":
                        offenders.append((path.relative_to(REPO_ROOT), node.lineno))
    assert offenders == [], offenders


def test_no_private_cluster_paths_or_usernames():
    """Scan source, configs, and scripts for leaked private paths or identifiers."""
    banned = [
        "/" + "Users/",
        "/" + "mnt/scratch",
        "/" + "work/upschrimpf",
        "/" + "scratch/izar",
        "/" + "capstor/",
        "ak" + "gokce",
        "jed" + ".epfl.ch",
        "clar" + "iden:",
    ]
    offenders = []
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in banned:
            if token in text:
                offenders.append((path.relative_to(REPO_ROOT), token))
    assert offenders == [], offenders


def test_data_prep_notebooks_are_clean():
    offenders = []
    for path in iter_data_prep_notebooks():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for idx, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None:
                offenders.append((path.relative_to(REPO_ROOT), idx, "execution_count"))
            if cell.get("outputs"):
                offenders.append((path.relative_to(REPO_ROOT), idx, "outputs"))
    assert offenders == [], offenders
