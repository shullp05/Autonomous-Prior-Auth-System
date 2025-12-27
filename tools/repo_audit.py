import json
import os
import re
import subprocess
from collections.abc import Iterable

from config import MAX_SEARCH_DEPTH


def _is_hidden(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    return any(p.startswith(".") for p in parts if p)


def _safe_find(root: str = ".") -> list[str]:
    # Hidden directories/files excluded via '*/.*'; depth limited by MAX_SEARCH_DEPTH.
    # We do not follow symlinks to avoid loops; fallback to os.walk respects followlinks=False.
    cmd = [
        "find",
        root,
        "-maxdepth",
        str(MAX_SEARCH_DEPTH),
        "-not",
        "-path",
        "*/.*",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        return lines
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"find failed: {e.output}")


def _walk_limited(root: str = ".", max_depth: int = MAX_SEARCH_DEPTH) -> list[str]:
    results: list[str] = []
    base_depth = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        if _is_hidden(dirpath):
            dirnames[:] = []
            continue
        depth = dirpath.rstrip("/").count("/") - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        results.append(dirpath)
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not _is_hidden(fp):
                results.append(fp)
    return results


def _py_imports(files: Iterable[str]) -> set[str]:
    imports: set[str] = set()
    pattern = re.compile(r"^(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))")
    for fp in files:
        if not fp.endswith(".py"):
            continue
        try:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = pattern.match(line.strip())
                    if m:
                        mod = m.group(1) or m.group(2)
                        if mod:
                            imports.add(mod.split(".")[0])
        except Exception:
            pass
    return imports


def _referenced_paths(all_paths: Iterable[str]) -> set[str]:
    refs: set[str] = set()
    for fp in all_paths:
        base = os.path.basename(fp)
        if base in {"Makefile", "pyproject.toml", "requirements.txt"}:
            try:
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                for candidate in all_paths:
                    bn = os.path.basename(candidate)
                    if bn and bn in text:
                        refs.add(candidate)
            except Exception:
                continue
        if fp.endswith("package.json") or fp.endswith("index.html"):
            try:
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                for candidate in all_paths:
                    bn = os.path.basename(candidate)
                    if bn and bn in text:
                        refs.add(candidate)
            except Exception:
                continue
    return refs


def _is_excluded(path: str) -> bool:
    ex_dirs = {"__pycache__", "node_modules", "chroma_db"}
    if any(f"/{d}/" in path.replace("\\", "/") for d in ex_dirs):
        return True
    if path.endswith(":Zone.Identifier"):
        return True
    return False


def audit_repo() -> dict:
    # Repository audit with hidden exclusion and depth limit; robust to find failures.
    try:
        found = _safe_find(".")
    except RuntimeError:
        found = _walk_limited(".", MAX_SEARCH_DEPTH)

    found = [p for p in found if not _is_excluded(p)]
    py_imports = _py_imports(found)
    refs = _referenced_paths(found)

    used: set[str] = set()
    for p in found:
        bn = os.path.basename(p)
        if bn.endswith(".py"):
            used.add(p)
        if bn in {"data_patients.csv", "data_medications.csv", "data_conditions.csv", "data_observations.csv"}:
            used.add(p)
        if bn in {"batch_runner.py", "agent_logic.py", "policy_engine.py", "governance_audit.py", "config.py", "policy_constants.py"}:
            used.add(p)

    used |= refs

    orphans = [p for p in found if p not in used]

    # Categorize orphans by file extension for easier triage
    from collections import Counter
    ext_counts = Counter(os.path.splitext(p)[1] or "(no ext)" for p in orphans)

    report = {
        "depth": MAX_SEARCH_DEPTH,
        "total_found": len(found),
        "used_count": len(used),
        "orphan_count": len(orphans),
        "orphan_extensions": dict(ext_counts),
        "orphans": orphans,
    }
    return report


if __name__ == "__main__":
    try:
        rep = audit_repo()
        with open("orphan_report.json", "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(f"Audit complete. Found {rep['total_found']} items; orphans: {rep['orphan_count']}")
    except Exception as e:
        print(f"ERROR: {e}")
        raise
