"""Shared file contracts only: no ML, retrieval or generation dependencies."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SCHEMA = "casml-b0/v1"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{n}") from exc
    return rows


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))


def load_config(path):
    import yaml
    result = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Config must be a YAML object")
    return result


def signature(stage, config, inputs, modules):
    # A generation edit cannot invalidate retrieval: only this stage's source
    # files and the shared contract are included, never the whole repository.
    names = set(modules) | {"artifacts.py", "contracts.py"}
    root = Path(__file__).parent
    return {"schema": SCHEMA, "stage": stage, "config": config, "inputs": inputs,
            "code": {n: file_hash(root / n) for n in sorted(names)}}


def begin(directory, sig, resumable=False):
    directory = Path(directory)
    identity = digest(sig)
    path = directory / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        if manifest.get("artifact_id") != identity:
            raise ValueError(f"Config/input/code changed for {directory}. Use a NEW --out directory; "
                             "upstream artifacts are reusable.")
        if manifest.get("complete"):
            load_artifact(directory, sig["stage"])
            return manifest, True
        if not resumable:
            raise ValueError(f"Incomplete artifact: {directory}. Use a new --out directory.")
        return manifest, False
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty directory {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "stage": sig["stage"], "artifact_id": identity,
                "signature": sig, "created_utc": datetime.now(timezone.utc).isoformat(),
                "complete": False, "files": {}}
    write_json(path, manifest)
    return manifest, False


def finish(directory, manifest, files, **extra):
    directory = Path(directory)
    manifest.update(extra)
    manifest["files"] = {str(p): file_hash(directory / p) for p in files}
    manifest["complete"] = True
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_artifact(directory, stage):
    directory = Path(directory)
    m = read_json(directory / "manifest.json")
    if m.get("schema") != SCHEMA or m.get("stage") != stage:
        raise ValueError(f"Expected {SCHEMA} {stage} artifact, got {directory}")
    if not m.get("complete"):
        raise ValueError(f"Artifact is incomplete: {directory}; resume that stage first")
    if digest(m["signature"]) != m.get("artifact_id"):
        raise ValueError(f"Manifest identity mismatch: {directory}")
    for name, checksum in m["files"].items():
        target = directory / name
        if not target.is_file() or file_hash(target) != checksum:
            raise ValueError(f"Artifact checksum mismatch: {target}")
    return m


def environment():
    packages = {}
    for name in ("numpy", "PyMuPDF", "transformers", "torch", "sentence-transformers", "faiss-cpu"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "packages": packages}
