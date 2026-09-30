"""Read ONLY the retrieval artifact, then generate/resume a separate run."""
from __future__ import annotations

import time
from pathlib import Path

from .artifacts import (SCHEMA, begin, digest, environment, finish, load_artifact, read_json,
                        read_jsonl, signature, write_json, write_jsonl)
from .contracts import validate_retrieval
from .context import pack_context
from .llm import make_generator


def load_prompts(config, config_path):
    root = Path(config_path).resolve().parent
    system = (root / config["system_prompt_file"]).read_text(encoding="utf-8").strip()
    user = (root / config["user_prompt_file"]).read_text(encoding="utf-8").strip()
    if "{question}" not in user or "{context}" not in user:
        raise ValueError("User template must contain {question} and {context}")
    user.format(question="test", context="test")
    return system, user


def generate(retrieval_dir, config, config_path, out, limit=None):
    retrieval_dir, out = Path(retrieval_dir), Path(out)
    parent = load_artifact(retrieval_dir, "retrieval")
    rows = read_jsonl(retrieval_dir / "retrieval.jsonl")
    validate_retrieval(rows)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if int(config["context_top_k"]) > int(parent["signature"]["config"].get("top_k", 20)):
        raise ValueError("context_top_k exceeds the retrieval cache's top_k. Build a larger cache first.")
    system, user = load_prompts(config, config_path)
    effective = {**config, "system_prompt_content": system, "user_prompt_content": user}
    sig = signature("generation", effective, {"retrieval_id": parent["artifact_id"], "selected_queries": digest(rows)},
                    ["generation.py", "context.py", "llm.py"])
    manifest, cached = begin(out, sig, resumable=True)
    if cached:
        return manifest
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    results, pending = {}, []
    for row in rows:
        key = digest(row["query_id"])
        path = checkpoint_dir / f"{key}.json"
        previous = read_json(path) if path.exists() else None
        if previous:
            checksum = previous.pop("record_sha256", None)
            if checksum != digest(previous) or previous.get("run_id") != manifest["artifact_id"] or previous.get("query_signature") != digest(row):
                raise ValueError(f"Invalid checkpoint for query {row['query_id']}")
            previous["record_sha256"] = checksum
        if previous and previous["status"] in ("ok", "insufficient_context"):
            results[row["query_id"]] = previous
        else:
            pending.append(row)
    resumed = len(results)
    backend = make_generator(config) if pending else None
    for row in pending:
        started = time.perf_counter()
        result = {"schema": SCHEMA, "run_id": manifest["artifact_id"], "query_signature": digest(row),
                  "query_id": row["query_id"], "question": row["question"],
                  "generator_backend": config.get("backend", "huggingface")}
        try:
            payload = pack_context(row, backend, config, system, user)
            result.update(payload)
            if payload["evidence"]:
                result.update(backend.generate(payload, config))
                if not result["answer"].strip():
                    raise ValueError("Model produced an empty answer")
                result["status"] = "ok"
            else:
                result.update(status="insufficient_context", answer="The provided excerpts do not contain enough information to answer this question.",
                              output_tokens=0, finish_reason="no_evidence")
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        result["seconds"] = round(time.perf_counter() - started, 4)
        result["record_sha256"] = digest(result)
        write_json(checkpoint_dir / f"{digest(row['query_id'])}.json", result)
        results[row["query_id"]] = result
        print(f"[{len(results)}/{len(rows)}] {row['query_id']}: {result['status']}", flush=True)
        if result["status"] == "error" and config.get("fail_fast", True):
            write_jsonl(out / "predictions.jsonl", [results[q["query_id"]] for q in rows if q["query_id"] in results])
            raise RuntimeError(f"Query {row['query_id']} failed: {result['error']}. Checkpoint saved; rerun to resume.")
    ordered = [results[row["query_id"]] for row in rows]
    write_jsonl(out / "predictions.jsonl", ordered)
    report = {"query_count": len(rows), "generated_this_call": len(pending), "resumed": resumed,
              "errors": [r["query_id"] for r in ordered if r["status"] == "error"],
              "length_limited": [r["query_id"] for r in ordered if r.get("finish_reason") == "length"],
              "total_generation_seconds": sum(r["seconds"] for r in ordered), "environment": environment(),
              "note": "Evidence records identify context supplied to the model, not verified factual support for every answer claim."}
    write_json(out / "report.json", report)
    if report["errors"]:
        raise RuntimeError("Some queries failed. Fix the runtime problem and resume before exporting.")
    files = ["predictions.jsonl", "report.json"] + [f"checkpoints/{digest(r['query_id'])}.json" for r in rows]
    return finish(out, manifest, files, doc_id=parent["doc_id"], source_pdf=parent["source_pdf"],
                  retrieval_backend=parent.get("backend"), generator_backend=config.get("backend", "huggingface"),
                  query_count=len(rows), full_retrieval_query_count=parent["query_count"])
