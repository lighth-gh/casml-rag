"""Frozen-input generation, human claim review, and explicitly non-official proxies."""
from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

from .artifacts import (begin, digest, file_hash, finish, load_artifact, read_json,
                        read_jsonl, signature, write_json, write_jsonl)
from .generation import load_prompts
from .llm import make_generator
from .reviewed import load_split, normalized, references

SPEC = "casml-local-proxy/v2-human-claims"
WEIGHTS = {"CP": .2, "AF": .2, "AC": .4, "RA": .2}


def training_checkpoint(directory, checkpoint):
    manifest = load_artifact(directory, "finetune")
    report = read_json(Path(directory) / "report.json")
    if report.get("workflow") != "reviewed-v2" or checkpoint not in report["checkpoints"]:
        raise ValueError("Choose an explicit reviewed-v2 checkpoint from report.json")
    return manifest, report, Path(directory).resolve() / "checkpoints" / checkpoint


def evaluate(dataset, split, config, config_path, out, training=None, checkpoint=None,
             _backend=None, _identity=None, context_mode="retrieved"):
    if split not in ("dev", "holdout"):
        raise ValueError("Evaluation only accepts dev or holdout")
    data, rows = load_split(dataset, split)
    if context_mode not in ("retrieved", "oracle"):
        raise ValueError("context_mode must be retrieved or oracle")
    if context_mode == "oracle":
        from .context import render_context
        rows = [{**r, "evidence": r["oracle_evidence"], "context": render_context(r["oracle_evidence"]),
                 "evidence_coverage": 1.0} for r in rows]
    system, user = load_prompts(config, config_path)
    effective = {**config, "system_prompt_content": system, "user_prompt_content": user}
    identity = {"kind": "base", "model_name": config["model_name"], "revision": config.get("revision", "main")}
    if config.get("finetune_artifact") or config.get("adapter_path"):
        raise ValueError("Use a base generation config and explicit --training/--checkpoint")
    if training:
        parent, report, adapter = training_checkpoint(training, checkpoint)
        for key in ("model_name", "revision"):
            if config.get(key) != report["base_config"].get(key):
                raise ValueError("Evaluation base model/revision differs from training")
        if data["artifact_id"] != report["dataset_id"]:
            raise ValueError("Candidate must be evaluated on its registered dataset")
        identity = {"kind": "adapter", "training_id": parent["artifact_id"], "checkpoint": checkpoint}
        effective["adapter_path"] = str(adapter)
    elif checkpoint:
        raise ValueError("--checkpoint requires --training")
    if _identity is not None:
        identity = _identity
    # Inputs and decoding, excluding only model-loading details, must match in A/B.
    decoding = {k: v for k, v in config.items() if k not in (
        "model_name", "revision", "device", "local_files_only", "system_prompt_file", "user_prompt_file")}
    payloads = [{"query_id": r["query_id"], "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user.format(question=r["question"], context=r["context"])}],
        "references": references(r["evidence"])} for r in rows]
    frozen = digest({"inputs": payloads, "decoding": decoding})
    sig = signature("eval_predictions", effective,
                    {"dataset_id": data["artifact_id"], "split": split, "model": identity,
                     "input_fingerprint": frozen, "context_mode": context_mode,
                     "base_model": {"model_name": config["model_name"], "revision": config.get("revision", "main")}},
                    ["evaluation.py", "llm.py", "reviewed.py", "generation.py"])
    manifest, cached = begin(out, sig, resumable=True)
    if cached:
        return manifest
    out = Path(out)
    (out / "checkpoints").mkdir(exist_ok=True)
    predictions, template, backend = [], [], _backend
    for row, fixed in zip(rows, payloads):
        path = out / "checkpoints" / (digest(row["query_id"]) + ".json")
        if path.exists():
            record = read_json(path)
            checksum = record.pop("record_sha256", None)
            if checksum != digest(record) or record.get("eval_id") != manifest["artifact_id"]:
                raise ValueError("Invalid evaluation checkpoint")
        else:
            if backend is None:
                backend = make_generator(effective)
            count = backend.count_messages(fixed["messages"])
            if count + int(config["max_new_tokens"]) > min(int(config["context_window"]), backend.context_window):
                raise ValueError("Reviewed prompt exceeds model budget; re-review a shorter context")
            answer = backend.generate({**fixed, "input_tokens": count, "evidence": row["evidence"]}, config)
            record = {**fixed, **answer, "eval_id": manifest["artifact_id"], "context": row["context"],
                      "question": row["question"], "split_group": row["bootstrap_group"],
                      "evidence_coverage": row["evidence_coverage"],
                      "required_facts": row["required_facts"], "gold_reference_sets": row["gold_reference_sets"]}
            write_json(path, {**record, "record_sha256": digest(record)})
        predictions.append(record)
        template.append({"query_id": row["query_id"], "prediction_sha256": digest(record),
                         "review_status": "pending", "reviewer": "", "context_relevant_spans": [],
                         "claims": [], "severe_error": None})
        print(f"Evaluation {split}: {len(predictions)}/{len(rows)}", flush=True)
    write_jsonl(out / "predictions.jsonl", predictions)
    write_jsonl(out / "review_template.jsonl", template)
    write_json(out / "report.json", {"model": identity, "split": split, "query_count": len(rows),
                "input_fingerprint": frozen, "context_mode": context_mode, "official_metric_verified": False,
                "status": "awaiting_human_claim_review"})
    files = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"]
    return finish(out, manifest, files)


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def set_f1(actual, expected):
    actual, expected = set(actual), set(expected)
    if not expected:
        raise ValueError("Gold reference sets must not be empty")
    common = len(actual & expected)
    return f1(common / len(actual) if actual else 0, common / len(expected))


def score_row(prediction, review):
    if review.get("review_status") != "approved" or not str(review.get("reviewer", "")).strip():
        raise ValueError("All evaluation rows require independent reviewer approval")
    if review.get("prediction_sha256") != digest(prediction):
        raise ValueError("Review refers to different/stale prediction")
    if type(review.get("severe_error")) is not bool:
        raise ValueError("severe_error must be an explicit boolean")
    claims = review.get("claims")
    if not isinstance(claims, list) or not claims or not prediction["answer"].strip():
        raise ValueError("Cannot score an empty answer or an answer with no reviewed claims")
    if normalized(" ".join(c.get("text", "") for c in claims)) != normalized(prediction["answer"]):
        raise ValueError("Review claims must cover the full generated answer, in order")
    fact_ids = {f["fact_id"] for f in prediction["required_facts"]}
    covered, correct, supported = set(), 0, 0
    for claim in claims:
        if not isinstance(claim.get("text"), str) or not claim["text"].strip():
            raise ValueError("Every reviewed claim must have non-empty text")
        if any(type(claim.get(k)) is not bool for k in ("correct", "supported")):
            raise ValueError("Claim correctness/support must be explicit booleans")
        matched = claim.get("fact_ids")
        if not isinstance(matched, list) or not set(matched) <= fact_ids:
            raise ValueError("Claim refers to unknown required fact IDs")
        if claim["correct"]:
            covered.update(matched)
        elif matched:
            raise ValueError("Incorrect claim cannot satisfy required facts")
        correct += claim["correct"]
        supported += claim["supported"]
    context = prediction["context"]
    spans = review.get("context_relevant_spans")
    if not isinstance(spans, list):
        raise ValueError("context_relevant_spans must be explicit character intervals")
    for span in spans:
        if (not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span)
                or not 0 <= span[0] < span[1] <= len(context)):
            raise ValueError("Invalid relevance span")
    tokens = list(re.finditer(r"\S+", context))
    if not tokens or not fact_ids:
        raise ValueError("Missing context or required facts")
    # A token counts once only when fully contained in a relevant span.
    relevant = sum(any(a <= t.start() and t.end() <= b for a, b in spans) for t in tokens)
    refs = prediction["references"]
    ra = max((set_f1(refs["pages"], gold["pages"]) + set_f1(refs["sections"], gold["sections"])) / 2
             for gold in prediction["gold_reference_sets"])
    metrics = {"CP": relevant / len(tokens), "AF": supported / len(claims),
               "AC": f1(correct / len(claims), len(covered) / len(fact_ids)), "RA": ra}
    return {"query_id": prediction["query_id"], "split_group": prediction["split_group"], **metrics,
            "S_proxy": sum(WEIGHTS[k] * metrics[k] for k in WEIGHTS),
            "severe_error": review["severe_error"],
            "evidence_coverage": prediction["evidence_coverage"],
            "schema_ok": prediction.get("finish_reason") == "eos" and bool(refs["pages"] and refs["sections"])}


def score_evaluation(predictions_dir, reviews_path, config, out):
    parent = load_artifact(predictions_dir, "eval_predictions")
    predictions = read_jsonl(Path(predictions_dir) / "predictions.jsonl")
    if not predictions:
        raise ValueError("Cannot score an empty prediction set")
    reviews = read_jsonl(reviews_path)
    by_id = {r["query_id"]: r for r in reviews}
    if len(by_id) != len(reviews) or set(by_id) != {p["query_id"] for p in predictions}:
        raise ValueError("Review IDs must match all predictions exactly, without duplicates")
    scores = [score_row(p, by_id[p["query_id"]]) for p in predictions]
    sig = signature("eval_scores", {"spec": SPEC, "weights": WEIGHTS},
                    {"predictions_id": parent["artifact_id"], "reviews_sha256": file_hash(reviews_path)},
                    ["evaluation.py", "reviewed.py"])
    manifest, cached = begin(out, sig)
    if cached:
        return manifest
    report = {"spec": SPEC, "weights": WEIGHTS, **parent["signature"]["inputs"],
              "query_count": len(scores), "official_metric_verified": False,
              "means": {k: sum(s[k] for s in scores) / len(scores) for k in (*WEIGHTS, "S_proxy", "evidence_coverage")},
              "severe_errors": sum(s["severe_error"] for s in scores),
              "schema_ok": all(s["schema_ok"] for s in scores)}
    write_jsonl(Path(out) / "scores.jsonl", scores)
    write_json(Path(out) / "report.json", report)
    return finish(out, manifest, ["scores.jsonl", "report.json"])


def paired_comparison(baseline_dir, candidate_dir, config, expected_split):
    bm = load_artifact(baseline_dir, "eval_scores")
    cm = load_artifact(candidate_dir, "eval_scores")
    base, candidate = [read_json(Path(d) / "report.json") for d in (baseline_dir, candidate_dir)]
    for key in ("spec", "weights", "dataset_id", "split", "input_fingerprint", "context_mode", "base_model"):
        if base[key] != candidate[key]:
            raise ValueError(f"A/B comparison changed {key}")
    if base["split"] != expected_split or base["model"]["kind"] != "base" or candidate["model"]["kind"] != "adapter":
        raise ValueError("Compare a base model against a candidate on the expected split")
    left = {s["query_id"]: s for s in read_jsonl(Path(baseline_dir) / "scores.jsonl")}
    right = {s["query_id"]: s for s in read_jsonl(Path(candidate_dir) / "scores.jsonl")}
    if not left or left.keys() != right.keys():
        raise ValueError("A/B query IDs differ")
    groups = defaultdict(list)
    for key in left:
        if left[key]["split_group"] != right[key]["split_group"]:
            raise ValueError("A/B group changed")
        if any(abs(left[key][k] - right[key][k]) > 1e-12 for k in ("CP", "RA")):
            raise ValueError("Frozen-context comparison changed CP/RA annotations")
        groups[left[key]["split_group"]].append(right[key]["S_proxy"] - left[key]["S_proxy"])
    rng = random.Random(int(config.get("seed", 42)))
    iterations = int(config.get("bootstrap_samples", 2000))
    if iterations < 100:
        raise ValueError("At least 100 bootstrap samples required")
    values = list(groups.values())
    samples = []
    for _ in range(iterations):
        draws = [rng.choice(values) for _ in values]
        samples.append(sum(map(sum, draws)) / sum(map(len, draws)))
    samples.sort()
    delta = {k: candidate["means"][k] - base["means"][k] for k in (*WEIGHTS, "S_proxy")}
    return {"baseline_id": bm["artifact_id"], "candidate_id": cm["artifact_id"],
            "model": candidate["model"], "base_model": base["model"], "dataset_id": base["dataset_id"],
            "query_count": len(left), "group_count": len(groups), "delta": delta,
            "ci95": [samples[int(.025 * (iterations - 1))], samples[int(.975 * (iterations - 1))]],
            "no_new_severe_errors": all(not right[k]["severe_error"] or left[k]["severe_error"] for k in left),
            "schema_ok": candidate["schema_ok"], "spec": SPEC}


def select_model(baseline, candidate, config, out, holdout_baseline=None, holdout_candidate=None):
    dev = paired_comparison(baseline, candidate, config, "dev")
    if read_json(Path(candidate) / "report.json")["context_mode"] != "retrieved":
        raise ValueError("Promotion requires retrieved-context scores; oracle is diagnostic only")
    holdout = None
    if bool(holdout_baseline) != bool(holdout_candidate):
        raise ValueError("Provide both holdout scores")
    if holdout_baseline:
        holdout = paired_comparison(holdout_baseline, holdout_candidate, config, "holdout")
        if read_json(Path(holdout_candidate) / "report.json")["context_mode"] != "retrieved":
            raise ValueError("Holdout confirmation requires retrieved-context scores")
        if any(holdout[k] != dev[k] for k in ("model", "base_model", "dataset_id")):
            raise ValueError("Holdout must confirm exactly the same candidate, base and dataset")
    enough = (dev["query_count"] >= int(config.get("min_dev_examples", 80)) and
              dev["group_count"] >= int(config.get("min_groups", 20)))
    gates = {"dev_schema": dev["schema_ok"], "no_new_severe_errors": dev["no_new_severe_errors"],
             "AF_AC_non_decreasing": all(dev["delta"][k] >= 0 for k in ("AF", "AC")),
             "minimum_gain": dev["delta"]["S_proxy"] >= float(config.get("min_gain", .02)),
             "positive_ci": dev["ci95"][0] > 0}
    if holdout:
        enough &= holdout["query_count"] >= int(config.get("min_holdout_examples", 80))
        gates["holdout_confirmation"] = (holdout["schema_ok"] and holdout["no_new_severe_errors"] and
            holdout["delta"]["S_proxy"] > 0 and all(holdout["delta"][k] >= 0 for k in ("AF", "AC")))
    status = "insufficient_evidence" if not enough or holdout is None else ("selected" if all(gates.values()) else "rejected")
    report = {"status": status, "model": dev["model"], "dataset_id": dev["dataset_id"],
              "dev": dev, "holdout": holdout, "gates": gates, "official_metric_verified": False,
              "note": "Human-reviewed local proxy; this does not establish private-score improvement."}
    sig = signature("model_selection", config, {"dev": dev, "holdout": holdout}, ["evaluation.py"])
    manifest, cached = begin(out, sig)
    if not cached:
        write_json(Path(out) / "report.json", report)
        manifest = finish(out, manifest, ["report.json"], selection_status=status)
    return manifest
