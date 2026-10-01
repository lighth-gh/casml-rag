"""Read ONLY the retrieval artifact, then generate/resume a separate run."""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from .artifacts import (SCHEMA, begin, digest, environment, finish, load_artifact, read_json,
                        read_jsonl, signature, write_json, write_jsonl)
from .contracts import validate_retrieval
from .context import pack_context
from .diagnosis import write_diagnosis
from .llm import make_generator
from .validation import (grounding_retry_instruction, remove_unsupported_sentences,
                         validate_answer_grounding)


def load_prompts(config, config_path):
    root = Path(config_path).resolve().parent
    system = (root / config["system_prompt_file"]).read_text(encoding="utf-8").strip()
    user = (root / config["user_prompt_file"]).read_text(encoding="utf-8").strip()
    if "{question}" not in user or "{context}" not in user:
        raise ValueError("User template must contain {question} and {context}")
    user.format(question="test", context="test")
    return system, user


def generate_answer(backend, payload, config, on_attempt=None, attempt_reason="initial"):
    """Retry cutoffs with concise instructions and decoding controls, preserving evidence."""
    tokens = int(config["max_new_tokens"])
    ceiling = int(config.get("retry_max_new_tokens", tokens))
    if ceiling < tokens:
        raise ValueError("retry_max_new_tokens must be >= max_new_tokens")
    window = min(int(config["context_window"]), backend.context_window)
    ceiling = min(ceiling, window - payload["input_tokens"])
    if tokens > ceiling:
        raise ValueError("Input + max_new_tokens exceeds context_window")
    attempts = []
    active_payload = payload
    active_config = dict(config)
    while True:
        answer = backend.generate(active_payload, {**active_config, "max_new_tokens": tokens})
        attempts.append({"reason": attempt_reason if not attempts else "length_retry",
                         "max_new_tokens": tokens, "output_tokens": answer["output_tokens"],
                         "finish_reason": answer["finish_reason"],
                         "input_tokens": active_payload["input_tokens"],
                         "repetition_penalty": active_config.get("repetition_penalty", 1.0),
                         "no_repeat_ngram_size": active_config.get("no_repeat_ngram_size", 0),
                         "answer_head": answer["answer"][:300], "answer_tail": answer["answer"][-300:]})
        result = {**active_payload, **answer, "generation_attempts": attempts, "max_new_tokens_used": tokens}
        if on_attempt is not None:
            on_attempt(result)
        if answer["finish_reason"] != "length":
            return result
        policy_changed = False
        if len(attempts) == 1:
            # Merely increasing length replays the same greedy sequence. Change
            # the retry policy once, and record the actual prompt used below.
            instruction = str(config.get("retry_instruction", "")).strip()
            retry_payload = dict(payload)
            if instruction:
                messages = [dict(m) for m in payload["messages"]]
                messages[-1]["content"] += "\n\n" + instruction
                retry_payload.update(messages=messages, input_tokens=backend.count_messages(messages))
                policy_changed = True
            retry_ceiling = min(int(config.get("retry_max_new_tokens", tokens)),
                                window - retry_payload["input_tokens"])
            if retry_ceiling < 1:
                result["retry_skipped"] = "Concise retry prompt leaves no output token budget"
                return result
            for key in ("repetition_penalty", "no_repeat_ngram_size"):
                if "retry_" + key in config:
                    policy_changed |= active_config.get(key) != config["retry_" + key]
                    active_config[key] = config["retry_" + key]
            active_payload = retry_payload
            ceiling = retry_ceiling
        if tokens >= ceiling and not policy_changed:
            return result
        next_tokens = min(tokens * 2, ceiling)
        print(f"Answer reached {tokens} tokens; retrying with {next_tokens} tokens"
              f" (repetition_penalty={active_config.get('repetition_penalty', 1.0)}, "
              f"no_repeat_ngram_size={active_config.get('no_repeat_ngram_size', 0)}, "
              f"concise_prompt={bool(config.get('retry_instruction'))}).", flush=True)
        tokens = next_tokens


def generate_grounded_answer(backend, payload, config, on_attempt=None):
    """Report grounding by default; rewriting requires explicit opt-in."""
    result = generate_answer(backend, payload, config, on_attempt=on_attempt)
    if not config.get("grounding_validator_enabled", False):
        result["grounding_mode"] = "disabled"
        if config.get("grounding_report_enabled", True):
            validation = validate_answer_grounding(
                result["answer"], payload["evidence"],
                max_words=int(config.get("grounding_repair_max_words", 140)))
            result.update(
                grounding_mode="report_only", answer_validation=validation,
                validation_attempts=[{"retry_index": 0, "validation": validation,
                                      "answer_head": result["answer"][:300],
                                      "answer_tail": result["answer"][-300:]}])
            if on_attempt is not None:
                on_attempt(result)
        return result
    result["grounding_mode"] = "enforce"
    if result.get("finish_reason") == "length":
        return result
    max_retries = int(config.get("grounding_validator_max_retries", 1))
    if max_retries < 0:
        raise ValueError("grounding_validator_max_retries must be non-negative")
    all_attempts = list(result.get("generation_attempts", []))
    validation_attempts = []
    drafts = []
    for retry_index in range(max_retries + 1):
        result["grounding_mode"] = "enforce"
        validation = validate_answer_grounding(
            result["answer"], payload["evidence"],
            max_words=int(config.get("grounding_repair_max_words", 140)))
        drafts.append({"retry_index": retry_index, "answer": result["answer"],
                       "validation": validation, "output_tokens": result["output_tokens"]})
        validation_attempts.append({"retry_index": retry_index, "validation": validation,
                                    "answer_head": result["answer"][:300],
                                    "answer_tail": result["answer"][-300:]})
        result["answer_validation"] = validation
        result["validation_attempts"] = validation_attempts
        result["generation_attempts"] = all_attempts
        if on_attempt is not None:
            on_attempt(result)
        if validation["valid"]:
            return result
        if retry_index >= max_retries:
            if config.get("grounding_deterministic_repair", True):
                repairs = []
                for draft in drafts:
                    candidate = remove_unsupported_sentences(
                        draft["answer"], draft["validation"], payload["evidence"],
                        max_words=int(config.get("grounding_repair_max_words", 140)))
                    candidate["source_retry_index"] = draft["retry_index"]
                    candidate["source_answer"] = draft["answer"]
                    candidate["source_output_tokens"] = draft["output_tokens"]
                    if candidate["answer"] and candidate["validation"]["valid"]:
                        repairs.append(candidate)
                repair = (min(repairs, key=lambda candidate: (
                    len(candidate["removed_sentences"]), len(candidate["removed_fragments"]),
                    candidate["word_count"])) if repairs else
                    {"answer": "", "validation": validation, "removed_sentences": [],
                     "removed_fragments": [], "word_count": 0, "source_retry_index": None,
                     "source_answer": result["answer"], "source_output_tokens": result["output_tokens"]})
                result["grounding_repair"] = {key: value for key, value in repair.items()
                                              if key not in ("answer", "source_answer")}
                if repair["answer"] and repair["validation"]["valid"]:
                    validation_attempts.append({"retry_index": "deterministic_repair",
                                                "validation": repair["validation"],
                                                "answer_head": repair["answer"][:300],
                                                "answer_tail": repair["answer"][-300:]})
                    result["model_answer_before_grounding_repair"] = repair["source_answer"]
                    result["model_output_tokens_before_grounding_repair"] = repair["source_output_tokens"]
                    result["answer"] = repair["answer"]
                    result["answer_validation"] = repair["validation"]
                    result["validation_attempts"] = validation_attempts
                    result["output_tokens"] = backend.count_text(repair["answer"])
                    result["finish_reason"] = "grounding_repair"
                    if on_attempt is not None:
                        on_attempt(result)
            return result
        instruction = grounding_retry_instruction(
            validation, config.get("grounding_validator_retry_instruction", ""))
        retry_payload = dict(payload)
        messages = [dict(message) for message in payload["messages"]]
        messages.append({"role": "assistant", "content": result["answer"]})
        messages.append({"role": "user", "content": instruction})
        retry_payload.update(messages=messages, input_tokens=backend.count_messages(messages))
        retry_config = dict(config)
        grounding_tokens = int(config.get("grounding_retry_max_new_tokens", config["max_new_tokens"]))
        retry_config["max_new_tokens"] = grounding_tokens
        retry_config["retry_max_new_tokens"] = grounding_tokens
        retry_config["retry_instruction"] = ""
        retry_config["repetition_penalty"] = config.get(
            "retry_repetition_penalty", config.get("repetition_penalty", 1.0))
        retry_config["no_repeat_ngram_size"] = config.get(
            "retry_no_repeat_ngram_size", config.get("no_repeat_ngram_size", 0))
        print("Unsupported evidence tokens detected; regenerating answer "
              f"({retry_index + 1}/{max_retries}).", flush=True)
        retry_result = generate_answer(backend, retry_payload, retry_config,
                                       attempt_reason="grounding_retry")
        all_attempts.extend(retry_result.get("generation_attempts", []))
        result = retry_result


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
    if int(config.get("retry_max_new_tokens", config["max_new_tokens"])) < int(config["max_new_tokens"]):
        raise ValueError("retry_max_new_tokens must be >= max_new_tokens")
    if int(config.get("grounding_validator_max_retries", 1)) < 0:
        raise ValueError("grounding_validator_max_retries must be non-negative")
    effective = {**config, "system_prompt_content": system, "user_prompt_content": user}
    sig = signature("generation", effective, {"retrieval_id": parent["artifact_id"], "selected_queries": digest(rows)},
                    ["generation.py", "context.py", "llm.py", "diagnosis.py", "validation.py"])
    manifest, cached = begin(out, sig, resumable=True)
    if cached:
        return manifest
    results, resumed, backend = {}, 0, None
    runtime = environment()

    def save_diagnosis(state="running", failure=None):
        write_diagnosis(out, manifest, rows, results, state=state, runtime=runtime,
                        resumed=resumed, model_context_window=getattr(backend, "context_window", None),
                        full_query_count=parent["query_count"], failure=failure)

    save_diagnosis()
    try:
        checkpoint_dir = out / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        pending = []
        for row in rows:
            key = digest(row["query_id"])
            path = checkpoint_dir / f"{key}.json"
            previous = read_json(path) if path.exists() else None
            if previous:
                checksum = previous.pop("record_sha256", None)
                if checksum != digest(previous) or previous.get("run_id") != manifest["artifact_id"] or previous.get("query_signature") != digest(row):
                    raise ValueError(f"Invalid checkpoint for query {row['query_id']}")
                previous["record_sha256"] = checksum
            if (previous and previous["status"] in ("ok", "insufficient_context")
                    and previous.get("finish_reason") != "length"):
                results[row["query_id"]] = previous
            else:
                pending.append(row)
        resumed = len(results)
        backend = make_generator(config) if pending else None
        save_diagnosis()
        consecutive_length_limited = 0
        abort_after_length = int(config.get("abort_after_consecutive_length_limited", 0))
        if abort_after_length < 0:
            raise ValueError("abort_after_consecutive_length_limited must be non-negative")
        for row in pending:
            started = time.perf_counter()
            result = {"schema": SCHEMA, "run_id": manifest["artifact_id"], "query_signature": digest(row),
                      "query_id": row["query_id"], "question": row["question"],
                      "generator_backend": config.get("backend", "huggingface"), "status": "running"}
            results[row["query_id"]] = result
            save_diagnosis()

            def record_attempt(attempt):
                result.update(attempt)
                save_diagnosis()

            try:
                payload = pack_context(row, backend, config, system, user)
                result.update(payload)
                if payload["evidence"]:
                    result.update(generate_grounded_answer(backend, payload, config, on_attempt=record_attempt))
                    if not result["answer"].strip():
                        raise ValueError("Model produced an empty answer")
                    result["status"] = ("ok" if result.get("grounding_mode") == "report_only"
                                        or result.get("answer_validation", {"valid": True})["valid"]
                                        else "unsupported_claims")
                else:
                    result.update(status="insufficient_context", answer="The provided excerpts do not contain enough information to answer this question.",
                                  output_tokens=0, finish_reason="no_evidence")
            except Exception as exc:
                result.update(status="error", error=f"{type(exc).__name__}: {exc}",
                              traceback=traceback.format_exc())
            result["seconds"] = round(time.perf_counter() - started, 4)
            result["record_sha256"] = digest(result)
            write_json(checkpoint_dir / f"{digest(row['query_id'])}.json", result)
            results[row["query_id"]] = result
            save_diagnosis()
            outcome = "length_limited" if result["status"] == "ok" and result.get("finish_reason") == "length" else result["status"]
            print(f"[{len(results)}/{len(rows)}] {row['query_id']}: {outcome}", flush=True)
            if result["status"] == "error" and config.get("fail_fast", True):
                write_jsonl(out / "predictions.jsonl", [results[q["query_id"]] for q in rows if q["query_id"] in results])
                raise RuntimeError(f"Query {row['query_id']} failed: {result['error']}. Checkpoint saved; rerun to resume.")
            if result.get("finish_reason") == "length":
                consecutive_length_limited += 1
            else:
                consecutive_length_limited = 0
            if abort_after_length and consecutive_length_limited >= abort_after_length:
                write_jsonl(out / "predictions.jsonl", [results[q["query_id"]] for q in rows if q["query_id"] in results])
                raise RuntimeError(
                    f"Aborting after {consecutive_length_limited} consecutive length-limited answer(s); "
                    "inspect diagnosis.json before spending GPU time on the remaining queries. "
                    "The length-limited checkpoint will be regenerated on the next run."
                )
        ordered = [results[row["query_id"]] for row in rows]
        write_jsonl(out / "predictions.jsonl", ordered)
        report = {"query_count": len(rows), "generated_this_call": len(pending), "resumed": resumed,
                  "errors": [r["query_id"] for r in ordered if r["status"] == "error"],
                  "unsupported_claims": [r["query_id"] for r in ordered if r["status"] == "unsupported_claims"],
                  "grounding_warnings": [r["query_id"] for r in ordered
                                        if r.get("grounding_mode") == "report_only"
                                        and not r["answer_validation"]["valid"]],
                  "grounding_warning_details": [
                      {"query_id": r["query_id"], "question": r["question"],
                       "answer_validation": r["answer_validation"]}
                      for r in ordered if r.get("grounding_mode") == "report_only"
                      and not r["answer_validation"]["valid"]],
                  "length_limited": [r["query_id"] for r in ordered if r.get("finish_reason") == "length"],
                  "retried_for_length": [r["query_id"] for r in ordered
                                          if any(a.get("reason") == "length_retry"
                                                 for a in r.get("generation_attempts", []))],
                  "length_limited_details": [
                      {"query_id": r["query_id"], "question": r["question"],
                       "answer_head": r["answer"][:500], "answer_tail": r["answer"][-500:],
                       "generation_attempts": r.get("generation_attempts", []),
                       "retry_skipped": r.get("retry_skipped")}
                      for r in ordered if r.get("finish_reason") == "length"],
                  "unsupported_claim_details": [
                      {"query_id": r["query_id"], "question": r["question"],
                       "answer": r["answer"], "answer_validation": r.get("answer_validation"),
                       "validation_attempts": r.get("validation_attempts", [])}
                      for r in ordered if r["status"] == "unsupported_claims"],
                  "total_generation_seconds": sum(r["seconds"] for r in ordered), "environment": runtime,
                  "note": "Evidence records identify context supplied to the model, not verified factual support for every answer claim."}
        write_json(out / "report.json", report)
        if report["errors"]:
            raise RuntimeError("Some queries failed. Fix the runtime problem and resume before exporting.")
        if report["unsupported_claims"]:
            raise RuntimeError("Some answers still contain numbers, years, or proper names absent from their evidence. "
                               "Inspect unsupported_claim_details, then adjust the prompt, retry policy, or evidence "
                               "and use a new run directory.")
        save_diagnosis("completed")
        files = ["predictions.jsonl", "report.json", "diagnosis.json"] + [f"checkpoints/{digest(r['query_id'])}.json" for r in rows]
        return finish(out, manifest, files, doc_id=parent["doc_id"], source_pdf=parent["source_pdf"],
                      retrieval_backend=parent.get("backend"), generator_backend=config.get("backend", "huggingface"),
                      query_count=len(rows), full_retrieval_query_count=parent["query_count"])
    except (Exception, KeyboardInterrupt) as exc:
        save_diagnosis("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
