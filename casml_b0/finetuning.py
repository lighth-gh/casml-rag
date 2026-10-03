"""Supervised LoRA training on explicit question/context/answer JSONL files."""
from __future__ import annotations

import random
from pathlib import Path

from .artifacts import (begin, environment, file_hash, finish, load_config, read_jsonl,
                        signature, write_json)
from .generation import load_prompts


def read_examples(path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Training/validation JSONL must not be empty")
    seen_ids, seen_questions = set(), set()
    result = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index}: expected an object")
        value = row.get("query_id")
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError(f"Row {index}: query_id must be a non-empty string or integer")
        item = {"query_id": str(value)}
        for key in ("question", "context", "answer"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Row {index}: {key} must be a non-empty string")
            item[key] = row[key].strip()
        question = " ".join(item["question"].casefold().split())
        if item["query_id"] in seen_ids or question in seen_questions:
            raise ValueError(f"Duplicate query_id/question in {path}: {item['query_id']}")
        seen_ids.add(item["query_id"])
        seen_questions.add(question)
        item["label_source"] = str(row.get("label_source", "user_supplied_unverified"))
        result.append(item)
    return result


def split_examples(rows, validation_rows=None, validation_fraction=0.1, seed=42):
    if validation_rows is not None:
        ids = {row["query_id"] for row in rows}
        questions = {" ".join(row["question"].casefold().split()) for row in rows}
        for row in validation_rows:
            if (row["query_id"] in ids or
                    " ".join(row["question"].casefold().split()) in questions):
                raise ValueError("Train/validation overlap in query_id or question")
        return rows, validation_rows
    if not 0 < validation_fraction < 1 or len(rows) < 2:
        raise ValueError("Need at least two examples and 0 < validation_fraction < 1")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    count = min(len(rows) - 1, max(1, round(len(rows) * validation_fraction)))
    return shuffled[count:], shuffled[:count]


def tokenize_example(row, tokenizer, system, user_template, max_length):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_template.format(
                    question=row["question"], context=row["context"])}]
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row["answer"]}],
        tokenize=True, add_generation_prompt=False)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template does not preserve the assistant prefix")
    if len(full_ids) > max_length:
        raise ValueError(f"Example {row['query_id']} has {len(full_ids)} tokens > {max_length}; "
                         "shorten context/answer or increase max_length. No silent truncation.")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(f"No assistant tokens for {row['query_id']}")
    # Preserve the chat-template end-of-turn token in the supervised answer.
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class AnswerCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        length = max(len(row["input_ids"]) for row in features)
        batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for row in features:
            padding = length - len(row["input_ids"])
            for key, value in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
                batch[key].append(row[key] + [value] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def training_schedule(examples, batch_size, accumulation, epochs):
    import math
    if any(type(v) is not int or v < 1 for v in (examples, batch_size, accumulation, epochs)):
        raise ValueError("Example count, batch, accumulation and epochs must be positive integers")
    batches = math.ceil(examples / batch_size)
    return {"examples_per_epoch": examples, "batches_per_epoch": batches,
            "steps_per_epoch": math.ceil(batches / accumulation),
            "optimizer_steps": math.ceil(batches / accumulation) * epochs,
            "effective_batch": batch_size * accumulation, "epochs": epochs, "world_size": 1}


def finetune(train_path, config, config_path, out, validation_path=None):
    """Train reviewed data only. Keep every candidate; never select by token loss."""
    import math
    import os
    import time
    from .reviewed import load_split
    from .artifacts import write_jsonl

    dataset = Path(train_path)
    if not dataset.is_dir() or validation_path:
        raise ValueError("v2 requires --dataset pointing to a prepare-sft artifact; raw/synthetic JSONL is not approved data")
    data, train_rows = load_split(dataset, "train")
    _, eval_rows = load_split(dataset, "dev")
    # Do not read holdout content into the training process.
    if len(train_rows) < int(config.get("min_train_examples", 200)) or len(eval_rows) < int(config.get("min_dev_examples", 80)):
        raise ValueError("Insufficient reviewed train/dev data; finish human annotation before training")
    if any(r.get("review_status") != "approved" for r in train_rows + eval_rows):
        raise ValueError("Unapproved training labels")
    if int(config.get("max_steps", -1)) != -1:
        raise ValueError("v2 trains complete integer epochs; max_steps must be -1")
    schedule = training_schedule(len(train_rows), config.get("batch_size", 1),
                                 config.get("gradient_accumulation_steps", 8), config.get("num_train_epochs", 1))
    config_root = Path(config_path).resolve().parent
    generation_path = (config_root / config["generation_config"]).resolve()
    generation = load_config(generation_path)
    if generation.get("backend", "huggingface") != "huggingface" or generation.get("finetune_artifact"):
        raise ValueError("Fine-tune requires the base Hugging Face generator")
    system, user = load_prompts(generation, generation_path)
    max_length = int(config.get("max_length", 2048))
    if not 2 <= max_length <= int(generation["context_window"]):
        raise ValueError("Invalid max_length")
    seed = int(config.get("seed", 42))
    sig = signature("finetune", {**config, "generation": generation, "system": system, "user": user},
                    {"dataset_id": data["artifact_id"]}, ["finetuning.py", "reviewed.py", "evaluation.py", "llm.py"])
    out = Path(out).resolve()
    manifest, cached = begin(out, sig)
    if cached:
        return manifest
    # This CLI runs in its own process; restrict the default pilot to one GPU.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.get("cuda_device", "0")))
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed
    from .evaluation import evaluate
    from .llm import HFGenerator

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v2 pilot requires exactly one visible CUDA GPU; set CUDA_VISIBLE_DEVICES=0")
    set_seed(seed)
    precision = config.get("dtype", "auto")
    if precision == "auto":
        precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if precision not in ("float32", "float16", "bfloat16"):
        raise ValueError("Invalid training dtype")
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bfloat16")
    options = {"revision": generation.get("revision", "main"), "trust_remote_code": False,
               "local_files_only": generation.get("local_files_only", False)}
    tokenizer = AutoTokenizer.from_pretrained(generation["model_name"], **options)
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("Need an instruct model with chat template and EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    train_data = [tokenize_example(r, tokenizer, system, user, max_length) for r in train_rows]
    # Validate dev length without exposing holdout or optimizing on dev labels.
    for row in eval_rows:
        tokenize_example(row, tokenizer, system, user, max_length)
    model = AutoModelForCausalLM.from_pretrained(generation["model_name"], torch_dtype=getattr(torch, precision),
        attn_implementation=generation.get("attn_implementation", "sdpa"), **options).to("cuda")
    if max_length > model.config.max_position_embeddings:
        raise ValueError("Sequence length exceeds the base model context window")
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=int(config.get("lora_r", 8)),
        lora_alpha=int(config.get("lora_alpha", 16)), lora_dropout=float(config.get("lora_dropout", .05)),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]), bias="none"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    model.config.use_cache = False
    backend = HFGenerator.from_loaded(model, tokenizer, generation)
    report = {"workflow": "reviewed-v2", "dataset_id": data["artifact_id"], "schedule": schedule,
              "train_examples": len(train_rows), "validation_examples": len(eval_rows),
              "base_config": generation, "checkpoints": [], "selection_status": "not_evaluated",
              "dtype": precision, "environment": environment(), "official_metric_verified": False}
    write_json(out / "report.json", report)
    print(f"Training schedule: {schedule}", flush=True)

    def dev_generation(name, identity):
        model.eval()
        model.config.use_cache = True
        evaluate(dataset, "dev", generation, generation_path, out / "dev_generations" / name,
                 _backend=backend, _identity=identity)
        model.config.use_cache = False
        model.train()

    # LoRA starts with a zero delta: evaluate base before any optimizer step.
    with model.disable_adapter():
        dev_generation("base", {"kind": "base", "model_name": generation["model_name"],
                                "revision": generation.get("revision", "main")})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(config.get("learning_rate", 1e-5)), weight_decay=.01)
    warmup = min(schedule["optimizer_steps"] - 1,
                 math.ceil(schedule["optimizer_steps"] * float(config.get("warmup_ratio", .05))))
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=warmup,
                              num_training_steps=schedule["optimizer_steps"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "float16")
    collator = AnswerCollator(tokenizer.pad_token_id)
    global_step, examples_seen, updates = 0, 0, []
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    batch_size, accumulation = config.get("batch_size", 1), config.get("gradient_accumulation_steps", 8)
    checkpoint_steps = int(config.get("checkpoint_steps", 0))
    if checkpoint_steps < 0:
        raise ValueError("checkpoint_steps must be >= 0")

    def save_candidate(name):
        target = out / "checkpoints" / name
        model.save_pretrained(target, safe_serialization=True)
        tokenizer.save_pretrained(target)
        report["checkpoints"].append(name)
        write_json(out / "report.json", report)
        dev_generation(name, {"kind": "adapter", "training_id": manifest["artifact_id"], "checkpoint": name})

    for epoch in range(schedule["epochs"]):
        order = list(range(len(train_data)))
        random.Random(seed + epoch).shuffle(order)
        accumulated_tokens, accumulated_loss = 0, 0.0
        for batch_index, start in enumerate(range(0, len(order), batch_size), 1):
            selected = order[start:start + batch_size]
            batch = {k: v.to("cuda") for k, v in collator([train_data[i] for i in selected]).items()}
            token_count = int((batch["labels"][:, 1:] != -100).sum().item())
            if token_count < 1:
                raise ValueError("Training batch has no supervised answer tokens")
            with torch.autocast("cuda", dtype=getattr(torch, precision), enabled=precision != "float32"):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            # Token-weighted accumulation also handles the final partial batch.
            scaler.scale(loss * token_count).backward()
            accumulated_tokens += token_count
            accumulated_loss += float(loss.detach()) * token_count
            examples_seen += len(selected)
            if batch_index % accumulation == 0 or batch_index == schedule["batches_per_epoch"]:
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_tokens)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite gradients; retry a new run with BF16/float32 or lower LR")
                lr = optimizer.param_groups[0]["lr"]
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                update = {"global_step": global_step, "epoch": epoch + batch_index / schedule["batches_per_epoch"],
                          "examples_seen": examples_seen, "learning_rate": lr,
                          "answer_token_loss": accumulated_loss / accumulated_tokens}
                updates.append(update)
                write_json(out / "trainer_state.json", {"loop": "explicit_token_weighted_v2", **update,
                                                          "schedule": schedule, "warmup_steps": warmup})
                print(update, flush=True)
                accumulated_tokens, accumulated_loss = 0, 0.0
                if checkpoint_steps and global_step % checkpoint_steps == 0 and batch_index != schedule["batches_per_epoch"]:
                    save_candidate(f"step-{global_step:06d}")
        save_candidate(f"epoch-{epoch + 1:03d}")
    if global_step != schedule["optimizer_steps"] or examples_seen != len(train_rows) * schedule["epochs"]:
        raise RuntimeError("Training schedule mismatch; refusing to mark this artifact complete")
    write_jsonl(out / "training_log.jsonl", updates)
    for name, content in (("system.txt", system), ("user.txt", user)):
        (out / name).write_text(content, encoding="utf-8")
    report.update(global_step=global_step, examples_seen=examples_seen, runtime_seconds=time.monotonic() - started,
                  note="All checkpoints are candidates. Review dev generations, score proxies, confirm holdout, then merge-model.")
    write_json(out / "report.json", report)
    files = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p != out / "manifest.json"]
    return finish(out, manifest, files)


def merge_selected(training, checkpoint, selection, config, out):
    """Publish only the explicit checkpoint confirmed by dev and holdout gates."""
    import yaml
    from .artifacts import load_artifact, read_json
    from .evaluation import training_checkpoint
    parent, report, adapter = training_checkpoint(training, checkpoint)
    decision = load_artifact(selection, "model_selection")
    selected = read_json(Path(selection) / "report.json")
    expected = {"kind": "adapter", "training_id": parent["artifact_id"], "checkpoint": checkpoint}
    if (selected.get("status") != "selected" or selected.get("model") != expected
            or selected.get("dataset_id") != report["dataset_id"]):
        raise ValueError("Model selection did not approve this exact training artifact/checkpoint")
    sig = signature("finetune", {"workflow": "selected-v2", **config},
                    {"training_id": parent["artifact_id"], "checkpoint": checkpoint,
                     "selection_id": decision["artifact_id"]}, ["finetuning.py", "evaluation.py", "llm.py"])
    manifest, cached = begin(out, sig)
    if cached:
        return manifest
    out = Path(out).resolve()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .artifacts import read_jsonl
    base = report["base_config"]
    options = {"revision": base.get("revision", "main"), "local_files_only": base.get("local_files_only", False),
               "trust_remote_code": False}
    model = AutoModelForCausalLM.from_pretrained(base["model_name"], torch_dtype=torch.float32, **options)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False).eval()
    probes = read_jsonl(Path(training) / "dev_generations" / checkpoint / "predictions.jsonl")[:3]
    if not probes:
        raise ValueError("Cannot verify merge without fixed dev probes")
    encoded = [tokenizer.apply_chat_template(p["messages"], tokenize=True, add_generation_prompt=True,
                                             return_tensors="pt") for p in probes]
    # CPU float32, last-token logits on several fixed dev prompts, before/after merge.
    with torch.inference_mode():
        before = [model(ids).logits[:, -1, :].clone() for ids in encoded]
        merged = model.merge_and_unload(safe_merge=True).eval()
        after = [merged(ids).logits[:, -1, :].clone() for ids in encoded]
    atol, rtol = float(config.get("merge_atol", .0001)), float(config.get("merge_rtol", .0001))
    if not all(torch.allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(before, after)):
        raise RuntimeError("Merged model failed fixed-prompt logit equivalence; artifact remains incomplete")
    merged.config.use_cache = True
    merged.save_pretrained(out / "model", safe_serialization=True)
    tokenizer.save_pretrained(out / "model")
    # Verify serialization too; a correct in-memory merge alone is insufficient.
    del merged, model
    import gc
    gc.collect()
    restored = AutoModelForCausalLM.from_pretrained(out / "model", local_files_only=True,
                                                  torch_dtype=torch.float32, trust_remote_code=False).eval()
    restored_tokenizer = AutoTokenizer.from_pretrained(out / "model", local_files_only=True)
    with torch.inference_mode():
        for probe, ids, expected_logits in zip(probes, encoded, after):
            restored_ids = restored_tokenizer.apply_chat_template(probe["messages"], tokenize=True,
                add_generation_prompt=True, return_tensors="pt")
            if not torch.equal(ids, restored_ids) or not torch.allclose(
                    expected_logits, restored(restored_ids).logits[:, -1, :], atol=atol, rtol=rtol):
                raise RuntimeError("Reloaded merged model/tokenizer failed equivalence check")
    for name in ("system.txt", "user.txt"):
        (out / name).write_text((Path(training) / name).read_text(encoding="utf-8"), encoding="utf-8")
    inference = {**base, "model_name": str(out / "model"), "revision": "main", "local_files_only": True,
                 "finetune_artifact": str(out), "system_prompt_file": "system.txt", "user_prompt_file": "user.txt"}
    (out / "generate.yaml").write_text(yaml.safe_dump(inference, sort_keys=False), encoding="utf-8")
    write_json(out / "selection.json", selected)
    write_json(out / "report.json", {"workflow": "selected-v2", "selection_status": "selected",
               "training_id": parent["artifact_id"], "checkpoint": checkpoint, "merge_verified": True,
               "reload_verified": True, "merge_probe_count": len(probes), "atol": atol, "rtol": rtol,
               "official_metric_verified": False})
    return finish(out, manifest, [str(p.relative_to(out)) for p in out.rglob("*")
                                if p.is_file() and p != out / "manifest.json"])


def selected_generation_config(directory):
    """Load a relocatable inference config only from a checked, selected export."""
    from .artifacts import load_artifact, load_config, read_json
    directory = Path(directory).resolve()
    manifest = load_artifact(directory, "finetune")
    report = read_json(directory / "report.json")
    if report.get("workflow") != "selected-v2":
        raise ValueError("Inference requires a selected-v2 merged artifact, not a training candidate")
    decision = read_json(directory / "selection.json")
    expected = {"kind": "adapter", "training_id": report.get("training_id"), "checkpoint": report.get("checkpoint")}
    if (report.get("workflow") != "selected-v2" or not report.get("merge_verified") or not report.get("reload_verified")
            or report.get("selection_status") != "selected" or decision.get("status") != "selected"
            or decision.get("model") != expected):
        raise ValueError("Inference requires a selected-v2 merged artifact with passing dev/holdout gates")
    config = load_config(directory / "generate.yaml")
    config.update(model_name=str(directory / "model"), finetune_artifact=str(directory), local_files_only=True,
                  system_prompt_file=str(directory / "system.txt"), user_prompt_file=str(directory / "user.txt"))
    return manifest, config
