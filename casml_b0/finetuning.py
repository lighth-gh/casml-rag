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


def finetune(train_path, config, config_path, out, validation_path=None):
    import yaml

    config_root = Path(config_path).resolve().parent
    generation_path = (config_root / config["generation_config"]).resolve()
    generation = load_config(generation_path)
    if generation.get("backend", "huggingface") != "huggingface" or generation.get("finetune_artifact"):
        raise ValueError("Fine-tune requires a base Hugging Face generation config")
    system, user = load_prompts(generation, generation_path)
    rows = read_examples(train_path)
    validation = read_examples(validation_path) if validation_path else None
    seed = int(config.get("seed", 42))
    train_rows, eval_rows = split_examples(rows, validation, float(config.get("validation_fraction", 0.1)), seed)
    max_length = int(config.get("max_length", 2048))
    if max_length < 2 or max_length > int(generation["context_window"]):
        raise ValueError("max_length must be between 2 and generation context_window")
    effective = {**config, "generation": generation, "system": system, "user": user}
    inputs = {"train_sha256": file_hash(train_path),
              "validation_sha256": file_hash(validation_path) if validation_path else None}
    sig = signature("finetune", effective, inputs, ["finetuning.py", "generation.py"])
    out = Path(out).resolve()
    manifest, cached = begin(out, sig)
    if cached:
        return manifest

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning requires a CUDA GPU. Enable GPU on Kaggle before running this stage.")
    set_seed(seed)
    precision = config.get("dtype", "auto")
    if precision == "auto":
        precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if precision not in ("bfloat16", "float16", "float32"):
        raise ValueError("dtype must be auto/bfloat16/float16/float32")
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; use dtype: auto")
    options = {"revision": generation.get("revision", "main"), "trust_remote_code": False,
               "local_files_only": generation.get("local_files_only", False)}
    tokenizer = AutoTokenizer.from_pretrained(generation["model_name"], **options)
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("Training requires a chat template and EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    train_data = [tokenize_example(row, tokenizer, system, user, max_length) for row in train_rows]
    eval_data = [tokenize_example(row, tokenizer, system, user, max_length) for row in eval_rows]
    model = AutoModelForCausalLM.from_pretrained(
        generation["model_name"], torch_dtype=getattr(torch, precision),
        attn_implementation=generation.get("attn_implementation", "sdpa"), **options)
    if max_length > model.config.max_position_embeddings:
        raise ValueError("max_length exceeds model context window")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)), lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]), bias="none"))
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=float(config.get("num_train_epochs", 3)),
        max_steps=int(config.get("max_steps", -1)),
        per_device_train_batch_size=int(config.get("batch_size", 1)), per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(config.get("learning_rate", 0.0002)), warmup_ratio=0.05,
        weight_decay=0.01, lr_scheduler_type="cosine", optim="adamw_torch",
        fp16=precision == "float16", bf16=precision == "bfloat16",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=1, report_to=[], seed=seed, data_seed=seed,
        dataloader_num_workers=0, label_names=["labels"], prediction_loss_only=True)
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=eval_data,
                      data_collator=AnswerCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    baseline_metrics = trainer.evaluate()
    result = trainer.train()
    eval_metrics = trainer.evaluate()
    model.save_pretrained(out / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(out / "adapter")
    # Merge on CPU so inference uses the existing HFGenerator without PEFT.
    model.to("cpu")
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(out / "model", safe_serialization=True)
    tokenizer.save_pretrained(out / "model")
    for name, content in (("system.txt", system), ("user.txt", user)):
        (out / name).write_text(content, encoding="utf-8")
    inference = {**generation, "model_name": str(out / "model"), "revision": "main",
                 "local_files_only": True, "finetune_artifact": str(out),
                 "system_prompt_file": "system.txt", "user_prompt_file": "user.txt"}
    (out / "generate.yaml").write_text(yaml.safe_dump(inference, sort_keys=False), encoding="utf-8")
    write_json(out / "report.json", {
        "train_examples": len(train_rows), "validation_examples": len(eval_rows),
        "train_ids": [row["query_id"] for row in train_rows],
        "validation_ids": [row["query_id"] for row in eval_rows],
        "baseline_metrics": baseline_metrics, "train_metrics": result.metrics,
        "validation_metrics": eval_metrics, "dtype": precision, "environment": environment(),
        "note": "Validation loss is teacher-forced answer loss, not a CASML score."})
    files = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()
             and "checkpoints" not in p.relative_to(out).parts and p.name != "manifest.json"]
    return finish(out, manifest, files)
