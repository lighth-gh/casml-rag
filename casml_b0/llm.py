"""Generation adapters. No embedding/index/retrieval import is allowed here."""
from __future__ import annotations

import re


class ExtractiveDemoGenerator:
    """Offline integration example, not an LLM and never a production fallback."""
    context_window = 32768

    def count_text(self, text):
        return len(re.findall(r"\S+", text))

    def count_messages(self, messages):
        return sum(self.count_text(m["content"]) + 4 for m in messages) + 4

    def generate(self, payload, config):
        # Demo explicitly copies a source sentence. Quality is not evaluated.
        text = payload["evidence"][0]["text"]
        sentence = re.split(r"(?<=[.!?])\s+", text)[0]
        words = sentence.split()
        count = int(config["max_new_tokens"])
        return {"answer": " ".join(words[:count]), "output_tokens": min(len(words), count),
                "finish_reason": "length" if len(words) > count else "demo_extract"}


class HFGenerator:
    def __init__(self, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = config.get("device", "auto")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = config.get("dtype", "auto")
        if precision == "auto":
            precision = "float16" if str(self.device).startswith("cuda") else "float32"
        if precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be auto/float32/float16/bfloat16")
        if str(self.device).startswith("cpu") and precision == "float16":
            raise ValueError("Use float32 on CPU")
        options = {"revision": config.get("revision", "main"),
                   "local_files_only": config.get("local_files_only", False),
                   "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], **options)
        if not self.tokenizer.chat_template:
            raise ValueError("Generator must have a chat template; use an instruct/chat model")
        model_options = dict(options)
        if config.get("attn_implementation"):
            model_options["attn_implementation"] = config["attn_implementation"]
        self.model = AutoModelForCausalLM.from_pretrained(config["model_name"],
                        torch_dtype=getattr(torch, precision), **model_options).to(self.device).eval()
        self.context_window = int(getattr(self.model.config, "max_position_embeddings", 4096))
        configured_eos = config.get("eos_token_ids")
        if configured_eos is None:
            configured_eos = self.model.generation_config.eos_token_id
        if isinstance(configured_eos, int):
            configured_eos = [configured_eos]
        self.eos_token_ids = [int(value) for value in configured_eos or []]
        if not self.eos_token_ids:
            raise ValueError("No EOS token configured; generation could run until max_new_tokens")
        vocab_size = int(getattr(self.model.config, "vocab_size", len(self.tokenizer)))
        if any(value < 0 or value >= vocab_size for value in self.eos_token_ids):
            raise ValueError(f"EOS token IDs outside model vocabulary: {self.eos_token_ids}")
        configured_pad = config.get("pad_token_id", self.tokenizer.pad_token_id)
        if configured_pad is None:
            configured_pad = self.eos_token_ids[-1]
        self.pad_token_id = int(configured_pad)
        self.tokenizer.pad_token_id = self.pad_token_id
        self.model.generation_config.eos_token_id = self.eos_token_ids
        self.model.generation_config.pad_token_id = self.pad_token_id
        print(f"Generation stop tokens: eos={self.eos_token_ids}, pad={self.pad_token_id}", flush=True)
        torch.manual_seed(int(config.get("seed", 42)))

    def count_text(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_messages(self, messages):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    def generate(self, payload, config):
        text = self.tokenizer.apply_chat_template(payload["messages"], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, add_special_tokens=False, truncation=False,
                                return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[-1]
        if input_length != payload["input_tokens"]:
            raise ValueError("Chat-template token count mismatch; refusing implicit truncation")
        kwargs = {"max_new_tokens": int(config["max_new_tokens"]), "do_sample": bool(config.get("do_sample", False)),
                  "pad_token_id": self.pad_token_id, "eos_token_id": self.eos_token_ids,
                  "repetition_penalty": float(config.get("repetition_penalty", 1.0)),
                  "no_repeat_ngram_size": int(config.get("no_repeat_ngram_size", 0))}
        if kwargs["do_sample"]:
            kwargs.update(temperature=float(config.get("temperature", 0.7)), top_p=float(config.get("top_p", 0.9)))
        else:
            # Neutralize sampling-only defaults stored in some model configs.
            kwargs.update(temperature=None, top_p=None, top_k=None)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)[0, input_length:]
        answer = self.tokenizer.decode(output, skip_special_tokens=True).strip()
        eos = set(self.eos_token_ids)
        reached_limit = len(output) >= kwargs["max_new_tokens"] and int(output[-1]) not in eos
        return {"answer": answer, "output_tokens": len(output),
                "finish_reason": "length" if reached_limit else "eos"}


def make_generator(config):
    backend = config.get("backend", "huggingface")
    if backend == "huggingface":
        return HFGenerator(config)
    if backend == "extractive_demo":
        return ExtractiveDemoGenerator()
    raise ValueError(f"Unknown generator backend {backend}")
