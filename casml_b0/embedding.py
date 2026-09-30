"""Embedding adapters. Only index/retrieve import this module."""
from __future__ import annotations

import hashlib
import re
import numpy as np


class HashDemoEmbedder:
    """Deterministic bag-of-words hashing for offline plumbing tests, NOT dense B0."""
    def __init__(self, config):
        self.dimension = int(config.get("demo_dimension", 1024))

    def encode(self, texts, query=False):
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                j = int(hashlib.sha256(word.encode()).hexdigest()[:12], 16) % self.dimension
                vectors[i, j] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


class SentenceEmbedder:
    def __init__(self, config):
        from sentence_transformers import SentenceTransformer
        self.config = config
        self.model = SentenceTransformer(config["model_name"], device=config.get("device", "cpu"),
                    revision=config.get("revision", "main"),
                    local_files_only=config.get("local_files_only", False))
        self.model.max_seq_length = int(config.get("max_sequence_length", 512))

    def encode(self, texts, query=False):
        prefix = self.config.get("query_prefix", "") if query else ""
        texts = [prefix + text for text in texts]
        # Silent embedding truncation would conceal retrieval problems.
        for text in texts:
            length = len(self.model.tokenizer(text, truncation=False)["input_ids"])
            if length > self.model.max_seq_length:
                raise ValueError(f"Embedding input has {length} tokens > {self.model.max_seq_length}. "
                                 "Reduce corpus chunk size or choose a longer-context embedder.")
        return self.model.encode(texts, batch_size=int(self.config.get("batch_size", 32)),
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=True).astype("float32")


def make_embedder(config):
    backend = config.get("backend", "sentence_transformers")
    if backend == "sentence_transformers":
        return SentenceEmbedder(config)
    if backend == "hash_demo":
        return HashDemoEmbedder(config)
    raise ValueError(f"Unknown embedding backend {backend}")
