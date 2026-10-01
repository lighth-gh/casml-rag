"""Dense/BM25 hybrid retrieval with optional cross-encoder reranking."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .artifacts import (SCHEMA, begin, digest, finish, load_artifact,
                        read_jsonl, signature, write_jsonl)
from .contracts import load_queries, validate_retrieval
from .embedding import make_embedder


TOKEN_PATTERN = re.compile(r"(?u)\b\w+(?:'\w+)?\b")


def lexical_tokens(text):
    """Small deterministic tokenizer suitable for an English textbook BM25 index."""
    return TOKEN_PATTERN.findall(text.casefold())


class BM25Index:
    """Dependency-free Okapi BM25; the corpus is small enough to build per retrieval run."""

    def __init__(self, texts, *, k1=1.5, b=0.75):
        self.k1, self.b = float(k1), float(b)
        self.documents = [lexical_tokens(text) for text in texts]
        self.lengths = np.asarray([len(tokens) for tokens in self.documents], dtype=np.float32)
        self.average_length = float(self.lengths.mean()) if len(self.lengths) else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.documents]
        document_frequency = Counter()
        for terms in self.term_frequencies:
            document_frequency.update(terms.keys())
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query):
        scores = np.zeros(len(self.documents), dtype=np.float32)
        if not self.documents or self.average_length <= 0:
            return scores
        for term in set(lexical_tokens(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self.term_frequencies):
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * self.lengths[index] / self.average_length
                )
                scores[index] += idf * frequency * (self.k1 + 1.0) / denominator
        return scores


def reciprocal_rank_fusion(dense_ids, bm25_ids, *, dense_weight=0.55,
                           bm25_weight=0.45, rrf_k=60):
    """Fuse rankings, not incomparable raw score scales."""
    if dense_weight < 0 or bm25_weight < 0 or dense_weight + bm25_weight <= 0:
        raise ValueError("Hybrid weights must be non-negative and not both zero")
    if int(rrf_k) < 1:
        raise ValueError("rrf_k must be positive")
    scores = {}
    for rank, index in enumerate(dense_ids, 1):
        scores[int(index)] = scores.get(int(index), 0.0) + dense_weight / (int(rrf_k) + rank)
    for rank, index in enumerate(bm25_ids, 1):
        scores[int(index)] = scores.get(int(index), 0.0) + bm25_weight / (int(rrf_k) + rank)
    return sorted(scores, key=lambda index: (-scores[index], index)), scores


def _dense_search(index_dir, parent, embeddings, pool_k):
    if parent["embedding_config"].get("engine", "faiss") == "faiss":
        import faiss
        index = faiss.read_index(str(index_dir / "index.faiss"))
        return index.search(embeddings, pool_k)
    matrix = np.load(index_dir / "embeddings.npy", allow_pickle=False)
    all_scores = embeddings @ matrix.T
    indices = np.argsort(-all_scores, axis=1, kind="stable")[:, :pool_k]
    return np.take_along_axis(all_scores, indices, axis=1), indices


def _make_reranker(config):
    backend = config.get("reranker_backend", "none")
    if backend in (None, "none"):
        return None
    if backend != "cross_encoder":
        raise ValueError("reranker_backend must be none or cross_encoder")
    import torch
    from sentence_transformers import CrossEncoder

    device = config.get("reranker_device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(
        config["reranker_model_name"],
        revision=config.get("reranker_revision", "main"),
        device=device,
        local_files_only=config.get("local_files_only", False),
        trust_remote_code=False,
    )


def retrieve(index_dir, queries_path, config, out):
    index_dir, out = Path(index_dir), Path(out)
    parent = load_artifact(index_dir, "index")
    queries = load_queries(queries_path)
    sig = signature("retrieval", config, {"index_id": parent["artifact_id"], "queries": digest(queries)},
                    ["retrieval.py", "embedding.py"])
    manifest, cached = begin(out, sig)
    if cached:
        return manifest
    chunks = read_jsonl(index_dir / "chunks.jsonl")
    top_k = int(config.get("top_k", 20))
    if top_k < 1:
        raise ValueError("top_k must be positive")
    top_k = min(top_k, len(chunks))
    method = config.get("method", "dense")
    if method not in ("dense", "hybrid"):
        raise ValueError("method must be dense or hybrid")

    dense_pool_k = top_k if method == "dense" else min(
        int(config.get("dense_pool_k", max(top_k, 50))), len(chunks)
    )
    if dense_pool_k < top_k:
        raise ValueError("dense_pool_k must be >= top_k")
    embeddings = make_embedder(parent["embedding_config"]).encode(
        [q["question"] for q in queries], query=True
    )
    dense_scores, dense_indices = _dense_search(index_dir, parent, embeddings, dense_pool_k)

    if method == "dense":
        rows = []
        for query, ids, values in zip(queries, dense_indices, dense_scores):
            candidates = [{**chunks[int(index)], "rank": rank, "retrieval_score": float(score),
                           "dense_rank": rank, "dense_score": float(score)}
                          for rank, (index, score) in enumerate(zip(ids[:top_k], values[:top_k]), 1)]
            rows.append({"schema": SCHEMA, **query, "candidates": candidates})
        backend = parent["embedding_config"].get("backend", "sentence_transformers")
    else:
        include_title = bool(config.get("bm25_include_section_title", True))
        lexical_texts = []
        rerank_texts = []
        for chunk in chunks:
            title = " / ".join(chunk["section_path"])
            text = f"{title}\n{chunk['text']}" if title and include_title else chunk["text"]
            lexical_texts.append(text)
            rerank_texts.append(text)
        bm25 = BM25Index(lexical_texts, k1=config.get("bm25_k1", 1.5),
                         b=config.get("bm25_b", 0.75))
        bm25_pool_k = min(int(config.get("bm25_pool_k", max(top_k, 50))), len(chunks))
        fusion_top_k = min(int(config.get("fusion_top_k", max(top_k, 30))), len(chunks))
        if bm25_pool_k < top_k or fusion_top_k < top_k:
            raise ValueError("bm25_pool_k and fusion_top_k must be >= top_k")
        reranker = _make_reranker(config)
        rows = []
        for query, dense_ids, dense_values in zip(queries, dense_indices, dense_scores):
            sparse_scores = bm25.scores(query["question"])
            sparse_ids = np.argsort(-sparse_scores, kind="stable")[:bm25_pool_k]
            fused_ids, fused_scores = reciprocal_rank_fusion(
                dense_ids, sparse_ids,
                dense_weight=float(config.get("dense_weight", 0.55)),
                bm25_weight=float(config.get("bm25_weight", 0.45)),
                rrf_k=int(config.get("rrf_k", 60)),
            )
            fused_ids = fused_ids[:fusion_top_k]
            dense_rank = {int(index): rank for rank, index in enumerate(dense_ids, 1)}
            dense_score = {int(index): float(score) for index, score in zip(dense_ids, dense_values)}
            bm25_rank = {int(index): rank for rank, index in enumerate(sparse_ids, 1)}

            if reranker is not None:
                pairs = [(query["question"], rerank_texts[index]) for index in fused_ids]
                reranker_scores = np.asarray(reranker.predict(
                    pairs,
                    batch_size=int(config.get("reranker_batch_size", 16)),
                    show_progress_bar=False,
                )).reshape(-1)
                reranker_score = {
                    index: float(reranker_scores[position])
                    for position, index in enumerate(fused_ids)
                }
                final_ids = sorted(
                    fused_ids,
                    key=lambda index: (-reranker_score[index], -fused_scores[index], index),
                )[:top_k]
            else:
                final_ids = fused_ids[:top_k]
                reranker_score = {}

            candidates = []
            for rank, index in enumerate(final_ids, 1):
                final_score = reranker_score.get(index, fused_scores[index])
                candidates.append({
                    **chunks[index],
                    "rank": rank,
                    "retrieval_score": float(final_score),
                    "dense_rank": dense_rank.get(index),
                    "dense_score": dense_score.get(index),
                    "bm25_rank": bm25_rank.get(index),
                    "bm25_score": float(sparse_scores[index]),
                    "fusion_score": float(fused_scores[index]),
                    "reranker_score": reranker_score.get(index),
                })
            rows.append({"schema": SCHEMA, **query, "candidates": candidates})
        backend = "hybrid_dense_bm25" + ("_cross_encoder" if reranker is not None else "")

    validate_retrieval(rows)
    write_jsonl(out / "retrieval.jsonl", rows)
    return finish(out, manifest, ["retrieval.jsonl"], query_count=len(rows), candidate_count=top_k,
                  doc_id=parent["doc_id"], source_pdf=parent["source_pdf"], backend=backend)
