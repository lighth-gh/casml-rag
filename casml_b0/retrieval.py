"""Dense index + questions -> SELF-CONTAINED retrieval cache (JSONL)."""
from pathlib import Path
import numpy as np

from .artifacts import (SCHEMA, begin, digest, finish, load_artifact,
                        read_jsonl, signature, write_jsonl)
from .contracts import load_queries, validate_retrieval
from .embedding import make_embedder


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
    k = int(config.get("top_k", 20))
    if k < 1:
        raise ValueError("top_k must be positive")
    k = min(k, len(chunks))
    embeddings = make_embedder(parent["embedding_config"]).encode([q["question"] for q in queries], query=True)
    if parent["embedding_config"].get("engine", "faiss") == "faiss":
        import faiss
        index = faiss.read_index(str(index_dir / "index.faiss"))
        scores, indices = index.search(embeddings, k)
    else:
        matrix = np.load(index_dir / "embeddings.npy", allow_pickle=False)
        all_scores = embeddings @ matrix.T
        indices = np.argsort(-all_scores, axis=1, kind="stable")[:, :k]
        scores = np.take_along_axis(all_scores, indices, axis=1)
    rows = []
    for q, ids, values in zip(queries, indices, scores):
        candidates = [{**chunks[int(idx)], "rank": rank, "retrieval_score": float(score)}
                      for rank, (idx, score) in enumerate(zip(ids, values), 1)]
        rows.append({"schema": SCHEMA, **q, "candidates": candidates})
    validate_retrieval(rows)
    write_jsonl(out / "retrieval.jsonl", rows)
    return finish(out, manifest, ["retrieval.jsonl"], query_count=len(rows), candidate_count=k,
                  doc_id=parent["doc_id"], source_pdf=parent["source_pdf"],
                  backend=parent["embedding_config"].get("backend", "sentence_transformers"))
