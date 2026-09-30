"""Corpus -> reusable dense index. No generation dependencies."""
from pathlib import Path
import numpy as np

from .artifacts import begin, finish, load_artifact, read_jsonl, signature, write_jsonl
from .embedding import make_embedder


def build_index(corpus, config, out):
    corpus, out = Path(corpus), Path(out)
    source = load_artifact(corpus, "corpus")
    sig = signature("index", config, {"corpus_id": source["artifact_id"]}, ["indexing.py", "embedding.py"])
    manifest, cached = begin(out, sig)
    if cached:
        return manifest
    chunks = read_jsonl(corpus / "chunks.jsonl")
    texts = []
    for chunk in chunks:
        title = "/".join(chunk["section_path"])
        texts.append(f"{title}\n{chunk['text']}" if title and config.get("include_section_title", True) else chunk["text"])
    embeddings = make_embedder(config).encode(texts)
    if embeddings.ndim != 2 or len(embeddings) != len(chunks) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding matrix")
    engine = config.get("engine", "faiss")
    if engine == "faiss":
        import faiss
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(out / "index.faiss"))
        index_file = "index.faiss"
    elif engine == "numpy":
        np.save(out / "embeddings.npy", embeddings, allow_pickle=False)
        index_file = "embeddings.npy"
    else:
        raise ValueError(f"Unknown index engine {engine}")
    write_jsonl(out / "chunks.jsonl", chunks)
    return finish(out, manifest, [index_file, "chunks.jsonl"], embedding_config=config,
                  doc_id=source["doc_id"], source_pdf=source["source_pdf"],
                  count=len(chunks), dimension=embeddings.shape[1])
