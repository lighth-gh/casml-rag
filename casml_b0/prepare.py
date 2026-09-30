"""PDF -> normalized pages + stable source-bearing chunks. No retriever/LLM."""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from .artifacts import (begin, digest, environment, file_hash, finish, signature,
                        write_jsonl, atomic_text, write_json)


def clean_text(text):
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


class RegexOffsets:
    """Explicit offline DEMO tokenizer; never selected by the real B0 config."""
    def offsets(self, text):
        return [m.span() for m in re.finditer(r"\S+", text)]


class HFOffsets:
    def __init__(self, config):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_model"], revision=config.get("tokenizer_revision", "main"),
            use_fast=True, local_files_only=config.get("local_files_only", False))
        if not self.tokenizer.is_fast:
            raise ValueError("Chunking needs a fast tokenizer with character offsets")

    def offsets(self, text):
        values = self.tokenizer(text, add_special_tokens=False, truncation=False,
                                return_offsets_mapping=True)["offset_mapping"]
        return [(a, b) for a, b in values if b > a]


def split_page(text, tokenizer, size, overlap):
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= chunk_overlap < chunk_tokens")
    offsets = tokenizer.offsets(text)
    start = 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        # Prefer a sentence ending near the nominal boundary.
        if end < len(offsets):
            lower = max(start + overlap + 1, start + int(size * 0.7))
            for candidate in range(end, lower, -1):
                if text[offsets[candidate - 1][0]:offsets[candidate - 1][1]].rstrip().endswith((".", "?", "!")):
                    end = candidate
                    break
        a, b = offsets[start][0], offsets[end - 1][1]
        yield a, b, text[a:b], end - start
        if end == len(offsets):
            break
        start = end - overlap


def read_overrides(path, page_count):
    if path is None:
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"pdf_page", "printed_page", "section_path"} <= set(reader.fieldnames or []):
            raise ValueError("Page map needs pdf_page,printed_page,section_path columns")
        result = {}
        for row in reader:
            page = int(row["pdf_page"])
            if not 1 <= page <= page_count or page in result:
                raise ValueError(f"Invalid/duplicate page map row: {page}")
            # Blank values explicitly mean unknown (never invent an offset).
            result[page] = {"printed_page": row["printed_page"].strip() or None,
                            "section_path": [s.strip() for s in row["section_path"].split("/") if s.strip()],
                            "section_method": "user_override"}
    return result


def prepare(pdf, config, out, page_map_override=None):
    import fitz
    pdf, out = Path(pdf), Path(out)
    doc_id = file_hash(pdf)
    sig = signature("corpus", config, {"pdf_sha256": doc_id,
                    "page_map_sha256": file_hash(page_map_override) if page_map_override else None},
                    ["prepare.py"])
    manifest, cached = begin(out, sig)
    if cached:
        return manifest
    backend = config.get("tokenizer_backend", "huggingface")
    if backend not in ("huggingface", "regex_demo"):
        raise ValueError(f"Unknown tokenizer backend {backend}")
    tokenizer = HFOffsets(config) if backend == "huggingface" else RegexOffsets()
    size, overlap = int(config["chunk_tokens"]), int(config["chunk_overlap"])
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk configuration")
    pages, chunks = [], []
    with fitz.open(pdf) as doc:
        if doc.needs_pass:
            raise ValueError("Encrypted PDF requires an unlocked local copy")
        overrides = read_overrides(page_map_override, len(doc))
        toc, stack, anchors = doc.get_toc(), [], []
        for level, title, page in toc:
            stack = stack[:max(0, level - 1)]
            stack.append(str(title).strip())
            if 1 <= page <= len(doc):
                anchors.append((page, list(stack)))
        anchors.sort(key=lambda a: a[0])
        explicit_labels = bool(doc.get_page_labels())
        for i, page in enumerate(doc, 1):
            text = clean_text(page.get_text("text", sort=True))
            prior = [path for start, path in anchors if start <= i]
            metadata = {"doc_id": doc_id, "source_pdf": pdf.name, "pdf_page": i,
                        "printed_page": (page.get_label() or None) if explicit_labels else None,
                        "section_path": prior[-1] if prior else [],
                        "section_method": "toc_page_approximation" if prior else "unknown"}
            metadata.update(overrides.get(i, {}))
            pages.append({**metadata, "text": text, "text_sha256": digest(text)})
            for a, b, body, n_tokens in split_page(text, tokenizer, size, overlap):
                chunk_id = "c_" + digest([doc_id, i, a, b, body])[:24]
                chunks.append({**metadata, "chunk_id": chunk_id, "text": body,
                               "text_sha256": digest(body), "char_start": a, "char_end": b,
                               "chunk_token_count": n_tokens})
    if not chunks:
        raise ValueError("No extractable text. This baseline needs OCR before ingesting scanned PDFs.")
    write_jsonl(out / "pages.jsonl", pages)
    write_jsonl(out / "chunks.jsonl", chunks)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["pdf_page", "printed_page", "section_path", "section_method"])
    writer.writeheader()
    for p in pages:
        writer.writerow({k: "/".join(p[k]) if k == "section_path" else p[k]
                         for k in writer.fieldnames})
    atomic_text(out / "page_map.csv", buffer.getvalue())
    report = {"page_count": len(pages), "chunk_count": len(chunks),
              "empty_pages": [p["pdf_page"] for p in pages if not p["text"]],
              "unmapped_printed_pages": sum(p["printed_page"] is None for p in pages),
              "approximate_sections": sum(p["section_method"] == "toc_page_approximation" for p in pages),
              "note": "TOC page ranges are approximate, especially at section boundaries. Inspect page_map.csv.",
              "environment": environment()}
    write_json(out / "report.json", report)
    return finish(out, manifest, ["pages.jsonl", "chunks.jsonl", "page_map.csv", "report.json"],
                  doc_id=doc_id, source_pdf=pdf.name, page_count=len(pages), chunk_count=len(chunks))
