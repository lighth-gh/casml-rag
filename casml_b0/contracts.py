"""Stable, JSON-only boundaries between stages."""
from __future__ import annotations

from .artifacts import digest, read_json, SCHEMA


def load_queries(path):
    rows = read_json(path)
    if isinstance(rows, dict):
        rows = rows.get("queries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("queries.json must contain a non-empty list of query_id/question objects")
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or "query_id" not in row or "question" not in row:
            raise ValueError("Every query needs query_id and question")
        value = row["query_id"]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("query_id must be a string or integer")
        query_id = str(value)
        question = row["question"]
        if not query_id.strip() or query_id in seen:
            raise ValueError(f"Empty/duplicate query_id: {query_id!r}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Empty question for {query_id}")
        seen.add(query_id)
        result.append({"query_id": query_id, "question": question.strip()})
    return result


def validate_evidence(e):
    required = {"chunk_id", "doc_id", "text", "text_sha256", "pdf_page", "source_pdf",
                "printed_page", "section_path", "section_method", "char_start", "char_end"}
    if not isinstance(e, dict) or not required <= e.keys():
        raise ValueError("Evidence record is missing fields")
    if not isinstance(e["pdf_page"], int) or e["pdf_page"] < 1:
        raise ValueError("pdf_page is always 1-based")
    if not isinstance(e["text"], str) or not e["text"].strip():
        raise ValueError("Empty evidence")
    if digest(e["text"]) != e["text_sha256"]:
        raise ValueError(f"Evidence text changed: {e['chunk_id']}")
    if not isinstance(e["section_path"], list) or not all(isinstance(s, str) for s in e["section_path"]):
        raise ValueError("section_path must be a list of strings")
    if e["char_start"] < 0 or e["char_end"] <= e["char_start"]:
        raise ValueError("Invalid evidence character range")


def validate_retrieval(rows):
    seen = set()
    for row in rows:
        if row.get("schema") != SCHEMA or row.get("query_id") in seen:
            raise ValueError("Invalid retrieval schema or duplicate query")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Missing retrieval question")
        seen.add(row["query_id"])
        chunks = set()
        for e in row["candidates"]:
            validate_evidence(e)
            if e["chunk_id"] in chunks:
                raise ValueError("Duplicate candidate chunk_id")
            chunks.add(e["chunk_id"])


def unique(values):
    return list(dict.fromkeys(values))
