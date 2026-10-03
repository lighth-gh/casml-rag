"""Reviewed, source-backed training data. No model judge can mark a row approved."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .artifacts import (begin, digest, file_hash, finish, load_artifact, read_jsonl,
                        signature, write_json, write_jsonl)
from .context import render_context
from .contracts import validate_evidence


def normalized(text):
    return " ".join(text.casefold().split())


def source_group(chunk):
    section = chunk.get("section_path") or [f"page-block-{(chunk['pdf_page'] - 1) // 5}"]
    return digest([chunk["doc_id"], section])


def review_digest(row):
    """Changing an answer, split, source or fact invalidates the review stamp."""
    keys = ("query_id", "question", "answer", "split", "split_group", "question_type",
            "evidence_ids", "required_facts", "answer_claims", "reference_sets", "source_corpus_id")
    return digest({key: row.get(key) for key in keys})


def validate_annotations(rows, chunks, minimums):
    by_chunk = {c["chunk_id"]: c for c in chunks}
    if len(by_chunk) != len(chunks):
        raise ValueError("Duplicate corpus chunk IDs")
    for chunk in chunks:
        validate_evidence(chunk)
    result = {name: [] for name in ("train", "dev", "holdout")}
    seen_ids, questions, assignments = set(), set(), {}
    group_members = {}
    skipped = Counter()

    def assign(key, split):
        if key in assignments and assignments[key] != split:
            raise ValueError(f"Split leakage: {key} occurs in {assignments[key]} and {split}")
        assignments[key] = split
        group_members.setdefault(key, set()).add(query_id)

    def supports(items):
        if not isinstance(items, list) or not items:
            raise ValueError("Every fact/claim needs source supports")
        ids = set()
        for item in items:
            chunk = by_chunk.get(item.get("chunk_id"))
            quote = item.get("quote")
            if chunk is None or not isinstance(quote, str) or not quote.strip() or quote not in chunk["text"]:
                raise ValueError("Source quote must occur exactly in a verified corpus chunk")
            ids.add(chunk["chunk_id"])
        return ids

    for row in rows:
        if row.get("review_status") != "approved":
            skipped[str(row.get("review_status", "missing"))] += 1
            continue
        if not str(row.get("reviewer", "")).strip() or row.get("review_sha256") != review_digest(row):
            raise ValueError("Approved annotation needs reviewer and a current review_sha256")
        split = row.get("split")
        if split not in result:
            raise ValueError("split must be train/dev/holdout")
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in seen_ids:
            raise ValueError("Annotation query_id must be a unique non-empty string")
        for key in ("question", "answer", "split_group", "question_type"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing annotation {key}")
        qkey = normalized(row["question"])
        if qkey in questions:
            raise ValueError("Duplicate normalized question")
        seen_ids.add(query_id)
        questions.add(qkey)
        evidence_ids = row.get("evidence_ids")
        if (not isinstance(evidence_ids, list) or not evidence_ids or
                len(set(evidence_ids)) != len(evidence_ids) or any(i not in by_chunk for i in evidence_ids)):
            raise ValueError("evidence_ids must identify distinct corpus chunks in prompt order")
        facts, claims = row.get("required_facts"), row.get("answer_claims")
        if not isinstance(facts, list) or not facts or not isinstance(claims, list) or not claims:
            raise ValueError("Approved data requires required_facts and answer_claims")
        if normalized(" ".join(c.get("text", "") for c in claims)) != normalized(row["answer"]):
            raise ValueError("answer_claims must cover the entire answer in order")
        fact_ids, fact_sources = set(), []
        all_sources = set(evidence_ids)
        for fact in facts:
            fact_id = fact.get("fact_id")
            if not isinstance(fact_id, str) or not fact_id or fact_id in fact_ids or not fact.get("text", "").strip():
                raise ValueError("Each required fact needs unique fact_id and text")
            fact_ids.add(fact_id)
            supported = supports(fact.get("supports"))
            fact_sources.append(supported)
            all_sources |= supported
            assign("fact:" + normalized(fact["text"]), split)
        for claim in claims:
            if not claim.get("text", "").strip():
                raise ValueError("Empty answer claim")
            if not supports(claim.get("supports")) <= set(evidence_ids):
                raise ValueError("Answer label uses evidence absent from its actual context")
        alternatives = row.get("reference_sets")
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError("reference_sets must list valid alternative sets of chunk IDs")
        gold_refs = []
        for ids in alternatives:
            if not isinstance(ids, list) or not ids or any(i not in by_chunk for i in ids):
                raise ValueError("Invalid reference alternative")
            if not all(set(ids) & support for support in fact_sources):
                raise ValueError("Each reference alternative must cover every required fact")
            all_sources.update(ids)
            gold_refs.append(references([by_chunk[i] for i in ids]))
        assign("group:" + row["split_group"], split)
        for chunk_id in all_sources:
            chunk = by_chunk[chunk_id]
            assign("source:" + source_group(chunk), split)
            assign("page:" + digest([chunk["doc_id"], chunk["pdf_page"]]), split)
            assign("text:" + normalized(chunk["text"]), split)
        evidence = [by_chunk[i] for i in evidence_ids]
        result[split].append({**row, "context": render_context(evidence), "evidence": evidence,
                              "oracle_evidence": [by_chunk[i] for i in alternatives[0]],
                              "evidence_coverage": sum(bool(set(evidence_ids) & ids) for ids in fact_sources) / len(fact_sources),
                              "gold_reference_sets": gold_refs, "label_source": "human_reviewed_source_grounded",
                              "annotation_sha256": digest(row)})
    # Bootstrap connected source/topic families, not arbitrary per-question IDs.
    leaders = {i: i for i in seen_ids}

    def leader(i):
        while leaders[i] != i:
            leaders[i] = leaders[leaders[i]]
            i = leaders[i]
        return i

    for members in group_members.values():
        members = sorted(members)
        for i in members[1:]:
            leaders[leader(i)] = leader(members[0])
    components = {}
    for i in sorted(seen_ids):
        components.setdefault(leader(i), []).append(i)
    for rows_in_split in result.values():
        for row in rows_in_split:
            row["bootstrap_group"] = digest(components[leader(row["query_id"])])
    for split, minimum in minimums.items():
        if len(result[split]) < int(minimum):
            raise ValueError(f"Need {minimum} approved {split} examples; found {len(result[split])}. "
                             "Synthetic drafts are not gold; complete review first.")
    return result, {"skipped": dict(skipped), "groups": assignments}


def references(evidence):
    pages, sections = set(), set()
    for e in evidence:
        page = e.get("printed_page")
        if page is None or not str(page).isdigit() or int(page) < 1 or not e.get("section_path"):
            raise ValueError("Reviewed references require verified positive printed pages and section paths")
        pages.add(int(page))
        sections.add("/".join(e["section_path"]))
    return {"pages": sorted(pages), "sections": sorted(sections)}


def prepare_reviewed(corpus_dir, annotations, config, out):
    parent = load_artifact(corpus_dir, "corpus")
    minimums = {name: config.get("min_" + name + "_examples", default)
                for name, default in (("train", 200), ("dev", 80), ("holdout", 80))}
    if any(int(v) < 1 for v in minimums.values()):
        raise ValueError("All reviewed splits must be non-empty")
    rows = read_jsonl(annotations)
    if any(r.get("review_status") == "approved" and r.get("source_corpus_id") != parent["artifact_id"] for r in rows):
        raise ValueError("Approved annotation source_corpus_id must match the reviewed corpus version")
    splits, report = validate_annotations(rows,
                                          read_jsonl(Path(corpus_dir) / "chunks.jsonl"), minimums)
    sig = signature("reviewed_sft", config, {"corpus_id": parent["artifact_id"],
                    "annotations_sha256": file_hash(annotations)}, ["reviewed.py", "context.py"])
    manifest, cached = begin(out, sig)
    if cached:
        return manifest
    out = Path(out)
    files = []
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)
        files.append(f"{name}.jsonl")
    write_json(out / "split_manifest.json", {"counts": {k: len(v) for k, v in splits.items()}, **report,
               "official_ground_truth": False, "review_type": "human_attested; source quotes checked by code"})
    return finish(out, manifest, files + ["split_manifest.json"])


def load_split(directory, split):
    if split not in ("train", "dev", "holdout"):
        raise ValueError("Unknown split")
    manifest = load_artifact(directory, "reviewed_sft")
    return manifest, read_jsonl(Path(directory) / f"{split}.jsonl")


def attach_context(drafts_path, retrieval_dir, config, config_path, out, _backend=None):
    """Freeze production-packed evidence BEFORE humans review the annotations."""
    from .context import pack_context
    from .generation import load_prompts
    parent = load_artifact(retrieval_dir, "retrieval")
    drafts = read_jsonl(drafts_path)
    retrieved = read_jsonl(Path(retrieval_dir) / "retrieval.jsonl")
    by_id = {str(r["query_id"]): r for r in retrieved}
    if (len(by_id) != len(retrieved) or len({r["query_id"] for r in drafts}) != len(drafts)
            or set(by_id) != {str(r["query_id"]) for r in drafts}):
        raise ValueError("Retrieve all draft question IDs exactly once; never use competition queries here")
    system, user = load_prompts(config, config_path)
    sig = signature("review_context", {**config, "system": system, "user": user},
                    {"retrieval_id": parent["artifact_id"], "drafts_sha256": file_hash(drafts_path)},
                    ["reviewed.py", "context.py"])
    manifest, cached = begin(out, sig)
    if cached:
        return manifest
    if _backend is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=config.get("revision", "main"),
            local_files_only=bool(config.get("local_files_only", False)), trust_remote_code=False)

        class TokenCounter:
            context_window = int(config["context_window"])

            def count_text(self, text):
                return len(tokenizer.encode(text, add_special_tokens=False))

            def count_messages(self, messages):
                return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

        _backend = TokenCounter()
    output = []
    for draft in drafts:
        row = by_id[str(draft["query_id"])]
        if row["question"] != draft["question"]:
            raise ValueError("Retrieval question differs from draft")
        packed = pack_context(row, _backend, config, system, user)
        if not packed["evidence"]:
            raise ValueError("No retrieved evidence for draft")
        output.append({**draft, "context": packed["context"],
                       "evidence_ids": [c["chunk_id"] for c in packed["evidence"]],
                       "review_status": "pending", "reviewer": "", "review_sha256": None})
    write_jsonl(Path(out) / "drafts.jsonl", output)
    write_json(Path(out) / "report.json", {"drafts": len(output), "status": "awaiting_human_review",
               "note": "Context changes invalidate approval. Check missing facts and cross-split source overlap."})
    return finish(out, manifest, ["drafts.jsonl", "report.json"])
