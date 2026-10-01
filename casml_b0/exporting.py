"""Generation artifact -> validated CSV + human-readable source audit."""
from __future__ import annotations

import csv
import html
import io
import json
import shutil
from pathlib import Path

from .artifacts import (atomic_text, begin, digest, file_hash, finish, load_artifact,
                        read_jsonl, signature, write_json)
from .contracts import load_queries, unique, validate_evidence
from .context import render_context
from .validation import validate_answer_grounding

COLUMNS = ["ID", "context", "answer", "references"]


def references_for(evidence, config):
    mode = config.get("page_mode", "pdf")
    if mode not in ("pdf", "printed"):
        raise ValueError("page_mode must be pdf or printed")
    pages, sections = [], []
    for e in sorted(evidence, key=lambda x: x["pdf_page"]):
        page = e["pdf_page"] if mode == "pdf" else e["printed_page"]
        if page is None or str(page).strip() == "":
            raise ValueError(f"Printed page is unknown for PDF page {e['pdf_page']}; supply a verified page map")
        if config.get("page_value_type", "string") == "integer":
            page = int(page)
        elif config.get("page_value_type", "string") == "string":
            page = str(page)
        else:
            raise ValueError("page_value_type must be string or integer")
        pages.append(page)
        if e["section_path"]:
            sections.append("/".join(e["section_path"]))
    return {"sections": unique(sections), "pages": unique(pages)}


def audit_html(rows, summary, with_pdf):
    esc = lambda value: html.escape(str(value))
    cards = []
    for row in rows:
        sources = []
        for e in row["evidence"]:
            page = e["pdf_page"]
            link = f'<a href="book.pdf#page={page}">Mở PDF trang {page}</a>' if with_pdf else f"PDF trang {page}"
            sources.append(f'<div class="source"><b>{link}</b> · trang in: {esc(e["printed_page"])}'
                f'<p>Mục: {esc(" / ".join(e["section_path"]) or "Chưa xác định")} '
                f'({esc(e["section_method"])})</p><p>{esc(e["text"])}</p>'
                f'<small>{esc(e["chunk_id"])} · ký tự {e["char_start"]}:{e["char_end"]}</small></div>')
        cards.append(f'<article><h2>{esc(row["query_id"])} — {esc(row["question"])}</h2>'
                     f'<p class="answer">{esc(row["answer"])}</p><details open><summary>Nguồn đã đưa vào LLM</summary>'
                     + "".join(sources) + '</details></article>')
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CASML B0 — Kiểm tra nguồn</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f3f5f8;color:#17253a;margin:0}main{max-width:1050px;margin:40px auto;padding:0 24px}h1{font-size:30px}h2{font-size:20px}.banner,article{background:white;padding:24px;margin:20px 0;border:1px solid #dce3ec;border-radius:12px}.answer{border-left:4px solid #26827d;padding:12px 18px;background:#f1faf8;white-space:pre-wrap}.source{border-top:1px solid #dce3ec;margin-top:16px;padding-top:16px}small{color:#53657b;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}a{color:#145ea8}</style><main>
<h1>CASML B0 — Kiểm tra đáp án và nguồn</h1><div class="banner"><p>Trang PDF luôn tính từ 1. Các trích đoạn dưới đây là context thực tế đã đưa cho model. Kiểm tra nguồn tồn tại không đồng nghĩa với xác nhận mọi ý trong đáp án là đúng.</p>
<p>Mục lấy từ bookmark chỉ là ánh xạ gần đúng theo trang; kiểm tra ranh giới mục trước khi nộp.</p><pre>''' + esc(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></div>' + "".join(cards) + '</main></html>'


def export(run, queries_path, config, out, sample=None, pdf=None):
    run, out = Path(run), Path(out)
    parent = load_artifact(run, "generation")
    queries = load_queries(queries_path)
    if not config.get("allow_demo", False) and (parent.get("generator_backend") == "extractive_demo" or parent.get("retrieval_backend") == "hash_demo"):
        raise ValueError("Demo backends are not competition B0. Use the real model configs, or explicitly allow_demo for a smoke test.")
    predictions = read_jsonl(run / "predictions.jsonl")
    ids = [q["query_id"] for q in queries]
    pred_ids = [p["query_id"] for p in predictions]
    if len(set(pred_ids)) != len(pred_ids) or set(pred_ids) != set(ids):
        raise ValueError("Prediction IDs must exactly match ALL input query IDs; do not export a partial smoke run")
    questions = {q["query_id"]: q["question"] for q in queries}
    headers = COLUMNS
    sample_verified = False
    if sample:
        with open(sample, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if headers != COLUMNS:
                raise ValueError(f"Official sample columns/order must be exactly {COLUMNS}; got {headers}")
            sample_ids = [r["ID"] for r in reader]
        if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != set(ids):
            raise ValueError("Sample IDs differ from query IDs")
        ids = sample_ids
        sample_verified = True
    by_id = {p["query_id"]: p for p in predictions}
    predictions = [by_id[qid] for qid in ids]
    csv_rows, truncated, unsupported_claims = [], [], []
    for p in predictions:
        copy = {k: v for k, v in p.items() if k != "record_sha256"}
        if p.get("record_sha256") != digest(copy):
            raise ValueError("Prediction record checksum mismatch")
        if p["question"] != questions[p["query_id"]] or p["status"] not in ("ok", "insufficient_context"):
            raise ValueError(f"Mismatched question or failed generation: {p['query_id']}")
        if not p["answer"].strip():
            raise ValueError("Empty answer")
        for e in p["evidence"]:
            validate_evidence(e)
            if e["doc_id"] != parent["doc_id"]:
                raise ValueError("Evidence points to a different PDF")
        if p["context"] != render_context(p["evidence"]):
            raise ValueError("CSV context and actual packed evidence differ")
        if p.get("finish_reason") == "length":
            truncated.append(p["query_id"])
        answer_validation = validate_answer_grounding(
            p["answer"], p["evidence"], max_words=int(config.get("max_answer_words", 140)))
        if not answer_validation["valid"]:
            unsupported_claims.append({"query_id": p["query_id"], "validation": answer_validation})
        refs = references_for(p["evidence"], config)
        csv_rows.append({"ID": p["query_id"], "context": p["context"], "answer": p["answer"],
                         "references": json.dumps(refs, ensure_ascii=False)})
    if truncated and config.get("fail_on_truncation", True):
        raise ValueError(f"Length-limited answers: {truncated[:10]}. Inspect {run / 'report.json'} "
                         "(length_limited_details) and predictions.jsonl. Use concise retry instructions "
                         "and repetition controls in a new generation run, reusing retrieval; "
                         "increasing max_new_tokens alone may repeat the same unfinished answer.")
    if unsupported_claims and config.get("fail_on_unsupported_claims", False):
        raise ValueError("Answers fail evidence-token or word-count validation: "
                         + json.dumps(unsupported_claims[:10], ensure_ascii=False))
    if pdf and file_hash(pdf) != parent["doc_id"]:
        raise ValueError("Audit PDF differs from the indexed PDF (SHA256 mismatch)")
    sig = signature("export", config, {"generation_id": parent["artifact_id"], "queries": digest(queries),
                    "sample": file_hash(sample) if sample else None, "with_pdf": bool(pdf)},
                    ["exporting.py", "context.py", "validation.py"])
    manifest, cached = begin(out, sig)
    if cached:
        return manifest
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=headers)
    writer.writeheader()
    writer.writerows(csv_rows)
    atomic_text(out / "submission.csv", buffer.getvalue())
    # Parse the emitted CSV/JSON again to catch quoting or row-shape problems.
    roundtrip = list(csv.DictReader(io.StringIO(buffer.getvalue())))
    if len(roundtrip) != len(csv_rows) or [r["ID"] for r in roundtrip] != ids:
        raise ValueError("CSV round-trip failed")
    for r in roundtrip:
        json.loads(r["references"])
    printed_methods = [e.get("printed_page_method", "unknown")
                       for p in predictions for e in p["evidence"]]
    summary = {"rows": len(csv_rows), "page_mode": config.get("page_mode", "pdf"),
               "generator_backend": parent["generator_backend"], "retrieval_backend": parent.get("retrieval_backend"),
               "source_pdf_sha256": parent["doc_id"], "generation_id": parent["artifact_id"],
               "length_limited_queries": truncated,
               "unsupported_claims": unsupported_claims,
               "approximate_section_evidence": sum(e["section_method"] == "toc_page_approximation" for p in predictions for e in p["evidence"]),
               "unknown_section_evidence": sum(not e["section_path"] for p in predictions for e in p["evidence"]),
               "printed_page_methods": {method: printed_methods.count(method) for method in sorted(set(printed_methods))},
               "schema_contract_verified": headers == COLUMNS,
               "official_sample_verified": sample_verified,
               "official_metric_verified": False,
               "note": (("Schema and ID order were verified against the supplied official sample. "
                         if sample_verified else
                         "No sample file was supplied; the documented ID, context, answer, references contract "
                         "and input query order were used. ")
                        + "The hidden competition metric cannot be reproduced locally.")}
    write_json(out / "validation.json", summary)
    atomic_text(out / "audit.html", audit_html(predictions, summary, bool(pdf)))
    files = ["submission.csv", "validation.json", "audit.html"]
    if pdf:
        shutil.copyfile(pdf, out / "book.pdf")
        files.append("book.pdf")
    return finish(out, manifest, files, row_count=len(csv_rows))
