"""Candidate cache -> exact context. This belongs to GENERATION, not retrieval."""
from __future__ import annotations


def render_context(evidence):
    parts = []
    for n, item in enumerate(evidence, 1):
        section = "/".join(item["section_path"]) or "section unavailable"
        parts.append(f"[E{n}] PDF page {item['pdf_page']} | {section}\n{item['text']}")
    return "\n\n".join(parts)


def messages_for(system, user_template, question, evidence):
    return [{"role": "system", "content": system},
            {"role": "user", "content": user_template.format(question=question, context=render_context(evidence))}]


def overlaps(candidate, selected, threshold):
    for other in selected:
        if candidate["text_sha256"] == other["text_sha256"]:
            return True
        if candidate["doc_id"] == other["doc_id"] and candidate["pdf_page"] == other["pdf_page"]:
            shared = max(0, min(candidate["char_end"], other["char_end"]) -
                         max(candidate["char_start"], other["char_start"]))
            length = min(candidate["char_end"] - candidate["char_start"],
                         other["char_end"] - other["char_start"])
            if shared / length >= threshold:
                return True
    return False


def pack_context(row, backend, config, system, user_template):
    max_new = int(config["max_new_tokens"])
    window = min(int(config["context_window"]), backend.context_window)
    budget = window - max_new
    limit = int(config["context_top_k"])
    context_budget = int(config["context_token_budget"])
    if max_new < 1 or limit < 1 or budget < 1 or context_budget < 1:
        raise ValueError("Invalid generation token budget")
    empty = messages_for(system, user_template, row["question"], [])
    if backend.count_messages(empty) > budget:
        raise ValueError("Question + instructions do not fit; increase context_window. Question is NEVER truncated.")
    evidence = []
    for candidate in row["candidates"]:
        if overlaps(candidate, evidence, float(config.get("duplicate_overlap", 0.75))):
            continue
        trial = evidence + [candidate]
        if backend.count_text(render_context(trial)) > context_budget:
            continue
        messages = messages_for(system, user_template, row["question"], trial)
        if backend.count_messages(messages) > budget:
            continue
        evidence = trial
        if len(evidence) == limit:
            break
    if row["candidates"] and not evidence:
        raise ValueError("No complete chunk fits the context budget. Increase budget or rebuild smaller chunks.")
    messages = messages_for(system, user_template, row["question"], evidence)
    return {"evidence": evidence, "context": render_context(evidence), "messages": messages,
            "input_tokens": backend.count_messages(messages),
            "context_tokens": backend.count_text(render_context(evidence)),
            "available_candidates": len(row["candidates"])}
