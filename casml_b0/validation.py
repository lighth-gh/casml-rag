"""Deterministic grounding checks for generated answers.

The validator is intentionally narrow: it catches numeric values, years and
proper-name-like tokens that are absent from the evidence text. It does not
claim to verify that a supported token is used in the correct relationship.
"""
from __future__ import annotations

import re
import unicodedata


_NUMBER_RE = re.compile(r"(?<![\w])\d+(?:,\d{3})*(?:\.\d+)?(?:st|nd|rd|th|'?s)?%?(?![\w])", re.IGNORECASE)
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'’.-]*\b")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
_BOLD_LABEL_RE = re.compile(r"(?m)^\s*(?:[-*]\s*)?\*\*[^*\n]+\*\*\s*:\s*")
_CONNECTORS = {"and", "of", "the", "de", "del", "van", "von", "la"}
_NAME_CONTEXT_WORDS = {
    "argued", "believed", "created", "developed", "discovered", "established", "introduced",
    "proposed", "published", "said", "suggested", "theorized", "was", "were",
}
_NON_NAME_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "they", "he", "she", "we", "i",
    "in", "on", "at", "by", "for", "from", "to", "with", "without", "during", "after", "before",
    "however", "therefore", "thus", "additionally", "moreover", "overall", "finally", "first", "second",
    "third", "also", "for example", "for instance", "according", "summary", "nature", "duration",
    "etiology", "symptoms", "impact", "classification", "key", "examples",
}


def _normalized_words(text):
    text = unicodedata.normalize("NFKC", str(text)).replace("’", "'").casefold()
    text = re.sub(r"\b(?:[a-z]\.){2,}", lambda match: match.group(0).replace(".", ""), text)
    text = re.sub(r"\b(?:[a-z]\.)+[a-z]\b", lambda match: match.group(0).replace(".", ""), text)
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
    return [word[:-2] if word.endswith("'s") and len(word) > 2 else word for word in words]


def _canonical_number(value):
    value = value.casefold().replace(",", "")
    value = re.sub(r"(st|nd|rd|th|'?s)(?=%?$)", "", value)
    suffix = "%" if value.endswith("%") else ""
    value = value.removesuffix("%")
    try:
        number = float(value)
        value = str(int(number)) if number.is_integer() else format(number, "g")
    except ValueError:
        pass
    return value + suffix


def _numbers(text):
    # List numbering is formatting, not a factual numeric claim.
    text = _LIST_MARKER_RE.sub("", str(text))
    return [(match.group(0), _canonical_number(match.group(0))) for match in _NUMBER_RE.finditer(text)]


def _is_year(number):
    plain = number.removesuffix("%")
    return plain.isdigit() and 1000 <= int(plain) <= 2100


def _is_proper_word(token):
    bare = token.rstrip(".'’")
    return (len(bare) >= 2 and bare[0].isupper() and
            (bare[1:].islower() or any(char.isupper() for char in bare[1:])))


def _is_sentence_start(text, start):
    prefix = re.sub(r"[*_`>#-]+$", "", text[:start].rstrip()).rstrip()
    return not prefix or prefix[-1] in ".!?\n:"


def _proper_name_candidates(answer):
    # Markdown list labels such as "**Duration**:" are semantic headings, not names.
    text = _BOLD_LABEL_RE.sub("", _LIST_MARKER_RE.sub("", str(answer)))
    matches = list(_WORD_RE.finditer(text))
    candidates = []
    index = 0
    while index < len(matches):
        match = matches[index]
        token = match.group(0).rstrip(".'’")
        if not _is_proper_word(token):
            index += 1
            continue
        sequence = [match]
        cursor = index + 1
        while cursor < len(matches):
            previous, current = matches[cursor - 1], matches[cursor]
            between = text[previous.end():current.start()]
            current_token = current.group(0).rstrip(".'’")
            if not between.isspace():
                break
            if current_token.casefold() in _CONNECTORS or _is_proper_word(current_token):
                sequence.append(current)
                cursor += 1
                continue
            break
        proper = [m.group(0).rstrip(".'’") for m in sequence if _is_proper_word(m.group(0).rstrip(".'’"))]
        original = text[sequence[0].start():sequence[-1].end()].strip()
        camel_or_acronym = any(tok.isupper() or any(char.isupper() for char in tok[1:]) for tok in proper)
        meaningful = [tok for tok in proper if tok.casefold() not in _NON_NAME_WORDS]
        next_word = matches[cursor].group(0).casefold() if cursor < len(matches) else ""
        name_context = (any(tok.casefold().endswith("'s") for tok in meaningful) or
                        next_word in _NAME_CONTEXT_WORDS)
        if len(meaningful) >= 2:
            candidates.append((original, meaningful))
        elif meaningful and (camel_or_acronym or name_context or
                             not _is_sentence_start(text, sequence[0].start())):
            candidates.append((original, meaningful))
        index = max(cursor, index + 1)
    return candidates


def validate_answer_grounding(answer, evidence):
    """Return unsupported numbers/years/proper names relative to evidence text."""
    evidence_text = "\n".join(str(item.get("text", "")) for item in evidence)
    evidence_numbers = {canonical for _, canonical in _numbers(evidence_text)}
    unsupported_numbers, unsupported_years = [], []
    for original, canonical in _numbers(answer):
        if canonical in evidence_numbers:
            continue
        target = unsupported_years if _is_year(canonical) else unsupported_numbers
        if original not in target:
            target.append(original)

    evidence_words = set(_normalized_words(evidence_text))
    def supported_word(word):
        return word in evidence_words or (len(word) >= 4 and any(word in source for source in evidence_words))

    unsupported_names = []
    for original, core_tokens in _proper_name_candidates(answer):
        normalized = [word for token in core_tokens for word in _normalized_words(token)]
        if normalized and any(not supported_word(word) for word in normalized):
            if original not in unsupported_names:
                unsupported_names.append(original)

    valid = not unsupported_numbers and not unsupported_years and not unsupported_names
    return {
        "valid": valid,
        "unsupported_numbers": unsupported_numbers,
        "unsupported_years": unsupported_years,
        "unsupported_proper_names": unsupported_names,
        "scope": "Evidence-token support only; relationships and general factual accuracy are not verified.",
    }


def grounding_retry_instruction(validation, base_instruction=""):
    blocked = (validation["unsupported_numbers"] + validation["unsupported_years"] +
               validation["unsupported_proper_names"])
    details = ", ".join(repr(value) for value in blocked[:20])
    prefix = (str(base_instruction).strip() + "\n") if str(base_instruction).strip() else ""
    return (prefix +
            "Rewrite the answer from scratch. Every number, year, and proper name in the answer must appear "
            "in the supplied textbook excerpts. Omit unsupported details, do not infer missing dates or names, "
            "and stay under 140 words. Do not output these unsupported items: " + details)
