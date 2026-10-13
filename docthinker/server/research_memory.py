"""Query-aware selection of explicit project memory, with no model or I/O.

Relevance is not truth or confidence. Saved findings/decisions remain user context,
not independent document evidence. Complete notes are preserved so negations and
conditions are never silently changed by truncation; the prompt layer may omit a
whole note later when its separate UTF-8 input budget is exhausted.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import re

from .research_retrieval import lexical_tokens

MAX_INPUT_MEMORIES = 100
MAX_SELECTED_MEMORIES = 20
MAX_NOTE_CHARS = 2_000
MAX_QUESTION_BYTES = 8_192
MAX_QUERY_TERMS = 64
MAX_DEFAULT_PREFERENCES = 2
_KINDS = {"preference", "decision", "finding"}
_PREFERENCE_QUERY = re.compile(
    r"我的偏好|表达偏好|回答偏好|偏好设置|你记得我的偏好|我喜欢什么|"
    r"\b(?:my preferences?|our preferences?|saved preferences?|preferred (?:language|style))\b",
    re.I,
)
_PROJECT_CONTEXT_QUERY = re.compile(
    r"^\s*(?:下一步|接下来|然后|项目进展|研究进展|记忆)[？?呢。！!\s]*$|"
    r"项目进展|研究进展|项目状态|下一步(?:做什么|怎么做|计划|是什么)|接下来(?:做什么|怎么做)|"
    r"上次(?:确定|决定|讨论)|之前(?:确定|决定)|你(?:还)?记得什么|有哪些(?:确认)?记忆|"
    r"已确认的(?:结论|决定|记忆)|我的记忆|"
    r"\b(?:project (?:progress|status)|next steps?|our decisions?|"
    r"what (?:did we decide|have we learned|do you remember))\b",
    re.I,
)


def _finite_number(value, *, default=0.0):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if math.isfinite(value):
                return float(value)
        except (OverflowError, ValueError):
            pass
    return default


def classify_memory_query(question: str) -> str:
    """Share narrow memory intent with the no-evidence gate; no model classifier."""
    if not isinstance(question, str):
        return "lexical_bm25_with_preferences"
    if _PREFERENCE_QUERY.search(question):
        return "explicit_preferences"
    if _PROJECT_CONTEXT_QUERY.search(question):
        return "recent_project_context"
    return "lexical_bm25_with_preferences"


def _eligible(rows: list[dict]) -> list[dict]:
    valid = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") not in {"active", "archived"}:
            continue
        if row.get("confirmed_by_user") is not True and row.get("confirmed") is not True:
            continue
        identifier, content, kind = row.get("id", row.get("memory_id")), row.get("text"), row.get("kind")
        if (not isinstance(identifier, str) or not identifier or len(identifier) > 128
                or not isinstance(content, str) or not content.strip()
                or len(content) > MAX_NOTE_CHARS or "\x00" in content
                or not isinstance(kind, str) or kind not in _KINDS):
            continue
        try:
            content.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            continue
        valid.append({"row": dict(row), "id": identifier, "kind": kind, "content": content,
                      "updated_at": _finite_number(row.get("updated_at")),
                      "revision": _finite_number(row.get("revision")),
                      "digest": hashlib.sha256((kind + "\n" + content).encode()).hexdigest()})
    # Defensive deduplication must not amplify a note's importance or confidence.
    # The newest current revision wins; the final digest breaks corrupt equal-ID
    # snapshots deterministically, never by the input iteration order.
    valid.sort(key=lambda item: (-item["updated_at"], -item["revision"], item["id"], item["digest"]))
    seen, unique = set(), []
    for item in valid:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)
    return unique


def select_confirmed_memories(question: str, rows: list[dict], *, max_candidates: int = MAX_SELECTED_MEMORIES) -> dict:
    """Rerank up to 20 whole notes from 100 indexed retrieval candidates.

    Ordinary research questions get relevant findings/decisions and at most two
    preferences. Unrelated findings are not added merely because they are recent.
    Only an explicit project-progress/memory question permits recent zero-overlap
    decisions/findings as background. A preference question may review more than
    two preferences. This function does not validate source entailment or infer
    preferences; the caller must already scope its input to one private project
    and the requested validity time. The indexed store searches the full retained
    lifetime before this bounded reranker runs; 100 is not a storage capacity.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty memory-selection question is required")
    try:
        if len(question.encode("utf-8", errors="strict")) > MAX_QUESTION_BYTES:
            raise ValueError("Memory-selection question exceeds its budget")
    except UnicodeEncodeError:
        raise ValueError("Memory-selection question must be valid UTF-8") from None
    if not isinstance(rows, list) or len(rows) > MAX_INPUT_MEMORIES:
        raise ValueError("Memory selection requires at most 100 project records")
    if type(max_candidates) is not int or not 1 <= max_candidates <= MAX_SELECTED_MEMORIES:
        raise ValueError("Memory selection allows 1 to 20 candidates")
    candidates = _eligible(rows)
    query = list(dict.fromkeys(lexical_tokens(question)))[:MAX_QUERY_TERMS]
    counters = [Counter(lexical_tokens(item["content"])) for item in candidates]
    count = len(candidates)
    average_length = max(1.0, sum(sum(counter.values()) for counter in counters) / max(1, count))
    df = {term: sum(bool(counter[term]) for counter in counters) for term in query}
    idf = {term: math.log(1.0 + (count - df[term] + 0.5) / (df[term] + 0.5)) for term in query}
    for item, counter in zip(candidates, counters):
        length = max(1, sum(counter.values()))
        matches = [term for term in query if counter[term]]
        score = sum(idf[term] * counter[term] * 2.2 /
                    (counter[term] + 1.2 * (0.25 + 0.75 * length / average_length))
                    for term in matches)
        # Coverage rewards connecting several requested concepts, not repetition
        # of a single term. Scores stay internal and are NOT memory confidence.
        score += 2.0 * len(matches) / max(1, len(query))
        item.update({"score": score, "matches": len(matches)})

    def rank(item):
        return (-item["score"], -item["updated_at"], -item["revision"], item["id"])

    preferences = sorted((item for item in candidates if item["kind"] == "preference"), key=rank)
    notes = sorted((item for item in candidates if item["kind"] != "preference"), key=rank)
    relevant = [item for item in notes if item["matches"]]
    query_strategy = classify_memory_query(question)
    explicit_preferences = query_strategy == "explicit_preferences"
    project_context = query_strategy == "recent_project_context"
    if explicit_preferences:
        selected = preferences[:max_candidates]
        selected.extend(relevant[:max_candidates - len(selected)])
        strategy = "explicit_preferences"
    else:
        if project_context:
            # Matching old decisions still beat newer unrelated decisions; recent
            # zero-overlap notes are included only because the user requested it.
            note_pool = relevant + sorted((item for item in notes if not item["matches"]),
                                          key=lambda item: (-item["updated_at"], -item["revision"], item["id"]))
            strategy = "recent_project_context"
        else:
            note_pool = relevant
            strategy = "lexical_bm25_with_preferences"
        preference_count = min(MAX_DEFAULT_PREFERENCES, len(preferences), max_candidates)
        # Even under a one-item caller cap, a relevant finding cannot be replaced
        # entirely by style preferences. Otherwise reserve at most two hints.
        if note_pool:
            preference_count = min(preference_count, max_candidates - 1)
        selected = note_pool[:max_candidates - preference_count]
        selected.extend(preferences[:preference_count])
    return {"memories": [dict(item["row"]) for item in selected],
            "omitted_count": len(candidates) - len(selected), "strategy": strategy}
