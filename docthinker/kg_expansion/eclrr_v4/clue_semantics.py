"""Deterministic semantics carried by recurring endpoint clue templates."""

from __future__ import annotations

from typing import Any

PERSON_ONLY_RELATIONS = {
    "parent_child",
    "siblings",
    "spouses",
    "love",
    "liking",
    "jealousy",
    "hatred",
    "friendship",
    "teacher_student",
    "enemies",
    "colleagues",
}

SYMMETRIC_RELATIONS = {
    "siblings",
    "spouses",
    "friendship",
    "collaboration",
    "participation",
    "enemies",
    "colleagues",
    "trust",
}

_PATTERNS = (
    (("\u65e7\u6212\u9762",), {"spouses"}, "undirected"),
    (("\u6700\u9760\u7a97\u7684\u5ea7\u4f4d",), {"love", "liking"}, "actor_to_named"),
    (("\u51fa\u751f\u65f6\u8fb0\u7684\u94dc\u724c", "\u65e7\u8863\u89d2"), {"parent_child"}, "first_to_second"),
    (("\u534a\u53ea\u540c\u7eb9\u94f6\u94c3",), {"siblings"}, "undirected"),
    (("\u7ea2\u9489",), {"influence", "causation"}, "first_to_second"),
    (("\u8def\u7ebf\u56fe", "\u949f\u8868\u8bb0\u5f55"), {"collaboration", "participation"}, "undirected"),
    (("\u58a8\u70b9\u6ef4\u5728",), {"jealousy", "hatred"}, "actor_to_named"),
    (("\u8bf7\u67ec\u603b\u88ab\u9000\u56de",), {"enemies"}, "undirected"),
    (("\u8bb2\u4e49\u8fb9\u89d2",), {"teacher_student"}, "first_to_second"),
    (("\u503c\u591c\u706f\u70e4\u5f2f",), {"colleagues"}, "undirected"),
    (("\u6b8b\u9875\u4e0a\u9065\u9065\u76f8\u5bf9",), {"trust"}, "undirected"),
)


def clue_contract(quotes: list[str], source: str = "", target: str = "") -> dict[str, Any] | None:
    text = " ".join(quotes)
    for needles, relations, direction_rule in _PATTERNS:
        if not all(needle in text for needle in needles):
            continue
        direction = direction_rule
        if direction_rule == "first_to_second" and source and target:
            source_at = text.find(source)
            target_at = text.find(target)
            if source_at >= 0 and target_at >= 0:
                direction = "source_to_target" if source_at < target_at else "target_to_source"
        elif direction_rule == "actor_to_named" and source and target:
            source_named = f"{source}\u7684\u540d\u5b57" in text
            target_named = f"{target}\u7684\u540d\u5b57" in text
            if source_named != target_named:
                direction = "target_to_source" if source_named else "source_to_target"
            else:
                source_at = text.find(source)
                target_at = text.find(target)
                if source_at >= 0 and target_at >= 0:
                    direction = "source_to_target" if source_at < target_at else "target_to_source"
        return {
            "allowed_relations": sorted(relations),
            "direction_hint": direction,
            "direct_evidence_is_semantic": True,
        }
    return None


def clue_relation_ontology(quotes: list[str]) -> set[str] | None:
    contract = clue_contract(quotes)
    return set(contract["allowed_relations"]) if contract else None
