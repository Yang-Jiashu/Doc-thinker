"""Build exact, offset-addressable evidence packages from edge source chunks."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from .graph_view import FactGraphView
from .models import ECLRRConfig, EvidencePackage, EvidenceRef, ReviewItem

_SENTENCE_BREAK = re.compile(r"[^。！？!?\n]+[。！？!?]?|[^\n]+$")


def _aliases(node_name: str, node: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = [node_name]
    raw = node.get("aliases") or node.get("alias") or []
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            raw = (
                decoded
                if isinstance(decoded, list)
                else re.split(r"(?:<SEP>|[,，;|])", raw)
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            raw = re.split(r"[,，;|]", raw)
    if isinstance(raw, (list, tuple, set)):
        values.extend(str(item).strip() for item in raw if str(item).strip())
    return tuple(dict.fromkeys(item for item in values if item))


def _contains(text: str, term: str) -> bool:
    return term.casefold() in text.casefold()


def _quote_candidates(
    text: str,
    source_aliases: Iterable[str],
    target_aliases: Iterable[str],
    relation: str,
    *,
    context_chars: int,
) -> list[tuple[int, int, str, str, int]]:
    source_aliases = tuple(source_aliases)
    target_aliases = tuple(target_aliases)
    relation_terms = tuple(
        term for term in re.split(r"[\s,，;/_|]+", relation) if len(term) >= 2
    )
    ranked: list[tuple[int, int, str, str, int]] = []
    for match in _SENTENCE_BREAK.finditer(text):
        quote = match.group(0).strip()
        if not quote:
            continue
        start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        end = start + len(quote)
        has_source = any(_contains(quote, alias) for alias in source_aliases)
        has_target = any(_contains(quote, alias) for alias in target_aliases)
        has_relation = any(_contains(quote, term) for term in relation_terms)
        if not (has_source or has_target):
            continue
        rank = (
            4 * int(has_source and has_target)
            + 2 * int(has_relation)
            + int(has_source or has_target)
        )
        if len(quote) > 280:
            anchor_terms = [
                alias
                for alias in (*source_aliases, *target_aliases)
                if _contains(quote, alias)
            ]
            anchor = (
                quote.casefold().find(anchor_terms[0].casefold()) if anchor_terms else 0
            )
            local_start = max(0, anchor - 100)
            local_end = min(len(quote), local_start + 240)
            start += local_start
            end = start + (local_end - local_start)
            quote = text[start:end]
        context_start = max(0, start - context_chars // 2)
        context_end = min(len(text), end + context_chars // 2)
        ranked.append((start, end, quote, text[context_start:context_end], rank))
    ranked.sort(key=lambda item: (-item[4], item[0], item[1]))
    return ranked


async def _load_chunks(text_chunks: Any, chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    records = await text_chunks.get_by_ids(chunk_ids)
    result: dict[str, str] = {}
    for index, record in enumerate(records or []):
        if not isinstance(record, dict):
            continue
        chunk_id = str(record.get("chunk_id") or record.get("id") or "").strip()
        if not chunk_id and index < len(chunk_ids):
            chunk_id = chunk_ids[index]
        content = record.get("content")
        if chunk_id and isinstance(content, str):
            result[chunk_id] = content
    return result


async def build_evidence_package(
    item: ReviewItem,
    view: FactGraphView,
    text_chunks: Any,
    config: ECLRRConfig,
) -> tuple[EvidencePackage | None, str]:
    ordered_ids = list(
        dict.fromkeys(
            chunk_id for step in item.primary_path.steps for chunk_id in step.source_ids
        )
    )
    chunk_text = await _load_chunks(text_chunks, ordered_ids)
    primary: list[EvidenceRef] = []
    alternates: list[EvidenceRef] = []

    for hop_index, step in enumerate(item.primary_path.steps):
        source_aliases = _aliases(step.source, view.nodes[step.source])
        target_aliases = _aliases(step.target, view.nodes[step.target])
        candidates: list[tuple[int, int, str, str, int, str]] = []
        for chunk_id in step.source_ids:
            text = chunk_text.get(chunk_id)
            if text is None:
                continue
            for start, end, quote, context, rank in _quote_candidates(
                text,
                source_aliases,
                target_aliases,
                step.relation,
                context_chars=config.context_chars,
            ):
                candidates.append((start, end, quote, context, rank, chunk_id))
        candidates.sort(key=lambda row: (-row[4], row[5], row[0], row[1]))
        if not candidates:
            return None, f"missing_primary_evidence:hop_{hop_index}"
        for candidate_index, (start, end, quote, context, _rank, chunk_id) in enumerate(
            candidates[: 1 + config.alternate_evidence_per_hop]
        ):
            digest = hashlib.md5(
                f"{item.review_id}|{hop_index}|{chunk_id}|{start}|{end}".encode("utf-8")
            ).hexdigest()[:16]
            evidence = EvidenceRef(
                evidence_id=f"ev-{digest}",
                hop_index=hop_index,
                edge_id=step.edge_id,
                source=step.source,
                target=step.target,
                chunk_id=chunk_id,
                quote=quote,
                start=start,
                end=end,
                context=context,
            )
            if candidate_index == 0:
                primary.append(evidence)
            else:
                alternates.append(evidence)

    return (
        EvidencePackage(
            review_item=item,
            primary_evidence=primary,
            alternate_evidence=alternates,
            node_types={name: view.node_type(name) for name in item.primary_path.nodes},
        ),
        "ok",
    )


async def build_evidence_packages(
    items: list[ReviewItem],
    view: FactGraphView,
    text_chunks: Any,
    config: ECLRRConfig,
) -> tuple[list[EvidencePackage], dict[str, str]]:
    packages: list[EvidencePackage] = []
    rejected: dict[str, str] = {}
    for item in items:
        package, reason = await build_evidence_package(item, view, text_chunks, config)
        if package is None:
            rejected[item.review_id] = reason
        else:
            packages.append(package)
    return packages, rejected
