"""Deterministic relation-aware beam search for evidence-carrying paths."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from .graph_view import FactGraphView
from .models import ECLRRConfig, ReviewItem, SearchPath


def _path_score(view: FactGraphView, path: SearchPath) -> float:
    qualities = []
    chunk_ids: set[str] = set()
    source_sets: list[set[str]] = []
    hub_penalty = 0.0
    conflict_penalty = 0.0
    person_bonus = 0.0
    for step in path.steps:
        edge = view.fact_edges[step.edge_id]
        qualities.append(0.55 * edge.grounding + 0.45 * edge.specificity)
        source_set = set(edge.source_ids)
        source_sets.append(source_set)
        chunk_ids.update(source_set)
        raw = edge.raw
        if str(raw.get("temporal_conflict") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            conflict_penalty += 0.6
        if str(raw.get("contradiction") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            conflict_penalty += 0.6
    for node in path.nodes[1:-1]:
        degree = view.degree(node)
        if degree > 8:
            hub_penalty += min(0.35, (degree - 8) * 0.015)
        if view.node_type(node) == "person":
            person_bonus += 0.025
    weakest = min(qualities, default=0.0)
    mean_quality = sum(qualities) / max(1, len(qualities))
    grounded_hops = sum(bool(items) for items in source_sets) / max(1, len(source_sets))
    diversity = min(1.0, len(chunk_ids) / max(1, len(path.steps)))
    inverse_penalty = 0.03 * sum(
        step.traversal_direction == "inverse" for step in path.steps
    )
    return (
        1.8 * weakest
        + mean_quality
        + 0.65 * grounded_hops
        + 0.25 * diversity
        + min(0.15, person_bonus)
        - hub_penalty
        - inverse_penalty
        - conflict_penalty
        - 0.025 * max(0, path.hops - 3)
    )


def _rank_key(path: SearchPath) -> tuple:
    return (-path.score, path.nodes, tuple(step.edge_id for step in path.steps))


def discover_review_items(
    view: FactGraphView,
    config: ECLRRConfig,
) -> list[ReviewItem]:
    """Enumerate non-adjacent endpoint pairs from stable 3-8 hop path slices."""
    by_pair: dict[tuple[str, str], list[SearchPath]] = defaultdict(list)
    for start in sorted(view.nodes):
        beam = [SearchPath(nodes=(start,), steps=(), score=0.0)]
        for _depth in range(1, config.max_hops + 1):
            expanded: list[SearchPath] = []
            for current in beam:
                candidates = view.adjacency.get(current.nodes[-1], ())[
                    : config.max_neighbours
                ]
                for step in candidates:
                    if step.traversal_target in current.nodes:
                        continue
                    path = SearchPath(
                        nodes=(*current.nodes, step.traversal_target),
                        steps=(*current.steps, step),
                        score=0.0,
                    )
                    path = SearchPath(path.nodes, path.steps, _path_score(view, path))
                    expanded.append(path)
                    if path.hops < config.min_hops:
                        continue
                    pair = tuple(sorted((path.nodes[0], path.nodes[-1])))
                    if path.nodes[0] != pair[0]:
                        continue
                    values = by_pair[pair]
                    if path.signature not in {item.signature for item in values}:
                        values.append(path)
                        values.sort(key=_rank_key)
                        del values[config.max_paths_per_item :]
            expanded.sort(key=_rank_key)
            beam = expanded[: config.beam_width]
            if not beam:
                break

    ranked_pairs = sorted(
        by_pair.items(),
        key=lambda item: (_rank_key(item[1][0]), item[0]),
    )[: config.max_review_items]
    review_items: list[ReviewItem] = []
    for pair, paths in ranked_pairs:
        digest = hashlib.md5((pair[0] + "\0" + pair[1]).encode("utf-8")).hexdigest()[
            :16
        ]
        review_items.append(
            ReviewItem(
                review_id=f"review-{digest}",
                source=pair[0],
                target=pair[1],
                primary_path=paths[0],
                supporting_paths=paths[1:],
                fuzzy_edge=view.fuzzy_edge(*pair),
            )
        )
    return review_items
