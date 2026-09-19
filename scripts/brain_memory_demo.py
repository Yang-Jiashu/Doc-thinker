#!/usr/bin/env python3
"""
DocThinker · Brain-Like Memory Demo
===================================
Demonstrates five pillars of brain-like memory:

  1. SEED   — Plant initial memories (isolated episodes)
  2. SLEEP  — Consolidation: replay & infer cross-links (test-time scaling)
  3. THINK  — Spreading Activation: find hidden paths via association
  4. AGE    — Decay & Pruning: "use it or lose it"
  5. INSPECT — Memory Inspector: transparency & controllability

Memory is alive: it consolidates during "sleep", activates through
association, ages with use-it-or-lose-it dynamics, and is fully transparent.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neuro_memory.models import Episode, EdgeType, MemoryEdge, get_decay_for_edge_type
from neuro_memory.graph_store import MemoryGraphStore, SECONDS_PER_DAY
from neuro_memory.consolidation import consolidate, build_structure_description
from neuro_memory.spreading_activation import spreading_activation_traced


# ═══ ANSI Colors ═══════════════════════════════════════════════════
class C:
    R    = '\033[0m'
    B    = '\033[1m'
    DIM  = '\033[2m'
    RED  = '\033[91m'
    GRN  = '\033[92m'
    YEL  = '\033[93m'
    BLU  = '\033[94m'
    MAG  = '\033[95m'
    CYN  = '\033[96m'
    WHT  = '\033[97m'
    GRY  = '\033[90m'
    BOLD = '\033[1m'
    UNDR = '\033[4m'


# ═══ Utilities ═════════════════════════════════════════════════════
def banner():
    print(f"""
{C.CYN}{'━'*72}{C.R}
{C.BOLD}{C.MAG}  DocThinker · Brain-Like Memory Demo{C.R}
{C.CYN}  Seed → Sleep → Think → Age → Inspect{C.R}
{C.CYN}  Memory is alive, not stored.{C.R}
{C.CYN}{'━'*72}{C.R}""")


def phase_header(num: int, title: str, subtitle: str = ""):
    print(f"\n{C.BOLD}{C.YEL}{'═'*72}{C.R}")
    print(f"{C.BOLD}{C.YEL}  Phase {num}: {title}{C.R}")
    if subtitle:
        print(f"{C.GRY}  {subtitle}{C.R}")
    print(f"{C.BOLD}{C.YEL}{'═'*72}{C.R}\n")


def box(lines: List[str], color: str = C.CYN) -> str:
    width = max(len(line) for line in lines) if lines else 0
    top    = f"{color}┌{'─'*(width+2)}┐{C.R}"
    bottom = f"{color}└{'─'*(width+2)}┘{C.R}"
    mid    = [f"{color}│ {line.ljust(width)} │{C.R}" for line in lines]
    return "\n".join([top] + mid + [bottom])


def edge_type_color(et: EdgeType) -> str:
    return {
        EdgeType.CONCEPT_LINK: C.CYN,
        EdgeType.SAME_THEME:   C.GRN,
        EdgeType.ANALOGOUS_TO: C.MAG,
        EdgeType.EPISODE_SIMILARITY: C.BLU,
        EdgeType.CO_ACTIVATED: C.YEL,
        EdgeType.MENTIONS:     C.GRY,
    }.get(et, C.WHT)


def edge_type_symbol(et: EdgeType) -> str:
    return {
        EdgeType.CONCEPT_LINK: "──",
        EdgeType.SAME_THEME:   "══",
        EdgeType.ANALOGOUS_TO: "≈≈",
        EdgeType.EPISODE_SIMILARITY: "──",
        EdgeType.CO_ACTIVATED: "++",
        EdgeType.MENTIONS:     "..",
    }.get(et, "──")


def short_id(nid: str) -> str:
    if nid.startswith("ep-"):
        return nid
    return nid


def node_label(nid: str, graph: MemoryGraphStore, episodes: Dict[str, Episode]) -> str:
    if nid in episodes:
        ep = episodes[nid]
        return f"{nid} [{ep.summary[:35]}]"
    node = graph.get_node(nid)
    if node and node.get("type") == "entity":
        return f"⟨{nid}⟩"
    return nid


def compute_path_score(path_edges: List[Tuple[str, EdgeType, float]],
                       seed_activation: float = 1.0) -> float:
    """Compute the activation score along a single path (not accumulated)."""
    score = seed_activation
    for i, (_, edge_type, edge_weight) in enumerate(path_edges):
        decay = get_decay_for_edge_type(edge_type)
        score *= edge_weight * (decay ** (i + 1))
    return score


def print_graph_stats(graph: MemoryGraphStore, label: str = ""):
    nodes = graph.get_all_nodes()
    edges = graph.get_all_edges()
    ep_count  = sum(1 for _, n in nodes if n.get("type") == "episode")
    ent_count = sum(1 for _, n in nodes if n.get("type") == "entity")
    print(f"  {C.DIM}Graph:{C.R} {C.BOLD}{len(nodes)}{C.R} nodes "
          f"({ep_count} episodes, {ent_count} entities), "
          f"{C.BOLD}{len(edges)}{C.R} edges"
          f"{f'  {label}' if label else ''}")


def weight_bar(w: float, width: int = 20) -> str:
    filled = int(w * width)
    return "█" * filled + "░" * (width - filled)


# ═══ Demo Data ═════════════════════════════════════════════════════
def create_episodes() -> Dict[str, Episode]:
    """Create 3 independent research episodes."""
    eps = {}

    eps["ep-001"] = Episode(
        episode_id="ep-001",
        timestamp=time.time() - 3 * 86400,
        source_type="chat",
        summary="RAG系统检索精度优化：发现向量召回的top-k中存在大量语义偏移，"
                "需要改进embedding模型和重排序策略",
        key_points=[
            "RAG检索精度不足",
            "embedding语义偏移问题",
            "需要重排序策略",
        ],
        concepts=["RAG", "retrieval", "precision", "embedding", "reranking"],
        entity_ids=["retrieval", "embedding"],
        relation_triples=[
            ("RAG", "uses", "retrieval"),
            ("retrieval", "depends_on", "embedding"),
            ("embedding", "has_issue", "semantic_drift"),
        ],
    )

    eps["ep-002"] = Episode(
        episode_id="ep-002",
        timestamp=time.time() - 2 * 86400,
        source_type="doc",
        summary="注意力机制计算效率分析：Transformer中self-attention的O(n^2)复杂度"
                "限制了大上下文窗口的扩展，研究了线性注意力近似方案",
        key_points=[
            "self-attention复杂度O(n^2)",
            "大上下文窗口受限",
            "线性注意力近似方案",
        ],
        concepts=["attention", "transformer", "efficiency", "embedding", "computation"],
        entity_ids=["attention", "transformer", "embedding"],
        relation_triples=[
            ("transformer", "uses", "attention"),
            ("attention", "operates_on", "embedding"),
            ("attention", "has_complexity", "O(n^2)"),
        ],
    )

    eps["ep-003"] = Episode(
        episode_id="ep-003",
        timestamp=time.time() - 1 * 86400,
        source_type="chat",
        summary="导师建议关注模型可解释性：不能只追求性能，需要理解模型决策过程，"
                "建议从注意力权重可视化入手",
        key_points=[
            "导师建议关注可解释性",
            "理解模型决策过程",
            "注意力权重可视化作为切入点",
        ],
        concepts=["interpretability", "advisor", "research_direction", "attention"],
        entity_ids=["interpretability", "attention"],
        relation_triples=[
            ("advisor", "suggests", "interpretability"),
            ("interpretability", "starts_from", "attention"),
        ],
    )

    for ep in eps.values():
        ep.structure_description = build_structure_description(ep)

    return eps


def seed_graph(graph: MemoryGraphStore, episodes: Dict[str, Episode]):
    """Add episodes and entities to graph with bidirectional entity links."""
    for ep_id, ep in episodes.items():
        graph.add_node(ep_id, "episode", {"episode_id": ep_id, "summary": ep.summary})
        for ent_id in ep.entity_ids:
            graph.add_node(ent_id, "entity", {})
            graph.add_edge(ep_id, ent_id, EdgeType.CONCEPT_LINK, weight=0.70)
            graph.add_edge(ent_id, ep_id, EdgeType.CONCEPT_LINK, weight=0.70)


# ═══ Mock Functions for Consolidation ═══════════════════════════════
async def mock_content_sim(a_id: str, b_id: str) -> float:
    sims = {
        ("ep-001", "ep-002"): 0.55,
        ("ep-001", "ep-003"): 0.48,
        ("ep-002", "ep-003"): 0.52,
    }
    key = (a_id, b_id) if (a_id, b_id) in sims else (b_id, a_id)
    return sims.get(key, 0.0)


async def mock_structure_sim(a_id: str, b_id: str) -> float:
    sims = {
        ("ep-001", "ep-002"): 0.35,
        ("ep-001", "ep-003"): 0.40,
        ("ep-002", "ep-003"): 0.38,
    }
    key = (a_id, b_id) if (a_id, b_id) in sims else (b_id, a_id)
    return sims.get(key, 0.0)


async def mock_llm(prompt: str) -> str:
    has_rag    = "RAG" in prompt or "检索精度" in prompt
    has_attn   = "注意力" in prompt or "Transformer" in prompt or "self-attention" in prompt
    has_interp = "可解释性" in prompt or "导师" in prompt or "决策过程" in prompt

    if has_rag and has_interp:
        return "relation: analogous_to\nreason: 都是从问题出发寻找技术切入点的研究路径"
    if has_rag and has_attn:
        return "relation: same_theme\nreason: 两者都是NLP技术研究中遇到的性能与精度问题"
    if has_attn and has_interp:
        return "relation: same_theme\nreason: 注意力机制是连接效率与可解释性的桥梁"
    return "relation: none\nreason: 无明显关联"


# ═══ Phase 1: Seed ═════════════════════════════════════════════════
def phase1_seed(graph: MemoryGraphStore, episodes: Dict[str, Episode]):
    phase_header(1, "Seeding Memories",
                 "Three independent conversations → three isolated episodes")

    print(f"  {C.DIM}Three research conversations on different days:{C.R}\n")

    for ep_id, ep in episodes.items():
        day = datetime.fromtimestamp(ep.timestamp).strftime("%b %d")
        print(f"  {C.BOLD}{C.CYN}{ep_id}{C.R}  {C.GRY}[{day}]{C.R}  {ep.summary[:55]}...")
        print(f"  {C.DIM}  concepts: {', '.join(ep.concepts)}{C.R}")
        for ent in ep.entity_ids:
            print(f"  {C.GRY}  └── ⟨{ent}⟩  (concept_link, w=0.70){C.R}")
        print()

    print_graph_stats(graph, f"{C.GRY}0 cross-episode links{C.R}")
    print(f"\n  {C.YEL}▶ Note:{C.R} Episodes are {C.BOLD}isolated{C.R} — no connections between them.")
    print(f"  {C.GRY}The graph is just entity hubs, not a true association network.{C.R}")


# ═══ Phase 2: Consolidation (Sleep) ═════════════════════════════════
async def phase2_sleep(graph: MemoryGraphStore, episodes: Dict[str, Episode]):
    phase_header(2, "Memory Consolidation (Sleep)",
                 "Like the brain replaying experiences during sleep...")

    edges_before = len(graph.get_all_edges())

    print(f"  {C.DIM}Replaying 3 episodes, inferring cross-episode relations...{C.R}\n")

    result = await consolidate(
        graph,
        episodes,
        recent_n=50,
        content_sim_threshold=0.45,
        structure_sim_threshold=0.30,
        llm_func=mock_llm,
        content_sim_fn=mock_content_sim,
        structure_sim_fn=mock_structure_sim,
    )

    edges_after = len(graph.get_all_edges())
    new_edges = edges_after - edges_before

    print(box([
        f"  pairs_processed:     {result['pairs_processed']}",
        f"  edges_added:         {result['edges_added']}",
        f"  edges_strengthened:  {result['edges_strengthened']}",
        f"  new cross-links:     +{new_edges}",
    ]))
    print()

    # Show new edges
    print(f"  {C.BOLD}{C.GRN}New connections formed during sleep:{C.R}\n")
    seen_pairs = set()
    for edge in graph.get_all_edges():
        if edge.edge_type in (EdgeType.SAME_THEME, EdgeType.ANALOGOUS_TO):
            if edge.source_id.startswith("ep-") and edge.target_id.startswith("ep-"):
                pair_key = tuple(sorted([edge.source_id, edge.target_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                c = edge_type_color(edge.edge_type)
                sym = edge_type_symbol(edge.edge_type)
                print(f"    {C.CYN}{edge.source_id}{C.R} {c}{sym}[{edge.edge_type.value}, "
                      f"w={edge.weight:.2f}]{sym}{C.R} {C.CYN}{edge.target_id}{C.R}")

    print()
    print_graph_stats(graph, f"{C.GRN}Δ +{new_edges} edges{C.R}")

    print(f"""
  {C.DIM}Before sleep (isolated):         {C.GRN}After sleep (consolidated):{C.R}
  {C.DIM}                                 {C.R}
  {C.DIM}  ep-001    ep-002               {C.GRN}  ep-001 ≈≈[analogous]≈≈ ep-003{C.R}
  {C.DIM}     │        │                  {C.GRN}    │  ╲               ╱  │{C.R}
  {C.DIM}     │        │                  {C.GRN}    │   ╲    ep-003   ╱   │{C.R}
  {C.DIM}  retrieval  attention           {C.DIM}    │    ╲    │    ╱     │{C.R}
  {C.DIM}  embedding  transformer         {C.DIM}    │     ╲   │   ╱      │{C.R}
  {C.DIM}             embedding           {C.GRN}    ══[same_theme]══      │{C.R}
  {C.DIM}  ep-003                         {C.DIM}    │        │           │{C.R}
  {C.DIM}     │                           {C.DIM}  retrieval  attention  transformer
  {C.DIM}  interpretability               {C.DIM}  embedding  embedding  embedding
  {C.DIM}                                 {C.DIM}             interpretability
  {C.DIM}                                 {C.BOLD}{C.YEL}↑ Sleep created 3 new cross-links{C.R}""")

    print(f"\n  {C.BOLD}Key insight:{C.R} The graph went from {C.BOLD}entity hubs{C.R} → "
          f"{C.BOLD}association network{C.R}.")
    print(f"  {C.GRY}ep-001 (RAG) ↔ ep-003 (interpretability) via 'analogous_to'{C.R}")
    print(f"  {C.GRY}— two different problems, same research pattern. Brain created it.{C.R}")


# ═══ Phase 3: Spreading Activation (Think) ══════════════════════════
def phase3_think(graph: MemoryGraphStore, episodes: Dict[str, Episode]):
    phase_header(3, "Spreading Activation (Think)",
                 "Day 4 · User asks about 'embedding drift'")

    print(f"  {C.DIM}User: \"最近在研究 embedding drift 的问题\"{C.R}")
    print(f"  {C.DIM}Note: user never mentioned RAG, retrieval, attention, or interpretability.{C.R}\n")

    seed_id = "embedding"
    print(f"  {C.MAG}▶ Seed: entity ⟨{seed_id}⟩  (activation = 1.0){C.R}")
    print(f"  {C.DIM}Activation propagating through the association graph...\n{C.R}")

    results = spreading_activation_traced(
        graph,
        [seed_id],
        max_hops=3,
        initial_activation=1.0,
        record_activation=True,
    )

    # Compute path-based scores (not accumulated) for clearer display
    by_hop: Dict[int, List] = {}
    for nid, accum_score, path_nodes, path_edges in results:
        hop = len(path_nodes) - 1
        path_score = compute_path_score(path_edges, 1.0)
        by_hop.setdefault(hop, []).append((nid, path_score, path_nodes, path_edges))

    for hop in sorted(by_hop.keys()):
        items = sorted(by_hop[hop], key=lambda x: -x[1])
        print(f"  {C.BOLD}{C.BLU}┌─ HOP {hop} {'─'*(60 - len(f'HOP {hop}'))}┐{C.R}")

        for nid, score, path_nodes, path_edges in items:
            label = node_label(nid, graph, episodes)

            if nid == seed_id:
                tag = f"{C.MAG}SEED{C.R}"
            elif nid in episodes:
                tag = f"{C.GRN}★ EPISODE HIT{C.R}"
            else:
                tag = f"{C.CYN}(entity){C.R}"

            # Build path display
            if path_edges:
                path_parts = []
                for i, (src, et, w) in enumerate(path_edges):
                    c = edge_type_color(et)
                    sym = edge_type_symbol(et)
                    path_parts.append(f"{C.GRY}{short_id(src)}{C.R}")
                    path_parts.append(f"{c}{sym}[{et.value},{w:.2f}]{sym}{C.R}")
                    if i == len(path_edges) - 1:
                        path_parts.append(f"{C.GRY}{short_id(path_nodes[-1])}{C.R}")
                path_str = " ".join(path_parts)
            else:
                path_str = f"{C.GRY}(seed){C.R}"

            print(f"  {C.BOLD}{C.BLU}│{C.R} {C.BOLD}{score:.4f}{C.R}  {label}")
            print(f"  {C.BOLD}{C.BLU}│{C.R}   {C.DIM}path: {path_str}{C.R}")
            print(f"  {C.BOLD}{C.BLU}│{C.R}   {tag}")
            print(f"  {C.BOLD}{C.BLU}│{C.R}")

        print(f"  {C.BOLD}{C.BLU}└{'─'*62}┘{C.R}\n")

    # Highlight the key finding
    print(box([
        "  ★ KEY FINDING: Hidden Association Discovered",
        "",
        "  User said:  'embedding drift'",
        "  Seed:       entity ⟨embedding⟩",
        "",
        "  The system found ep-003 (interpretability) at HOP 2:",
        "    ⟨embedding⟩ → ep-001 (RAG) → ep-003 (interpretability)",
        "",
        "  This path uses the 'analogous_to' edge CREATED during sleep.",
        "  Before sleep, ⟨embedding⟩ could NOT reach 'interpretability'.",
        "",
        "  This is ASSOCIATION RECALL, not keyword matching.",
    ], C.YEL))

    print(f"\n  {C.BOLD}vs ChatGPT Memory:{C.R}")
    print(f"  {C.GRY}ChatGPT stores flat text snippets and does keyword/BM25 retrieval.{C.R}")
    print(f"  {C.GRY}It cannot find 'interpretability' when you say 'embedding drift'{C.R}")
    print(f"  {C.GRY}because the words don't overlap at all.{C.R}")
    print(f"  {C.GRY}DocThinker's association graph finds it in 2 hops.{C.R}")


# ═══ Phase 4: Memory Aging (Evolve) ═════════════════════════════════
def phase4_age(graph: MemoryGraphStore, episodes: Dict[str, Episode]):
    phase_header(4, "Memory Aging (Evolution)",
                 "Simulating 30 days — use it or lose it")

    # Snapshot edge state before aging
    edges_before = {}
    for e in graph.get_all_edges():
        edges_before[e.edge_key()] = (e.weight, e.edge_type, e.source_id, e.target_id)

    print(f"  {C.DIM}Simulating 30 days of memory usage...{C.R}\n")
    print(f"  {C.DIM}Day  1: All edges fresh, weights at initial values{C.R}")
    print(f"  {C.DIM}Day  7: ep-001 & ep-003 recalled during queries → edges refreshed{C.R}")
    print(f"  {C.DIM}         ep-002 NEVER recalled → all edges aging{C.R}")

    # Reset ALL edges to old
    old_time = time.time() - 40 * SECONDS_PER_DAY
    now = time.time()
    for e in graph.get_all_edges():
        e.last_activated_at = old_time
        e.created_at = old_time

    # Mark ONLY ep-001 & ep-003 edges as recently activated
    # IMPORTANT: skip any edge touching ep-002 (it was never recalled)
    for e in graph.get_all_edges():
        if "ep-002" in (e.source_id, e.target_id):
            continue  # ep-002 was not recalled — let it age
        if "ep-001" in (e.source_id, e.target_id) or "ep-003" in (e.source_id, e.target_id):
            e.last_activated_at = now

    # Strengthen recently activated edges (consolidation bonus from repeated recall)
    for e in graph.get_all_edges():
        if e.last_activated_at >= now - 1:
            e.weight = min(1.0, e.weight + 0.12)

    print(f"  {C.DIM}Day 15: ep-002 edges decaying (weight × 0.4 per week){C.R}")

    # Aggressive decay for old edges
    total_decayed = 0
    for week in range(3):
        decayed = graph.decay_edges(decay_factor=0.4, max_age_days=7.0)
        total_decayed += decayed
        if decayed > 0:
            print(f"          Week {week+1}: {C.YEL}{decayed}{C.R} edges decayed")

    print(f"  {C.DIM}Day 30: Pruning edges below weight 0.25{C.R}\n")

    pruned = graph.prune_edges(min_weight=0.25)

    # Results
    print(f"  {C.BOLD}Aging Results:{C.R}\n")
    print(f"    Total decayed:  {C.YEL}{total_decayed}{C.R} edge-rounds")
    print(f"    Total pruned:   {C.RED}{pruned}{C.R} edges removed")
    print()

    # Per-episode detailed status
    print(f"  {C.BOLD}Memory Evolution Summary:{C.R}\n")
    print(f"    {'Episode':<12} {'Description':<30} {'avg_w':>6}  {'Bar':<22} {'Status'}")
    print(f"    {'─'*12} {'─'*30} {'─'*6}  {'─'*22} {'─'*10}")

    for ep_id in ["ep-001", "ep-003", "ep-002"]:
        ep = episodes[ep_id]
        surviving = 0
        pruned_count = 0
        total_weight = 0.0
        for e_key, (w, et, src, tgt) in edges_before.items():
            if ep_id in (src, tgt):
                current = graph._edge_index.get(e_key)
                if current:
                    surviving += 1
                    total_weight += current.weight
                else:
                    pruned_count += 1

        avg_w = total_weight / max(1, surviving)
        bar = weight_bar(avg_w)

        if pruned_count > 0 and avg_w < 0.5:
            status = f"{C.RED}WEAKENED{C.R}"
        elif avg_w > 0.7:
            status = f"{C.GRN}STRONG{C.R}"
        else:
            status = f"{C.YEL}STABLE{C.R}"

        desc = ep.summary[:28]
        print(f"    {ep_id:<12} {desc:<30} {avg_w:>6.2f}  {bar}  {status}")
        if pruned_count:
            print(f"    {'':12} {C.GRY}pruned: {pruned_count} edges, surviving: {surviving}{C.R}")

    print()
    print(box([
        "  BRAIN-LIKE MEMORY PRINCIPLE",
        "",
        "  \"Neurons that fire together wire together.\"",
        "  \"Neurons that never fire fade and die.\"",
        "",
        "  ep-001 & ep-003: recalled during use → edges strengthened → STRONG",
        "  ep-002: never recalled → ALL 10 edges decayed & pruned (0 surviving)",
        "",
        "  The memory graph is ALIVE: it self-optimizes during use.",
        "  This is test-time scaling for MEMORY, not for the model.",
    ], C.MAG))


# ═══ Phase 5: Memory Inspector (Transparency & Control) ═════════════
def phase5_inspect(graph: MemoryGraphStore, episodes: Dict[str, Episode]):
    phase_header(5, "Memory Inspector (Transparency)",
                 "Full graph state — every node, every edge, fully visible")

    print(f"  {C.DIM}Unlike ChatGPT's black-box memory, DocThinker's memory is fully transparent.{C.R}")
    print(f"  {C.DIM}You can inspect every node, every edge, every weight.{C.R}\n")

    # Show all nodes
    nodes = graph.get_all_nodes()
    ep_nodes = [(nid, n) for nid, n in nodes if n.get("type") == "episode"]
    ent_nodes = [(nid, n) for nid, n in nodes if n.get("type") == "entity"]

    print(f"  {C.BOLD}{C.CYN}Nodes ({len(nodes)} total):{C.R}\n")
    print(f"    {C.BOLD}Episodes:{C.R}")
    for nid, n in ep_nodes:
        ep = episodes.get(nid)
        recall = f"recalled {ep.retrieval_count}×" if ep and ep.retrieval_count > 0 else "never recalled"
        print(f"      {C.CYN}{nid}{C.R}  {C.GRY}{ep.summary[:45]}...{C.R}")
        print(f"      {C.DIM}concepts: {', '.join(ep.concepts)}  |  {recall}{C.R}")
    print()
    print(f"    {C.BOLD}Entities:{C.R}")
    for nid, n in ent_nodes:
        # Count connections
        out_edges = graph.get_out_edges(nid)
        print(f"      ⟨{C.CYN}{nid}{C.R}⟩  {C.GRY}({len(out_edges)} connections){C.R}")
    print()

    # Show all edges by type
    edges = graph.get_all_edges()
    by_type: Dict[EdgeType, List[MemoryEdge]] = {}
    for e in edges:
        by_type.setdefault(e.edge_type, []).append(e)

    print(f"  {C.BOLD}{C.CYN}Edges by type ({len(edges)} total):{C.R}\n")
    for et in sorted(by_type.keys(), key=lambda x: -len(by_type[x])):
        elist = by_type[et]
        c = edge_type_color(et)
        avg_w = sum(e.weight for e in elist) / len(elist)
        print(f"    {c}{et.value}{C.R}  ({len(elist)} edges, avg_w={avg_w:.2f})")

    print()

    # Show edge details for cross-episode links
    print(f"  {C.BOLD}{C.CYN}Cross-episode edges (the ones sleep created):{C.R}\n")
    for et in [EdgeType.ANALOGOUS_TO, EdgeType.SAME_THEME]:
        if et not in by_type:
            continue
        c = edge_type_color(et)
        sym = edge_type_symbol(et)
        seen = set()
        for e in by_type[et]:
            if not (e.source_id.startswith("ep-") and e.target_id.startswith("ep-")):
                continue
            pair = tuple(sorted([e.source_id, e.target_id]))
            if pair in seen:
                continue
            seen.add(pair)
            print(f"    {e.source_id} {c}{sym}[{et.value}, w={e.weight:.2f}]{sym}{C.R} {e.target_id}")

    print()

    # Show the key difference: transparency
    print(box([
        "  TRANSPARENCY & CONTROLLABILITY",
        "",
        "  ChatGPT Memory:                    DocThinker Memory:",
        "  ┌──────────────────────┐           ┌──────────────────────┐",
        "  │ Black box            │           │ Full graph visible   │",
        "  │ Flat text snippets   │           │ Typed edges + weights│",
        "  │ No relationship info │           │ Association network  │",
        "  │ Can't edit structure │           │ Can edit/delete nodes│",
        "  │ No activation trace  │           │ Full activation path │",
        "  │ Static (never changes)│          │ Dynamic (evolves)    │",
        "  └──────────────────────┘           └──────────────────────┘",
        "",
        "  You can: inspect any node, edit any edge, delete any memory,",
        "  trace activation paths, and watch the graph evolve over time.",
    ], C.CYN))


# ═══ Main ═══════════════════════════════════════════════════════════
async def main():
    banner()

    # Setup
    graph = MemoryGraphStore()
    episodes = create_episodes()
    seed_graph(graph, episodes)

    # Phase 1: Seed
    phase1_seed(graph, episodes)

    # Phase 2: Consolidation (Sleep)
    await phase2_sleep(graph, episodes)

    # Phase 3: Spreading Activation (Think)
    phase3_think(graph, episodes)

    # Record retrieval for episodes actually USED to answer queries
    # ep-001 & ep-003 were recalled; ep-002 was activated but NOT used
    for ep_id in ["ep-001", "ep-003"]:
        if ep_id in episodes:
            episodes[ep_id].record_retrieval()
            episodes[ep_id].record_retrieval()  # recalled twice

    # Phase 4: Memory Aging (Evolve)
    phase4_age(graph, episodes)

    # Phase 5: Memory Inspector (Transparency)
    phase5_inspect(graph, episodes)

    # Final summary
    print(f"\n{C.CYN}{'━'*72}{C.R}")
    print(f"{C.BOLD}{C.MAG}  Summary: Why This Memory is Different{C.R}\n")
    print(f"  {C.BOLD}1. Dynamic (not static):{C.R} Memory restructures during 'sleep'.")
    print(f"     {C.GRY}Consolidation creates new connections between episodes.{C.R}")
    print(f"  {C.BOLD}2. Associative (not keyword):{C.R} Spreading activation finds hidden paths.")
    print(f"     {C.GRY}⟨embedding⟩ → ep-001 (RAG) → ep-003 (interpretability){C.R}")
    print(f"     {C.GRY}No keyword overlap. Pure association.{C.R}")
    print(f"  {C.BOLD}3. Self-optimizing (use it or lose it):{C.R} Memories age and prune.")
    print(f"     {C.GRY}Frequently recalled → stronger. Never recalled → decayed & pruned.{C.R}")
    print(f"  {C.BOLD}4. Transparent & controllable:{C.R} Full graph is visible and editable.")
    print(f"     {C.GRY}Every node, every edge, every weight — inspectable.{C.R}")
    print(f"\n  {C.BOLD}This is test-time scaling for MEMORY, not for the model.{C.R}")
    print(f"  {C.GRY}The model doesn't change. The memory evolves.{C.R}")
    print(f"\n{C.CYN}{'━'*72}{C.R}\n")


if __name__ == "__main__":
    asyncio.run(main())
