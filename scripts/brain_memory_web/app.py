#!/usr/bin/env python3
"""
DocThinker · Brain-Like Memory Web Demo
=======================================
Interactive web demo showcasing:
  - Memory graph visualization (D3.js force-directed graph)
  - Sleep consolidation (new edges appear)
  - Spreading activation (activation wave animation)
  - Memory aging (edges decay and disappear)
  - Memory inspector (full transparency)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from flask import Flask, jsonify, render_template, request

from neuro_memory.models import Episode, EdgeType, MemoryEdge, get_decay_for_edge_type
from neuro_memory.graph_store import MemoryGraphStore, SECONDS_PER_DAY
from neuro_memory.consolidation import consolidate, build_structure_description
from neuro_memory.spreading_activation import spreading_activation_traced

app = Flask(__name__, template_folder="templates")

# ═══ Global State ═══════════════════════════════════════════════════
class DemoState:
    def __init__(self):
        self.graph: MemoryGraphStore = MemoryGraphStore()
        self.episodes: Dict[str, Episode] = {}
        self.conversation_log: List[Dict] = []
        self.phase: str = "idle"  # idle | seeded | consolidated | queried | aged
        self.reset()

    def reset(self):
        self.graph = MemoryGraphStore()
        self.episodes = {}
        self.conversation_log = []
        self.phase = "idle"

    def create_episodes(self):
        eps = {}
        eps["ep-001"] = Episode(
            episode_id="ep-001",
            timestamp=time.time() - 3 * 86400,
            source_type="chat",
            summary="RAG系统检索精度优化：发现向量召回的top-k中存在大量语义偏移，需要改进embedding模型和重排序策略",
            key_points=["RAG检索精度不足", "embedding语义偏移问题", "需要重排序策略"],
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
            summary="注意力机制计算效率分析：Transformer中self-attention的O(n^2)复杂度限制了大上下文窗口的扩展",
            key_points=["self-attention复杂度O(n^2)", "大上下文窗口受限", "线性注意力近似方案"],
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
            summary="导师建议关注模型可解释性：不能只追求性能，需要理解模型决策过程，建议从注意力权重可视化入手",
            key_points=["导师建议关注可解释性", "理解模型决策过程", "注意力权重可视化作为切入点"],
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

    def seed_graph(self):
        for ep_id, ep in self.episodes.items():
            self.graph.add_node(ep_id, "episode", {"episode_id": ep_id, "summary": ep.summary})
            for ent_id in ep.entity_ids:
                self.graph.add_node(ent_id, "entity", {})
                self.graph.add_edge(ep_id, ent_id, EdgeType.CONCEPT_LINK, weight=0.70)
                self.graph.add_edge(ent_id, ep_id, EdgeType.CONCEPT_LINK, weight=0.70)

    def graph_to_json(self) -> Dict[str, Any]:
        nodes = []
        for nid, ndata in self.graph.get_all_nodes():
            node = {
                "id": nid,
                "type": ndata.get("type", "unknown"),
                "label": nid,
            }
            if nid in self.episodes:
                ep = self.episodes[nid]
                node["summary"] = ep.summary
                node["concepts"] = ep.concepts
                node["retrieval_count"] = ep.retrieval_count
                node["source_type"] = ep.source_type
            nodes.append(node)

        edges = []
        seen = set()
        for e in self.graph.get_all_edges():
            key = tuple(sorted([e.source_id, e.target_id]) + [e.edge_type.value])
            if key in seen:
                continue
            seen.add(key)
            edges.append({
                "source": e.source_id,
                "target": e.target_id,
                "type": e.edge_type.value,
                "weight": round(e.weight, 4),
                "last_activated": e.last_activated_at,
            })

        return {"nodes": nodes, "edges": edges, "phase": self.phase}


state = DemoState()


# ═══ Mock Functions ═════════════════════════════════════════════════
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


def compute_path_score(path_edges, seed_activation=1.0):
    score = seed_activation
    for i, (_, edge_type, edge_weight) in enumerate(path_edges):
        decay = get_decay_for_edge_type(edge_type)
        score *= edge_weight * (decay ** (i + 1))
    return score


# ═══ API Routes ═════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reset", methods=["POST"])
def api_reset():
    state.reset()
    return jsonify({"status": "ok", "message": "Memory reset"})


@app.route("/api/seed", methods=["POST"])
def api_seed():
    state.reset()
    state.episodes = state.create_episodes()
    state.seed_graph()
    state.phase = "seeded"

    state.conversation_log.append({
        "role": "system",
        "content": "3 个独立研究对话已写入记忆系统",
        "timestamp": time.time(),
        "phase": "seed",
    })
    state.conversation_log.append({
        "role": "user",
        "content": "我在做RAG系统检索精度优化，发现embedding语义偏移严重",
        "timestamp": time.time() - 3 * 86400,
    })
    state.conversation_log.append({
        "role": "user",
        "content": "分析了Transformer中self-attention的O(n^2)复杂度问题",
        "timestamp": time.time() - 2 * 86400,
    })
    state.conversation_log.append({
        "role": "user",
        "content": "导师建议我从注意力权重可视化入手研究可解释性",
        "timestamp": time.time() - 1 * 86400,
    })

    return jsonify({
        "status": "ok",
        "graph": state.graph_to_json(),
        "conversation": state.conversation_log,
        "message": "3 个记忆已播种（互相独立，无跨事件连接）",
    })


@app.route("/api/sleep", methods=["POST"])
def api_sleep():
    if not state.episodes:
        return jsonify({"error": "请先播种记忆"}), 400

    edges_before = len(state.graph.get_all_edges())

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(consolidate(
        state.graph,
        state.episodes,
        recent_n=50,
        content_sim_threshold=0.45,
        structure_sim_threshold=0.30,
        llm_func=mock_llm,
        content_sim_fn=mock_content_sim,
        structure_sim_fn=mock_structure_sim,
    ))
    loop.close()

    edges_after = len(state.graph.get_all_edges())
    new_edges = edges_after - edges_before

    # Collect new cross-episode edges
    new_links = []
    for e in state.graph.get_all_edges():
        if e.edge_type in (EdgeType.SAME_THEME, EdgeType.ANALOGOUS_TO):
            if e.source_id.startswith("ep-") and e.target_id.startswith("ep-"):
                new_links.append({
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.edge_type.value,
                    "weight": round(e.weight, 4),
                })

    state.phase = "consolidated"
    state.conversation_log.append({
        "role": "system",
        "content": f"睡眠巩固完成：新增 {new_edges} 条跨事件连接，加强 {result['edges_strengthened']} 条边",
        "timestamp": time.time(),
        "phase": "sleep",
    })

    return jsonify({
        "status": "ok",
        "graph": state.graph_to_json(),
        "new_links": new_links,
        "stats": {
            "pairs_processed": result["pairs_processed"],
            "edges_added": result["edges_added"],
            "edges_strengthened": result["edges_strengthened"],
        },
        "conversation": state.conversation_log,
        "message": f"睡眠巩固完成！大脑在后台创建了 {len(new_links)//2} 对新连接",
    })


@app.route("/api/query", methods=["POST"])
def api_query():
    if not state.episodes:
        return jsonify({"error": "请先播种记忆"}), 400

    data = request.json or {}
    query_text = data.get("query", "embedding drift")
    seed_entity = data.get("seed", "embedding")

    # If seed not in graph, use embedding as default
    if not state.graph.has_node(seed_entity):
        seed_entity = "embedding"

    results = spreading_activation_traced(
        state.graph,
        [seed_entity],
        max_hops=3,
        initial_activation=1.0,
        record_activation=True,
    )

    # Format results
    activations = []
    for nid, accum_score, path_nodes, path_edges in results:
        path_score = compute_path_score(path_edges, 1.0)
        node_info = {
            "node_id": nid,
            "score": round(path_score, 6),
            "accumulated_score": round(accum_score, 6),
            "hop": len(path_nodes) - 1,
            "path": path_nodes,
            "path_edges": [
                {"source": s, "type": et.value, "weight": round(w, 4)}
                for s, et, w in path_edges
            ],
            "is_episode": nid in state.episodes,
            "is_seed": nid == seed_entity,
        }
        if nid in state.episodes:
            node_info["summary"] = state.episodes[nid].summary
            node_info["concepts"] = state.episodes[nid].concepts
        activations.append(node_info)

    state.phase = "queried"
    state.conversation_log.append({
        "role": "user",
        "content": query_text,
        "timestamp": time.time(),
    })

    # Find the hidden association
    hidden = None
    for a in activations:
        if a["node_id"] == "ep-003" and a["hop"] >= 2:
            hidden = a
            break

    if hidden:
        state.conversation_log.append({
            "role": "system",
            "content": f"通过扩散激活找到隐藏关联：{hidden['path']}（{hidden['score']:.4f}）",
            "timestamp": time.time(),
            "phase": "activation",
        })

    return jsonify({
        "status": "ok",
        "seed": seed_entity,
        "query": query_text,
        "activations": activations,
        "graph": state.graph_to_json(),
        "conversation": state.conversation_log,
        "hidden_finding": hidden,
        "message": f"从 ⟨{seed_entity}⟩ 出发，激活了 {len(activations)} 个节点",
    })


@app.route("/api/age", methods=["POST"])
def api_age():
    if not state.episodes:
        return jsonify({"error": "请先播种记忆"}), 400

    edges_before = {}
    for e in state.graph.get_all_edges():
        edges_before[e.edge_key()] = (e.weight, e.edge_type, e.source_id, e.target_id)

    # Simulate 30 days: ep-001 & ep-003 recalled, ep-002 never recalled
    old_time = time.time() - 40 * SECONDS_PER_DAY
    now = time.time()

    for e in state.graph.get_all_edges():
        e.last_activated_at = old_time
        e.created_at = old_time

    for e in state.graph.get_all_edges():
        if "ep-002" in (e.source_id, e.target_id):
            continue
        if "ep-001" in (e.source_id, e.target_id) or "ep-003" in (e.source_id, e.target_id):
            e.last_activated_at = now

    for e in state.graph.get_all_edges():
        if e.last_activated_at >= now - 1:
            e.weight = min(1.0, e.weight + 0.12)

    total_decayed = 0
    for week in range(3):
        decayed = state.graph.decay_edges(decay_factor=0.4, max_age_days=7.0)
        total_decayed += decayed

    pruned = state.graph.prune_edges(min_weight=0.25)

    # Collect per-episode status
    episode_status = []
    for ep_id in ["ep-001", "ep-003", "ep-002"]:
        surviving = 0
        pruned_count = 0
        total_weight = 0.0
        for e_key, (w, et, src, tgt) in edges_before.items():
            if ep_id in (src, tgt):
                current = state.graph._edge_index.get(e_key)
                if current:
                    surviving += 1
                    total_weight += current.weight
                else:
                    pruned_count += 1
        avg_w = total_weight / max(1, surviving)
        status = "STRONG" if avg_w > 0.7 else ("WEAKENED" if avg_w < 0.5 else "STABLE")
        episode_status.append({
            "episode_id": ep_id,
            "summary": state.episodes[ep_id].summary[:40],
            "avg_weight": round(avg_w, 4),
            "surviving_edges": surviving,
            "pruned_edges": pruned_count,
            "status": status,
        })

    state.phase = "aged"
    state.conversation_log.append({
        "role": "system",
        "content": f"记忆衰老完成：{total_decayed} 轮衰减，{pruned} 条边被剪枝",
        "timestamp": time.time(),
        "phase": "age",
    })

    return jsonify({
        "status": "ok",
        "graph": state.graph_to_json(),
        "episode_status": episode_status,
        "stats": {
            "total_decayed": total_decayed,
            "total_pruned": pruned,
        },
        "conversation": state.conversation_log,
        "message": f"30天模拟完成：{pruned} 条边被剪枝，记忆图谱已自我优化",
    })


@app.route("/api/graph", methods=["GET"])
def api_graph():
    return jsonify({"graph": state.graph_to_json(), "conversation": state.conversation_log})


@app.route("/api/auto", methods=["POST"])
def api_auto():
    """Run the full demo automatically."""
    results = {}

    # Step 1: Seed
    state.reset()
    state.episodes = state.create_episodes()
    state.seed_graph()
    state.phase = "seeded"
    state.conversation_log.append({
        "role": "system",
        "content": "3 个独立研究对话已写入记忆系统",
        "timestamp": time.time(),
        "phase": "seed",
    })
    results["seed"] = {"graph": state.graph_to_json(), "conversation": state.conversation_log}

    # Step 2: Sleep
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(consolidate(
        state.graph, state.episodes,
        recent_n=50, content_sim_threshold=0.45, structure_sim_threshold=0.30,
        llm_func=mock_llm, content_sim_fn=mock_content_sim, structure_sim_fn=mock_structure_sim,
    ))
    loop.close()
    state.phase = "consolidated"
    state.conversation_log.append({
        "role": "system",
        "content": f"睡眠巩固完成：新增 {result['edges_added']} 条跨事件连接",
        "timestamp": time.time(),
        "phase": "sleep",
    })
    results["sleep"] = {"graph": state.graph_to_json(), "stats": result}

    # Step 3: Query
    sa_results = spreading_activation_traced(
        state.graph, ["embedding"], max_hops=3, initial_activation=1.0, record_activation=True,
    )
    activations = []
    for nid, accum, path_nodes, path_edges in sa_results:
        ps = compute_path_score(path_edges, 1.0)
        activations.append({
            "node_id": nid, "score": round(ps, 6), "hop": len(path_nodes)-1,
            "path": path_nodes, "is_episode": nid in state.episodes, "is_seed": nid=="embedding",
        })
    state.phase = "queried"
    results["query"] = {"activations": activations, "graph": state.graph_to_json()}

    # Step 4: Age
    old_time = time.time() - 40 * SECONDS_PER_DAY
    now = time.time()
    for e in state.graph.get_all_edges():
        e.last_activated_at = old_time
        e.created_at = old_time
    for e in state.graph.get_all_edges():
        if "ep-002" in (e.source_id, e.target_id):
            continue
        if "ep-001" in (e.source_id, e.target_id) or "ep-003" in (e.source_id, e.target_id):
            e.last_activated_at = now
    for e in state.graph.get_all_edges():
        if e.last_activated_at >= now - 1:
            e.weight = min(1.0, e.weight + 0.12)
    total_decayed = sum(state.graph.decay_edges(decay_factor=0.4, max_age_days=7.0) for _ in range(3))
    pruned = state.graph.prune_edges(min_weight=0.25)
    state.phase = "aged"
    results["age"] = {"graph": state.graph_to_json(), "decayed": total_decayed, "pruned": pruned}

    return jsonify({"status": "ok", "steps": results})


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DocThinker · Brain-Like Memory Web Demo")
    print("  Open http://127.0.0.1:5137 in your browser")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=5137, debug=False, use_reloader=False)
