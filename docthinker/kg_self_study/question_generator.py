"""Question generation for the KG self-study loop.

Takes P1 subgraph analysis results and dispatches to P2-A..P2-F prompt
templates to produce a batch of typed questions for the current round.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List

from .prompts import (
    BRIDGE_QUESTION_PROMPT,
    COMPARISON_QUESTION_PROMPT,
    COMPONENT_BRIDGING_PROMPT,
    CONTRADICTION_QUESTION_PROMPT,
    EDGE_VALIDATION_PROMPT,
    SUBGRAPH_ANALYSIS_PROMPT,
    TWO_HOP_INFERENCE_PROMPT,
)
from .work_budget import StudyBudgetStop

_log = logging.getLogger("docthinker.kg_self_study.question_gen")


def _safe_json_parse(raw: str) -> Any:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start: end + 1])
            except json.JSONDecodeError:
                continue
    return None


class QuestionGenerator:
    """Generate self-study questions from a selected subgraph."""

    def __init__(
        self,
        llm_func: Callable,
        questions_per_strategy: int = 3,
    ):
        self.llm_func = llm_func
        self.n_questions = questions_per_strategy

    async def analyze_subgraph(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """P1: Analyze subgraph structure and identify opportunities."""
        entities_json = json.dumps(
            [{"name": e.get("id") or e.get("entity_id"),
              "type": e.get("entity_type", "unknown"),
              "description": e.get("description") or "",
              "source_ids": str(e.get("source_id", "")),
              "degree": 0}
             for e in entities],
            ensure_ascii=False, indent=2,
        )
        relations_json = json.dumps(
            [{"source": r.get("source") or r.get("src_id"),
              "target": r.get("target") or r.get("tgt_id"),
              "keywords": r.get("keywords", ""),
              "description": r.get("description") or "",
              "source_id": str(r.get("source_id", "")),
              "is_discovered": r.get("is_discovered", "0")}
             for r in relations],
            ensure_ascii=False, indent=2,
        )

        prompt = SUBGRAPH_ANALYSIS_PROMPT.format(
            entities_json=entities_json,
            relations_json=relations_json,
        )

        try:
            raw = await self.llm_func(prompt)
            analysis = _safe_json_parse(raw)
            if not isinstance(analysis, dict):
                _log.warning("[question_gen] P1 needs a JSON object, using defaults")
                analysis = {}
            for key in (
                "bridge_candidates", "potential_contradictions", "two_hop_gaps",
                "same_type_pairs", "weak_edges", "isolated_clusters",
            ):
                if key in analysis:
                    value = analysis[key]
                    analysis[key] = [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
        except StudyBudgetStop:
            raise
        except Exception as exc:
            _log.error("[question_gen] P1 LLM call failed: %s", exc)
            analysis = {}

        return analysis

    async def generate_questions(
        self,
        strategy: str,
        analysis: Dict[str, Any],
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """P2: Generate questions for the given strategy type."""
        generator_fn = {
            "bridge_entity": self._gen_bridge,
            "two_hop_completion": self._gen_two_hop,
            "comparison_alignment": self._gen_comparison,
            "weak_component": self._gen_component,
            "hub_enrichment": self._gen_bridge,
            "evidence_chain": self._gen_edge_validation,
        }.get(strategy, self._gen_bridge)

        if self.n_questions <= 0:
            return []
        questions = self._bounded_questions(await generator_fn(analysis, entities, relations))
        if len(questions) < self.n_questions:
            questions.extend(await self._gen_contradiction(analysis))
        return self._bounded_questions(questions)

    def _bounded_questions(self, questions: List) -> List[Dict]:
        selected, seen = [], set()
        for item in questions:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            if question and question not in seen:
                seen.add(question)
                selected.append(item)
            if len(selected) >= self.n_questions:
                break
        return selected

    async def _gen_bridge(
        self, analysis: Dict, entities: List[Dict], relations: List[Dict],
    ) -> List[Dict]:
        bridges = analysis.get("bridge_candidates", [])
        if not bridges:
            bridges = [{"entity": e.get("id") or e.get("entity_id"),
                         "description": e.get("description", "")}
                        for e in entities[:3]]

        entity_map = {
            (e.get("id") or e.get("entity_id") or ""): e for e in entities
        }
        edge_by_node: Dict[str, List[str]] = {}
        for r in relations:
            s = str(r.get("source") or r.get("src_id", ""))
            t = str(r.get("target") or r.get("tgt_id", ""))
            edge_by_node.setdefault(s, []).append(t)
            edge_by_node.setdefault(t, []).append(s)

        all_questions: List[Dict] = []
        for bridge in bridges[:3]:
            if len(all_questions) >= self.n_questions:
                break
            b_name = bridge.get("entity", "")
            if not isinstance(b_name, str):
                continue
            b_desc = bridge.get("description") or (
                entity_map.get(b_name, {}).get("description", "")
            )
            neighbors = edge_by_node.get(b_name, [])

            prompt = BRIDGE_QUESTION_PROMPT.format(
                bridge_entity=b_name,
                description=b_desc,
                neighbors_from_doc1=", ".join(neighbors[:5]),
                neighbors_from_doc2=", ".join(neighbors[5:10]),
                n_questions=self.n_questions - len(all_questions),
            )

            try:
                raw = await self.llm_func(prompt)
                parsed = _safe_json_parse(raw)
                if isinstance(parsed, list):
                    all_questions = self._bounded_questions(all_questions + parsed)
            except StudyBudgetStop:
                raise
            except Exception as exc:
                _log.warning("[question_gen] bridge gen failed: %s", exc)

        return all_questions

    async def _gen_two_hop(
        self, analysis: Dict, entities: List[Dict], relations: List[Dict],
    ) -> List[Dict]:
        gaps = analysis.get("two_hop_gaps", [])
        if not gaps:
            return []

        prompt = TWO_HOP_INFERENCE_PROMPT.format(
            two_hop_paths_json=json.dumps(gaps[:5], ensure_ascii=False),
        )

        try:
            raw = await self.llm_func(prompt)
            parsed = _safe_json_parse(raw)
            return parsed if isinstance(parsed, list) else []
        except StudyBudgetStop:
            raise
        except Exception as exc:
            _log.warning("[question_gen] two_hop gen failed: %s", exc)
            return []

    async def _gen_comparison(
        self, analysis: Dict, entities: List[Dict], relations: List[Dict],
    ) -> List[Dict]:
        pairs = analysis.get("same_type_pairs", [])
        if not pairs:
            return []

        prompt = COMPARISON_QUESTION_PROMPT.format(
            same_type_pairs_json=json.dumps(pairs[:5], ensure_ascii=False),
        )

        try:
            raw = await self.llm_func(prompt)
            parsed = _safe_json_parse(raw)
            return parsed if isinstance(parsed, list) else []
        except StudyBudgetStop:
            raise
        except Exception as exc:
            _log.warning("[question_gen] comparison gen failed: %s", exc)
            return []

    async def _gen_contradiction(self, analysis: Dict) -> List[Dict]:
        candidates = analysis.get("potential_contradictions", [])
        if not candidates:
            return []

        prompt = CONTRADICTION_QUESTION_PROMPT.format(
            contradiction_candidates_json=json.dumps(
                candidates[:5], ensure_ascii=False,
            ),
        )

        try:
            raw = await self.llm_func(prompt)
            parsed = _safe_json_parse(raw)
            return parsed if isinstance(parsed, list) else []
        except StudyBudgetStop:
            raise
        except Exception as exc:
            _log.warning("[question_gen] contradiction gen failed: %s", exc)
            return []

    async def _gen_edge_validation(
        self, analysis: Dict, entities: List[Dict], relations: List[Dict],
    ) -> List[Dict]:
        weak = analysis.get("weak_edges", [])
        if not weak:
            return []

        prompt = EDGE_VALIDATION_PROMPT.format(
            weak_edges_json=json.dumps(weak[:5], ensure_ascii=False),
        )

        try:
            raw = await self.llm_func(prompt)
            parsed = _safe_json_parse(raw)
            return parsed if isinstance(parsed, list) else []
        except StudyBudgetStop:
            raise
        except Exception as exc:
            _log.warning("[question_gen] edge validation gen failed: %s", exc)
            return []

    async def _gen_component(
        self, analysis: Dict, entities: List[Dict], relations: List[Dict],
    ) -> List[Dict]:
        isolated = analysis.get("isolated_clusters", [])
        if not isolated:
            return []

        prompt = COMPONENT_BRIDGING_PROMPT.format(
            isolated_clusters_json=json.dumps(isolated[:3], ensure_ascii=False),
            nearest_main_entities_json=json.dumps(
                [{"name": e.get("id") or e.get("entity_id"),
                  "description": e.get("description") or ""}
                 for e in entities[:5]],
                ensure_ascii=False,
            ),
        )

        try:
            raw = await self.llm_func(prompt)
            parsed = _safe_json_parse(raw)
            return parsed if isinstance(parsed, list) else []
        except StudyBudgetStop:
            raise
        except Exception as exc:
            _log.warning("[question_gen] component gen failed: %s", exc)
            return []
