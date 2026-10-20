"""Deterministic, bounded stage proposals from frozen method schemas.

No model, network, file, store or background-loop access. The host grants
retrieval, executes actions, records their IDs and provides current sources.
Proposed/seen instructions never count as executed or semantically verified.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence

from .insight_schema import MAX_SCHEMAS, METHOD_POLICIES, adapt_insight_schemas
from .method_planner import MAX_SOURCES, compile_method_plan
from .research_source_contract import SOURCE_LABEL_PATTERN
from .research_citation_syntax import parse_citation_groups

MAX_ACTIONS_PER_STAGE = 2
MAX_PROPOSAL_BYTES = 5_000
MAX_CONTEXT_BYTES = 1_200
MAX_QUERY_BYTES = 900
STAGES = frozenset({"before_retrieval", "after_evidence", "before_delivery"})
_SOURCE_LABEL = SOURCE_LABEL_PATTERN


def _sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).encode("utf-8"))


def _text(value, limit=1024):
    if not isinstance(value, str) or len(value) > limit:
        return ""
    try:
        value.encode("utf-8")
    except UnicodeError:
        return ""
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _ids(value):
    return {_text(item, 256) for item in value[:128]} - {""} if _sequence(value) else set()


class InsightRuntime:
    """One request's frozen schemas; snapshots carry the host execution ledger.

    ``on_stage`` does not mutate the snapshot, schemas, or an internal ledger.
    Repeating a snapshot yields identical action IDs. The host passes executed
    IDs and consumed supplement rounds on subsequent calls. A revision changes
    IDs; it does not reset the per-request supplemental-retrieval budget.
    """

    def __init__(self, question, support, *, mode="auto", literal=False):
        if mode not in {"auto", "off"}:
            raise ValueError("Unknown insight runtime mode")
        self.question = _text(question, 8192).strip()
        self.disabled = mode == "off" or literal or not self.question
        self.schemas = () if self.disabled else adapt_insight_schemas(support)

    def schema_refs(self):
        return [schema.reference() for schema in self.schemas]

    def _sources(self, snapshot):
        raw = snapshot.get("sources", [])
        rows = list(raw[:MAX_SOURCES + 1]) if _sequence(raw) else []
        revoked_ids = _ids(snapshot.get("revoked_source_ids"))
        revoked_docs = _ids(snapshot.get("revoked_document_ids"))
        kept, revoked_labels = [], []
        for row in rows:
            if not isinstance(row, Mapping):
                kept.append(row)
                continue
            identifier, document = _text(row.get("id"), 256), _text(row.get("document_id"), 256)
            unavailable = (identifier in revoked_ids or document in revoked_docs
                           or row.get("revoked") is True or row.get("available") is False
                           or row.get("status") in ("revoked", "unavailable", "deleted"))
            if unavailable:
                label = _text(row.get("label"), 8)
                if _SOURCE_LABEL.fullmatch(label):
                    revoked_labels.append(label)
            else:
                kept.append(row)
        return kept, sorted(set(revoked_labels))

    def on_stage(self, stage, snapshot):
        if stage not in STAGES:
            raise ValueError("Unknown insight stage")
        if not isinstance(snapshot, Mapping):
            raise ValueError("Insight snapshot must be an object")
        schemas = self.schemas
        if "active_method_labels" in snapshot:
            labels = _ids(snapshot["active_method_labels"])
            schemas = tuple(schema for schema in schemas if schema.label in labels)
        result = {"kind": "insight_proposal", "version": 1, "stage": stage,
                  "status": "ready", "actions": [], "checks": {},
                  "trace": {"algorithm": "bounded_stage_policy_v1", "model_calls": 0,
                            "network_calls": 0, "memory_writes": 0, "schema_count": len(schemas),
                            "max_schemas": MAX_SCHEMAS, "max_actions": MAX_ACTIONS_PER_STAGE,
                            "benefit_estimated": False, "execution_claimed": False}}
        if self.disabled or not schemas:
            result["status"] = "disabled" if self.disabled else "not_applicable"
            return result
        if snapshot.get("provenance_valid") is False:
            result["status"] = "source_unavailable"
            result["checks"] = {"source_access": "unavailable", "entailment_verified": False}
            return result

        rows, revoked = self._sources(snapshot)
        cards = [schema.method_card() for schema in schemas]
        plan = compile_method_plan(cards, rows, skill_id=schemas[0].pack_id)
        result["checks"] = {
            "structural_gaps": plan["structural_gaps"],
            "evidence": plan["evidence"], "source_labels": plan["source_labels"],
            "revoked_labels": revoked,
            "semantic_premises": "unknown", "stop_conditions": "unknown",
            "unknown_premise_count": sum(len(schema.required_facts) for schema in schemas),
            "entailment_verified": False, "evidence_independence_verified": False,
        }
        # Include task constraints, exact source text/geometry and revocation
        # status in the fingerprint, but never disclose source text in receipts.
        source_fingerprint = []
        for row in rows:
            if not isinstance(row, Mapping):
                source_fingerprint.append({"invalid": True})
                continue
            source_fingerprint.append({key: _text(row.get(key), 64000)
                                       for key in ("id", "document_id", "label", "content")})
            source_fingerprint[-1]["geometry"] = [str(row.get(key, ""))[:64]
                                                   for key in ("excerpt_start", "excerpt_end", "original_content_length")]
        revision = _text(str(snapshot.get("task_revision", "")), 256)
        state = _hash({"question": self.question, "revision": revision, "sources": source_fingerprint,
                       "active_methods": [schema.label for schema in schemas],
                       "revoked": revoked, "answer": _text(snapshot.get("answer"), 120000)})
        executed = _ids(snapshot.get("executed_action_ids"))
        actions = []
        selected_count = 0

        def propose(schema, kind, payload):
            nonlocal selected_count
            # Choose a fixed bounded set before replay suppression. Completing
            # its first two actions must not admit an unbounded tail on replay.
            if selected_count >= MAX_ACTIONS_PER_STAGE:
                return
            selected_count += 1
            identifier = _hash({"schema": schema.version, "stage": stage,
                                "state": state, "kind": kind, "payload": payload})
            if identifier not in executed:
                actions.append({"id": identifier, "kind": kind, "schema_id": schema.id,
                                "schema_version": schema.version, "payload": payload,
                                "selection_basis": "selected_method_check_coverage_not_measured_utility"})

        if stage == "before_retrieval":
            rounds = snapshot.get("supplement_rounds", 0)
            allowed = snapshot.get("allow_supplemental_retrieval") is True
            if type(rounds) is not int or rounds < 0 or rounds >= 1 or not allowed:
                result["status"] = "budget_exhausted" if type(rounds) is int and rounds >= 1 else "no_action"
            else:
                anchor = " ".join(self.question.split())[:180]
                for schema in schemas:
                    dimension = " ".join(schema.dimension_terms)
                    query = anchor + " " + dimension
                    if len(query.encode("utf-8")) > MAX_QUERY_BYTES:
                        continue
                    propose(schema, "supplemental_query", {
                        "query": query, "dimension": schema.dimension,
                        "dimension_terms": list(schema.dimension_terms),
                        "requires_task_anchor": True})
            # No source selection has happened at this stage.
            result["checks"] = {"semantic_premises": "unknown", "entailment_verified": False}
        elif stage == "after_evidence":
            for schema in schemas:
                propose(schema, "analysis_constraint", {
                    "code": schema.method_id, "text": schema.analysis_check,
                    "method_label": schema.label, "semantic_premises": "unknown"})
        else:
            citations = parse_citation_groups(snapshot.get("answer"))
            used = set(citations['labels'])
            # A declared label whose excerpt is invalid does not become valid
            # merely because it appeared in the selected source array.
            source_labels = {row["label"] for row in rows if isinstance(row, Mapping)
                             and row.get("label") in plan["source_labels"]
                             and compile_method_plan([], [row], skill_id="")["evidence"]["usable_sources"] == 1}
            method_labels = {schema.label for schema in schemas}
            payload = {"unknown_source_labels": sorted(label for label in used
                                                        if label.startswith("S") and label not in source_labels)[:32],
                       "unknown_method_labels": sorted(label for label in used
                                                        if label.startswith("K") and label not in method_labels)[:32],
                       "missing_source_citations": bool(source_labels and not used.intersection(source_labels)),
                       "citation_parse_truncated": citations['truncated'],
                       "unknown_premise_count": result["checks"]["unknown_premise_count"],
                       "revoked_labels": revoked, "structural_gaps": plan["structural_gaps"],
                       "entailment_verified": False}
            propose(schemas[0], "delivery_check", payload)

        result["actions"] = actions
        if not actions and result["status"] == "ready":
            result["status"] = "no_action"
        while _size(result) > MAX_PROPOSAL_BYTES and result["actions"]:
            result["actions"].pop()
            result["trace"]["budget_omitted"] = True
        if not result["actions"] and result["trace"].get("budget_omitted"):
            result["status"] = "budget_omitted"
        assert _size(result) <= MAX_PROPOSAL_BYTES
        return result

    @staticmethod
    def render_context(proposal):
        """Render only known analysis checks, for the existing method budget.

        The host must put this inside its existing 3,500-byte method allowance
        and fit it against actual retained sources and its total prompt budget.
        """
        if not isinstance(proposal, Mapping) or proposal.get("status") != "ready":
            return ""
        rows = proposal.get("actions", [])
        if not _sequence(rows):
            return ""
        checks = []
        for action in rows[:MAX_ACTIONS_PER_STAGE]:
            if not isinstance(action, Mapping) or action.get("kind") != "analysis_constraint":
                continue
            payload = action.get("payload")
            if not isinstance(payload, Mapping):
                continue
            code = payload.get("code")
            policy = METHOD_POLICIES.get(code) if isinstance(code, str) else None
            label = _text(payload.get("method_label"), 8)
            if policy and re.fullmatch(r"K[1-9][0-9]{0,3}", label):
                checks.append({"method": label, "code": code, "check": policy[3]})
        if not checks:
            return ""
        structural = proposal.get("checks", {})
        gaps = structural.get("structural_gaps", []) if isinstance(structural, Mapping) else []
        known_gaps = {"invalid_sources", "source_limit_exceeded", "invalid_source_labels",
                      "duplicate_source_labels", "empty_excerpts", "missing_document_ids",
                      "invalid_source_ids", "invalid_excerpt_geometry", "no_usable_excerpts",
                      "insufficient_distinct_documents"}
        data = {"kind": "insight_stage_checks", "semantic_premises": "unknown",
                "entailment_verified": False, "checks": checks,
                "structural_gaps": [gap for gap in gaps[:12] if isinstance(gap, str) and gap in known_gaps]
                if _sequence(gaps) else []}
        context = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return context if len(context.encode("utf-8")) <= MAX_CONTEXT_BYTES else ""
