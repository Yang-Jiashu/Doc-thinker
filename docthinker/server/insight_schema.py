"""Immutable adapters for the already selected, data-only method cards.

Schemas preserve conditions as text. Adapting a card neither verifies its
premises nor grants access to any document or tool.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

MAX_SCHEMAS = 3
MAX_CARD_BYTES = 3_500
_LABEL = re.compile(r"K[1-9][0-9]{0,3}\Z")

# Explicit implementation policy, not instructions interpreted from a card.
# The first tuple element is its parent pack; the others define a bounded
# evidence-search dimension and the analysis check the method suggests.
METHOD_POLICIES = MappingProxyType({
    "compare-like-with-like": ("literature-synthesis", "study_conditions",
        ("对象", "对照", "指标", "样本", "conditions", "controls", "metrics", "dataset"),
        "比较对象、对照、指标和测量条件；关键条件未核对时列为不可比，不强排效果优劣。"),
    "trace-conflicts-and-shared-evidence": ("literature-synthesis", "conflicting_results",
        ("分歧", "反例", "数据来源", "实验批次", "conflict", "counterevidence", "provenance", "replication"),
        "对照结论方向及适用条件；共享数据或转述不能按独立研究计票，来源依赖不明时保留未知。"),
    "bound-the-evidence-gap": ("literature-synthesis", "search_scope",
        ("检索", "范围", "局限", "search", "scope", "limitations"),
        "未找到证据仅表示本次已查材料的缺口；说明检索范围，不推断整个领域尚无研究。"),
    "observation-to-testable-hypothesis": ("research-ideation", "observations",
        ("观察", "结果", "变量", "测量", "observation", "result", "variable", "measurement"),
        "分开原文观察、候选解释和可检验预测；解释及其前提未验证，需列出可能反驳它的结果。"),
    "transfer-structure-not-vocabulary": ("research-ideation", "mechanism",
        ("机制", "角色", "关系", "约束", "mechanism", "relation", "constraint"),
        "类比先对照角色、关系及约束；相同词语不能证明可迁移，迁移只作为带失效条件的候选假设。"),
    "cheapest-discriminating-check": ("research-ideation", "discriminating_test",
        ("对照", "混杂", "反证", "验证", "control", "confound", "falsification", "validation"),
        "先检查不同解释能否产生可区分的预测，再比较验证成本；缺少数据时列待验证项。"),
    "audience-claim-evidence": ("evidence-writing", "claim_support",
        ("证据", "结果", "论点", "用途", "evidence", "result", "claim", "audience"),
        "按交付用途组织论点与证据；缺少支撑的判断标为待证，删除不影响判断的重复背景。"),
    "claim-boundary-counterevidence": ("evidence-writing", "counterevidence",
        ("反例", "样本", "局限", "相关", "counterevidence", "sample", "limitation", "correlation"),
        "保留结论的样本范围和反证；相关性不得直接写成因果，未解释的相反证据列为争议。"),
    "deliverable-readiness-check": ("evidence-writing", "delivery_requirements",
        ("要求", "数字", "日期", "引用", "requirement", "number", "date", "citation"),
        "逐项对照用户交付要求，核对数字、日期和引用；仍无法验证的项目明确列出。"),
    "constraints-before-ranking": ("decision-review", "decision_constraints",
        ("约束", "成本", "收益", "周期", "constraint", "cost", "benefit", "period"),
        "比较方案先核对硬约束和数据口径；未知约束保持待核实，不用加权总分抵消明确限制。"),
    "sensitivity-and-reversibility": ("decision-review", "sensitivity",
        ("假设", "范围", "撤回", "损失", "assumption", "range", "reversal", "loss"),
        "检查关键假设在有依据的范围内变化时是否改变排序；撤回容易不等于错误代价低。"),
    "stage-gates-before-scale": ("decision-review", "stage_gates",
        ("基线", "验证", "门槛", "损失", "baseline", "validation", "threshold", "loss"),
        "试点列明同口径基线/对照、样本与时间、指标测法和通过/停止/数据不足门槛；建议阈值须标待确认，不以显著或可接受代替可测条件。区分拟议验证与论文已测结果，不事后降低门槛。"),
    "investigate-before-judgment": ("mao-workplace", "timeline",
        ("时间线", "日期", "记录", "延期", "timeline", "date", "record", "delay"),
        "把事实、来源和解释分开；没有对应记录的解释保持待验证，不根据单方叙述推断动机。"),
    "identify-current-bottleneck": ("mao-workplace", "dependencies",
        ("依赖", "瓶颈", "阻塞", "资源", "dependency", "bottleneck", "constraint", "resource"),
        "先核对交付目标与依赖记录，再提出候选瓶颈；没有依赖证据时不把相关因素当决定原因。"),
    "test-and-revise": ("mao-workplace", "trial_baseline",
        ("基线", "指标", "试点", "期限", "baseline", "metric", "trial", "deadline"),
        "提出试点时区分基线、预测和否定条件；权限、风险或指标未知时仅给出待核查建议。"),
    "coordinate-with-shared-context": ("mao-workplace", "handoff",
        ("负责人", "交付", "期限", "依赖", "owner", "delivery", "deadline", "dependency"),
        "核对负责人、交付标准、期限和前置依赖；信息不足或分歧保持待决定项，不代替他人承诺。"),
})


def _strings(value, *, limit=8):
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= limit:
        return None
    if any(not isinstance(item, str) or not item.strip() or len(item) > 500 for item in value):
        return None
    return tuple(value)


@dataclass(frozen=True)
class InsightSchema:
    id: str
    version: str
    pack_id: str
    method_id: str
    label: str
    required_facts: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    dimension: str
    dimension_terms: tuple[str, ...]
    analysis_check: str

    def reference(self):
        return {"id": self.id, "version": self.version, "label": self.label,
                "validation_status": "design_unvalidated"}

    def method_card(self):
        return {"id": self.method_id, "label": self.label,
                "required_facts": list(self.required_facts),
                "stop_conditions": list(self.stop_conditions)}


def adapt_insight_schemas(support):
    """Freeze at most three already selected methods; never read a pack or file.

    All card data and the fixed stage policy contribute to its version hash,
    including boundaries and provenance. Changing a query dimension or check
    invalidates past action fingerprints even when the card stays unchanged.
    Unknown policies and malformed or duplicate identities fail closed.
    Natural-language requirements remain unknown in the runtime.
    """
    if not isinstance(support, Mapping):
        return ()
    trace, skill, rows = support.get("trace"), support.get("skill"), support.get("sources")
    if (not isinstance(trace, Mapping) or trace.get("status") != "ready"
            or not isinstance(skill, Mapping) or not isinstance(rows, Sequence)
            or isinstance(rows, (str, bytes)) or not support.get("prompt")):
        return ()
    pack_id, pack_version = skill.get("id"), skill.get("version")
    if not isinstance(pack_id, str) or not isinstance(pack_version, str):
        return ()
    rows = rows[:MAX_SCHEMAS]
    labels = [row.get("label") for row in rows if isinstance(row, Mapping)]
    identifiers = [row.get("id") for row in rows if isinstance(row, Mapping)]
    schemas = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        method_id, label = row.get("id"), row.get("label")
        policy = METHOD_POLICIES.get(method_id) if isinstance(method_id, str) else None
        if (policy is None or policy[0] != pack_id or not isinstance(label, str)
                or not _LABEL.fullmatch(label) or labels.count(label) != 1
                or identifiers.count(method_id) != 1):
            continue
        required, stops = _strings(row.get("required_facts")), _strings(row.get("stop_conditions"))
        if required is None or stops is None:
            continue
        try:
            data = json.dumps({"adapter_version": 1, "pack_version": pack_version,
                               "pack_id": pack_id, "card": dict(row),
                               "stage_policy": {"dimension": policy[1],
                                                "dimension_terms": policy[2],
                                                "analysis_check": policy[3]}}, sort_keys=True,
                              ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            continue
        if len(data) > MAX_CARD_BYTES:
            continue
        schemas.append(InsightSchema(
            id=f"{pack_id}/{method_id}", version=hashlib.sha256(data).hexdigest(),
            pack_id=pack_id, method_id=method_id, label=label,
            required_facts=required, stop_conditions=stops, dimension=policy[1],
            dimension_terms=policy[2], analysis_check=policy[3]))
    return tuple(schemas)
