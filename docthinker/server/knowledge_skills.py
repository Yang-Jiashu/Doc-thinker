"""Harness-routed, bounded retrieval from data-only product knowledge skills.

The catalog is cheap metadata. Selecting a catalog ID permits local lexical
lookup, not arbitrary filesystem access, code execution, network access, or a
memory write. ``SKILL.md`` is packaging documentation and is never executed or
injected into the model. K references are methodological references; they must
never be relabeled as the user's S evidence or as facts about their situation.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from .research_retrieval import lexical_tokens

MAX_PROMPT_BYTES = 3_500
MAX_SOURCES = 3
MAX_PACK_BYTES = 65_536
MAX_ENTRIES = 32
MAX_QUERY_BYTES = 8_192
MAX_QUERY_TERMS = 128
KNOWLEDGE_SCOPE = "method_reference_not_user_evidence"

_REGISTRY = MappingProxyType({
    "mao-workplace": {
        "id": "mao-workplace",
        "name": "职场方法 · 毛选启发",
        "version": "2",
        "description": "按任务查询原创的事实核实、优先级、试点与协作方法；历史原文只作启发出处。",
        "kind": "knowledge_skill",
        "knowledge_scope": KNOWLEDGE_SCOPE,
        "authorship": "project_original_method_design",
        "validation_status": "design_unvalidated",
    },
    **{identifier: {
        "id": identifier, "name": name, "version": "2" if identifier == "decision-review" else "1", "description": description,
        "kind": "knowledge_skill", "knowledge_scope": KNOWLEDGE_SCOPE,
        "authorship": "project_original_method_design", "validation_status": "design_unvalidated",
    } for identifier, name, description in (
        ("research-ideation", "科研选题 · 可检验假设", "把研究观察组织成可证伪假设、迁移条件和最小验证；不把想法冒充已证实结论。"),
        ("literature-synthesis", "论文综合 · 证据对照", "跨论文对齐对象、指标和条件，整理冲突、重复证据及未覆盖问题。"),
        ("evidence-writing", "长文写作 · 论点与证据", "为报告和长文组织受众、论点、证据及反例，检查交付是否完整可用。"),
        ("decision-review", "方案决策 · 取舍与验证", "比较工作方案的硬约束、敏感因素和可逆验证步骤；不代替重大决策或专业意见。"),
    )},
})
_ORIGINAL_SKILLS = frozenset(_REGISTRY) - {"mao-workplace"}
_PACK_ROOT = Path(__file__).resolve().parents[1] / "knowledge_skills"
_SOURCE_HOSTS = frozenset({"www.12371.cn", "news.12371.cn"})
_ID_RE = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_GENERIC_TERMS = frozenset(
    "毛选 职场 工作 公司 老板 领导 同事 员工 问题 处理 事情 这个 这种 通过 找到 "
    "进行 应该 方法 分析 建议 认为 里面 今天 现在 情况 结合 根据 参考 帮助 能够 "
    "怎样 咋办 怎么办 怎么 根据 请你 不要 不能 可以 是否 用到 我的 我该 一些 "
    "work workplace company boss colleague colleagues problem problems method methods "
    "use using help advice need want think situation today".split()
)
_TERM_FORMS = {'hypotheses': 'hypothesis', 'predictions': 'prediction', 'claims': 'claim',
               'reports': 'report', 'plans': 'plan', 'papers': 'paper', 'studies': 'study',
               'tradeoffs': 'tradeoff', 'constraints': 'constraint', 'vendors': 'vendor',
               'conditions': 'condition', 'experiments': 'experiment', 'gaps': 'gap'}
_ENTRY_KEYS = frozenset({
    "id", "title", "source_title", "source_url", "published", "excerpt",
    "commentary", "retrieval_keywords", "boundaries",
    "problem_pattern", "required_facts", "decision_rule", "action_template",
    "stop_conditions", "counterexample", "reasoning_role", "task_signals",
})
_PACK_KEYS = frozenset({
    "schema_version", "version", "kind", "knowledge_scope", "id", "title",
    "description", "license_note", "entries",
    "authorship", "validation_status",
})
_OPT_OUT_RE = re.compile(
    r"(?:不要|不用|无需|不需要|不想|别|禁止|关闭|禁用).{0,12}(?:毛选|毛泽东|方法参考)|"
    r"(?:毛选|毛泽东|方法参考).{0,6}(?:不用|不要用|先别用|关闭)|"
    r"(?:do\s+not|don['’]t|no)\s+(?:use\s+)?mao\b", re.I,
)
_KNOWLEDGE_OPT_OUT_RE = re.compile(
    r"(?:不要|不用|无需|不需要|不想|别|禁止|关闭|禁用).{0,12}(?:知识辅助|知识包|知识技能|方法辅助|辅助推理|skills?)|"
    r"(?:知识辅助|知识包|知识技能|方法辅助|辅助推理|skills?).{0,6}(?:关闭|禁用|不用|不要用)|"
    r"(?:do\s+not|don['’]t|no)\s+(?:use\s+)?(?:knowledge\s+(?:skills?|assistance)|skills?)\b",
    re.I,
)
_WORK_OPT_OUT_RE = re.compile(r"(?:不要|不用|无需|不需要|别).{0,8}(?:工作方法|职场方法|职场建议)")
_QUOTED_RE = re.compile(r"```[\s\S]*?```|`[^`]*`|“[^”]*”|‘[^’]*’|\"[^\"]*\"|'[^'\n]*'")
# Only explicitly delimited material is removed. This is a conservative local
# intent view, not a parser that can prove arbitrary prose is trusted. It is
# bounded by MAX_QUERY_BYTES and never changes the question/evidence sent to
# the writer. An unclosed marked source remains source through the end.
_MARKED_SOURCE_RE = re.compile(
    r"<(source|document|untrusted_text)\b[^>]*>[\s\S]*?(?:</\1\s*>|\Z)|"
    r"(?:^|\n)[ \t]*(?:#{1,6}[ \t]*)?(?:(?:请看|请阅读)[ \t]*)?"
    r"(?:以下资料|资料原文|参考资料|材料如下|资料|材料|原文|excerpt|"
    r"source(?:\s+(?:excerpt|text))?|document(?:\s+(?:excerpt|text))?)[ \t]*[:：]"
    r"[\s\S]*?(?=\n[ \t]*(?:我的(?:要求|问题)|用户(?:要求|请求)|"
    r"(?:my|user)\s+(?:request|question))[ \t]*[:：]|\Z)", re.I)
_BLOCK_QUOTE_RE = re.compile(r"^[ \t]*>[^\n]*(?:\n|$)", re.M)
_LITERAL_RE = re.compile(
    r"翻译|改写|润色|逐字|仅引用|只引用|解释这(?:句|段)|"
    r"(?:作者|原文|论文)[^，,。.;；!?！？\n]{0,12}(?:说(?!明)|表述|描述|认为|指出|建议|提到|看法)|"
    r"(?:总结|概括|复述).{0,12}(?:这篇|这段|作者|原文|论文)|"
    r"\b(?:translate|rewrite|paraphrase|proofread|quote\s+verbatim)\b|"
    r"\b(?:paper|author|source)\b[^，,。.;；!?！？\n]{0,25}\b(?:say|says|said|suggests?|describes?|states?)\b",
    re.I,
)
_TECHNICAL_RE = re.compile(
    r"科研|论文|实验|模型|算法|代码|程序|训练|编程|数据库|向量|检索|公式|数学|"
    r"(?<![A-Za-z0-9_])(?:gpu|tokens?|api|sql|rag|llm|transformers?|python|javascript|latency|benchmark)(?![A-Za-z0-9_])",
    re.I,
)
_TECHNICAL_OBJECT = (
    r"(?:优先级队列|调度算法|代码|算法|模型(?:训练)?|程序|数据结构|数据库|"
    r"SQL(?:查询)?|API(?:请求)?|GPU|python|javascript|priority\s+queue|scheduler)"
)
_TECHNICAL_IMPLEMENTATION_RE = re.compile(
    # A boss/customer mention must not turn an implementation request into
    # workplace advice. Prefer missing an ambiguous mixed request to adding an
    # unrelated historical method to a technical answer.
    rf"(?:实现|编写|调试|修复|优化|加速|计算|推导).{{0,12}}{_TECHNICAL_OBJECT}|"
    rf"{_TECHNICAL_OBJECT}(?!团队|部门|人员)[^，。；;!?！?]{{0,24}}(?:怎么|如何|怎样).{{0,6}}(?:实现|编写|计算|求解|推导|调试|修复|优化|加速)|"
    rf"\b(?:implement|debug|optimi[sz]e|code|compute)\b.{{0,24}}{_TECHNICAL_OBJECT}|"
    r"\b(?:write|create|build|generate)\s+(?:(?:a|an|the|some|new)\s+)?"
    r"(?:(?:python|javascript|typescript|java|rust|golang|go|c\+\+|c#|ruby|php|swift)\s+)?"
    r"(?:code|programs?|functions?|scripts?)\b", re.I,
)
_NEGATED_WORK_TOPIC_RE = re.compile(
    r"(?:不是|并非|不涉及).{0,3}(?:沟通|协作|协同|分工|汇报|会议|职场)(?:方面的|的)?(?:问题|任务|话题|讨论)"
)
_NEGATED_REQUEST_RE = re.compile(
    r"(?:不要|不用|无需|不需要|不必|别|不是要|不是让你)[^，。；;!?！？\n]{0,100}|"
    r"\b(?:don't|do\s+not|no\s+need\s+to)\b[^,;.!?\n]{0,160}", re.I,
)
_CALCULATION_ONLY_RE = re.compile(
    r"(?:只|仅|仅仅).{0,8}(?:计算|算一下|算出|求值|求解|换算)|"
    r"\b(?:just|only)\s+(?:calculate|compute|evaluate)\b", re.I,
)
_DEFINITION_ONLY_RE = re.compile(
    r"(?:这个词|这个术语|这个概念).{0,10}(?:意思|含义|定义)|"
    r"解释.{0,6}(?:这个词|这个术语|这个概念)|是什么含义|是什么意思|"
    r"什么是(?:决策矩阵|证伪|文献综述)|\b(?:what\s+does\b.{0,60}\bmean|define\s+)\b", re.I,
)
_HUMAN_WORK_RE = re.compile(
    r"同事|跨部门|领导|老板|客户|员工|人手|人力|负责人|职场|协同|分工|汇报|会议|"
    r"团队.{0,8}(?:沟通|协作|冲突|分歧)|\b(?:coworker|colleague|boss|stakeholder|meeting)s?\b",
    re.I,
)
_WORK_CONTEXT_RE = re.compile(
    r"项目|团队|同事|部门|领导|老板|客户|业务|交付|需求|排期|人手|人力|预算|职场|"
    r"工作|任务|流程|会议|负责人|\b(?:project|team|workplace|task|deadline|manager|department)s?\b",
    re.I,
)
_TASK_PATTERNS = (
    ("coordination", re.compile(r"跨部门|沟通|协作|协同|分工|汇报|会议|对齐|信息不同|信息不一致|"
                               r"\b(?:coordinat\w*|communicat\w*|handoff|meetings?)\b", re.I)),
    ("prioritization", re.compile(r"优先级|瓶颈|资源不足|人手不足|目标冲突|取舍|先做哪个|先做什么|"
                                 r"\b(?:prioriti\w*|bottleneck|tradeoff|trade-off)\b", re.I)),
    ("bounded_trial", re.compile(r"试点|试行|小范围|停止条件|验证方案|方案验证|反馈修正|"
                                 r"\b(?:pilot|trial|test\s+(?:the\s+)?plan)\b", re.I)),
    ("fact_finding", re.compile(r"项目延期|延期原因|调查|核实|甩锅|误会|信息不足|信息不完整|"
                                r"\b(?:investigat\w*|verify\s+facts|project\s+delay|blam\w*)\b", re.I)),
)

# Specific deliverables take precedence over incidental workplace words. One
# pack per request keeps lookup/model input bounded as the catalog grows. This
# is a transparent priority rule, not a calibrated semantic intent classifier.
_SCIENCE_CONTEXT = re.compile(r"科研|论文|文献|研究|实验|\b(?:research|papers?|literature|scientific|experiment)s?\b", re.I)
_TASK_SKILL_ROUTES = (
    ("research-ideation", "research_ideation", _SCIENCE_CONTEXT, re.compile(
        r"(?:提出|构思|设计|生成|寻找|探索|挖掘|推荐|想出|帮我找).{0,24}(?:选题|研究方向|研究问题|假设|idea)|"
        r"(?:选题|研究方向|研究问题|假设).{0,12}(?:怎么|如何).{0,6}(?:找|提|设|构思|验证)|"
        r"(?:帮我|请).{0,8}(?:科研选题|研究选题)|"
        r"\b(?:propose|generate|develop|brainstorm|design)\b.{0,45}\b(?:hypothes\w*|research\s+(?:ideas?|questions?|directions?))\b", re.I)),
    ("literature-synthesis", "literature_synthesis", re.compile(
        r"(?:这些|这几篇|多篇|几篇|两篇|三篇|不同|这两篇).{0,8}(?:论文|文献|研究)|文献综述|跨论文|"
        r"\b(?:papers|studies|literature)\b", re.I), re.compile(
        r"综合|综述|对比|比较|分歧|冲突|证据矩阵|研究缺口|列成表|整理表格|证据对照|"
        r"\b(?:synthesi[sz]e|synthesis|review|compare|contradict\w*|conflict\w*|gaps?)\b", re.I)),
    ("evidence-writing", "evidence_writing", re.compile(
        r"报告|长文|提案|策划书|分析文章|论证|论点|\b(?:reports?|proposals?|essays?|arguments?)\b", re.I), re.compile(
        r"(?:写|起草|撰写|组织|梳理|搭建|规划).{0,25}(?:报告|长文|提案|策划书|分析文章|论证|论点)|"
        r"(?:报告|长文|提案|论证|论点).{0,20}(?:怎么写|如何写|怎么组织|证据|结构)|"
        r"(?:检查|核查|审查|审阅).{0,30}(?:报告|长文|提案|论点|引用|证据|交付|验收)|"
        r"\b(?:write|draft|compose|structure|outline|check|review)\b.{0,45}\b(?:reports?|proposals?|essays?|arguments?)\b", re.I)),
    ("decision-review", "decision_review", re.compile(
        r"方案|项目|产品|供应商|架构|路线|选项|\b(?:plans?|options?|vendors?|proposals?|alternatives?)\b", re.I), re.compile(
        r"(?:比较|对比|评估).{0,12}(?:两[个种套]|几[个种套]|多个|不同|这[些两]).{0,12}(?:方案|项目|产品|供应商|架构|路线|选项)|"
        r"(?:方案|项目|产品|供应商|架构|路线|选项).{0,12}(?:怎么选|如何选|取舍|抉择|权衡)|决策矩阵|"
        r"\b(?:compare|choose|choosing|decide|evaluate|tradeoffs?)\b.{0,45}\b(?:plans?|options?|vendors?|proposals?|alternatives?)\b", re.I)),
)
_HIGH_STAKES_DECISION_RE = re.compile(
    r"医疗|用药|诊断|治疗|手术|法律|诉讼|投资|股票|基金|贷款|理财|保险|"
    r"\b(?:medical|medication|diagnosis|treatment|surgery|legal|lawsuit|invest\w*|stocks?|loans?|insurance)\b", re.I)

# A decision deliverable need not explicitly compare two pre-existing options.
# Require a work/product object and an actual request for a bounded planning
# artifact or ordering recommendation, not just the word "advice"/"roadmap".
# This branch keeps the existing route order and all earlier exclusion gates.
_DECISION_DELIVERABLE_CONTEXT_RE = re.compile(
    r"项目|产品|业务|功能|服务|交付|上线|发布|实施|落地|"
    r"\b(?:projects?|products?|features?|services?|delivery|roll[- ]?out|launch|implementation)\b", re.I)
_DECISION_REQUEST_RE = re.compile(
    r"(?:^|[，,。；;!?！？\n])\s*(?:(?:请(?:你)?|麻烦(?:你)?|帮我|帮我们|能否|可否|需要你|想请你).{0,16})?"
    r"(?:(?:为|针对|就)[^，。；;!?！？\n]{1,24}?)?"
    r"(?:写|起草|撰写|制定|规划|给出|提供|给(?:我|我们)?)[^，。；;!?！？\n]{0,36}"
    r"(?:路线图|路线建议|(?:实施|落地|上线|发布|交付)(?:路线|计划|建议))|"
    r"(?:^|[，,。；;!?！？\n])\s*(?:(?:请(?:你)?|帮我|帮我们).{0,12})?"
    r"(?:推荐|建议|给出)[^，。；;!?！？\n]{0,32}(?:优先做|先做|暂缓|推迟|后做)|"
    r"(?:^|[,.!?;\n])\s*(?:(?:please|can\s+you|could\s+you|would\s+you|help\s+(?:me|us))\s+)?"
    r"(?:write|draft|create|develop|design|provide|give|plan|recommend)\b[^.;!?\n]{0,70}"
    r"\b(?:roadmaps?|(?:product\s+)?roll[- ]?out(?:\s+(?:plans?|advice|guidance|recommendations?|strategy))?|"
    r"(?:implementation|launch|delivery)\s+(?:plans?|advice|guidance|recommendations?|strategy))\b|"
    r"(?:^|[,.!?;\n])\s*(?:(?:please|can\s+you|could\s+you|help\s+(?:me|us))\s+)?"
    r"(?:recommend|advise)\b[^.;!?\n]{0,70}\b(?:prioriti[sz]\w*|first|defer\w*|postpone\w*)\b", re.I)
_DECISION_OTHER_ARTIFACT_RE = re.compile(
    r"(?:写|创作)(?:一(?:首|篇|个|段))?(?:(?:关于|有关)[^，,。.;；!?！？\n]{1,36})?(?:诗(?:歌)?|故事|笑话)|"
    r"(?:介绍|解释|讲解)(?:一下)?[^，,。.;；!?！？\n]{0,24}"
    r"(?:路线图|路线建议|(?:实施|落地|上线|发布|交付)(?:路线|计划|建议))|"
    r"\b(?:write|draft|create|compose|give|provide)\s+(?:(?:me|us)\s+)?"
    r"(?:(?:a|an|the|short|brief|funny)\s+){0,3}"
    r"(?:poems?|stor(?:y|ies)|songs?|jokes?|descriptions?|explanations?|introductions?)\b", re.I)

# Natural planning deliverables often omit the exact noun "roadmap". Keep an
# imperative/request skeleton and a concrete artifact, rather than promoting
# every occurrence of design/research/advice into mandatory method routing.
_NATURAL_REQUEST_RE = re.compile(
    r"(?:^|[，,。；;!?！？\n:：])[ \t]*"
    r"(?:(?:请(?:你)?|麻烦(?:你)?|帮(?:我|我们)|能否|可否|需要你|想请你)[ \t]*)?"
    r"(?:(?:根据|结合|针对|为|给)[^，,。；;!?！？\n]{0,28}?)?"
    r"(?:写|起草|撰写|制定|拟定|规划|设计|给出|提供|给(?:我|我们|团队)?)"
    r"[^，,。；;!?！？\n]{0,120}|"
    r"(?:^|[,.!?;\n:])[ \t]*(?:(?:please|can\s+you|could\s+you|would\s+you|help\s+(?:me|us))\s+)?"
    r"(?:write|draft|create|develop|design|provide|give|plan|recommend)\b[^,.;!?\n]{0,160}", re.I)
_DESIGN_CONTEXT_RE = re.compile(
    r"模块|系统|组件|平台|流程|项目|产品|业务|功能|服务|"
    r"\b(?:modules?|systems?|components?|platforms?|workflows?|projects?|products?|features?|services?)\b", re.I)
_DESIGN_ARTIFACT_RE = re.compile(
    r"(?:设计|架构|改进|验证|试用|试点|测试)(?:建议|方案|计划)|"
    r"\b(?:design|architectural|improvement|validation|trial|pilot|test)\s+"
    r"(?:recommendations?|advice|plans?|proposals?|strategy)\b", re.I)
_DESIGN_TARGET_RE = re.compile(
    r"\s+for\s+(?:our|your|their|the|this|that|a|an)\s+"
    r"[A-Za-z][A-Za-z0-9_-]{0,47}\b", re.I)
_RESEARCH_PLAN_RE = re.compile(
    r"(?:研究|实验)(?:验证)?(?:方案|计划|设计|协议)|"
    r"(?:为|针对)[^，,。；;!?！？\n]{0,12}(?:研究|实验)(?:设计|制定|拟定)(?:验证|测试)(?:方案|计划)|"
    r"\b(?:research|experimental|experiment|study)(?:\s+validation)?\s+(?:plans?|protocols?|designs?)\b|"
    r"\b(?:validation|test)\s+plans?\s+for\s+(?:(?:this|the|our)\s+)?(?:study|experiment|research)\b", re.I)


def _request_view(question):
    """Exclude marked material/other genres without interpreting instructions."""
    question = _MARKED_SOURCE_RE.sub('\n', question)
    question = _BLOCK_QUOTE_RE.sub('\n', question)
    question = _QUOTED_RE.sub(' ', question)
    # A truncated/unclosed code fence cannot supply a fresh user request.
    question = question.split('```', 1)[0]
    return question


def _natural_deliverable(question, artifact, *, context=None, target=None):
    known_context = context is None or bool(context.search(question))
    for request in _NATURAL_REQUEST_RE.finditer(question):
        clause = request.group()
        match = artifact.search(clause)
        if match is None or _DECISION_OTHER_ARTIFACT_RE.search(clause):
            continue
        # "Design recommendations for our <named object>" is an explicit
        # artifact/object relationship even when the object is not in a domain
        # noun list. It must follow this artifact in the same bounded request;
        # an unrelated "for ..." elsewhere cannot nominate a task.
        if known_context or (target is not None and target.match(clause[match.end():])):
            return True
    return False


def _exclude_other_artifact_clauses(question):
    # A poem *about* a proposal is not a proposal. Preserve a later real
    # deliverable instead of disabling the whole mixed request.
    return '\n'.join(clause for clause in re.split(r'[，,。.;；!?！？\n]', question)
                     if not _DECISION_OTHER_ARTIFACT_RE.search(clause))


def _requests_decision_deliverable(question):
    if not _DECISION_DELIVERABLE_CONTEXT_RE.search(question):
        return False
    for match in _DECISION_REQUEST_RE.finditer(question):
        # The explicit object can be a story/explanation *about* a roadmap.
        # Inspect only this bounded request clause, not genres elsewhere in the
        # question; a later genuine planning request remains eligible. Earlier
        # report/research routes retain their original priority and exclusions.
        clause = question[match.start():match.start() + 180].lstrip(' \t\r\n，,。.;；!?！？')
        clause = re.split(r'[，,。.;；!?！？\n]', clause, maxsplit=1)[0]
        if not _DECISION_OTHER_ARTIFACT_RE.search(clause):
            return True
    return False


def route_knowledge_skill(question: Any, mode: str = "auto") -> dict[str, Any]:
    """Conservative, local task routing, independent of personal profiling.

    A specific supported deliverable must be present outside quotes/code and
    negated requests. Literal extraction and technical implementation remain
    on their existing paths; ambiguous or unrecognized tasks can go without.
    """
    if mode not in ("auto", "off"):
        raise ValueError("Unknown knowledge skill mode")

    def result(status: str, task_type: str | None = None, skill_id: str | None = None) -> dict[str, Any]:
        return {
            "skill_id": skill_id if status == "selected" else None,
            "trace": {"algorithm": "local_knowledge_task_router_v4", "status": status,
                      "task_type": task_type, "model_calls": 0, "network_calls": 0,
                      "memory_writes": 0},
        }

    if mode == "off":
        return result("disabled")
    if not isinstance(question, str):
        return result("not_applicable")
    try:
        question = question.encode("utf-8")[:MAX_QUERY_BYTES].decode("utf-8", errors="ignore")
    except UnicodeError:
        return result("not_applicable")
    question = _request_view(question)
    if _KNOWLEDGE_OPT_OUT_RE.search(question):
        return result("disabled")
    workplace_disabled = bool(_OPT_OUT_RE.search(question) or _WORK_OPT_OUT_RE.search(question))
    if _LITERAL_RE.search(question) or _DEFINITION_ONLY_RE.search(question):
        return result("not_applicable")
    unquoted = _NEGATED_REQUEST_RE.sub(' ', question)
    # An explicitly negated topic is not a human-collaboration signal. This
    # does not remove lack-of-communication statements such as "没有沟通".
    unquoted = _NEGATED_WORK_TOPIC_RE.sub(" ", unquoted)
    if _TECHNICAL_IMPLEMENTATION_RE.search(unquoted) or _CALCULATION_ONLY_RE.search(unquoted):
        return result("disabled" if workplace_disabled else "not_applicable")
    unquoted = _exclude_other_artifact_clauses(unquoted)
    for skill_id, task_type, context, action in _TASK_SKILL_ROUTES:
        if ((context.search(unquoted) and action.search(unquoted))
                or (skill_id == 'research-ideation' and _natural_deliverable(unquoted, _RESEARCH_PLAN_RE))
                or (skill_id == 'decision-review' and (
                    _requests_decision_deliverable(unquoted) or
                    _natural_deliverable(unquoted, _DESIGN_ARTIFACT_RE, context=_DESIGN_CONTEXT_RE,
                                         target=_DESIGN_TARGET_RE)))):
            if skill_id == 'decision-review' and _HIGH_STAKES_DECISION_RE.search(unquoted):
                return result('not_applicable')
            return result('selected', task_type, skill_id)
    if workplace_disabled:
        return result('disabled')
    if not _WORK_CONTEXT_RE.search(unquoted):
        return result("not_applicable")
    if _TECHNICAL_RE.search(unquoted) and not _HUMAN_WORK_RE.search(unquoted):
        return result("not_applicable")
    scored = [(len(pattern.findall(unquoted)), index, task_type)
              for index, (task_type, pattern) in enumerate(_TASK_PATTERNS)]
    count, _, task_type = max(scored, key=lambda row: (row[0], -row[1]))
    return result("selected", task_type, "mao-workplace") if count else result("not_applicable")


def knowledge_skill_catalog() -> list[dict[str, str]]:
    """Return fresh, public metadata; this does not open the knowledge packs."""
    return [dict(value) for value in _REGISTRY.values()]


def _text(value: Any, max_chars: int, *, required: bool = True) -> str:
    if not isinstance(value, str) or len(value) > max_chars:
        raise ValueError("Invalid knowledge skill text")
    if required and not value.strip():
        raise ValueError("Empty knowledge skill text")
    if any(ord(char) < 32 and char not in "\n\t" for char in value):
        raise ValueError("Invalid knowledge skill control character")
    value.encode("utf-8", errors="strict")
    return value


def _strings(value: Any, *, max_items: int, max_chars: int) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= max_items:
        raise ValueError("Invalid knowledge skill list")
    return [_text(item, max_chars) for item in value]


def _source_url(value: Any) -> str:
    value = _text(value, 500)
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or parsed.hostname not in _SOURCE_HOSTS
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 443) or not parsed.path
            or parsed.fragment or any(char.isspace() for char in value)):
        raise ValueError("Invalid knowledge skill source URL")
    return value


def _read_pack(skill_id: str) -> bytes:
    # skill_id is first checked against the fixed registry; even a selected
    # value never supplies a filename. Symlinked packs are not followed.
    path = _PACK_ROOT / skill_id / "knowledge.json"
    if path.resolve() != path.absolute():
        raise ValueError("Knowledge skill resource must be a regular packaged path")
    with path.open("rb") as resource:
        data = resource.read(MAX_PACK_BYTES + 1)
    if len(data) > MAX_PACK_BYTES:
        raise ValueError("Knowledge skill resource exceeds the size limit")
    return data


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate knowledge skill JSON key")
        value[key] = item
    return value


def _invalid_constant(_: str) -> None:
    raise ValueError("Non-finite JSON constant in knowledge skill resource")


def _load_pack(skill_id: str) -> list[dict[str, Any]]:
    data = _read_pack(skill_id)
    if not isinstance(data, bytes) or len(data) > MAX_PACK_BYTES:
        raise ValueError("Invalid knowledge skill resource")
    pack = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)
    if (not isinstance(pack, dict) or set(pack) != _PACK_KEYS
            or type(pack.get("schema_version")) is not int
            or pack["schema_version"] != 1 or pack.get("id") != skill_id
            or pack.get("version") != _REGISTRY[skill_id]["version"]
            or pack.get("kind") != "knowledge_skill"
            or pack.get("knowledge_scope") != KNOWLEDGE_SCOPE
            or pack.get("authorship") != "project_original_method_design"
            or pack.get("validation_status") != "design_unvalidated"):
        raise ValueError("Unsupported knowledge skill resource")
    # Descriptive packaging fields are validated but are not model instructions.
    _text(pack.get("title"), 100)
    _text(pack.get("description"), 1_000)
    _text(pack.get("license_note"), 2_000)
    entries = pack.get("entries")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_ENTRIES:
        raise ValueError("Invalid knowledge skill entries")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {'reference_kind'} != _ENTRY_KEYS:
            raise ValueError("Invalid knowledge skill entry schema")
        reference_kind = entry.get('reference_kind', 'historical_inspiration')
        expected_kind = 'project_original' if skill_id in _ORIGINAL_SKILLS else 'historical_inspiration'
        if reference_kind != expected_kind:
            raise ValueError('Invalid knowledge skill reference kind')
        if reference_kind == 'project_original':
            # Original methods have a local author/title, not invented quotes
            # or URLs that imply external evidence for the rule's efficacy.
            if entry.get('source_url') != '' or entry.get('excerpt') != '':
                raise ValueError('Original methods cannot impersonate external quotations')
            if not _text(entry.get('source_title'), 100).startswith('DocThinker · '):
                raise ValueError('Original methods must identify their project provenance')
        identifier = _text(entry.get("id"), 64)
        if not _ID_RE.fullmatch(identifier) or identifier in seen:
            raise ValueError("Invalid or duplicate knowledge skill entry ID")
        seen.add(identifier)
        if entry.get("reasoning_role") not in ("analogy", "deduction", "induction"):
            raise ValueError("Invalid knowledge skill reasoning role")
        result.append({
            "id": identifier,
            "title": _text(entry.get("title"), 100),
            "source_title": _text(entry.get("source_title"), 100),
            "source_url": '' if reference_kind == 'project_original' else _source_url(entry.get("source_url")),
            "published": _text(entry.get("published"), 40),
            "excerpt": _text(entry.get("excerpt"), 120, required=reference_kind != 'project_original'),
            "reference_kind": reference_kind,
            "commentary": _text(entry.get("commentary"), 600),
            "retrieval_keywords": _strings(entry.get("retrieval_keywords"), max_items=40, max_chars=60),
            "boundaries": _strings(entry.get("boundaries"), max_items=8, max_chars=300),
            "problem_pattern": _text(entry.get("problem_pattern"), 200),
            "required_facts": _strings(entry.get("required_facts"), max_items=8, max_chars=150),
            "decision_rule": _text(entry.get("decision_rule"), 300),
            "action_template": _strings(entry.get("action_template"), max_items=8, max_chars=150),
            "stop_conditions": _strings(entry.get("stop_conditions"), max_items=8, max_chars=150),
            "counterexample": _text(entry.get("counterexample"), 200),
            "reasoning_role": entry["reasoning_role"],
            "task_signals": _strings(entry.get("task_signals"), max_items=40, max_chars=60),
        })
    return result


def _terms(text: str) -> list[str]:
    # Small, task-vocabulary inflection normalization, shared by card and query.
    # It neither translates arbitrary text nor changes document/S retrieval.
    return [_TERM_FORMS.get(term, term) for term in lexical_tokens(text)
            if term not in _GENERIC_TERMS and len(term) > 1]


def _rank(question: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Weighted lexical BM25; a positive topical overlap is required.

    Handwritten retrieval keywords improve common task phrasing without
    inferring facts, using user history, or making a query-expansion model call.
    This is not semantic understanding or a guaranteed best method selector.
    """
    query_terms = set(_terms(question)[:MAX_QUERY_TERMS])
    if not query_terms:
        return []
    documents = []
    for entry in entries:
        counts: Counter[str] = Counter()
        for text, weight in ((" ".join(entry["retrieval_keywords"] + entry["task_signals"]), 3),
                             (entry["title"], 2), (entry["excerpt"], 1),
                             (entry["commentary"], 1), (entry["problem_pattern"], 2)):
            counts.update({term: count * weight for term, count in Counter(_terms(text)).items()})
        documents.append(counts)
    average = sum(sum(counts.values()) for counts in documents) / len(documents) or 1.0
    frequencies = Counter(term for counts in documents for term in counts)
    ranked = []
    for entry, counts in zip(entries, documents):
        overlap = query_terms.intersection(counts)
        if not overlap:
            continue
        length = sum(counts.values())
        score = 0.0
        for term in overlap:
            idf = math.log(1 + (len(documents) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            tf = counts[term]
            score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / average))
        ranked.append((score, entry["id"], entry))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    if not ranked:
        return []
    # Do not fill three slots with weakly matching background just because room
    # remains. Budgeting below may also omit whole, otherwise relevant entries.
    threshold = ranked[0][0] * 0.3
    return [entry for score, _, entry in ranked if score >= threshold]


def _payload(skill: dict[str, str], sources: list[dict[str, Any]], *, method_plan=None) -> str:
    return json.dumps({
        "kind": "knowledge_skill_reference",
        "knowledge_scope": KNOWLEDGE_SCOPE,
        "authorship": "project_original_method_design",
        "validation_status": "design_unvalidated",
        "skill": skill,
        "usage_note": "项目自编方法仅供任务参考，不证明用户或资料事实；历史启发若有则另行署名，不把方法当作已验证结论。",
        "sources": sources,
        **({"method_plan": method_plan} if method_plan is not None else {}),
    }, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _result(status: str, skill: dict[str, str] | None, *, sources=None,
            prompt: str = "", candidates: int = 0, budget_omitted: int = 0) -> dict[str, Any]:
    return {
        "skill": skill,
        "sources": sources or [],
        "prompt": prompt,
        "trace": {
            "algorithm": "bounded_knowledge_skill_bm25_v1",
            "status": status,
            "model_calls": 0,
            "network_calls": 0,
            "memory_writes": 0,
            "graph_writes": 0,
            "knowledge_scope": KNOWLEDGE_SCOPE,
            "candidate_count": candidates,
            "selected_count": len(sources or []),
            "prompt_bytes": len(prompt.encode("utf-8")),
            "budget_omitted_count": budget_omitted,
            "semantic_relevance_verified": False,
        },
    }


def retrieve_knowledge_skill(skill_id: str | None, question: Any) -> dict[str, Any]:
    """Retrieve at most three complete K references after a harness selection.

    Unknown IDs are programming/request errors. Missing, malformed or unsupported
    packaged data fail closed with a metadata-only unavailable result. Ordinary
    irrelevant questions remain no_match even if the router selected a skill.
    """
    if skill_id is None:
        return _result("disabled", None)
    if not isinstance(skill_id, str) or skill_id not in _REGISTRY:
        raise ValueError("Unknown knowledge skill ID")
    skill = {key: _REGISTRY[skill_id][key] for key in ("id", "name", "version")}
    if not isinstance(question, str):
        return _result("no_match", skill)
    try:
        question = question.encode("utf-8")[:MAX_QUERY_BYTES].decode("utf-8", errors="ignore")
    except UnicodeError:
        return _result("no_match", skill)
    if (not _terms(question) or _KNOWLEDGE_OPT_OUT_RE.search(question)
            or (skill_id == 'mao-workplace' and (_OPT_OUT_RE.search(question) or _WORK_OPT_OUT_RE.search(question)))):
        return _result("no_match", skill)
    try:
        entries = _load_pack(skill_id)
    except (OSError, ValueError, UnicodeError, TypeError, RecursionError):
        return _result("unavailable", skill)
    ranked = _rank(question, entries)
    selected: list[dict[str, Any]] = []
    omitted = 0
    for entry in ranked:
        if len(selected) >= MAX_SOURCES:
            break
        source = {key: value for key, value in entry.items()
                  if key not in ("retrieval_keywords", "task_signals")}
        source.update(label=f"K{len(selected) + 1}", kind="method_reference", knowledge_scope=KNOWLEDGE_SCOPE)
        candidate = _payload(skill, [*selected, source])
        if len(candidate.encode("utf-8")) > MAX_PROMPT_BYTES:
            omitted += 1
            continue
        selected.append(source)
    if not selected:
        return _result("no_match", skill, candidates=len(ranked), budget_omitted=omitted)
    return _result("ready", skill, sources=selected, prompt=_payload(skill, selected),
                   candidates=len(ranked), budget_omitted=omitted)
