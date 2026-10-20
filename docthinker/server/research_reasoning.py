"""Bounded, source-grounded symbolic reasoning with no I/O or model calls.

The parser is intentionally incomplete: short Chinese/English relations and
explicit ``If ... then ...`` / ``如果 ... 则 ...`` Horn rules are supported.
Unparsed prose is not turned into a fact. Causal relations are never promoted to
implication rules. Analogy implements a small, bounded structure-mapping search,
not the full SME algorithm; induction reports source groups, not statistical
independence or a universal law.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

PARSER_VERSION = "literal-cnen-relations-v3"
MAX_CLAIMS = 192
MAX_CHUNKS = 64
MAX_CHUNK_CHARS = 20_000
MAX_SENTENCE_CHARS = 1_000
MAX_CONCLUSIONS = 8
MAX_DERIVATIONS = 64
MAX_DEPTH = 8
MAX_CASES = 12
MAX_CASE_EDGES = 16
MAX_MAPPING_STATES = 256
MAX_MAPPING_EXPANSIONS = 8_192
MAX_RESULT_BYTES = 24_576
_MODES = {"auto", "evidence", "analogy", "deduction", "induction"}
_SOURCE_PROVENANCES = {"llm", "self_study", "self_evolution", "hypothesis", "inferred", "memory", "llm_expansion"}
_SENTENCES = re.compile(r"[^\n。！？!?；;]+(?:[。！？!?；;]|(?=\n|$))")
_EN_SENTENCES = re.compile(r".+?(?:\.(?=\s|$)|$)")
_PREFIX = re.compile(r"^(?:observation|observed|fact|rule|观察|观测|事实|规则)\s*[:：]\s*", re.I)
_CASE = re.compile(r"^(?:case|案例|情境)\s*[:：]\s*([^:：\n]{1,120})(?:[:：]\s*(.*))?$", re.I)
_ROLE = re.compile(r"^(?:role|角色|类型)\s*[:：]\s*(.{1,200}?)\s*(?:=|是|:)\s*(.{1,120})$", re.I)
_UNCERTAIN = re.compile(r"\b(?:may|might|could|can|should|would|will|must|perhaps|possibly|probably|suggests?|hypothe[st]|not necessarily)\b|可能|也许|或许|假设|推测|未必|不一定|尚未证明|未证实|能够|可以|应当|应该|将会|将要", re.I)
_UNSUPPORTED = re.compile(r"\b(?:unless|either|or|but|only if|not only|except|approximately|roughly|sometimes|often|usually|rarely|always|never)\b|除非|或者|或是|仅当|只有|通常|往往|有时|大约|近似|总是|不仅|不但|然而", re.I)
_RELATIONS = (
    ("reduces", r"reduces?|reduced|decreases?|decreased|lowers?|lowered|降低|减少|减小|抑制"),
    ("increases", r"increases?|increased|raises?|raised|improves?|improved|提高|增加|增大|提升"),
    ("causes", r"causes?|caused|leads?\s+to|导致|引起|造成"),
    ("prevents", r"prevents?|prevented|阻止|防止"),
    ("supports", r"supports?|supported|支持"),
    ("contains", r"contains?|contained|包含|包括"),
    ("requires", r"requires?|required|需要|要求"),
    ("depends_on", r"depends?\s+on|relies?\s+on|依赖于|依赖"),
    ("is", r"is|are|was|were|是|属于"),
)
_RELATION_PATTERNS = tuple((name, re.compile(
    r"^(.{1,200}?)\s*(?:(does\s+not|do\s+not|did\s+not|doesn't|don't|didn't|not|不|没有|并非)\s*)?"
    r"(?:" + "|".join(r"(?<![A-Za-z])" + part + r"(?![A-Za-z])" if part[0].isascii() else part
                       for part in pattern.split("|")) + r")\s*(.{1,200})$", re.I
)) for name, pattern in _RELATIONS)
_CONDITION = re.compile(r"\s+(?:when|under|provided that|in the case of|in)\s+(.+)$", re.I)
_CN_CONDITION = re.compile(r"^在(.{1,160}?)(?:条件下|情况下|时)[，,]?\s*(.+)$")
_NUMBER = re.compile(r"(?<![A-Za-z])[+-]?\d+(?:\.\d+)?(?:\s*[%％])?")
_OPPOSITE_TRENDS = {"increases": "reduces", "reduces": "increases",
                    "state_increase": "state_decrease", "state_decrease": "state_increase"}


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return ""
    return value


def _normal(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip(" \t\r\n.,。；;，:：" )).casefold()
    return re.sub(r"^(?:the|an|a)\s+", "", value)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _limit(value: Any, default: int, cap: int) -> int:
    return min(cap, max(0, value)) if isinstance(value, int) and not isinstance(value, bool) else default


def _gap(code: str, **fields: Any) -> dict:
    messages = {
        "source_truth_not_verified": "推导只检查规则和前提是否衔接；原文陈述本身的真实性仍需核对。",
        "no_explicit_rules": "没有找到明确的条件规则；普通相关或因果描述不会自动当成逻辑规则。",
        "contradictory_premises": "发现方向相反或互相否定的前提，依赖这些前提的推导已停止。",
        "missing_premises": "部分规则缺少可核对的前提，尚不能据此补全推理链。",
        "reasoning_budget_reached": "本次推理达到计算或展示预算，只保留完整结果；未展示不代表不存在。",
        "analogy_is_not_entailment": "结构相似只能提出迁移假设，不能证明目标场景也有相同结果。",
        "unmodeled_case_differences": "案例可能还有原文未表达的差异，迁移前需要在目标场景验证。",
        "no_connected_structural_match": "没有找到至少两条相连关系的一一对应，不能只凭词语相似建立类比。",
        "opposite_relation_rejected": "发现映射后的关系方向相反，相关类比已排除。",
        "case_omitted_as_whole": "某些案例超过本次预算，已整体省略，避免漏掉反例后得出结论。",
        "source_independence_not_verified": "已去除相同文档和内容的重复计数，但无法据此证明不同来源在统计上独立。",
        "selection_bias_not_excluded": "归纳仅描述本次检索到的样本，尚未排除选择偏差，也不代表普遍规律。",
        "insufficient_comparable_observations": "相同对象、关系和条件下的可比较观察不足，尚不能形成样本规律。",
        "duplicate_observations_collapsed": "同一文档或重复内容的观察已合并，不会重复算作独立支持。",
        "claim_budget_reached": "完整前提超过本次推理窗口，已停止使用截断后的前提集合。",
        "requested_target_not_proved": "本次没有完整推得问题中点名的目标；中间环节不能替代目标结论。",
    }
    return {"code": code, "message": messages.get(code, "本次推理存在尚未核实的条件，请核对原文。"), **fields}


def choose_mode(question: str, requested: str = "auto") -> str:
    """An explicit mode wins; otherwise only explicit reasoning intent opts in."""
    if requested not in _MODES:
        raise ValueError("Unsupported reasoning mode")
    if requested != "auto":
        return requested
    text = _text(question)[:8_192]
    if re.search(r"类比|迁移|结构相似|\banalog(?:y|ies|ical)|\btransfer\b", text, re.I):
        return "analogy"
    if re.search(r"演绎|推导|逻辑推理|根据.*(?:规则|前提)|\bdeduc|\bderive\b|\blogical entail", text, re.I):
        return "deduction"
    if re.search(r"归纳|共同规律|共同模式|\binduct|\bregularit|\bcommon pattern", text, re.I):
        return "induction"
    return "evidence"


def _atom(text: str, *, explicit: bool = False) -> dict | None:
    value = _PREFIX.sub("", text.strip()).strip(" .。；;")
    if not value or len(value) > MAX_SENTENCE_CHARS or _UNCERTAIN.search(value) or _UNSUPPORTED.search(value):
        return None
    # No conjunction is silently swallowed into an entity. Horn antecedents are
    # split separately; quantifiers, numerical comparisons and disjunction need
    # a richer parser and are not claimed as solved here.
    if re.search(r"\b(?:and|if|then|all|every|some)\b|如果|那么|并且|且|所有|每个", value, re.I):
        return None
    if re.search(r"(?:不会|不能|不可|不应|会|能|并)(?:降低|减少|增加|提高|提升|导致|依赖|支持|需要)", value):
        return None
    leading_negative = re.match(r"^(?:not\s+|并非)(.+)$", value, re.I)
    if leading_negative:
        if not explicit:
            return None
        value = leading_negative.group(1)
    conditions: list[str] = []
    cn_condition = _CN_CONDITION.match(value)
    if cn_condition:
        conditions.append(_normal(cn_condition.group(1)))
        value = cn_condition.group(2)
    match = _CONDITION.search(value)
    if match:
        conditions.append(_normal(match.group(1)))
        value = value[:match.start()]
    if not value or len(conditions) > 2:
        return None
    # Common Chinese trend statements are complete unary observations, e.g.
    # 国内供给增加 / 进口依赖下降. In particular, 依赖 in 进口依赖 is part of
    # the quantity's name here, not a binary depends_on predicate.
    cn_trend = re.match(r"^(.{1,200}?)(?:(不|没有|未)\s*)?(增加|增长|提高|提升|改善|下降|减少|降低|恶化)$", value)
    if cn_trend:
        subject, negative, trend = cn_trend.groups()
        if not re.search(r"降低|减少|增加|提高|导致|支持|需要|包含|阻止", subject):
            if negative and leading_negative:
                return None
            return {"subject": _normal(subject),
                    "relation": "state_increase" if trend in {"增加", "增长", "提高", "提升", "改善"} else "state_decrease",
                    "object": "", "polarity": -1 if negative or leading_negative else 1,
                    "conditions": sorted(conditions)}
    for relation, pattern in _RELATION_PATTERNS:
        match = pattern.match(value)
        if match:
            subject, negative, obj = match.groups()
            if negative and leading_negative:
                return None
            # "X is not Y" puts negation after the copula.
            if relation == "is":
                negated = re.match(r"^(not|不|非)\s*(.+)$", obj, re.I)
                if negated:
                    if negative or leading_negative:
                        return None
                    negative, obj = negated.groups()
            subject, obj = _normal(subject), _normal(obj)
            # A second relation or a numerical qualifier inside a role means the
            # sentence is not the simple binary assertion this grammar models.
            if not subject or not obj or re.search(r"[,:：，<>≤≥=]", subject + obj):
                return None
            if any(re.fullmatch(r"it|they|he|she|this|that|these|those|其|它|他们|它们|这|该", role, re.I) for role in (subject, obj)):
                return None
            if _NUMBER.search(subject + obj):
                return None
            return {"subject": subject, "relation": relation, "object": obj,
                    "polarity": -1 if negative or leading_negative else 1, "conditions": sorted(conditions)}
    # Unary trends are distinct predicates, never conflated with binary causes.
    match = re.match(r"^(.{1,200}?)\s+(?:(does not|do not|did not)\s+)?(increases?|decreases?|grows?|falls?)$", value, re.I)
    if match:
        subject, negative, trend = match.groups()
        if negative and leading_negative:
            return None
        return {"subject": _normal(subject), "relation": "state_increase" if trend.lower().startswith(("increase", "grow")) else "state_decrease",
                "object": "", "polarity": -1 if negative or leading_negative else 1, "conditions": sorted(conditions)}
    # Explicit observations/rule propositions allow named boolean conditions,
    # e.g. "Observation: grid ready" and "If grid ready then supply grows".
    if explicit and len(value) <= 200 and not re.search(r"[,:：，<>≤≥=]", value) and not _NUMBER.search(value):
        polarity = -1 if leading_negative else 1
        negative = re.match(r"^(?:not\s+|非|并非)(.+)$", value, re.I)
        if negative:
            value, polarity = negative.group(1), -1
        return {"subject": _normal(value), "relation": "holds", "object": "",
                "polarity": polarity, "conditions": sorted(conditions)}
    return None


def _rule(text: str) -> dict | None:
    value = _PREFIX.sub("", text.strip()).strip(" .。；;")
    match = re.match(r"^if\s+(.+?)(?:,?\s+then\s+|,\s*)(.+)$", value, re.I)
    if not match:
        match = re.match(r"^(?:如果|若)(.+?)[，,]?\s*(?:那么|则)(.+)$", value)
    if not match or _UNCERTAIN.search(value) or _UNSUPPORTED.search(value):
        return None
    before, after = match.groups()
    parts = re.split(r"\s+and\s+|\s*&\s*|并且|且", before, flags=re.I)
    if not 1 <= len(parts) <= 4:
        return None
    premises = [_atom(part, explicit=True) for part in parts]
    consequence = _atom(after, explicit=True)
    if not consequence or any(not premise or premise["conditions"] for premise in premises) or consequence["conditions"]:
        return None
    return {"premises": premises, "consequence": consequence}


def _sentences(content: str):
    for broad in _SENTENCES.finditer(content):
        for inner in _EN_SENTENCES.finditer(broad.group(0)):
            value = inner.group(0)
            stripped = value.strip()
            if not stripped:
                continue
            start = broad.start() + inner.start() + len(value) - len(value.lstrip())
            yield stripped, start, start + len(stripped)


def extract_claims(chunks: Sequence[Mapping[str, Any]], *, max_claims: int = MAX_CLAIMS,
                   _case_labels: Mapping[str, str] | None = None) -> list[dict]:
    """Extract literal records. Offsets refer to the original source chunk.

    Optional ``content_hash``/``document_hash`` identifies duplicate uploads. With
    no document hash, a normalized complete chunk fingerprint provides a weaker,
    explicitly disclosed duplicate filter. The caller supplies authorized, live
    source chunks; this function performs no account/database lookups.
    """
    cap = _limit(max_claims, MAX_CLAIMS, MAX_CLAIMS)
    if not cap:
        return []
    rows: list[dict] = []
    seen: set[str] = set()
    valid = [row for row in chunks[:MAX_CHUNKS] if isinstance(row, Mapping)]
    for raw in sorted(valid, key=lambda row: (_text(row.get("document_id")), _text(row.get("id")))):
        identifier, document_id, content = _text(raw.get("id")), _text(raw.get("document_id")), _text(raw.get("content"))
        if not identifier or not document_id or len(identifier) > 256 or len(document_id) > 256 or not content:
            continue
        if raw.get("is_generated") in (True, "true", "1") or raw.get("is_discovered") in (True, "true", "1"):
            continue
        if _text(raw.get("provenance")) in _SOURCE_PROVENANCES or _text(raw.get("state")) in {"deleted", "failed", "invalidated"}:
            continue
        base = raw.get("excerpt_start", 0)
        if not isinstance(base, int) or isinstance(base, bool) or base < 0:
            continue
        # A cut mid-sentence can drop a negation or conditional prefix. Only the
        # complete interior sentences of a clipped excerpt may become claims.
        start_cut = bool(base)
        full_length = raw.get("original_content_length", base + len(content) + 1)
        if not isinstance(full_length, int) or isinstance(full_length, bool):
            continue
        end_cut = raw.get("truncated") is True and full_length > base + len(content)
        fingerprint = _text(raw.get("content_hash")) or _text(raw.get("document_hash"))
        fingerprint_kind = "document" if re.fullmatch(r"[a-fA-F0-9]{64}", fingerprint) else "chunk"
        if fingerprint_kind == "chunk":
            fingerprint = _digest(_normal(content))
        case_label = (_case_labels or {}).get(identifier, "")
        sentences = list(_sentences(content[:MAX_CHUNK_CHARS]))
        for index, (quote, left, right) in enumerate(sentences):
            if (start_cut and index == 0) or ((end_cut or len(content) > MAX_CHUNK_CHARS) and index == len(sentences) - 1):
                continue
            label = _CASE.match(quote.strip(" .。；;"))
            if label:
                case_label = _normal(label.group(1))
                # A standalone label scopes subsequent statements, not a claim.
                if not label.group(2):
                    continue
                remainder = label.group(2)
                offset = quote.find(remainder)
                quote, left = quote[offset:], left + offset
            if len(quote) > MAX_SENTENCE_CHARS:
                continue
            role = _ROLE.match(quote.strip(" .。；;"))
            rule = None if role else _rule(quote)
            atom = ({"subject": _normal(role.group(1)), "relation": "is", "object": _normal(role.group(2)),
                     "polarity": 1, "conditions": [], "type_annotation": True} if role else
                    None if rule else _atom(quote, explicit=bool(_PREFIX.match(quote))))
            if rule is None and atom is None:
                continue
            metadata: dict[str, Any] = {
                "kind": "rule" if rule else "observation", "case_id": f"{document_id}:{case_label}" if case_label else document_id,
                "case_label": case_label, "scope": case_label or "project",
                "source_ids": [identifier], "document_id": document_id,
                "source_fingerprint": fingerprint.lower(), "fingerprint_kind": fingerprint_kind,
                "quote": quote, "excerpt_start": base + left, "excerpt_end": base + right,
                "offset_unit": "characters_in_source_chunk", "parser": PARSER_VERSION,
            }
            page = raw.get("page")
            if isinstance(page, int) and not isinstance(page, bool) and page > 0:
                metadata["page"] = page
            metadata.update(rule if rule else atom)
            metadata["id"] = "claim_" + _digest(metadata)[:24]
            if metadata["id"] not in seen:
                rows.append(metadata)
                seen.add(metadata["id"])
            if len(rows) >= cap:
                return rows
    return rows


def extract_document_claims(chunks: Sequence[Mapping[str, Any]], *, max_claims: int = 1536) -> dict:
    """Index a complete bounded document, preserving named cases across chunks.

    Callers must provide the document's complete canonical chunks, not retrieval
    excerpts. If a document exceeds the structured window, no partial case is
    returned: an omitted counterexample must never make a case look complete.
    Ordinary text retrieval can still use that document independently.
    """
    limit = _limit(max_claims, 1536, 1536)
    if len(chunks) > 600 or not limit:
        return {"claims": [], "limited": bool(chunks)}
    ordered = sorted(enumerate(chunks), key=lambda pair: (
        _text(pair[1].get("document_id")) if isinstance(pair[1], Mapping) else "",
        pair[1].get("ordinal", pair[0]) if isinstance(pair[1], Mapping) and type(pair[1].get("ordinal", pair[0])) is int else pair[0],
    ))
    context: dict[str, str] = {}
    fingerprint_parts: dict[str, list] = defaultdict(list)
    for _, row in ordered:
        if isinstance(row, Mapping):
            fingerprint_parts[_text(row.get("document_id"))].append([row.get("page"), _normal(_text(row.get("content")))])
    fingerprints = {doc: _digest(parts) for doc, parts in fingerprint_parts.items()}
    result = []
    for _, row in ordered:
        if not isinstance(row, Mapping):
            continue
        content, document = _text(row.get("content")), _text(row.get("document_id"))
        if len(content) > MAX_CHUNK_CHARS or row.get("truncated") is True or row.get("excerpt_start", 0):
            return {"claims": [], "limited": True}
        if not re.fullmatch(r"[a-fA-F0-9]{64}", _text(row.get("content_hash")) or _text(row.get("document_hash"))):
            row = {**row, "document_hash": fingerprints[document]}
        claims = extract_claims([row], _case_labels={_text(row.get("id")): context.get(document, "")})
        if len(claims) >= MAX_CLAIMS or len(result) + len(claims) > limit:
            return {"claims": [], "limited": True}
        result.extend(claims)
        for quote, _, _ in _sentences(content):
            label = _CASE.match(quote.strip(" .。；;"))
            if label:
                context[document] = _normal(label.group(1))
    return {"claims": result, "limited": False}


def _atom_key(atom: Mapping, scope: str = "") -> tuple:
    return (scope, atom.get("subject", ""), atom.get("relation", ""), atom.get("object", ""),
            tuple(atom.get("conditions", ())))


def _fact_key(atom: Mapping, scope: str = "") -> tuple:
    return (*_atom_key(atom, scope), atom.get("polarity", 1))


def _atom_text(atom: Mapping) -> str:
    relation = {"reduces": "降低", "increases": "增加", "causes": "导致", "depends_on": "依赖",
                "prevents": "阻止", "supports": "支持", "contains": "包含", "requires": "需要",
                "is": "是", "state_increase": "增加", "state_decrease": "减少", "holds": "成立"}[atom["relation"]]
    return f"{atom['subject']} {'不' if atom['polarity'] < 0 else ''}{relation} {atom['object']}".strip()


def _valid_atom(atom: Any) -> bool:
    if not isinstance(atom, Mapping):
        return False
    return (bool(_text(atom.get("subject"))) and len(atom["subject"]) <= 200
            and isinstance(atom.get("object"), str) and len(atom["object"]) <= 200
            and isinstance(atom.get("relation"), str)
            and atom["relation"] in {name for name, _ in _RELATIONS} | {"holds", "state_increase", "state_decrease"}
            and type(atom.get("polarity")) is int and atom["polarity"] in {-1, 1}
            and isinstance(atom.get("conditions"), list) and len(atom["conditions"]) <= 2
            and all(isinstance(item, str) and len(item) <= 200 for item in atom["conditions"]))


def _valid_claim(claim: Any) -> bool:
    if not isinstance(claim, Mapping) or claim.get("parser") != PARSER_VERSION:
        return False
    if not all(_text(claim.get(field)) and len(claim[field]) <= limit for field, limit in
               (("id", 256), ("document_id", 256), ("quote", MAX_SENTENCE_CHARS),
                ("scope", 200), ("case_id", 512), ("source_fingerprint", 64))):
        return False
    if not isinstance(claim.get("source_ids"), list) or not 1 <= len(claim["source_ids"]) <= 4 or not all(_text(item) and len(item) <= 256 for item in claim["source_ids"]):
        return False
    if claim.get("kind") == "observation":
        return _valid_atom(claim)
    return (claim.get("kind") == "rule" and isinstance(claim.get("premises"), list)
            and 1 <= len(claim["premises"]) <= 4 and all(_valid_atom(atom) for atom in claim["premises"])
            and _valid_atom(claim.get("consequence")))


def _deduction(claims: list[dict], cap: int) -> tuple[list[dict], list[dict], dict]:
    observed = [row for row in claims if row["kind"] == "observation"]
    rules = [row for row in claims if row["kind"] == "rule"]
    blocked: set[tuple] = set()
    for key, group in itertools.groupby(sorted(observed, key=lambda row: _atom_key(row, row["scope"])), key=lambda row: _atom_key(row, row["scope"])):
        if {row["polarity"] for row in group} == {-1, 1}:
            blocked.add(key)
    observed_positive = {_atom_key(row, row["scope"]) for row in observed if row["polarity"] > 0}
    for key in observed_positive:
        if key[2] in _OPPOSITE_TRENDS:
            opposite = (key[0], key[1], _OPPOSITE_TRENDS[key[2]], key[3], key[4])
            if opposite in observed_positive:
                blocked.update((key, opposite))
    facts: dict[tuple, dict] = {}
    derived: list[dict] = []
    iterations = 0
    exhausted = False
    # Contradictions discovered in a later round invalidate the entire dependent
    # proof by rebuilding closure. Monotonic blocking is deliberately cautious.
    for _restart in range(MAX_DEPTH + 1):
        facts = {}
        derived = []
        for row in observed:
            if _atom_key(row, row["scope"]) not in blocked:
                facts.setdefault(_fact_key(row, row["scope"]), {**row, "depth": 0, "premise_ids": [], "proof": []})
        restarted = False
        for depth in range(1, MAX_DEPTH + 1):
            iterations += 1
            proposals: dict[tuple, dict] = {}
            for rule in rules:
                scope = rule["scope"]
                target = rule["consequence"]
                if _atom_key(target, scope) in blocked or _fact_key(target, scope) in facts:
                    continue
                premises = [facts.get(_fact_key(atom, scope)) for atom in rule["premises"]]
                if any(premise is None for premise in premises):
                    continue
                if any(_atom_key(atom, scope) in blocked for atom in rule["premises"]):
                    continue
                actual_depth = 1 + max(premise["depth"] for premise in premises)
                if actual_depth > MAX_DEPTH:
                    exhausted = True
                    continue
                source_ids = sorted({source for row in [rule, *premises] for source in row["source_ids"]})
                premise_ids = [rule["id"], *[row["id"] for row in premises]]
                proof = {node["id"]: node for row in premises for node in row["proof"]}
                for row in [rule, *premises]:
                    proof[row["id"]] = {"id": row["id"], "kind": row["kind"], "source_ids": row["source_ids"],
                                         "premise_ids": row.get("premise_ids", []),
                                         "text": (row.get("text") or row.get("quote") or _atom_text(row))[:800]}
                candidate = {**target, "scope": scope, "id": "deduction_" + _digest([scope, target, premise_ids])[:24],
                             "kind": "deduced", "text": _atom_text(target), "premise_ids": premise_ids,
                             "source_ids": source_ids, "proof": list(proof.values()), "depth": actual_depth,
                             "epistemic_status": "conditional_on_source_premises"}
                proposals.setdefault(_fact_key(target, scope), candidate)
                if len(proposals) >= MAX_DERIVATIONS:
                    exhausted = True
                    break
            new_conflicts = {_atom_key(row, row["scope"]) for key, row in proposals.items()
                             if (*key[:-1], -key[-1]) in facts or (*key[:-1], -key[-1]) in proposals}
            for key, row in proposals.items():
                if row["polarity"] > 0 and row["relation"] in _OPPOSITE_TRENDS:
                    opposite_key = (key[0], key[1], _OPPOSITE_TRENDS[key[2]], key[3], key[4], 1)
                    if opposite_key in facts or opposite_key in proposals:
                        new_conflicts.update((key[:-1], opposite_key[:-1]))
            if new_conflicts - blocked:
                blocked.update(new_conflicts)
                restarted = True
                break
            if not proposals:
                break
            for key, value in proposals.items():
                if len(derived) >= MAX_DERIVATIONS:
                    exhausted = True
                    break
                facts[key] = value
                derived.append(value)
            if len(derived) >= MAX_DERIVATIONS:
                break
        if not restarted:
            break
    else:
        # Incomplete conflict stabilization cannot be presented as a valid proof.
        derived = []
        exhausted = True
    if any(_atom_key(rule["consequence"], rule["scope"]) not in blocked
           and _fact_key(rule["consequence"], rule["scope"]) not in facts
           and all(_fact_key(atom, rule["scope"]) in facts for atom in rule["premises"])
           for rule in rules):
        exhausted = True
    gaps = [_gap("source_truth_not_verified")]
    if not rules:
        gaps.append(_gap("no_explicit_rules"))
    if blocked:
        gaps.append(_gap("contradictory_premises", count=len(blocked)))
    missing = []
    for rule in rules:
        unmet = [atom for atom in rule["premises"] if _fact_key(atom, rule["scope"]) not in facts]
        if unmet:
            missing.append({"rule_id": rule["id"], "premises": [_atom_text(atom) for atom in unmet]})
    if missing:
        gaps.append(_gap("missing_premises", rules=missing[:4], rule_count=len(missing)))
    if exhausted or len(derived) > cap:
        gaps.append(_gap("reasoning_budget_reached"))
    return derived[:cap], gaps, {"rules": len(rules), "observations": len(observed), "closure_iterations": iterations,
                               "derived_count": len(derived), "contradictions": len(blocked), "max_depth": MAX_DEPTH}


def _components(edges: list[dict]) -> list[list[dict]]:
    remaining = list(edges)
    components = []
    while remaining:
        group = [remaining.pop(0)]
        nodes = {group[0]["subject"], group[0]["object"]}
        changed = True
        while changed:
            changed = False
            for edge in remaining[:]:
                if nodes.intersection((edge["subject"], edge["object"])):
                    group.append(edge)
                    nodes.update((edge["subject"], edge["object"]))
                    remaining.remove(edge)
                    changed = True
        components.append(group)
    return components


def _mapping(base: list[dict], target: list[dict], base_types: Mapping, target_types: Mapping,
             expansion_budget: int) -> tuple[list[tuple[dict, dict]], dict[str, str], int, bool]:
    # Beam states are partial, one-to-one node maps plus relation pairings. A
    # match is accepted only if its mapped relations form a connected component.
    states: list[tuple[dict, tuple, frozenset]] = [({}, (), frozenset())]
    expansions = 0
    exhausted = False
    for left in base:
        next_states = list(states)
        for mapping, pairs, used in states:
            for index, right in enumerate(target):
                if index in used or (left["relation"], left["polarity"]) != (right["relation"], right["polarity"]):
                    continue
                if expansions >= expansion_budget:
                    exhausted = True
                    break
                expansions += 1
                candidate = dict(mapping)
                compatible = True
                for a, b in ((left["subject"], right["subject"]), (left["object"], right["object"])):
                    if (a in candidate and candidate[a] != b) or (a not in candidate and b in candidate.values()):
                        compatible = False
                        break
                    if a in base_types and b in target_types and base_types[a] != target_types[b]:
                        compatible = False
                        break
                    candidate[a] = b
                if compatible:
                    next_states.append((candidate, (*pairs, (left, right)), used | {index}))
            if exhausted:
                break
        unique = {}
        for mapping, pairs, used in next_states:
            signature = (tuple(sorted(mapping.items())), tuple(sorted((a["id"], b["id"]) for a, b in pairs)))
            unique.setdefault(signature, (mapping, pairs, used))
        states = sorted(unique.values(), key=lambda state: (-len(state[1]), -len(state[0]), tuple(sorted(state[0].items()))))[:MAX_MAPPING_STATES]
        if exhausted:
            break
    candidates = []
    for mapping, pairs, _ in states:
        if len(pairs) < 2:
            continue
        # A two-edge minimum means isolated word/relation overlap is insufficient.
        for group in _components([pair[0] for pair in pairs]):
            ids = {edge["id"] for edge in group}
            component_pairs = [(a, b) for a, b in pairs if a["id"] in ids]
            if len(component_pairs) >= 2:
                selected_nodes = {node for a, _ in component_pairs for node in (a["subject"], a["object"])}
                candidates.append((component_pairs, {a: b for a, b in mapping.items() if a in selected_nodes}))
    if not candidates:
        return [], {}, expansions, exhausted
    pairs, mapping = max(candidates, key=lambda item: (len(item[0]), len(item[1]), tuple(sorted(item[1].items()))))
    return pairs, mapping, expansions, exhausted


def _analogy(claims: list[dict], cap: int) -> tuple[list[dict], list[dict], dict]:
    cases: dict[str, list[dict]] = defaultdict(list)
    types: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in claims:
        if row.get("type_annotation") is True and row["kind"] == "observation" and row["relation"] == "is" and row["polarity"] == 1:
            types[row["case_id"]][row["subject"]].add(row["object"])
        if row["kind"] == "observation" and row["object"] and row["relation"] != "is":
            cases[row["case_id"]].append(row)
    omitted_cases = sum(len(rows) > MAX_CASE_EDGES for rows in cases.values())
    eligible_cases = [(key, rows) for key, rows in sorted(cases.items()) if len(rows) <= MAX_CASE_EDGES]
    omitted_cases += max(0, len(eligible_cases) - MAX_CASES)
    cases = dict(eligible_cases[:MAX_CASES])
    results = []
    total_expansions = 0
    directions_examined = 0
    exhausted = False
    rejected_conflicts = 0
    # An analogy is directional: only the base can supply a candidate inference.
    # Document UUID ordering must not decide which case supplies that inference.
    # Examine both orientations consecutively, reserving half the remaining
    # search window for the reverse direction of each unordered pair.
    oriented_pairs = [orientation for pair in itertools.combinations(cases.items(), 2)
                      for orientation in (pair, tuple(reversed(pair)))]
    for direction_index, ((left_case, base), (right_case, target)) in enumerate(oriented_pairs):
        if len(base) < 2 or len(target) < 2:
            continue
        # Duplicate uploads cannot manufacture a second analogy case.
        if {row["source_fingerprint"] for row in base} == {row["source_fingerprint"] for row in target}:
            continue
        if total_expansions >= MAX_MAPPING_EXPANSIONS:
            exhausted = True
            break
        remaining = MAX_MAPPING_EXPANSIONS - total_expansions
        orientation_budget = max(1, remaining // 2) if direction_index % 2 == 0 else remaining
        pairs, mapping, expansions, limited = _mapping(base, target, types[left_case], types[right_case], orientation_budget)
        directions_examined += 1
        total_expansions += expansions
        exhausted |= limited
        if not pairs:
            continue
        target_by_key = defaultdict(list)
        for row in target:
            target_by_key[(row["subject"], row["relation"], row["object"])].append(row)
        mismatches = []
        conflicts = []
        candidates = []
        paired_ids = {a["id"] for a, _ in pairs}
        for row in base:
            if row["subject"] not in mapping or row["object"] not in mapping:
                continue
            counterparts = target_by_key.get((mapping[row["subject"]], row["relation"], mapping[row["object"]]), [])
            if any(item["polarity"] != row["polarity"] for item in counterparts):
                conflicts.append(row["id"])
            opposite = target_by_key.get((mapping[row["subject"]], _OPPOSITE_TRENDS.get(row["relation"]), mapping[row["object"]]), [])
            if row["polarity"] > 0 and any(item["polarity"] > 0 and item["conditions"] == row["conditions"] for item in opposite):
                conflicts.append(row["id"])
            if row["id"] not in paired_ids:
                mismatches.append({"code": "unmatched_base_relation", "claim_id": row["id"]})
                if not counterparts and len(candidates) < 4:
                    projected = {"subject": mapping[row["subject"]], "relation": row["relation"],
                                 "object": mapping[row["object"]], "polarity": row["polarity"],
                                 "conditions": list(row["conditions"])}
                    candidates.append({**projected, "text": _atom_text(projected), "base_claim_id": row["id"],
                                       "source_ids": row["source_ids"], "target_conditions_verified": False,
                                       "epistemic_status": "unverified_transfer_hypothesis"})
        if conflicts:
            rejected_conflicts += 1
            continue
        for a, b in pairs:
            if a["conditions"] != b["conditions"]:
                mismatches.append({"code": "different_conditions", "base_id": a["id"], "target_id": b["id"],
                                   "base_conditions": a["conditions"], "target_conditions": b["conditions"]})
        if any(a not in types[left_case] or b not in types[right_case] for a, b in mapping.items()):
            mismatches.append({"code": "role_types_not_fully_specified"})
        used = [row for pair in pairs for row in pair]
        projected_ids = {candidate["base_claim_id"] for candidate in candidates}
        used.extend(row for row in base if row["id"] in projected_ids)
        used.extend(row for row in claims if row.get("type_annotation") is True and (
            (row["case_id"] == left_case and row["subject"] in mapping)
            or (row["case_id"] == right_case and row["subject"] in mapping.values())))
        # Different named entities are necessary for a cross-case analogy; exact
        # repetition belongs to evidence comparison/induction instead.
        if all(a == b for a, b in mapping.items()):
            continue
        result = {"id": "analogy_" + _digest([left_case, right_case, sorted(mapping.items())])[:24],
                  "kind": "analogy_hypothesis", "text": f"两个案例有 {len(pairs)} 条相连关系可作结构对应，并产生 {len(candidates)} 条待验证的迁移假设；结构对应不能证明目标案例的未观察结果。",
                  "base_case": left_case, "target_case": right_case,
                  "mappings": [{"base": a, "target": b} for a, b in sorted(mapping.items())],
                  "matched_relations": [{"base_id": a["id"], "target_id": b["id"], "relation": a["relation"], "polarity": a["polarity"]} for a, b in pairs],
                  "candidate_inferences": candidates,
                  "differences": mismatches[:8], "premise_ids": sorted({row["id"] for row in used}),
                  "source_ids": sorted({source for row in used for source in row["source_ids"]}),
                  "epistemic_status": "hypothesis_requires_target_validation"}
        results.append(result)
    def mapping_rank(row):
        return (-len(row["matched_relations"]), -len(row["candidate_inferences"]), len(row["differences"]),
                tuple((item["base"], item["target"]) for item in row["mappings"]))
    # Keep one best direction per case pair; equal structural matches prefer the
    # direction with an explicit transferable base edge, never a guessed edge.
    best_directions = {}
    for row in sorted(results, key=mapping_rank):
        best_directions.setdefault(tuple(sorted((row["base_case"], row["target_case"]))), row)
    results = list(best_directions.values())
    gaps = [_gap("analogy_is_not_entailment"), _gap("unmodeled_case_differences")]
    if not results:
        gaps.append(_gap("no_connected_structural_match"))
    if rejected_conflicts:
        gaps.append(_gap("opposite_relation_rejected", count=rejected_conflicts))
    if omitted_cases:
        gaps.append(_gap("case_omitted_as_whole", count=omitted_cases))
    if exhausted or len(results) > cap:
        gaps.append(_gap("reasoning_budget_reached"))
    return results[:cap], gaps, {"cases": len(cases), "mapping_expansions": total_expansions,
                               "matched_cases": len(results), "mapping_directions_examined": directions_examined,
                               "minimum_connected_relations": 2}


def _induction(claims: list[dict], cap: int) -> tuple[list[dict], list[dict], dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in claims:
        if row["kind"] == "observation" and row["object"] and row["relation"] != "is":
            groups[_atom_key(row)].append(row)
    results = []
    removed = 0
    for key, same_pattern_rows in sorted(groups.items()):
        rows = list(same_pattern_rows)
        # Increasing and reducing the same quantity under the same conditions
        # are opposing observed directions. "Does not increase" is NOT silently
        # converted into support for reduction (it may mean no change).
        if key[2] in _OPPOSITE_TRENDS:
            opposite_key = (key[0], key[1], _OPPOSITE_TRENDS[key[2]], key[3], key[4])
            rows.extend({**row, "counterexample_of_relation": key[2]} for row in groups.get(opposite_key, []) if row["polarity"] > 0)
        def is_counter(row):
            return row["polarity"] < 0 or "counterexample_of_relation" in row
        # Connected components over equal document OR equal content fingerprints
        # avoid double-counting overlapping chunks and reuploads simultaneously.
        source_groups: list[list[dict]] = []
        for row in rows:
            matches = [group for group in source_groups if any(other["document_id"] == row["document_id"]
                       or other["source_fingerprint"] == row["source_fingerprint"] for other in group)]
            if matches:
                merged = [row]
                for group in matches:
                    merged.extend(group)
                    source_groups.remove(group)
                source_groups.append(merged)
            else:
                source_groups.append([row])
        removed += len(rows) - len(source_groups)
        positive = [group for group in source_groups if not any(is_counter(row) for row in group)]
        negative = [group for group in source_groups if any(is_counter(row) for row in group)]
        if len(positive) < 2:
            continue
        used = [min(group, key=lambda row: row["id"]) for group in positive]
        used.extend(min((row for row in group if is_counter(row)), key=lambda row: row["id"]) for group in negative)
        exemplar = positive[0][0]
        pattern = {field: exemplar[field] for field in ("subject", "relation", "object", "conditions")}
        pattern_text = _atom_text(exemplar)
        if exemplar["conditions"]:
            pattern_text += "（适用条件：" + "；".join(exemplar["conditions"]) + "）"
        results.append({"id": "induction_" + _digest([key, sorted(row["id"] for row in used)])[:24],
                        "kind": "inductive_candidate", "text": f"在已检索的去重来源组中，{len(positive)} 组报告“{pattern_text}”，{len(negative)} 组包含反例。这里只描述样本内规律，不能推出普遍规律或因果性。",
                        "pattern": pattern, "support_count": len(positive), "counterexample_count": len(negative),
                        "source_group_count": len(source_groups), "premise_ids": [row["id"] for row in used],
                        "source_ids": sorted({source for row in used for source in row["source_ids"]}),
                        "counterexample_ids": [row["id"] for row in used if is_counter(row)],
                        "epistemic_status": "sample_regularity_not_universal", "scope": "retrieved_source_groups"})
    results.sort(key=lambda row: (-row["source_group_count"], row["id"]))
    gaps = [_gap("source_independence_not_verified"), _gap("selection_bias_not_excluded")]
    if not results:
        gaps.append(_gap("insufficient_comparable_observations"))
    if removed:
        gaps.append(_gap("duplicate_observations_collapsed", count=removed))
    if len(results) > cap:
        gaps.append(_gap("reasoning_budget_reached"))
    return results[:cap], gaps, {"comparable_patterns": len(groups), "candidate_patterns": len(results),
                               "minimum_support_groups": 2, "duplicate_observations": removed}


def reason(question: str, claims: Sequence[Mapping[str, Any]], mode: str = "auto", *,
           max_conclusions: int = MAX_CONCLUSIONS) -> dict[str, Any]:
    """Run one bounded operator on live, literal source claims.

    The caller must revalidate source ownership/revocation before displaying or
    persisting results. Returned conclusions must not be inserted as observed
    facts or as automatic memories. ``evidence`` extracts no extra inference.
    """
    selected_mode = choose_mode(question, mode)
    cap = _limit(max_conclusions, MAX_CONCLUSIONS, MAX_CONCLUSIONS)
    valid = sorted({row["id"]: dict(row) for row in claims[:MAX_CLAIMS] if _valid_claim(row)}.values(), key=lambda row: row["id"])
    trace = {"algorithm": "bounded_literal_reasoning_v1", "parser": PARSER_VERSION,
             "claim_count": len(valid), "model_calls": 0, "semantic_entailment_verified": False,
             "max_claims": MAX_CLAIMS, "max_conclusions": cap}
    if selected_mode == "evidence" or not cap:
        return {"mode": selected_mode, "status": "not_requested" if selected_mode == "evidence" else "bounded",
                "conclusions": [], "gaps": [], "trace": trace}
    if len(claims) > MAX_CLAIMS:
        return {"mode": selected_mode, "status": "insufficient_evidence", "conclusions": [],
                "gaps": [_gap("claim_budget_reached")], "trace": trace}
    operation = {"deduction": _deduction, "analogy": _analogy, "induction": _induction}[selected_mode]
    conclusions, gaps, details = operation(valid, MAX_CLAIMS)
    # An explicit proof request about a named source entity is answered with
    # conclusions involving that entity, not a successfully derived but unrelated
    # earlier link. Broad "what follows from ..." requests retain forward closure.
    goal_question = _normal(_text(question)[:8_192])
    explicit_goal = bool(re.search(r"能否|是否|可否|能不能|推导|证明|验证|\b(?:prove|derive|deduce|verify)\b", goal_question, re.I))
    if re.search(r"\b(?:derive|deduce)\s+from\b|(?:从|根据).*(?:推导|演绎).*(?:什么|哪些)", goal_question, re.I):
        explicit_goal = False
    if selected_mode == "deduction" and explicit_goal:
        entities = {atom[field] for claim in valid for atom in
                    ([claim] if claim['kind'] == 'observation' else [*claim['premises'], claim['consequence']])
                    for field in ("subject", "object") if atom[field]}
        def mentioned(entity: str) -> bool:
            if re.search(r"[\u3400-\u9fff]", entity):
                return len(entity) >= 2 and entity in goal_question
            return bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(entity) + r"(?![A-Za-z0-9])", goal_question))
        targets = {entity for entity in entities if mentioned(entity)}
        if targets:
            matching = [row for row in conclusions if row["subject"] in targets or row["object"] in targets]
            details["query_goal_entities"] = sorted(targets)[:8]
            details["unrelated_derivations_omitted"] = len(conclusions) - len(matching)
            conclusions = matching
            if not conclusions:
                gaps.insert(0, _gap("requested_target_not_proved", targets=sorted(targets)[:8],
                                    message="本次没有完整推得问题中点名的目标；已推得的中间环节不作为该目标的证明。"))
    # Query relevance chooses which complete proof/case comparison to show. It
    # never changes the truth value of a premise or creates a graph edge.
    from .research_retrieval import lexical_tokens
    terms = set(lexical_tokens(_text(question)[:8_192]))
    def rank(row: dict) -> tuple:
        semantic_fields = {key: row[key] for key in ("subject", "object", "pattern", "mappings") if key in row}
        overlap = len(terms & set(lexical_tokens(json.dumps(semantic_fields, ensure_ascii=False))))
        return (-overlap, -row.get("depth", 0), -len(row.get("matched_relations", [])), row["id"])
    conclusions.sort(key=rank)
    if len(conclusions) > cap:
        gaps.append(_gap("reasoning_budget_reached"))
    conclusions = conclusions[:cap]
    if len(claims) > MAX_CLAIMS:
        gaps.append(_gap("claim_budget_reached"))
    trace.update(details)
    trace["max_result_bytes"] = MAX_RESULT_BYTES
    trace["parser_scope"] = "short_literal_relations_and_explicit_conditionals"
    result = {"mode": selected_mode, "status": "derived" if conclusions else "insufficient_evidence",
              "conclusions": conclusions, "gaps": gaps[:8], "trace": trace}
    while len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > MAX_RESULT_BYTES and conclusions:
        conclusions.pop()
        if not any(gap["code"] == "reasoning_budget_reached" for gap in result["gaps"]):
            result["gaps"] = [*result["gaps"][:7], _gap("reasoning_budget_reached")]
    result["status"] = "derived" if conclusions else "insufficient_evidence"
    return result
