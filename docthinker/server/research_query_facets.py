"""Bounded literal query facets for local research retrieval, without an LLM.

Facets describe what the user asked, not what a paper is expected to say.  They
are neither generated search expansions nor an entailment/causality judgement.
Document names only supply coverage hints; they do not supply answer facts.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .research_sentence_windows import sentence_spans
from .research_document_titles import initial_title_hints

MAX_QUERY_BYTES = 8_192
MAX_FACETS = 8
MAX_FACET_BYTES = 768
MAX_METADATA_ROWS = 4_096
MAX_DOCUMENTS = 256
MAX_DOCUMENT_HINTS = 8
MAX_NAMES_PER_DOCUMENT = 4
MAX_TECHNICAL_ANCHORS = 32

_HAN = r"\u3400-\u4dbf\u4e00-\u9fff"
_WORDS = re.compile(r"[a-zA-Z0-9]+|[" + _HAN + r"]+")
_NAME_STOP = frozenset("a an the in on of for to and with at by from".split())
_GENERIC_FILE_STEMS = frozenset(
    'map cache memory paper report document notes note readme draft untitled summary review plan plans '
    'method methods results appendix data study research survey introduction abstract'.split())
_IDENTIFIED_FILE_STEM = re.compile(
    r'(?P<name>[A-Za-z][A-Za-z0-9-]{3,63})[_ -](?:'
    r'\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}(?:v[1-9]\d*)?|'
    r'(?:19|20)\d{2}[-_](?:0[1-9]|1[0-2])[-_](?:0[1-9]|[12]\d|3[01]))\Z', re.I)
_TRIM = " \t\r\n，,。.;；:：?？!！"
_POLITE = re.compile(
    r"^(?:(?:请(?:你)?|帮我|给我|麻烦你|能否|可以)(?:帮我)?\s*|"
    r"(?:please|could you|can you|would you)\s+)", re.I,
)
_CLAUSE_BREAK = re.compile(r"[?？!！\n]+|。(?!\d)")
# English full stops are ambiguous (decimals, e.g., initials). Split only
# before an explicit new question/command and after checking condition scope.
_EN_SENTENCE_BREAK = re.compile(
    r"(?<!\.)(?<!\b[A-Za-z])\.(?=\s+(?:(?:please\s+)?(?:explain|describe|compare|contrast|identify|discuss|trace|assess|evaluate|outline|determine)|how|why|what|which|when|where|whether)\b)", re.I,
)
# These are deliberately conservative syntactic dependencies, not a claim to
# resolve arbitrary pronouns or prove that two requests are semantically free.
_OPEN_SCOPE = re.compile(
    r"^(?:only\s+(?:if|when|under)|if\b|unless\b|provided\b|assuming\b|given\b|without\b|"
    r"under\s+(?:the\s+)?(?:assumption|condition))|^(?:如果|若|除非|仅当|只有|只在|在.{0,40}(?:条件|前提)下)", re.I,
)
_GLOBAL_ASSUMPTION = re.compile(r"^(?:assume|suppose)\b|^(?:假设|假定)", re.I)
_REFERENTIAL_START = re.compile(
    r"^(?:(?:please\s+)?(?:explain|describe|compare|contrast|identify|discuss|trace|assess|evaluate|outline|determine)\s+)?"
    r"(?:(?:how|why|what|which|when|where|whether)\s+)?"
    r"(?:(?:can|could|does|do|did|is|are|was|were|would|will|should|has|have)\s+)?"
    r"(?:it|its|they|their|them|this|that|these|those|such|the\s+(?:former|latter|above))\b|"
    r"^(?:(?:请|帮我)?(?:解释|说明|比较|对比|分析|讨论|评估|核对))?\s*"
    r"(?:它|其|这(?:个|些|种|一|项)|那(?:个|些|种)|该|上述|前者|后者)", re.I,
)
_QUESTION_HEAD = (
    r"(?:how|why|what|which|when|where|whether)\b|"
    r"(?:under|in|to)\s+(?:what|which)\b|"
    r"如何|怎么|怎样|为什么|为何|是否|能否|哪些|什么|哪里|何处|何时|何种"
)
_SUBQUESTION_BREAK = re.compile(
    r"[,，]\s*(?:(?:以及|并且|还有|同时)\s*|and\s+)?(?=(?:"
    + _QUESTION_HEAD + r")|"
    r"(?:它|其|该方法|该模型|该系统|这些实验|这项实验|论文中|文中|研究中|实验中|作者)\s*(?:"
    + _QUESTION_HEAD + r")|"
    r"实验|方法|结果|局限|限制|证据)", re.I,
)
_EN_QUESTION_BREAK = re.compile(r"\s+and\s+(?=(?:" + _QUESTION_HEAD + r"))", re.I)
_GUARD = re.compile(
    r"\b(?:if|unless|except|only|no|not|never|neither|nor|cannot|without|provided|assuming|rather than)\b|"
    r"\b\w+n['’]t\b|如果|若(?:是|仅|在|不)?|除非|只有|仅当|只在|"
    r"(?:仅|只)(?:依据|根据|基于|使用|用|引用)|不|未|无|没有|否则|而非", re.I,
)
_QUOTED = re.compile(r'“([^”\n]{2,160})”|"([^"\n]{2,160})"|`([^`\n]{2,160})`|《([^》\n]{2,160})》')
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*(?:[ \t]+[A-Za-z][A-Za-z0-9_-]*){0,11}")
_PHRASE_EDGE = frozenset(
    "a an and are as at be by can compare describe do does explain for from how "
    "i in is it me of on or please show summarize tell the their these this to was "
    "what when which why will with work works would you your".split()
)
_ROLE_PATTERNS = (
    ("experiment_setup", r"\b(?:experiment\w*|setup|benchmark\w*|evaluat\w*|ablation\w*|dataset\w*|test(?:ed|ing)?)\b|实验|评测|评估|数据集|测试|消融|检查|控制变量|设置|实际验证|验证情况|实证(?:检验|验证|结果)"),
    ("mechanism", r"\b(?:how|mechanism\w*|method\w*|algorithm\w*|implement\w*|procedure\w*|process\w*|architectur\w*|data flow)\b|如何|怎么|怎样|机制|方法|算法|实现|流程|步骤|架构|数据流"),
    ("result", r"\b(?:result\w*|finding\w*|effect\w*|outcome\w*|improv\w*|perform\w*|empirically\s+test\w*)\b|结果|结论|发现|效果|性能|提升|改善|实际验证|验证情况"),
    ("limitation", r"\b(?:limit\w*|fail\w*|caveat\w*|counterexample\w*|condition\w*|assum\w*|trade.?off\w*)\b|局限|限制|失败|反例|条件|假设|代价|前提"),
    ("comparison", r"\b(?:compar\w*|contrast\w*|differ\w*|versus|vs)\b|对比|比较|区别|异同|不同"),
)


def is_role_only_phrase(phrase: str, question: str) -> bool:
    """Recognize only generic English request language, not technical meaning.

    Unknown or mixed-language phrases and explicitly quoted terms stay as-is.
    This controls redundant query views, never document scope or evidence.
    """
    if not isinstance(phrase, str) or not re.fullmatch(r'[A-Za-z\s-]+', phrase.strip()):
        return False
    if any(any(group and group.casefold() == phrase.casefold() for group in match.groups())
           for match in _QUOTED.finditer(question[:MAX_QUERY_BYTES])):
        return False
    remainder = phrase
    found_role = False
    for _, pattern in _ROLE_PATTERNS:
        if re.search(pattern, remainder, re.I):
            found_role = True
            remainder = re.sub(pattern, ' ', remainder, flags=re.I)
    generic = _PHRASE_EDGE | {'task', 'tasks', 'answer', 'answers', 'study', 'studies',
                            'paper', 'papers', 'metric', 'metrics', 'its', 'actual', 'measured'}
    return found_role and set(re.findall(r'[a-z]+', remainder.casefold())) <= generic
_META_ONLY = re.compile(
    r"^(?:(?:请|帮我|给我|仅|只|务必|please)\s*)*(?:"
    r"(?:依据|根据|基于)(?:上传的|这些|所给|提供的)?(?:论文|资料|原文|文献)(?:回答|作答|说明)?|"
    r"(?:提供|给出|给|标注|附上|注明)(?:准确的|详细的)?(?:引用|出处|来源|参考文献)|"
    r"(?:不要|不得)(?:编造|虚构)(?:事实|结论|引用|来源|内容)?|"
    r"(?:用|使用)(?:中文|英文|markdown|Markdown)(?:回答|输出|格式)?|"
    r"(?:with|include|provide|add|give)(?:\s+(?:the|source|accurate))?\s+(?:citations?|references?|sources?)|"
    r"(?:answer|respond)(?:\s+only)?\s+(?:from|using|based on)\s+(?:the\s+)?(?:papers?|sources?|documents?)|"
    r"(?:related work|literature review|summary|abstract|introduction|paragraph|综述|摘要|相关工作))$", re.I,
)
_GENRE_PREFIX = re.compile(
    r"^(?:(?:请|帮我)\s*)?(?:写|撰写|起草|整理|生成)(?:一段|一篇|一份|一节)?\s*"
    r"(?:related work|literature review|综述|摘要|相关工作|介绍)(?:段落)?\s*[,，:：]\s*|"
    r"^(?:please\s+)?(?:write|draft|compose)\s+(?:a\s+)?(?:related work|literature review|summary|abstract)"
    r"(?:\s+(?:section|paragraph))?\s+(?:about|on|covering)\s+", re.I,
)
_EVIDENCE_PREFIX = re.compile(
    r"^(?:依据|根据|基于)(?:上传的|这些|所给|提供的)?(?:论文|资料|原文|文献)(?:回答|作答|说明)?\s*[,，:：]\s*", re.I,
)
_META_ANCHORS = frozenset({"related work", "literature review", "summary", "abstract", "introduction", "paragraph", "markdown", "citation", "citations", "references"})
_AUDIENCE_INTRO = re.compile(
    r"^\s*(?:(?:我想|我要|我希望|请|帮我)\s*)?"
    r"(?:(?:向|给|为)\s*[^，,。.;；!?？!]{0,24}(?:读者|受众|听众|新手|初学者|同学|同事|学生|用户|人员)\s*)?"
    r"(?:介绍|概述|总结)\s*$|^\s*(?:introduce|summari[sz]e)\s*$", re.I,
)

# A literal content request can be imperative or interrogative. These labels
# only route a bounded scoring budget; they never claim a source proves it.
_CONTENT_QUESTION = re.compile(_QUESTION_HEAD + r"|吗(?:[？?]|$)", re.I)
_CONTENT_COMMAND = re.compile(
    r"^(?:(?:请|帮我)\s*)?(?:解释|说明|比较|对比|分析|列出|讨论|追踪|梳理|评估|核对)|"
    r"^(?:please\s+)?(?:explain|describe|compare|contrast|identify|discuss|trace|assess|evaluate|outline)\b", re.I,
)
_VERIFICATION_DEMAND = re.compile(
    r"(?:是否|有没有|能否).{0,32}(?:实现|验证|证明|检验|测试|评估|控制|排除|成立)|"
    r"(?:论文|研究|实验|作者).{0,16}(?:已经|尚未|实际).{0,16}(?:实现|验证|证明|检验|测试|评估)|"
    r"\b(?:whether|did|does|has|have)\b.{0,100}\b(?:implement\w*|demonstrat\w*|verif\w*|establish\w*|achiev\w*|test\w*|evaluat\w*|control\w*|rule out)\b", re.I,
)

# This deliberately narrow syntax is a scoring view, not a claim extractor or
# a general quotation parser. Ambiguous apostrophes/nesting fail closed.
_AUDIT_PREFIX = re.compile(
    r"^(?:in\s+English\s*,\s*)?(?:please\s+)?(?:revise|correct|fact-check|audit)\s+"
    r"(?:this|the|the\s+following)\s+(?P<modifiers>(?:[A-Za-z][A-Za-z-]{0,30}\s+){0,3})(?:claim|statement)\s+"
    r"(?:using|against|based\s+on|according\s+to)\s+[^:;.!?\n]{2,160}:\s*$", re.I,
)
_AUDIT_SCOPE = re.compile(
    r"\b(?:if|unless|except|only|provided|assuming|assume|suppose|given|without|"
    r"conditional|conditioned|rather\s+than)\b|\bunder\s+(?:the\s+)?(?:condition|assumption)", re.I,
)
_AUDIT_PREFIX_SCOPE = re.compile(
    r"\b(?:when|whenever|where|while|once|until|before|after|during|exclud\w*|omit\w*|"
    r"ignor\w*|remov\w*|restrict\w*|condition\w*|depending|contingent)\b|"
    r"\b(?:subject\s+to|as\s+long\s+as|in\s+case)\b", re.I,
)
_AUDIT_MULTI_DOC = re.compile(
    r"\b(?:papers|studies|documents|both|versus|vs)\b|\b(?:and|or)\b|[&/]", re.I,
)
_AUDIT_REFERENCE = re.compile(r"\b(?:it|its|they|their|them|this|that|these|those|such|former|latter|above|same)\b", re.I)
_AUDIT_REQUEST = re.compile(r"^(?:please\s+)?(?:explain|describe|compare|contrast|identify|distinguish)\s+", re.I)


def quoted_audit_request(question: str) -> dict[str, Any] | None:
    """Join the literal prefix and request while eliding only the assertion.

    The quoted assertion remains in the original question. This optional view
    can rank a source for the explicit outside material request; it is neither
    an independent need nor support for the assertion being audited.
    Its two exact source spans are not a contiguous quotation of the question.
    """
    text = _safe_text(question)
    if not text or len(text.encode("utf-8")) > MAX_QUERY_BYTES or re.search("[" + _HAN + "]", text):
        return None
    quotes = [(index, char) for index, char in enumerate(text) if char in "\"'“”‘’`《》"]
    if len(quotes) != 2:
        return None
    (start, opening), (end, closing) = quotes
    if {"\"": "\"", "'": "'", "“": "”", "‘": "’"}.get(opening) != closing:
        return None
    prefix, claim = text[:start], text[start + 1:end]
    audit_prefix = _AUDIT_PREFIX.fullmatch(prefix)
    if (not audit_prefix or len(prefix.encode("utf-8")) > 256
            or _AUDIT_MULTI_DOC.search(prefix) or _AUDIT_SCOPE.search(text)
            # The whole prefix, including the material reference,
            # must be free of limiting scope. Checking claim modifiers alone
            # cannot establish whether the removed assertion owns that scope.
            or _GUARD.search(prefix) or _AUDIT_PREFIX_SCOPE.search(prefix)
            or len(claim.split()) < 4 or "\n" in claim):
        return None
    modifiers = audit_prefix.group("modifiers")
    if _GUARD.search(modifiers) or re.search(r"\b(?:conditional|non-\w+|assumed|hypothetical)\b", modifiers, re.I):
        return None
    # Accept a terminal mark inside or immediately outside the closing quote.
    # No other intervening clause is removed to manufacture a standalone view.
    tail = text[end + 1:]
    boundary = re.match(r"[.!?]?\s+", tail)
    if boundary is None or not (claim.rstrip().endswith((".", "!", "?")) or tail[:1] in ".!?"):
        return None
    request_start = end + 1 + boundary.end()
    request_end = len(text.rstrip())
    request = text[request_start:request_end]
    if (not _AUDIT_REQUEST.match(request) or len(request.encode("utf-8")) > MAX_FACET_BYTES
            or _AUDIT_REFERENCE.search(request) or re.search(r"[.!?]\s+\S|[;\n]", request)
            or re.search(r"\b(?:papers|studies|documents|both|versus|vs)\b", request, re.I)
            or re.search(r"\b(?:another|other|second)\s+(?:paper|study|document)\b", request, re.I)
            or re.search(r"\b(?:[Pp]aper|[Ss]tudy|[Dd]ocument)\s+[A-Z0-9]", request)):
        return None
    # Require explicit material nouns, not merely "Explain why the claim is
    # wrong". This lexical floor is a conservative gate, not semantic parsing.
    stop = set("please explain describe compare contrast identify distinguish how why what which when where whether "
               "the a an is are was were be been has have had do does did can could would should will "
               "claim statement wrong right correct incorrect true false paper study source evidence "
               "cite citing citation citations using according from in on of to at by for with and or "
               "conflates conflate not no never".split())
    material = {word.casefold() for word in re.findall(r"[A-Za-z][A-Za-z_-]+", request)} - stop
    if len(material) < 3:
        return None
    prefix_end = len(prefix.rstrip())
    query_spans = [[0, prefix_end], [request_start, request_end]]
    query = " ".join(text[left:right] for left, right in query_spans)
    if len(query.encode("utf-8")) > MAX_FACET_BYTES:
        return None
    return {"text": query, "query_spans": query_spans,
            "request_span": [request_start, request_end],
            "claim_span": [start, end + 1]}


def valid_quoted_audit_view(view: Mapping[str, Any], original: Mapping[str, Any]) -> bool:
    """Recompute literal mapping; aliases or copied metadata are insufficient."""
    expected = quoted_audit_request(original.get("text"))
    if (expected is None or original.get("kind") != "original" or original.get("query_only")
            or original.get("derived_translation")
            or original.get("id") != "qf_" + hashlib.sha256(("original\0" + _key(original["text"])).encode()).hexdigest()[:12]):
        return False
    scope = original.get("document_ids", ())
    if (not isinstance(scope, (list, tuple)) or not all(isinstance(doc, str) for doc in scope)
            or len(set(scope)) > 1):
        return False
    query_spans = view.get("query_spans")
    if not isinstance(query_spans, list) or len(query_spans) != 2:
        return False
    if any(not isinstance(span, list) or len(span) != 2 or any(type(endpoint) is not int for endpoint in span)
           for span in query_spans + [view.get("request_span"), view.get("claim_span")]):
        return False
    identifier = "qf_" + hashlib.sha256(("query_view\0" + _key(expected["text"])).encode()).hexdigest()[:12]
    return (view.get("id") == identifier and view.get("kind") == "query_view"
            and view.get("purpose") == "quoted_claim_audit" and view.get("query_only") is True
            and view.get("counts_as_requirement") is False and not view.get("derived_translation")
            and not view.get("query_alias_of") and not view.get("translation_full_query")
            and view.get("query_view_of") == original.get("id")
            and view.get("text") == expected["text"]
            and view.get("query_spans") == expected["query_spans"]
            and view.get("request_span") == expected["request_span"]
            and view.get("claim_span") == expected["claim_span"]
            and isinstance(view.get("document_ids", ()), (list, tuple))
            and all(isinstance(doc, str) for doc in view.get("document_ids", ()))
            and set(view.get("document_ids", ())) == set(scope))


def literal_requirement_kind(text: str) -> str | None:
    """Classify an explicit source-content need without generated expansions."""
    if not isinstance(text, str) or not text.strip() or _META_ONLY.fullmatch(text.strip(_TRIM)):
        return None
    roles = _roles(text)
    if not (_CONTENT_QUESTION.search(text) or _CONTENT_COMMAND.search(text)
            or roles and re.search(r"解释|说明|比较|分析|讨论", text)):
        return None
    if _VERIFICATION_DEMAND.search(text):
        return "verification"
    if "limitation" in roles or re.search(r"边界|适用|有效范围|\b(?:boundar\w*|applicab\w*|validity)\b", text, re.I):
        return "limitation"
    if "experiment_setup" in roles:
        return "experiment"
    if "comparison" in roles:
        return "comparison"
    return "content"


def _safe_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return ""
    return value


def _key(text: str) -> str:
    return " ".join(text.casefold().split()).strip(_TRIM)


def _roles(text: str, *, material_validation=True) -> list[str]:
    roles = [name for name, pattern in _ROLE_PATTERNS if re.search(pattern, text, re.I)]
    # A request to use a paper's validation is not merely a future product
    # trial. Only local material-attribution grammar supplies this query hint;
    # quotations/explicit exclusion do not add a positive requirement.
    surface = _QUOTED.sub(' ', text)
    if material_validation:
        from .research_operation_stages import operation_request_clauses
        positive = ' '.join(operation_request_clauses(text))
        mechanism_pattern = next(pattern for name, pattern in _ROLE_PATTERNS if name == 'mechanism')
        if 'mechanism' in roles and not re.search(mechanism_pattern, positive, re.I):
            roles.remove('mechanism')
    if material_validation and re.search(r'\b(?:how|explain|describe)\b|如何|怎样|说明|解释', surface, re.I):
        # Reuse the bounded positive-operation parser rather than promoting
        # every occurrence of "retrieval" (e.g. scores) to an architecture ask.
        from .research_operation_stages import requested_operation_stages
        from .research_component_coverage import requests_measurement_without_structure
        if requested_operation_stages(text) and not requests_measurement_without_structure(text):
            roles = list(dict.fromkeys([*roles, 'mechanism']))
    validation_match = re.search(
        r'(?:论文|研究)(?:里|中)?(?:的)?(?:实际)?验证|'
        r'\b(?:papers?|studies|authors?)\s+(?:actually\s+)?(?:validated|verified|tested)\b', surface, re.I)
    if material_validation and validation_match:
        prefix = re.split(r'[。；;!?？！\n]', surface[:validation_match.start()])[-1]
        if not re.search(r'(?:不要|无需|不必|不讨论|不使用|排除)|\b(?:do not|without|exclude|ignore)\b', prefix, re.I):
            roles = list(dict.fromkeys([*roles, 'experiment_setup', 'result']))
    return roles


def _trim_fluff(text: str) -> str:
    text = text.strip(_TRIM)
    for _ in range(3):
        stripped = _POLITE.sub("", text, count=1).strip(_TRIM)
        if stripped == text:
            break
        text = stripped
    text = _EVIDENCE_PREFIX.sub("", text, count=1).strip(_TRIM)
    return _GENRE_PREFIX.sub("", text, count=1).strip(_TRIM)


def _scoped_sentences(text: str) -> list[tuple[str, bool]]:
    """Split explicit sentence boundaries before checking local negation.

    Unclosed condition fragments bind the remaining instructions. A following
    referential request stays with its antecedent in one original contiguous
    span. This sacrifices some decomposition when scope is unclear instead of
    inventing a subject or stripping a condition from a new scoring query.
    """
    result = []
    # Existing question-mark/newline/Han-full-stop routing is unchanged. This
    # repair adds English full-stop boundaries inside previously guarded
    # paragraphs, not a new cross-question coreference resolver.
    for section in _CLAUSE_BREAK.split(text):
        real_ends = {end for _, end in sentence_spans(section)}
        boundaries = [match.span() for match in _EN_SENTENCE_BREAK.finditer(section) if match.end() in real_ends]
        spans, cursor = [], 0
        for left, right in boundaries:
            spans.append((cursor, left))
            cursor = right
        spans.append((cursor, len(section)))
        guard = bool(_GUARD.search(section))
        groups, open_scope = [], False
        for left, right in spans:
            value = _trim_fluff(section[left:right])
            if not value:
                continue
            dependent = bool(_REFERENTIAL_START.search(value))
            if groups and guard and (open_scope or dependent):
                groups[-1] = (groups[-1][0], right, True)
            else:
                groups.append((left, right, False))
            # Do not guess that an unclosed leading condition fragment stops
            # applying at a full stop. A visible comma/semicolon separates a
            # leading condition from its requested action, but is not proof.
            open_scope = open_scope or bool(_GLOBAL_ASSUMPTION.search(value)) or bool(
                _OPEN_SCOPE.search(value) and not re.search(r"[,，;；]", value))
        result.extend((section[left:right], bound) for left, right, bound in groups)
    return result


def _balanced(values: list[str], count: int) -> list[str]:
    """Retain spread across explicit needs instead of always truncating the tail."""
    if count <= 0:
        return []
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(index * (len(values) - 1) / (count - 1))] for index in range(count)]


def _name_tokens(text: str) -> list[str]:
    words = [item.group(0).casefold() for item in _WORDS.finditer(text)]
    if len(words) >= 2:
        words = [word for word in words if word not in _NAME_STOP]
    return words


def _material_clauses(values: list[str], count: int) -> list[str]:
    """Prioritize literal evidence needs, without editing their bound scope.

    Background and output style remain in the original question. They cannot
    consume focused retrieval slots while explicit material needs are omitted.
    If genuine needs alone exceed the cap, retain a deterministic spread and
    report the omitted count; no semantic decomposition or paraphrase is made.
    """
    focused = [value for value in values if literal_requirement_kind(value)]
    selected = _balanced(focused, count)
    if len(selected) < count:
        selected.extend(_balanced([value for value in values if value not in focused], count - len(selected)))
    return [value for value in values if value in selected]


def _document_mentions(question: str, chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Stored metadata and narrowly located first-page titles are locator data,
    # not source instructions or query expansions. Ordinary body is not read.
    names: dict[str, set[str]] = {}
    aliases: dict[str, set[str]] = {}
    for index, row in enumerate(chunks):
        if index >= MAX_METADATA_ROWS:
            break
        if not isinstance(row, Mapping):
            continue
        identifier = _safe_text(row.get("document_id"))
        if not identifier or len(identifier) > 256:
            continue
        if identifier not in names and len(names) >= MAX_DOCUMENTS:
            continue
        bucket = names.setdefault(identifier, set())
        for field in ("filename", "title", "document_title"):
            name = _safe_text(row.get(field))[:256]
            if field == "filename":
                name = re.split(r"[/\\]", name)[-1]
                name = re.sub(r"\.(?:pdf|txt|md|docx?|html?|epub)$", "", name, flags=re.I)
                suffix = _IDENTIFIED_FILE_STEM.fullmatch(name)
                if suffix and suffix['name'].casefold() not in _GENERIC_FILE_STEMS:
                    aliases.setdefault(suffix['name'].casefold(), set()).add(identifier)
            if name:
                bucket.add(name)
                if len(bucket) > MAX_NAMES_PER_DOCUMENT:
                    # Stable bounded metadata work even when every chunk has a
                    # different title. No document text is scanned here.
                    bucket.remove(max(bucket))
    title_aliases: dict[str, set[str]] = {}
    for hint in initial_title_hints(chunks):
        if hint['document_id'] in names:
            title_aliases.setdefault(hint['name'].casefold(), set()).add(hint['document_id'])
    # The shortened spelling is metadata, not inferred from paper claims.
    # A shared prefix across versions/documents is ambiguous: retain ordinary
    # full-name matching, but never choose one arbitrary owner for the alias.
    for alias, owners in aliases.items():
        if len(owners) == 1:
            bucket = names[next(iter(owners))]
            if len(bucket) < MAX_NAMES_PER_DOCUMENT:
                bucket.add(alias)
    matches: dict[tuple[int, int], set[str]] = {}
    # Compact technical names are exact spellings, not stop-word-tokenized
    # titles: A-QUEUE must never make the unrelated word QUEUE a document name.
    for alias, owners in title_aliases.items():
        if len(owners) == 1:
            for found in re.finditer(r'(?<![\w-])' + re.escape(alias) + r'(?![\w-])', question, re.I):
                matches.setdefault(found.span(), set()).update(owners)
    query_tokens = list(_WORDS.finditer(question))
    for identifier in sorted(names):
        for name in sorted(names[identifier]):
            tokens = _name_tokens(name)
            if not tokens or len(tokens) > 20:
                continue
            if len(tokens) == 1 and (len(tokens[0]) < (2 if re.search("[" + _HAN + "]", tokens[0]) else 3)
                                     or tokens[0] in _NAME_STOP):
                continue
            # Exact Han names may occur inside a continuous Chinese sentence.
            if len(tokens) == 1 and re.fullmatch("[" + _HAN + "]+", tokens[0]):
                for found in re.finditer(re.escape(tokens[0]), question.casefold()):
                    matches.setdefault(found.span(), set()).add(identifier)
                continue
            for start, match in enumerate(query_tokens):
                if match.group(0).casefold() != tokens[0]:
                    continue
                consumed = []
                end = start
                for end in range(start, min(len(query_tokens), start + len(tokens) + 8)):
                    word = query_tokens[end].group(0).casefold()
                    if len(tokens) >= 2 and word in _NAME_STOP:
                        continue
                    consumed.append(word)
                    if consumed != tokens[:len(consumed)]:
                        break
                    if consumed == tokens:
                        matches.setdefault((match.start(), query_tokens[end].end()), set()).add(identifier)
                        break
    # Longest match first at a given position; omit strict substrings so one
    # filename cannot overwhelm every facet. Different IDs remain explicit.
    selected = []
    for (start, end), identifiers in sorted(matches.items(), key=lambda item: (item[0][0], -item[0][1])):
        if any(start >= prior["start"] and end <= prior["end"] for prior in selected):
            continue
        selected.append({"text": question[start:end], "start": start, "end": end,
                         "document_ids": sorted(identifiers)})
        if len(selected) >= MAX_DOCUMENT_HINTS:
            break
    return selected


def build_query_facets(question: str, chunks: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Return at most eight literal facets, plus non-sensitive planning counts.

    Derived facet text always occurs verbatim in the bounded user question.
    Negations/conditions inhibit decomposing their clause into independent
    phrase queries. Roles are soft question-intent hints, never evidence labels.
    """
    raw = _safe_text(question).strip()
    encoded = raw.encode("utf-8")
    truncated = len(encoded) > MAX_QUERY_BYTES
    text = encoded[:MAX_QUERY_BYTES].decode("utf-8", errors="ignore").strip()
    facets: list[dict[str, Any]] = []
    seen: set[str] = set()
    mentions = _document_mentions(text, chunks) if text else []
    all_document_ids = sorted({identifier for mention in mentions for identifier in mention["document_ids"]})
    skipped_long = 0
    protected_clauses = 0

    def add(value: str, kind: str, weight: float, *, documents=(), anchors=()) -> None:
        nonlocal skipped_long
        value = value.strip(_TRIM) if kind != "original" else value
        key = _key(value)
        if not key or key in seen or len(facets) >= MAX_FACETS:
            return
        if kind != "original" and len(value.encode("utf-8")) > MAX_FACET_BYTES:
            skipped_long += 1
            return
        if kind != "original" and len(value) < 3:
            return
        row: dict[str, Any] = {
            "id": "qf_" + hashlib.sha256((kind + "\0" + key).encode("utf-8")).hexdigest()[:12],
            "text": value, "kind": kind, "weight": weight,
            "roles": _roles(value, material_validation=kind in {'original', 'clause'}),
        }
        if kind == "clause" and (requirement := literal_requirement_kind(value)):
            row["requirement_kind"] = requirement
        if documents:
            row["document_ids"] = sorted(set(documents))
        if anchors:
            row["anchors"] = list(dict.fromkeys(anchors))[:MAX_TECHNICAL_ANCHORS]
        facets.append(row)
        seen.add(key)

    clauses: list[str] = []
    for sentence, bound_scope in _scoped_sentences(text):
        sentence = _trim_fluff(sentence)
        if not sentence:
            continue
        if bound_scope or _GUARD.search(sentence):
            protected_clauses += 1
            clauses.append(sentence)
        else:
            for part in re.split(r"[;；]+", sentence):
                for clause in _EN_QUESTION_BREAK.split(part):
                    clauses.extend(_SUBQUESTION_BREAK.split(clause))
    classified_clauses = []
    for clause in clauses:
        value = _trim_fluff(clause)
        residue = value
        for mention in mentions:
            residue = residue.replace(mention["text"], " ")
        classified_clauses.append((value, bool(_META_ONLY.fullmatch(value)), bool(_AUDIENCE_INTRO.fullmatch(residue))))
    has_specific_need = any(value and not meta and not introduction for value, meta, introduction in classified_clauses)
    substantive_clauses = []
    meta_clauses = 0
    for value, meta, introduction in classified_clauses:
        # Audience/intro wording shapes the answer, not the source ranking, if
        # another clause states the actual evidence need. A standalone request
        # to introduce a document remains a substantive request.
        if meta or (introduction and has_specific_need):
            meta_clauses += 1
            continue
        if value and _key(value) not in {_key(item) for item in substantive_clauses}:
            substantive_clauses.append(value)
    # Concise technical phrases are most useful in mixed-language questions.
    # No source vocabulary, translation dictionary, or reference answer is used.
    phrase_candidates: list[str] = []
    for clause in substantive_clauses:
        if _GUARD.search(clause):
            continue
        for match in _QUOTED.finditer(clause):
            phrase_candidates.append(next(group for group in match.groups() if group is not None))
        latin_values = [part for match in _LATIN_RUN.finditer(clause)
                        for part in re.split(r"\s+(?:and|or|versus|vs)\s+", match.group(0), flags=re.I)]
        for value in latin_values:
            # Classify the intact genre before trimming edge words. Otherwise
            # "related work" loses "work" and becomes a spurious "related"
            # technical query that actively retrieves background sections.
            if _key(value) in _META_ANCHORS:
                continue
            words = list(re.finditer(r"[A-Za-z][A-Za-z0-9_-]*", value))
            left, right = 0, len(words)
            while left < right and words[left].group(0).casefold() in _PHRASE_EDGE:
                left += 1
            while right > left and words[right - 1].group(0).casefold() in _PHRASE_EDGE:
                right -= 1
            if left >= right:
                continue
            phrase = value[words[left].start():words[right - 1].end()]
            count = right - left
            mixed = bool(re.search("[" + _HAN + "]", clause))
            listed = bool(re.search(r"[,;:]", clause))
            if 1 <= count <= 6 and (mixed or listed or "-" in phrase or "_" in phrase):
                phrase_candidates.append(phrase)
    technical_phrases = []
    for phrase in phrase_candidates:
        if (_key(phrase) in _META_ANCHORS or any(_key(phrase) == _key(mention["text"]) for mention in mentions)
                or _key(phrase) in {_key(item) for item in technical_phrases}):
            continue
        technical_phrases.append(phrase)
    anchors = _balanced(technical_phrases, MAX_TECHNICAL_ANCHORS)
    if text:
        add(text, "original", 0.5, documents=all_document_ids, anchors=anchors)
    # Reserve at least two independent phrase queries when a long multi-part
    # instruction would otherwise exhaust all eight query embeddings on prose.
    clause_budget = min(4 if len(anchors) >= 2 else 7, len(substantive_clauses))
    retained_clauses = _material_clauses(substantive_clauses, clause_budget)
    for value in retained_clauses:
        documents = [identifier for mention in mentions if mention["text"] in value
                     for identifier in mention["document_ids"]]
        add(value, "clause", 1.0, documents=documents or all_document_ids,
            anchors=[phrase for phrase in anchors if phrase in value])
    audit = quoted_audit_request(text) if not truncated and len(all_document_ids) <= 1 else None
    # Keep all existing literal needs. A view uses at most one phrase slot;
    # do not duplicate an already present composed query.
    if audit and len(facets) < MAX_FACETS and _key(audit["text"]) not in seen:
        facets.append({**audit, "id": "qf_" + hashlib.sha256(("query_view\0" + _key(audit["text"])).encode()).hexdigest()[:12],
                       "kind": "query_view", "weight": .5, "roles": [],
                       "purpose": "quoted_claim_audit", "query_only": True,
                       "counts_as_requirement": False, "query_view_of": facets[0]["id"],
                       "document_ids": all_document_ids})
        seen.add(_key(audit["text"]))
    for phrase in _balanced(anchors, MAX_FACETS - len(facets)):
        documents = [identifier for clause in substantive_clauses if phrase in clause
                     for mention in mentions if mention["text"] in clause
                     for identifier in mention["document_ids"]]
        add(phrase, "technical_phrase", 0.9, documents=documents, anchors=[phrase])
    for mention in mentions:
        add(mention["text"], "document", 0.45, documents=mention["document_ids"], anchors=[mention["text"]])
    return {
        "facets": facets,
        "document_mentions": mentions,
        "trace": {
            "status": "ready" if text else "empty_query", "version": "literal-facets-v4",
            "facet_count": len(facets), "max_facets": MAX_FACETS,
            "query_truncated": truncated, "query_bytes": len(text.encode("utf-8")),
            "max_query_bytes": MAX_QUERY_BYTES, "long_facets_skipped": skipped_long,
            "protected_clause_count": protected_clauses,
            "matched_document_count": len(all_document_ids),
            "technical_anchor_count": len(anchors), "technical_anchors_omitted": max(0, len(technical_phrases) - len(anchors)),
            "meta_clauses_removed": meta_clauses,
            "material_clauses_omitted": sum(bool(literal_requirement_kind(value))
                                             for value in substantive_clauses if value not in retained_clauses),
            "quoted_audit_view_count": sum(row.get("purpose") == "quoted_claim_audit" for row in facets),
            "provider_calls": 0, "llm_tokens": 0, "generated_expansions": False,
        },
    }
