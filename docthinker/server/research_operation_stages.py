"""Three explicit operation needs, using bounded surface syntax, not semantics.

Stages locate opportunities in unchanged source windows. They do not prove a
system performs an operation, that conditions are complete, or that two papers
describe interchangeable processes. Unknown forms intentionally return empty.
"""
from __future__ import annotations

import re

from .research_sentence_windows import sentence_spans, is_section_heading
from .research_query_facets import _QUOTED, _document_mentions

STAGES = ('write', 'change', 'read')
_BACKGROUND = re.compile(r'\b(?:abstract|introduction|related works?|prior works?|references|background)\b|摘要|引言|相关工作|背景|参考文献', re.I)
_EVALUATION = re.compile(r'\b(?:experiment\w*|evaluation|results?|benchmark\w*)\b|实验|评测|结果', re.I)
_PLANNED = re.compile(r'\b(?:plan\w* to|will|would|could|future (?:work|experiment)|proposed (?:experiment|study)|prior (?:work|studies))\b|计划|拟议|未来将|先前研究', re.I)
_NONCOVERAGE = re.compile(r'\b(?:does|do|did) not (?:describe|explain|discuss|cover|address)\b|(?:本节|本文|本段).{0,5}不(?:描述|解释|讨论|涉及|涵盖)', re.I)
_NEGATED_REQUEST = re.compile(r'^\s*(?:(?:please\s+)?(?:do not|don.t|skip|avoid)\b|(?:请)?(?:不要|不必|无需|不需要|不讨论|不解释|只字不提))', re.I)
_ARTIFACT = re.compile(r'^(?:(?:the|a|an|this|my|our)\s+)?'
    r'(?:(?!(?:and|or|then|to|into|in|from|with|for|how|that|which)\b)[a-z][\w-]*\s+){0,4}'
    r'(?:report|review|draft|essay|story|poem|answer|response|proposal|paper|article)\b|'
    r'^(?:一份|这份|该)?(?:报告|草稿|文章|故事|答案|回复|建议|论文)', re.I)
_METRIC_OBJECT = re.compile(r'^(?:[- ]+(?:scores?|accuracy|latency|metrics?|measurements?|rates?)\b|(?:的)?(?:准确率|得分|指标|延迟|分布))', re.I)
_QUERY = {
    'write': re.compile(r'\b(?:ingest\w*|insert\w*|writ\w*|store\w*|incoming|new (?:records?|information|data|entries|facts?))\b|写入|录入|新增(?:信息|记录|数据|事实)|新(?:信息|记录|数据).{0,8}(?:到来|进来|进入)|接收.{0,12}(?:信息|记录|数据)', re.I),
    'change': re.compile(r'\b(?:revis\w*|updat\w*|invalidat\w*|obsolete|changes?)\b|修订|变更|更新|失效|处理变化|替换旧', re.I),
    'read': re.compile(r'\b(?:retriev\w*|read\w*|look\s*up|subsequent (?:use|request)|later use)\b|读取|检索|取用|(?:下次|下一次|后续).{0,18}(?:用上|使用|查询|读取|检索)', re.I),
}
_VERBS = {
    'write': r'writes?|wrote|ingests?|ingested|inserts?|inserted|appends?|appended|adds?|added|creates?|created',
    'change': r'updates?|updated|revises?|revised|invalidates?|invalidated|replaces?|replaced|expires?|expired|reconciles?|reconciled',
    'read': r'retrieves?|retrieved|reads?|read|loads?|loaded|searches?|searched|looks? up',
}
_BAD_OBJECT = re.compile(r'^(?:the\s+)?(?:and|or|but|not|in|on|at|to|by|for|from|with|when|after|before|only|operations?|methods?|steps?|phases?|topics?|accuracy|latency|performance|report|draft|response|answer)\b', re.I)
_ASSIGNED_TARGET = re.compile(r'^\s*(?:for\s+|对\s*)([A-Za-z][\w-]{2,79})(?=\s*[,，]|\s+(?:only|explain|describe)\b|\s*只)', re.I)
_UNRESOLVED_TARGET = re.compile(r'^\s*(?:for\s+(?:(?:that|this)\s+(?:system|paper|document|method)|it)\b|对\s*(?:该系统|该论文|它))', re.I)
_TEMPLATE_HEADING = re.compile(r'\b(?:prompt\s+(?:templates?|examples?)|system\s+prompt)\b|提示模板|系统提示', re.I)
_TEMPLATE_ROLE = re.compile(r'\byou are\b|你是|你作为', re.I)
_LOCAL_TRIGGER = re.compile(r'\b(?:when|if|unless|only after|upon|on receipt|while)\b|当|如果|除非|仅当|只有|收到.{0,20}后', re.I)
_LOCAL_STATE = re.compile(r'\b(?:previous|prior|earlier|obsolete|existing|valid|invalidated|historical)\s+'
    r'(?:records?|versions?|revisions?|states?|facts?|entries|edges|intervals?)\b|'
    r'\b(?:current|new)\b.{0,60}\b(?:states?|versions?|records?)\b|'
    r'(?:之前|先前|旧|失效|有效|历史)(?:的)?(?:记录|版本|状态|事实|条目)|保留.{0,12}(?:历史|版本)', re.I)
_ASSIGN_RULE = re.compile(r'\bby\s+(?:setting|assigning)\s+[\w′\u2032\s]{1,60}\s+to\s+[\w′\u2032]+', re.I)
_OTHER_ASSIGNMENT = re.compile(r'\b(?:while|and|but)\s+(?:(?:the|a|an|another)\s+)?\w+\s+(?:sets?|assigns?)\b', re.I)


def _action_sentence_spans(text):
    # Evidence units may intentionally bind two antecedent-related sentences.
    # Their quotes stay together, but their operation qualifiers are separate.
    for left, right in sentence_spans(text):
        start = left
        for match in re.finditer(r'(?<=[.!?。！？])\s+', text[left:right]):
            end = left + match.start()
            yield start, end
            start = left + match.end()
        if start < right:
            yield start, right


def operation_request_clauses(question):
    """Bounded positive request views only; never alter source or user data."""
    if not isinstance(question, str) or len(question) > 8192:
        return ()
    if re.search(r'\b(?:count|identify)\s+(?:the\s+)?(?:verbs?|words?|terms?)\b|\btranslate\s*[:：]|翻译|统计.{0,8}(?:词语|动词|字数)', question, re.I):
        return ()
    text = _QUOTED.sub(' ', re.sub(r'(?m)^\s*>[^\n]*', ' ', question))
    clauses = []
    for unit in re.split(r'[;；。！？!?\n]|(?<=\.)\s+', text):
        parts = re.split(r'(?<=[,，])', unit)
        allow_list = False
        for part in parts:
            if not part.strip() or _NEGATED_REQUEST.match(part):
                allow_list = False
                continue
            # A bare comma-list item inherits only its own explicit operation
            # request. Keep the exact original substring, never manufacture a
            # new instruction or carry scope across sentences/assigned owners.
            item = re.sub(r'^\s*(?:and|or)\s+', '', part, flags=re.I).strip(' ,，.')
            if allow_list and any(pattern.fullmatch(item) for pattern in _QUERY.values()):
                previous = clauses[-1]
                request = re.search(r'\b(?:explain|describe)\b', previous, re.I)
                blocked = request and re.search(r'\b(?:about|on|hypothes\w*|literature|report|papers?)\b', previous[request.end():], re.I)
                tail = previous.rstrip(' ,，')
                last_operation = any(match.end() == len(tail)
                                     for pattern in _QUERY.values() for match in pattern.finditer(tail))
                if request and not blocked and previous.rstrip().endswith((',', '，')) and last_operation:
                    clauses[-1] += part
                    continue
            clauses.append(part)
            allow_list = True
    return tuple(clauses)


def requested_operation_stages(question):
    """At most three user-explicit operations; never infer missing stages."""
    found = set()
    for clause in operation_request_clauses(question):
        for stage, pattern in _QUERY.items():
            for match in pattern.finditer(clause):
                following = clause[match.end():]
                if _ARTIFACT.match(following.lstrip()) or _METRIC_OBJECT.match(following):
                    continue
                if stage == 'read' and match.group().casefold() in ('retrieval', 'retriever', 'retrievers'):
                    # A topic (papers on retrieval / a hypothesis about it)
                    # is not an explicit request to explain a read operation.
                    before = clause[:match.start()]
                    request = re.search(r'\b(?:how|explain|describe|explaining|describing)\b', before, re.I)
                    if not request or re.search(r'\b(?:about|on|hypothes\w*|literature|report|papers?)\b', before[request.end():], re.I):
                        continue
                found.add(stage)
    return tuple(stage for stage in STAGES if stage in found)


def operation_stages_by_document(question, chunks):
    """Clause-bound locator scope, not document permissions or semantic intent.

    Reuse the same bounded, ambiguity-preserving metadata matcher as query
    facets. A common request is shared; an explicit A/B allocation is not.
    Unbound clauses remain common rather than resolving pronouns speculatively.
    """
    if not requested_operation_stages(question):
        return {}
    units = [' '.join(operation_request_clauses(part)) for part in
             re.split(r'[;；。！？!?\n]|(?<=\.)\s+', question)]
    positive = '\n'.join(units)
    mentions = _document_mentions(positive, chunks)
    named = {doc for item in mentions for doc in item['document_ids']}
    scope = named or {row.get('document_id') for row in chunks[:4096] if row.get('document_id')}
    result = {}
    offset = 0
    for unit in units:
        local_mentions = [item for item in mentions if offset <= item['start'] < offset + len(unit)]
        local_docs = {doc for item in local_mentions for doc in item['document_ids']}
        assigned = _ASSIGNED_TARGET.match(unit)
        if (not local_docs and (_UNRESOLVED_TARGET.match(unit) or (assigned and
                re.search(r'[a-z][A-Z]|[A-Z].*-|^[A-Z]{3,}$', assigned[1])))):
            # An explicitly assigned but unresolved technical name is not a
            # common request about the known documents. Do not invent its ID
            # or transfer its requested operations to another material.
            offset += len(unit) + 1
            continue
        for part in operation_request_clauses(unit):
            stages = requested_operation_stages(part)
            if not stages:
                continue
            start = unit.find(part)
            direct = {doc for item in local_mentions
                      if offset + start <= item['start'] < offset + start + len(part)
                      for doc in item['document_ids']}
            # "For A, explain ..." binds within that same unit; multi-owner
            # common clauses keep every owner rather than choosing one ID.
            for doc in direct or local_docs or scope:
                result.setdefault(doc, set()).update(stages)
        offset += len(unit) + 1
    return {doc: tuple(stage for stage in STAGES if stage in stages)
            for doc, stages in result.items()}


def source_operation_qualities(row):
    """Only source-local actions with a visible object, never an asserted fact.

    Static component names/definitions cannot substitute for input processing.
    Observed negative behavior is retained; planned or explicitly undiscussed
    operations do not count as a method stage. All source bytes stay unchanged.
    """
    if row.get('status') in ('revoked', 'deleted', 'invalidated'):
        return {}
    text = row.get('content', '')
    section = row.get('section', '')
    if not isinstance(text, str) or len(text) > 8000 or _BACKGROUND.search(section) or _EVALUATION.search(section):
        return {}
    if _TEMPLATE_HEADING.search(section + '\n' + text[:200]) and _TEMPLATE_ROLE.search(text[:500]):
        # An embedded role/prompt template remains retrievable source data;
        # its instructions are not a narrated observation of system behavior.
        return {}
    if 'experiment_setup' in row.get('outline_roles', ()) and 'mechanism' not in row.get('outline_roles', ()):
        return {}
    found = {}
    for left, right in _action_sentence_spans(text):
        part = text[left:right].strip()
        if (not part or is_section_heading(part) or _PLANNED.search(part) or _NONCOVERAGE.search(part)
                or not re.search(r'[.!?。！？][\])”"\']*\s*$', part)):
            continue
        # Quoted examples are data, not an assertion of this system's behavior.
        matching = _QUOTED.sub(' ', part)
        local = set()
        assigned = set()
        for stage, verbs in _VERBS.items():
            for action in re.finditer(r'\b(?:' + verbs + r')\b', matching, re.I):
                before, after = matching[:action.start()], matching[action.end():].lstrip()
                actor = bool(re.search(r'\b(?:the|this|our|we|it|they)\b', before[-100:], re.I)
                             or re.search(r'\b[A-Z][a-z]+[A-Z]\w*\b', before[-60:]))
                obj = re.sub(r'^(?:(?:the|a|an|each|new|incoming|current|obsolete|affected|existing)\s+)+', '', after, flags=re.I)
                if actor and re.match(r'[A-Za-z][\w-]*', obj) and not _BAD_OBJECT.match(obj):
                    local.add(stage)
                    tail = re.split(r'\b(?:while|and|but)\s+(?:the|a|an|another)\b|[;。.!?]',
                                    _OTHER_ASSIGNMENT.split(after, maxsplit=1)[0], maxsplit=1, flags=re.I)[0]
                    if _ASSIGN_RULE.search(tail):
                        assigned.add(stage)
        if re.search(r'\b(?:cannot|does not|can.t)\s+preserve\b.{0,45}\b(?:earlier|previous|prior|old)\s+(?:revisions?|versions?|records?)\b', matching, re.I):
            local.add('change')
        if re.search(r'(?:系统|服务|流程|代理|模块|我们|它).{0,35}(?:写入|录入|接收|新增).{0,15}(?:信息|记录|数据|事实|条目)', matching):
            local.add('write')
        if re.search(r'(?:系统|服务|流程|代理|模块|我们|它).{0,15}(?:信息|记录|数据|事实|条目).{0,12}(?:写入|录入)', matching):
            local.add('write')
        if re.search(r'(?:系统|服务|流程|代理|模块|我们|它).{0,35}(?:更新|修订|替换|使.{0,8}失效).{0,15}(?:信息|记录|数据|事实|条目|版本)', matching):
            local.add('change')
        if re.search(r'(?:系统|服务|流程|代理|模块|我们|它).{0,35}(?:读取|检索|取用|搜索).{0,15}(?:信息|记录|数据|事实|条目)', matching):
            local.add('read')
        # One anchor sentence owns its qualifiers. Another task's sentence,
        # even in the same window, cannot complete this operation by union.
        quality_view = _OTHER_ASSIGNMENT.split(matching, maxsplit=1)[0]
        for stage in local:
            quality = .5 + .25 * bool(_LOCAL_TRIGGER.search(quality_view)) + .25 * bool(
                _LOCAL_STATE.search(quality_view) or stage in assigned)
            found[stage] = max(found.get(stage, 0.), quality)
    return {stage: found[stage] for stage in STAGES if stage in found}


def source_operation_stages(row):
    return tuple(source_operation_qualities(row))


def source_operation_quality(row, stage):
    """A bounded local shape score, never support/truth probability."""
    return source_operation_qualities(row).get(stage, 0.)
