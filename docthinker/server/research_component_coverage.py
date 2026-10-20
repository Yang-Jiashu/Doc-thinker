"""Source-bound functional-definition markers for bounded retrieval diversity.

These markers are grammatical locators, not extracted facts or an entailment
model. A name mentioned by a caption/overview does not cover its definition.
Quoted text is never changed, and marker text is never injected as new evidence.
"""
from __future__ import annotations

import json
import re

from .research_sentence_windows import evidence_units, opening_reference_unverified, section_heading_spans
from .research_source_contract import MAX_ANSWER_SOURCES, SOURCE_LABEL_PATTERN

MAX_COMPONENTS_PER_DOCUMENT = 8
MAX_DEFINITIONS_PER_WINDOW = 4
MAX_ORDERED_METHOD_CELLS_PER_DOCUMENT = 2
MAX_WRITER_COMPONENT_LOCATORS = 4
MAX_WRITER_COMPONENT_BYTES = 1200
MAX_WRITER_SOURCE_CHARS = 12000

_HEAD = r'(?:memory|storage|context|buffer|queue|cache|index|encoder|decoder|retriever|generator|scheduler|controller|processor|module|layer|datastore|database|instructions|component)'
_NAME = rf'(?P<name>(?:[A-Za-z][A-Za-z-]*\s+){{0,4}}{_HEAD})'
_OPERATION = (
    r'(?:(?:does?|can|must|may)\s+(?:not\s+)?|not\s+)?'
    r'(?:refer(?:s)?\s+to|serv(?:e|es)\s+as|consist(?:s)?\s+of|'
    r'stor(?:e|es)|hold(?:s)?|contain(?:s)?|keep(?:s)?|accept(?:s)?|'
    r'return(?:s)?|retriev(?:e|es)|search(?:es)?|manage(?:s)?|'
    r'is\s+(?:not\s+)?(?:read.only|writ(?:e?able)|used\s+to))\b(?![-\u00ad]\s*[A-Za-z])'
)
_DEFINITION = re.compile(rf'\b{_NAME}(?:\s*,\s*which)?\s+(?P<predicate>{_OPERATION})', re.I)
_HAN_DEFINITION = re.compile(
    r'(?P<name>[\u3400-\u9fffA-Za-z0-9]{0,10}(?:存储|缓存|缓冲区|队列|索引|编码器|解码器|检索器|生成器|调度器|控制器|处理器|模块|数据库|上下文|记忆))'
    r'(?P<predicate>(?:不|仅|主要)?(?:用于|负责|保存|存储|包含|接收|返回|指的是|是指))')
_NON_BODY = re.compile(
    r'\b(?:references|bibliography|related works?|prior works?|acknowledg\w*)\b|参考文献|相关工作|致谢', re.I)
_CAPTION = re.compile(r'^\s*(?:figure|fig\.|table)\s*\d|^\s*[图表]\s*\d', re.I)
_LEADING = frozenset('a an the and or but this that these those in of with for to from by as our its their'.split())
_COMPARISON = re.compile(r'\b(?:compar\w*|contrast\w*|versus|vs)\b|比较|对比|异同', re.I)
_STRUCTURAL_REQUEST = re.compile(
    r'\b(?:architectur\w*|mechanism\w*|components?|implementation|data flow|design)\b|'
    r'架构|机制|组件|组成|实现|数据流|原理|设计', re.I)
_METHOD_REQUEST = re.compile(r'\b(?:methods?|procedures?|algorithms?|operation)\b|方法|步骤|算法|工作流程', re.I)
_MEASUREMENT = re.compile(
    r'\b(?:accuracy|latency|throughput|scores?|metrics?|numbers?|percent(?:age)?|precision|recall|'
    r'f1|rouge(?:-\w+)?|exact.match|measurements?|p.values?)\b|准确率|延迟|吞吐|分数|指标|数值|数字|百分比|召回率', re.I)
_NEGATED_STRUCTURE = re.compile(
    r'\b(?:do not|don\x27t|without|avoid|no need to)\s+(?:(?:explain\w*|describ\w*|discuss\w*|the|any)\s+){0,4}'
    r'(?:architectur\w*|mechanism\w*|components?|design|implementation|methods?)\b|'
    r'\b(?:not|excluding|exclude)\s+(?:(?:the|any|its|their)\s+)?'
    r'(?:architectur\w*|mechanism\w*|components?|design|implementation|methods?)\b|'
    r'(?:不要|无需|不必|不需要|不含|不涉及)(?:解释|说明|讨论|展开|介绍|讲解|讲)?(?:架构|机制|组件|设计|实现|方法)|'
    r'(?:架构|机制|组件|设计|实现|方法)除外', re.I)
_PURPOSE_COMPARISON = re.compile(
    r'\b(?:compar\w*|contrast\w*).{0,220}\b(?:for|as)\b|(?:比较|对比).{0,80}(?:用于|用作|适用|场景)', re.I)
_EXPLAIN_FUNCTION = re.compile(
    r'\b(?:explain|describe|how)\b.{0,80}\b(?:memory|storage|retrieval|indexing|cache|buffer|queue|works?|operates?|functions?)\b|'
    r'(?:解释|说明|如何|怎样).{0,30}(?:存储|记忆|检索|工作|运行)', re.I)


def requests_component_explanation(question):
    """Conservative task gate, not 'every comparison needs architecture'.

    Require an explicit structural/method request or functional comparison for
    an application. Numeric-result comparisons and explicit exclusions leave
    the existing selector alone. Unknown intent creates no new obligation.
    """
    if not isinstance(question, str) or _NEGATED_STRUCTURE.search(question):
        return False
    functional = bool(_STRUCTURAL_REQUEST.search(question) or _EXPLAIN_FUNCTION.search(question))
    if _MEASUREMENT.search(question) and not functional:
        return False
    return functional or bool(_METHOD_REQUEST.search(question)
                              or (_COMPARISON.search(question) and _PURPOSE_COMPARISON.search(question)))


def requests_measurement_without_structure(question):
    """Explicit measurement/exclusion scope; unknown tasks are not reclassified."""
    if isinstance(question, str) and not _NEGATED_STRUCTURE.search(question):
        # A separate measurement deliverable does not cancel an explicit
        # operation explanation. Reuse positive clauses and stage-object
        # guards, so a metric *about* an operation gains no new method goal.
        from .research_operation_stages import operation_request_clauses, requested_operation_stages
        if any(re.search(r'\b(?:how|explain|describe)\b|如何|怎样|说明|解释', clause, re.I)
               and not _MEASUREMENT.search(clause) and requested_operation_stages(clause)
               for clause in operation_request_clauses(question)):
            return False
    return (isinstance(question, str) and not requests_component_explanation(question)
            and bool(_MEASUREMENT.search(question) or _NEGATED_STRUCTURE.search(question)))


def ordered_method_steps(row):
    """Locate an explicit ordered process, not an inferred causal chain.

    Up to two literal verb-pair cells per document prevent later-stage definitions
    from covering an explicitly described earlier/later process. No operation
    names or reconstructed steps are added to the evidence.
    """
    text, section = row.get('content', ''), row.get('section', '')
    if not isinstance(text, str) or _NON_BODY.search(section) or _CAPTION.match(text):
        return ()
    action = r'(?:build\w*|construct\w*|deriv\w*|extract\w*|index\w*|encod\w*|retriev\w*|'
    action += r'(?:pre-\s*)?generat\w*|summari[sz]\w*|combin\w*|partition\w*|select\w*|stor\w*|'
    action += r'load\w*|evict\w*|swap\w*|pass\w*|us(?:e|es|ing))'
    match = re.search(rf'\bfirst\s+(?:to\s+)?(?P<first>{action})\b[^.!?]{{8,420}}'
                      rf'\bthen\s+(?:to\s+)?(?P<then>{action})\b', text, re.I)
    if match:
        return (match['first'].casefold(), match['then'].casefold())
    return ()


def functional_definitions(row):
    """Return at most four exact name/predicate spans within this source window.

    Negation stays in the predicate and full quote. The presence of this marker
    does not decide whether the operation is supported, desirable or possible.
    Unrecognized paraphrases simply receive no extra structural coverage.
    """
    text, section = row.get('content', ''), row.get('section', '')
    if not isinstance(text, str) or not text or _NON_BODY.search(section) or _CAPTION.match(text):
        return []
    result, seen = [], set()
    for expression in (_DEFINITION, _HAN_DEFINITION):
        for match in expression.finditer(text):
            start, end = match.span('name')
            while start < end:
                first = re.match(r'[A-Za-z]+\s+', text[start:end])
                if not first or first.group(0).strip().lower() not in _LEADING:
                    break
                start += first.end()
            name = text[start:end]
            key = ' '.join(name.casefold().split())
            if not name or len(name) > 96 or key in seen or re.search(r'\b(?:and|or|but)\b', key):
                continue
            # A generic head on its own ("memory stores ...") does not
            # identify a distinct component that needs another output slot.
            if re.fullmatch(_HEAD, key, re.I):
                continue
            pred_start, pred_end = match.span('predicate')
            result.append({'key': key, 'name_start': start, 'name_end': end,
                           'predicate_start': pred_start, 'predicate_end': pred_end})
            seen.add(key)
            if len(result) >= MAX_DEFINITIONS_PER_WINDOW:
                return result
    return result


def component_definition_locators(question, source_quotes):
    """Locate functional wording in the actual fitted source-quote projection.

    No generated descriptions, new sources or completed-function claim are
    returned. Coordinates are relative to each unchanged quote's ``content``;
    a caller may add its ``excerpt_start`` to locate the original chunk. Only
    complete visible evidence units are pointed to, with their qualifications.
    This remains a bounded grammatical cue, not semantic or context validation.
    """
    if not requests_component_explanation(question) or not isinstance(source_quotes, (list, tuple)):
        return []
    queues, seen_labels = [], set()
    for row in source_quotes[:MAX_ANSWER_SOURCES]:
        if not isinstance(row, dict):
            continue
        text, label = row.get('content'), row.get('label')
        start, end = row.get('excerpt_start'), row.get('excerpt_end')
        if (not isinstance(text, str) or not text or len(text) > MAX_WRITER_SOURCE_CHARS
                or not isinstance(label, str) or not SOURCE_LABEL_PATTERN.fullmatch(label)
                or label in seen_labels or type(start) is not int or type(end) is not int
                or start < 0 or end - start != len(text)):
            continue
        seen_labels.add(label)
        headings = section_heading_spans(text)
        heading_index, section, queue, names = 0, '', [], set()
        for left, right in evidence_units(text):
            while heading_index < len(headings) and headings[heading_index][0] <= left:
                hleft, hright = headings[heading_index]
                section = text[hleft:hright]
                heading_index += 1
            unit = text[left:right]
            if (not re.search(r'[.!?。！？]["\'”’»)\]}]*\s*$', unit)
                    or re.search(r'[.!?]{2,}["\'”’»)\]}]*\s*$', unit)
                    or opening_reference_unverified(unit)):
                continue
            for marker in functional_definitions({'content': unit, 'section': section}):
                # A verb without an object (or an unfinished source tail) is
                # not a definition locator. Mention-only prose never qualifies.
                tail = re.split(r'[.!?。！？]', unit[marker['predicate_end']:], maxsplit=1)[0]
                if not re.search(r'[A-Za-z\u3400-\u9fff0-9]', tail) or marker['key'] in names:
                    continue
                name_start, name_end = left + marker['name_start'], left + marker['name_end']
                queue.append({'name': text[name_start:name_end], 'source_label': label,
                              'name_start': name_start, 'name_end': name_end,
                              'definition_start': left, 'definition_end': right,
                              'offset_unit': 'characters_in_content'})
                names.add(marker['key'])
                if len(queue) >= MAX_WRITER_COMPONENT_LOCATORS:
                    break
            if len(queue) >= MAX_WRITER_COMPONENT_LOCATORS:
                break
        if queue:
            queues.append(queue)
    # A source's long glossary cannot occupy all opportunities before another
    # source gets one. This exposes locators, never independent evidence votes.
    def metadata_bytes(locators):
        grouped = {}
        for locator in locators:
            grouped.setdefault(locator['source_label'], []).append(locator)
        return sum(len(',"component_definition_locators":'.encode())
                   + len(json.dumps(items, ensure_ascii=False, separators=(',', ':')).encode())
                   for items in grouped.values())

    result = []
    for index in range(MAX_WRITER_COMPONENT_LOCATORS):
        for queue in queues:
            if index >= len(queue):
                continue
            candidate = [*result, queue[index]]
            if metadata_bytes(candidate) > MAX_WRITER_COMPONENT_BYTES:
                continue
            result = candidate
            if len(result) >= MAX_WRITER_COMPONENT_LOCATORS:
                return result
    return result


def component_nomination_queue(rows):
    """Prefer new defined components/subsections, never mere name mentions.

    This reserves bounded candidate opportunities only, not final source slots.
    All ties use source content/geometry rather than upload UUIDs.
    """
    remaining = [row for row in rows if row.get('_component_definitions')]
    selected, names, sections = [], set(), set()
    while remaining and len(names) < MAX_COMPONENTS_PER_DOCUMENT and len(selected) < MAX_COMPONENTS_PER_DOCUMENT:
        eligible = [row for row in remaining if {item['key'] for item in row['_component_definitions']} - names]
        if not eligible:
            break
        eligible.sort(key=lambda row: (
            -len({item['key'] for item in row['_component_definitions']} - names),
            -(row.get('section', '') not in sections), -row['pool_score'],
            len(row['content']), row.get('filename', ''), row.get('ordinal', 0),
            row['excerpt_start'], row['content']))
        row = eligible[0]
        selected.append(row)
        remaining.remove(row)
        names.update(item['key'] for item in row['_component_definitions'])
        sections.add(row.get('section', ''))
    return selected
