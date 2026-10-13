"""Bounded source-window competition when local neural assets are absent.

Reuse the live-source window pool from the neural lane, then select bounded
document/role/topic coverage without claiming neural or support scores.
No embeddings, translations, answer generation, or inferred facts are made.
"""
from __future__ import annotations

import time
import re
import math
import hashlib
from collections import Counter

from .research_context_assist import route_context_assist, _EXACT_ONLY
from .research_evidence_selection import prepare_evidence_candidates, literal_query_surface
from .research_fallback_sources import filter_fallback_sources, source_filter_reason
from .research_query_facets import build_query_facets, _document_mentions, _QUOTED, is_role_only_phrase
from .research_retrieval import lexical_tokens
from .research_sentence_windows import opening_reference_unverified, sentence_spans, is_figure_caption_start, is_section_heading
from .research_figure_axes import requests_graph_data
from .research_operation_stages import requested_operation_stages, source_operation_qualities, operation_stages_by_document
from .research_component_coverage import (
    functional_definitions, component_nomination_queue, MAX_COMPONENTS_PER_DOCUMENT,
    requests_component_explanation, requests_measurement_without_structure, ordered_method_steps,
    MAX_ORDERED_METHOD_CELLS_PER_DOCUMENT,
)


def matching_tokens(text):
    """Cheap inflection matching, not synonym or cross-language expansion.

    Split Latin compounds, reduce regular plurals/verb endings and a few
    productive nominal endings. This locates e.g. retrieve/retriever/retrieval
    without injecting a guessed method, expected result, or new search topic.
    Han bigrams and short identifiers retain their original form.
    """
    result = []
    for token in lexical_tokens(text):
        for word in re.split('[-_]', token) if token.isascii() else (token,):
            if not word.isascii() or not word.isalpha() or len(word) < 5:
                result.append(word)
                continue
            if word.endswith('ies') and len(word) > 5:
                word = word[:-3] + 'y'
            elif word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
                word = word[:-1]
            for suffix in ('ing', 'ed', 'er', 'or', 'al'):
                if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                    # Nouns like flower/corner are not inflections of flow/corn.
                    # Keep uncertain agent/nominal endings intact; only the
                    # narrow productive verb-shaped roots receive this alias.
                    if suffix in ('er', 'or', 'al') and word[:-len(suffix)][-1] not in 'dtv':
                        continue
                    word = word[:-len(suffix)]
                    break
            if word.endswith('ion') and len(word) > 6 and word[-4] in 'st':
                word = word[:-3]
            if word.endswith('e') and not word.endswith('ee') and len(word) > 4:
                word = word[:-1]
            result.append(word)
    return result


_ACTIONS = re.compile(
    r'\b(?:encod\w*|retriev\w*|index\w*|combin\w*|select\w*|partition\w*|'
    r'marginali[sz]\w*|generat\w*|pass\w*|construct\w*|stor\w*|access\w*|'
    r'load\w*|evict\w*|swap\w*|pag(?:e|es|ed|ing)|call\w*|search\w*)\b|'
    r'编码|检索|索引|组合|选择|划分|存储|访问|载入|换出|分页|调用|搜索', re.I)
_OBSERVATION = re.compile(
    r'\b(?:we (?:find|found|show|observed)|results? (?:show|indicat|suggest)|'
    r'achiev\w*|outperform\w*|degrad\w*|decreas\w*|increas\w*|'
    r'accuracy|performance|observed|improv\w*)\b|发现|结果表明|准确率|性能|提升|下降|退化', re.I)
_EXPERIMENT = re.compile(
    r'\b(?:evaluat\w*|experiment\w*|benchmark\w*|dataset\w*|ablation\w*|'
    r'participants?|randomi[sz]\w*|controlled?|baseline\w*|test questions)\b|'
    r'评测|实验|数据集|消融|随机|对照|控制变量', re.I)
_STUDY_DESIGN = re.compile(
    r'\b(?:baselines?|controlled?|randomi[sz]\w*|held[- ]out|participants?|datasets?|'
    r'independent\w* sampl\w*|fixed (?:model|setting|load|output)|'
    r'\d+\s+(?:(?:distinct|virtual|probing|test|sampled)\s+)?'
    r'(?:users?|questions?|requests?|samples?|trials?|days?|tasks?|examples?))\b|'
    r'基线|对照|随机|独立采样|留出|参与者|数据集|固定(?:模型|参数|设置)|\d+\s*(?:名用户|个样本|个问题|轮试验)', re.I)
_QUALIFIER = re.compile(
    r'\b(?:limit\w*|only|unless|however|although|cannot|not|fail\w*|'
    r'conditions?|assum\w*|trade.?offs?|under|even)\b|局限|限制|仅|除非|但是|不能|失败|条件|前提', re.I)
_STRUCTURE = re.compile(
    r'\b(?:we (?:combin\w*|introduc\w*|propos\w*|present\w*)|'
    r'(?:system|model|memory|context|architecture)\s+(?:is |are )?(?:divid\w*|partition\w*|consist\w*|combin\w*)|'
    r'(?:retriever|encoder|generator|index|memory|context)\b.{0,110}\b(?:with|into|between|from|to)\b.{0,110}\b'
    r'(?:retriever|encoder|generator|index|memory|context)|hierarch\w*|tiers? of memory)\b|'
    r'系统.{0,20}(?:分为|包含|由)|记忆层级|我们(?:结合|提出|设计)|(?:编码|检索).{0,30}(?:生成|输出)', re.I)
_DIRECT_FINDING = re.compile(
    r'\b(?:we (?:find|found|show|observed|demonstrat\w*)|results? (?:show|indicat\w*|suggest)|'
    r'(?:accuracy|performance)\s+(?:is |was )?(?:highest|lowest|better|worse|degrad\w*|decreas\w*|increas\w*)|'
    r'outperform\w*)\b|发现|结果表明|准确率.{0,12}(?:更高|更低|下降|提升)', re.I)
_APPENDIX = re.compile(r'^\s*(?:appendi(?:x|ces)\b|[A-Z](?:\.\d+)*\s+(?:implementation|human evaluation|additional|hyperparameter))', re.I)
_OVERVIEW_SECTION = re.compile(r'\b(?:abstract|introduction|conclusions?|discussion|related works?|background)\b|摘要|引言|结论|讨论|相关工作|背景', re.I)
_MEASUREMENT_DEFINITION = re.compile(
    r'\b(?:metrics?|measurements?|annotators?|assessors?|raters?|assessment|evaluation protocol)\b.{0,180}'
    r'\b(?:scor\w*|assess\w*|measur\w*|judg\w*|scale|label\w*|determin\w*)\b|'
    r'\b(?:defin\w*|comput\w*|calculat\w*|measur\w*)\b.{0,65}'
    r'\b(?:as (?:the )?(?:fraction|ratio|proportion|number)|as .{0,70}divided by|on (?:a|the) .{0,20}scale)\b|'
    r'\bmeasured in\b.{0,55}\b(?:using|by)\b|'
    r'\bis\s+(?:the\s+)?(?:fraction|ratio|proportion)\s+of\b|'
    r'\bis\s+(?:the\s+)?number\b[^.!?]{1,90}\bdivided by\b|'
    r'\b(?:candidate|option|segment|item)\b[^.!?]{0,50}\bwith\s+(?:the\s+)?'
    r'(?:lower|higher|lowest|highest|minimum|maximum|best)\b[^.!?]{1,65}'
    r'\b(?:is|are)\s+(?:selected|chosen)\s+as\s+(?:the|a)\s+(?:predicted\s+)?(?:label|answer|prediction)\b|'
    r'\b(?:choose|select)\s+(?:the|a)\s+(?:candidate|option|segment|item)\b[^.!?]{0,50}'
    r'\bwith\s+(?:the\s+)?(?:lower|higher|lowest|highest|minimum|maximum|best)\b[^.!?]{1,65}'
    r'\bas\s+(?:the|a)\s+(?:predicted\s+)?(?:label|answer|prediction)\b|'
    r'(?:得分|分值|分数).{0,15}(?:最低|最高|更低|更高).{0,25}(?:候选|选项|片段).{0,15}'
    r'(?:选为|选作).{0,10}(?:标签|答案|预测)|'
    r'(?:指标|标注者|评分员|评价方法).{0,60}(?:计算|评定|评分|判断|量表|标签)|'
    r'(?:定义|计算).{0,25}(?:比例|比率|分数|数量)', re.I | re.S)
_REPORTED_COMPARISON = re.compile(
    r'\b(?:result\w*|finding\w*|insights?)\b.{0,600}'
    r'\b(?:higher|lower|better|worse|high performance|low performance|no (?:difference|improvement)|'
    r'outperform\w*|underperform\w*|increas\w*|decreas\w*)\b|'
    r'(?:结果|发现).{0,100}(?:更高|更低|无差异|没有提升|优于|劣于|提高|降低)', re.I | re.S)
_NOT_REPORTED_OBSERVATION = re.compile(
    r'\b(?:plan\w*|proposed (?:experiment|evaluation|trial|test|study)|aim\w*|goals?|hope\w*|will|would|could|'
    r'prior (?:studies|work)|previous (?:studies|work)|earlier (?:studies|work))\b|'
    r'拟议|计划|旨在|目标是|将会|先前研究|以往研究', re.I)
_MISSING_MEASUREMENT = re.compile(
    r'\b(?:no|without)\b.{0,60}\b(?:annotation|scoring|measurement|evaluation)\b|'
    r'\b(?:annotators?|assessors?|raters?)\b.{0,25}\b(?:did not|do not|never)\s+(?:score|assess|judge)|'
    r'未(?:报告|提供|说明).{0,20}(?:标注|评分|测量)|没有.{0,15}(?:评分标准|标注标准)', re.I)
_OBSERVED_NULL = re.compile(
    r'\b(?:study|trial|experiment)\s+(?:found|observed|showed|reported)\b.{0,160}'
    r'\b(?:no (?:improvement|difference)|unchanged|not improve|zero|lower|higher)\b|'
    r'\b(?:(?:observed|measured) difference|observed change)\b.{0,80}\b(?:zero|unchanged|no)\b|'
    r'实验结果(?:显示|表明).{0,100}(?:没有提升|无差异|未改善)', re.I | re.S)
_NAMED_REPORTED_COMPARISON = re.compile(
    r'\b(?:[A-Z][A-Za-z0-9_-]{2,63}|(?:[Tt]he|[Oo]ur|[Tt]his)\s+(?:method|system|model|approach|service))\s+'
    r'(?:demonstrates?|achieved?|showed|produced|yielded|exhibited|obtained)\b.{0,150}'
    r'\b(?:improvements?|higher|lower|better|worse|increases?|decreases?|no (?:difference|improvement))\b.{0,100}'
    r'\b(?:than|compared (?:with|to)|relative to|over|versus)\s+(?:the\s+|a\s+)?[A-Za-z][\w-]*', re.S)
_ILLUSTRATIVE_FINDING = re.compile(r'\b(?:suppose|imagine|hypothetical\w*|for example|e\.g\.)\b|假设|假想|举例|例如', re.I)
_SETUP_DETAILS = re.compile(
    r'\b(?:randomi[sz]\w*|held[- ]out|recruit\w*|independent\w* (?:samples?|volunteers?|participants?))\b|'
    r'\b\d+\s+(?:(?:distinct|virtual|probing|test|sampled)\s+){0,2}'
    r'(?:users?|questions?|requests?|samples?|trials?|days?|tasks?|examples?)\b|'
    r'\b(?:same|fixed)\s+(?:model|output|token|time|parameter|input|data)\w*\b|'
    r'随机|留出|招募|独立采样|\d+\s*(?:名用户|个样本|个问题|天)|固定(?:模型|参数|设置)', re.I)


def _explicit_label_definition(text):
    """Local assessment plus at least two literal value-to-meaning bindings.

    This recognizes a scoring procedure, not the adequacy of its scale. A
    number list, a business-code dictionary or another sentence's labels do
    not establish the assessment in this sentence. No values are interpreted.
    """
    marker = re.search(r'\b(?:labels?|ratings?|scale)\s*[:=]|(?:标签|等级|量表)\s*[:：]', text, re.I)
    if not marker:
        return False
    before = text[max(0, marker.start() - 220):marker.start()]
    if not re.search(r'\b(?:assess\w*|evaluat\w*|judg\w*|scor\w*)\b|'
                     r'\bdetermin\w*\s+(?:whether|if)\b|评定|判断|评估|评分', before, re.I):
        return False
    pairs = re.finditer(
        r'(?:^|[,;，；{])\s*[\"\']?'
        r'(?:[-+]?\d+(?:\.\d+)?|[A-Za-z][\w-]{0,24}|[\u3400-\u9fff]{1,6})'
        r'[\"\']?\s*[:=：]\s*[\"\']?'
        r'(?:[A-Za-z]|[\u3400-\u9fff])[^,;，；}\])）]{0,90}',
        text[marker.end():marker.end() + 400])
    return sum(1 for _ in pairs) >= 2


def _measurement_anchor(text):
    return bool(_MEASUREMENT_DEFINITION.search(text) or _explicit_label_definition(text))


def _outcome_anchor(part):
    """A reported local comparison, not significance or causal validation."""
    return bool(_DIRECT_FINDING.search(part) or _REPORTED_COMPARISON.search(part) or _OBSERVED_NULL.search(part)
                or (_NAMED_REPORTED_COMPARISON.search(part) and not _ILLUSTRATIVE_FINDING.search(part)))


def evaluation_cells(row):
    """Three bounded visible evidence shapes, never experimental validity.

    Digits, headings, success language and repeated evaluation words alone do
    not fill a cell. No metric values or causal conclusions are reconstructed.
    """
    text = row.get('content', '')
    if re.search(r'\b(?:related works?|references|background)\b|相关工作|参考文献|背景',
                 row.get('section', ''), re.I):
        return []  # A third-party summary is not this document's study cell.
    cells = []
    if (_EXPERIMENT.search(text) and _SETUP_DETAILS.search(text) and re.search(
            r'\b(?:evaluat\w*|recruit\w*|assign\w*|randomi[sz]\w*|conduct\w*|used|use|'
            r'construct\w*|synthesi\w*|collect\w*|generat\w*|answered|compared?|design\w*)\b|'
            r'招募|分配|随机|使用|构造|生成|采集|评测|比较|设计', text, re.I)):
        cells.append('setup')
    sentences = [text[left:right] for left, right in sentence_spans(text)]
    if any(_measurement_anchor(part) and not _MISSING_MEASUREMENT.search(part)
           and not _NOT_REPORTED_OBSERVATION.search(part) for part in sentences):
        cells.append('measurement')
    if any(_outcome_anchor(part)
           and not _NOT_REPORTED_OBSERVATION.search(part) for part in sentences):
        cells.append('outcome')
    return cells


def _evaluation_quality(row, cell):
    # Three source-local protocol cues bound setting coverage; a mention of a
    # baseline alone, or one probing-question count, cannot exhaust all setup
    # utility. This is NOT sample adequacy or experimental quality scoring.
    if _OVERVIEW_SECTION.search(row.get('section', '')):
        return .2
    if cell == 'setup':
        return min(1., .35 * len(list(_SETUP_DETAILS.finditer(row['content']))))
    raw = row['content']
    spans = sentence_spans(raw)
    local = []
    for index, (left, right) in enumerate(spans):
        part = raw[left:right]
        anchor = ((_measurement_anchor(part) and not _MISSING_MEASUREMENT.search(part))
                  if cell == 'measurement' else
                  _outcome_anchor(part))
        if not anchor or _NOT_REPORTED_OBSERVATION.search(part):
            continue
        context = [part]
        if index + 1 < len(spans):
            next_left, next_right = spans[index + 1]
            following = raw[next_left:next_right]
            # Only an adjacent continuation in the same paragraph can supply
            # a scope cue. Never borrow another task's failure handling or a
            # later section's limitation from elsewhere in the source window.
            if (not re.search(r'\n\s*\n', raw[right:next_left] + following[:len(following) - len(following.lstrip())])
                    and not is_section_heading(following.strip())
                    and re.match(r'\s*(?:this|these|that|those|the (?:result|finding|measurement|observation|metric|score)|'
                                 r'however|nonetheless|failed|unanswered|timeouts?|该|此|这些|未回答|失败|但是)\b', following, re.I)):
                context.append(following)
        local.append(' '.join(context))
    return max((_evaluation_snippet_quality(text, cell) for text in local), default=0.)


def _evaluation_snippet_quality(text, cell):
    # Each anchor and its permitted adjacent qualifier stands alone. Taking
    # the best complete local unit must not union different tasks' cues.
    if cell == 'measurement':
        # A named metric is not its measuring procedure. Fraction, physical
        # unit/instrument, label scale and choice rule are alternative visible
        # forms, not a requirement that every metric have a denominator.
        defined = _explicit_label_definition(text) or bool(re.search(
            r'\b(?:divided by|denominator|measured in|labels?|scale|correct (?:suffix|option)|'
            r'choose\w*|select\w*|identify\w*)\b|比例|分母|量表|标签|选择|识别', text, re.I))
        handling = bool(re.search(
            r'\b(?:failed|unanswered|ambiguous|excluded|timeouts?|blinded|incorrect|'
            r'monotonic|independent annotators?)\b|失败|未回答|排除|超时|盲评|错误', text, re.I))
        return .5 + .25 * defined + .25 * handling
    # Observable task/comparison scope and stated limits improve the same
    # outcome cell, regardless of whether the reported result is positive or
    # negative. This is not a test of significance, validity or truth.
    finding_parts = [text[left:right] for left, right in sentence_spans(text)
                     if _outcome_anchor(text[left:right])
                     and not _NOT_REPORTED_OBSERVATION.search(text[left:right])]
    finding_text = ' '.join(finding_parts)
    comparator = bool(re.search(
        r'\b(?:compar\w*|baseline|than|relative to|versus|against|confidence|uncertainty)\b|'
        r'相比|比较|基线|对照|置信|不确定', finding_text, re.I))
    scope = bool(re.search(
        r'\b(?:only|under|not (?:tested|establish|generaliz\w*)|were not|'
        r'limitation\w*|although|however|nonetheless)\b|仅|条件|未测试|不能|限制|但是', text, re.I))
    return .5 + .25 * comparator + .25 * scope


def _state_maintenance(row):
    """One visible bounded-state operation, not a proof of memory capacity."""
    text = row['content']
    if _NOT_REPORTED_OBSERVATION.search(text):
        return False
    bounded_storage = any(re.search(
        r'\b(?:memory|queue|buffer|cache|records?)\b.{0,180}'
        r'\b(?:at most|bounded|capacity|latest|fixed[- ]size|retention limit)\b|'
        r'\b(?:bounded|fixed[- ]size)\s+(?:memory|queue|buffer|cache)\b|'
        r'(?:存储|队列|缓存|记录)[^。；;]{0,60}(?:容量|最多|有限|保留期限)',
        text[left:right], re.I | re.S) for left, right in sentence_spans(text))
    return bool(bounded_storage
                and re.search(r'\b(?:evict\w*|remov\w*|discard\w*|overwrit\w*|replac\w*|append\w*)\b|淘汰|移除|覆盖|替换|追加', text, re.I))
_DANGLING_ANTECEDENT = re.compile(
    r'^\s*(?:to\s+(?:overcome|address|resolve|handle|avoid|mitigate)\s+(?:this|these|that|those)'
    r'(?:\s+(?:problem|issue|obstacle|difficulty|limitation)s?)?\s*[,;:]|(?:为此|因此)\s*[，,:：])', re.I)


def _role_strength(row, ordered_steps=(), positive_evidence_request=False):
    """Transparent passage-shape priors, never factual support probabilities."""
    text, section = row['content'], row.get('section', '')
    actions = len(set(matching_tokens(' '.join(match.group(0) for match in _ACTIONS.finditer(text)))))
    method_section = bool(re.search(r'method|approach|architecture|implementation|方法|架构|实现', section, re.I))
    structure = bool(_STRUCTURE.search(text))
    # A long result/appendix with many operation words is not automatically a
    # full method. Composition/data flow is a stronger nomination shape; this
    # is still a fallible local proxy, never entailment or factual correctness.
    strengths = {
        'mechanism': min(1., min(.5, .18 * actions) + .25 * method_section + .5 * structure),
        'experiment_setup': min(1., .3 * len(set(match.group(0).casefold() for match in _EXPERIMENT.finditer(text)))),
        'result': .9 if _DIRECT_FINDING.search(text) else .4 if _OBSERVATION.search(text) else 0.,
        'limitation': .7 if _QUALIFIER.search(literal_query_surface(text,
                        positive_evidence_request=positive_evidence_request)) else 0.,
    }
    if ordered_steps and not _NOT_REPORTED_OBSERVATION.search(text):
        # Reuse the already recognized operation contract: extracting/building
        # is no less a visible method than indexing/retrieving. Apply this
        # BEFORE overview/evaluation guards, not as a bypass around them.
        strengths['mechanism'] = max(.9, strengths['mechanism'])
    outline = set(row.get('outline_roles', ()))
    if 'mechanism' in outline and actions:
        strengths['mechanism'] = min(1., strengths['mechanism'] + .5)
    elif 'experiment_setup' in outline:
        # Operations in a worked response/evaluation are not interchangeable
        # with a method explanation. A source's own positive outline supplies
        # a bounded role prior, not a verdict that the experiment is valid.
        strengths['mechanism'] *= .35
    if 'experiment_setup' in outline and strengths['experiment_setup']:
        strengths['experiment_setup'] = (min(1., strengths['experiment_setup'] + .4)
            if _STUDY_DESIGN.search(text) else strengths['experiment_setup'] * .5)
    if _OVERVIEW_SECTION.search(section):
        # An overview remains eligible (and can be the only available method),
        # but repeated operation/evaluation words must not outrank detailed
        # source-located method and experiment sections on word count alone.
        strengths['mechanism'] *= .5
        strengths['experiment_setup'] *= .5
    cells = set(row.get('_evaluation_cells', ()))
    if 'measurement' in cells:
        strengths['experiment_setup'] = max(.8, strengths['experiment_setup'])
    if 'outcome' in cells:
        strengths['result'] = max(.9, strengths['result'])
    return strengths


def _nomination_prior(row):
    """Permit a low-overlap architecture passage to enter named synthesis.

    This nomination is only a candidate opportunity, not an answer slot. The
    same bounded selector still decides whether it adds useful coverage.
    """
    roles = _role_strength(row)
    maintenance = _state_maintenance(row)
    return max(roles['mechanism'], 1. if maintenance else 0.) if (
        _STRUCTURE.search(row['content']) or maintenance) else 0.


def _stable_key(row):
    # Ingest UUIDs are provenance IDs, not relevance. Equal content must not
    # receive different utility just because a clone generated new UUIDs.
    return (row.get('filename', ''), row.get('ordinal', 0), row['excerpt_start'],
            hashlib.sha256(row['content'].encode()).hexdigest())


def _terminal_boundary_observed(content):
    # PDF page numbers are not a grammatical ending. This check only warns
    # about an unverified end; punctuation itself does not prove completeness.
    text = re.sub(r'\n\s*\d{1,4}\s*$', '', content).rstrip()
    return bool(re.search(r'[.!?。！？][\]\)"\u201d\u2019]*$', text))


def requested_stage_scopes(question, candidates):
    """Literal two-sided stage request plus two source-local process headings.

    Distinguishing facts from speculation is not a workflow request merely
    because another part of the question mentions a method. Unknown phrasing
    leaves the original mechanism priority unchanged.
    """
    surface = re.sub(r'`[^`]*`|"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’', ' ', question[:8192])
    match = re.search(
        r'\bseparate\s+(?P<a>[^.;!?，。；\n]{1,70}?)\s+from\s+(?P<b>[^.;!?，。；\n]{1,90})|'
        r'\bdistinguish\s+between\s+(?P<c>[^.;!?，。；\n]{1,70}?)\s+and\s+(?P<d>[^.;!?，。；\n]{1,90})|'
        r'(?:区分|分别说明)\s*(?P<e>[^。；，\n]{1,35}?)(?:与|和|以及)(?P<f>[^。；，\n]{1,45})',
        surface, re.I)
    if not match:
        return {}
    if re.search(r"\b(?:not|never|don't)\s*$|(?:不要|不用|不必|无需)\s*$",
                 surface[max(0, match.start() - 24):match.start()], re.I):
        return {}
    left, right = next((match[a], match[b]) for a, b in (('a', 'b'), ('c', 'd'), ('e', 'f')) if match[a])
    right = re.split(r'\band\s+(?:explain|describe|report|evaluate)\b|并(?:解释|说明|描述)', right, maxsplit=1, flags=re.I)[0]
    a, b = set(matching_tokens(left)), set(matching_tokens(right))
    result = {}
    for doc in {row['document_id'] for row in candidates}:
        headings = {row.get('section', '') for row in candidates if row['document_id'] == doc and ordered_method_steps(row)}
        first = {heading for heading in headings if set(matching_tokens(heading)) & a}
        second = {heading for heading in headings if set(matching_tokens(heading)) & b}
        if first and second and any(x != y for x in first for y in second):
            result[doc] = a | b
    return result


def evaluation_relevance_documents(question, candidates):
    """Only an explicit broad evaluation request may share lexical relevance.

    A named document is an applicability boundary, not evidence of a claim.
    Unknown metric/task wording conservatively keeps per-window relevance;
    this does not resolve arbitrary technical synonyms or implicit scopes.
    """
    text = _QUOTED.sub(' ', question[:8192])
    if re.search(r"\b(?:only|solely|exclusively|focus|not|never|quote|verbatim)\b|"
                 r"\b(?:don't|do not)\b|仅|只(?:问|讲|说明|解释|关注|看|评估)|不要|不必|逐字|原样引用", text, re.I):
        return set()
    mentions = _document_mentions(text, candidates)
    if not mentions:
        return set()
    applicable, context = set(), set()
    for fragment in re.finditer(r'[^,，;；。.!?？！\n:：]+', text):
        local = [item for item in mentions if fragment.start() <= item['start'] and item['end'] <= fragment.end()]
        if local:
            context = {doc for item in local for doc in item['document_ids']}
        clause = fragment.group()
        for item in reversed(local):
            left, right = item['start'] - fragment.start(), item['end'] - fragment.start()
            clause = clause[:left] + ' ' + clause[right:]
        clause = clause.strip()
        if (re.search(r'\b(?:evaluat\w*|experiment\w*|tested|testing|validation)\b', clause, re.I)
                and is_role_only_phrase(clause, question)):
            applicable.update(context)
        if re.fullmatch(r'(?:请)?(?:解释|说明)?(?:论文|研究|文中)?'
                        r'(?:给出|提供|进行|给了|有)?(?:了)?(?:哪些|什么)?'
                        r'(?:实际验证|验证情况|实验结果|评测结果)', clause):
            applicable.update(context)
    return applicable


def _select_windows(question, candidates, facets, baseline):
    """Greedy document/requirement coverage with diminishing returns and MMR.

    Independent document-role cells retain complementary literal utility.
    Repeating background cannot fill every slot by repeating a keyword. Each
    document has the same soft first-source bonus, never a hard eligibility
    filter. An earlier UUID-dependent baseline ranking cannot bias this bonus.
    """
    demand = {role for facet in facets['facets'] for role in facet.get('roles', ())}
    measurement_only = requests_measurement_without_structure(question)
    component_task = requests_component_explanation(question)
    if 'comparison' in demand:
        demand.update(('experiment_setup', 'result'))
        if not measurement_only:
            demand.add('mechanism')
    if measurement_only:
        demand.discard('mechanism')
        demand.update(('result', 'experiment_setup'))
    if re.search(r'hypothes|source.backed observation|假设|原文观察', question, re.I):
        demand.update(('mechanism', 'result', 'experiment_setup'))
    weights = {'mechanism': 1.3, 'experiment_setup': .8, 'result': 1., 'limitation': .65}
    roles = {role: weight for role, weight in weights.items() if role in demand}
    operation_needs = requested_operation_stages(question) if 'mechanism' in roles else ()
    operation_documents = (facets.get('operation_document_stages') or
                           operation_stages_by_document(question, candidates)) if operation_needs else {}
    evaluation_documents = {row['document_id'] for row in candidates if row.get('_evaluation_cells')}
    generic = set(matching_tokens(
        'explain describe compare comparison method methods mechanism mechanisms evaluation experiment '
        'conditions limitations evidence sources source backed based concise write report papers hypothesis '
        'propose state research literature review conclusions strongest alternative explanation '
        '介绍 解释 比较 方法 机制 评测 实验 条件 局限 证据 来源 报告 论文 假设 提出'))
    topic = set(matching_tokens(question)) - generic
    # A document name already binds its metadata scope. Repeating that name in
    # an abstract is not a second substantive topic requirement.
    topic -= {term for facet in facets['facets'] if facet.get('kind') == 'document'
              for term in matching_tokens(facet['text'])}
    topic -= {term for mention in _document_mentions(question, candidates)
              for term in matching_tokens(mention['text'])}
    topic -= {term for mention in facets.get('document_mentions', ())
              for term in matching_tokens(mention['text'])}
    named_documents = {doc for facet in facets['facets'] for doc in facet.get('document_ids', ())}
    component_keys = {}
    if demand & {'mechanism', 'comparison'}:
        for doc in {row['document_id'] for row in candidates}:
            nominated = component_nomination_queue([row for row in candidates if row['document_id'] == doc])
            keys = list(dict.fromkeys(item['key'] for row in nominated
                                     for item in row.get('_component_definitions', ())))
            component_keys[doc] = set(keys[:MAX_COMPONENTS_PER_DOCUMENT])
    order = sorted(candidates, key=lambda row: (-row['pool_score'], *_stable_key(row)))
    ordered_keys = {}
    stage_scopes = requested_stage_scopes(question, candidates) if component_task else {}
    if component_task:
        stage_order = sorted(order, key=lambda row: (
            -len(set(matching_tokens(row.get('section', ''))) & topic),
            -row['pool_score'], *_stable_key(row)))
        for row in stage_order:
            steps = ordered_method_steps(row)
            keys = ordered_keys.setdefault(row['document_id'], [])
            if steps and steps not in keys:
                keys.append(steps)
    ranked = []
    family_scales = {}
    for doc in {row['document_id'] for row in candidates}:
        doc_operations = operation_documents.get(doc, ())
        doc_rows = [row for row in candidates if row['document_id'] == doc]
        section_counts = {role: min(2, len({tuple(row.get('outline_path', ())) for row in doc_rows
                            if row.get('outline_path') and role in row.get('outline_roles', ())}))
                          for role in roles}
        method_ceiling = (1.3 * 1.4 + .25 * 1.3 * section_counts.get('mechanism', 0)
            + .45 * len(component_keys.get(doc, ()))
            + min(MAX_ORDERED_METHOD_CELLS_PER_DOCUMENT, len(ordered_keys.get(doc, ())))
            + .55 * any(_state_maintenance(row) for row in doc_rows))
        evaluation_fallback = .25 if doc in evaluation_documents else 1.
        evaluation_ceiling = evaluation_fallback * sum(
            weight * (1.4 + .25 * section_counts.get(role, 0))
            for role, weight in roles.items() if role in {'experiment_setup', 'result'})
        if doc in evaluation_documents:
            evaluation_ceiling += 1.3 * 3
        family_scales[doc] = {
            # Within the SAME 1.3 family cap, explicit requested operations
            # precede extra implementation detail. This fixed 3/4 allocation
            # is a development heuristic, not an optimality/semantic claim.
            'mechanism': 1.3 * (.25 if doc_operations else 1.) / method_ceiling,
            'operation': 1.3 * .75 / len(doc_operations) if doc_operations else 0.,
            'evaluation': 1.3 / evaluation_ceiling if evaluation_ceiling else 0.,
            'evaluation_fallback': evaluation_fallback,
        }
    definition_sources = {(row['document_id'], row['source_id']) for row in candidates
                          if row.get('_component_definitions') and not is_figure_caption_start(row['content'])}
    for row in order:
        if source_filter_reason(question, row['content'], section=row.get('section', '')) is not None:
            continue
        q = .2 + .8 * row['pool_score']
        if row['document_id'] in named_documents:
            q = max(.25, q)
        if ('mechanism' in demand and not requests_graph_data(question)
                and is_figure_caption_start(row['content'])
                and (row['document_id'], row['source_id']) in definition_sources):
            # A keyword-rich caption cannot stand in for the same source's
            # actual functional explanation. Both remain eligible; requests
            # for the figure itself keep their original opportunity.
            q *= .5
        if _APPENDIX.search(row.get('section', '')) and not re.search(
                r'implementation|hyperparameter|training details|human evaluation|实现细节|超参数|人类评测', question, re.I):
            q *= .8  # Soft priority only: appendix evidence remains eligible.
        strength = _role_strength(row, ordered_method_steps(row) if component_task else (),
                                 positive_evidence_request=bool(demand & {'mechanism', 'experiment_setup', 'result'}))
        if measurement_only:
            # An incidental word match (e.g. "checksum matches") cannot turn
            # an architecture definition into requested measurement evidence.
            if not (strength['result'] or strength['experiment_setup']):
                continue
            if re.search(r'\d', row['content']) and _DIRECT_FINDING.search(row['content']):
                q = max(q, .4)  # Explicit numeric observation, not support probability.
        ranked.append({**row, '_q': q, '_roles': strength,
                       '_operation_stages': set(row.get('_operation_stages', ())) & set(operation_documents.get(row['document_id'], ())),
                       '_operation_quality': row.get('_operation_quality', {}) or source_operation_qualities(row) if operation_needs else {},
                       '_state_maintenance': 'mechanism' in demand and _state_maintenance(row),
                       '_evaluation_quality': {cell: _evaluation_quality(row, cell)
                           for cell in row.get('_evaluation_cells', ())},
                       '_ordered_steps': ordered_method_steps(row) if component_task and
                           ordered_method_steps(row) in ordered_keys.get(row['document_id'], ()) else (),
                       '_ordered_quality': (1. if row['document_id'] not in stage_scopes
                            or set(matching_tokens(row.get('section', ''))) & stage_scopes[row['document_id']] else .5),
                       '_components': {item['key'] for item in row.get('_component_definitions', ())}
                           & component_keys.get(row['document_id'], set()),
                       '_tokens': set(matching_tokens(literal_query_surface(row['content'],
                           positive_evidence_request=bool(demand & {'mechanism', 'experiment_setup', 'result'})))),
                       '_bytes': len(row['content'].encode())})
    unique_rows = list({(row.get('filename', ''), row['content']): row for row in ranked}.values())
    shared_documents = evaluation_relevance_documents(question, candidates) if 'experiment_setup' in demand else set()
    cell_relevance = {}
    for row in ranked:
        if row['document_id'] in shared_documents:
            for cell in row.get('_evaluation_cells', ()):
                key = row['document_id'], cell
                cell_relevance[key] = max(cell_relevance.get(key, 0.), row['_q'])
    for row in ranked:
        row['_evaluation_relevance'] = {cell: cell_relevance.get((row['document_id'], cell), row['_q'])
                                       for cell in row.get('_evaluation_cells', ())}
    counts = Counter(term for row in unique_rows for term in row['_tokens'] & topic)
    topic_weights = {term: math.log1p(len(unique_rows) / max(1, counts[term])) for term in topic}
    topic_total = sum(topic_weights.values()) or 1.
    # Cache pair similarities once. Recognized complementary cells are not
    # duplicate content just because the surrounding vocabulary is shared.
    for row in ranked:
        signatures = {('evaluation', cell) for cell in row.get('_evaluation_cells', ())}
        signatures.update(('component', key) for key in row['_components'])
        signatures.update(('operation', stage) for stage in row['_operation_stages'])
        if row['_ordered_steps']:
            signatures.add(('ordered', row['_ordered_steps']))
        if row['_state_maintenance']:
            signatures.add(('state', 'bounded'))
        row['_signatures'] = signatures
    all_ranked = list(ranked)
    pair_cost = {}
    for index, left in enumerate(all_ranked):
        for right in all_ranked[index + 1:]:
            similarity = len(left['_tokens'] & right['_tokens']) / max(1, len(left['_tokens'] | right['_tokens']))
            a, b = left['_signatures'], right['_signatures']
            overlap = len(a & b) / max(1, len(a | b)) if a and b else 1.
            if (left['document_id'] != right['document_id'] and left['document_id'] in named_documents
                    and right['document_id'] in named_documents and left['content'] == right['content']):
                # A comparison needs each requested document's provenance.
                # Identical text remains explicitly non-independent evidence.
                overlap = 0.
            pair_cost[id(left), id(right)] = pair_cost[id(right), id(left)] = similarity * overlap

    def set_value(rows):
        """One bounded, source-shape set objective shared by both stages.

        A single greedy pass followed by one exchange pass is a heuristic,
        not an optimal set cover or a semantic completeness certificate.
        """
        value = 0.
        for doc, scales in family_scales.items():
            subset = [row for row in rows if row['document_id'] == doc]
            if not subset:
                continue
            value += .3 * max(row['_q'] for row in subset)
            for stage in operation_documents.get(doc, ()):
                value += scales['operation'] * max((row['_q'] * row['_operation_quality'].get(stage, 0.) for row in subset
                    if stage in row['_operation_stages']), default=0.)
            for role, original_weight in roles.items():
                weight = original_weight
                if role == 'mechanism':
                    weight *= scales['mechanism']
                elif role in {'experiment_setup', 'result'}:
                    weight *= scales['evaluation'] * scales['evaluation_fallback']
                strengths = sorted((row['_q'] * row['_roles'][role] *
                    (row['_ordered_quality'] if role == 'mechanism' else 1.) for row in subset), reverse=True)
                value += weight * sum(a * b for a, b in zip((1., .4), strengths))
                sections = {}
                for row in subset:
                    path = tuple(row.get('outline_path', ()))
                    if path and doc in named_documents and role in row.get('outline_roles', ()):
                        sections[path] = max(sections.get(path, 0.), row['_q'] * row['_roles'][role] *
                            (row['_ordered_quality'] if role == 'mechanism' else 1.))
                value += .25 * weight * sum(sorted(sections.values(), reverse=True)[:2])
            for key in component_keys.get(doc, ()):
                value += scales['mechanism'] * .45 * max((row['_q'] * row['_ordered_quality'] for row in subset if key in row['_components']), default=0.)
            operations = {}
            for row in subset:
                if row['_ordered_steps']:
                    key = row['_ordered_steps']
                    operations[key] = max(operations.get(key, 0.), row['_q'] * row['_ordered_quality'])
            value += scales['mechanism'] * sum(sorted(operations.values(), reverse=True)[:MAX_ORDERED_METHOD_CELLS_PER_DOCUMENT])
            value += scales['mechanism'] * .55 * max((row['_q'] * row['_ordered_quality'] for row in subset if row['_state_maintenance']), default=0.)
            for cell in ('setup', 'measurement', 'outcome'):
                value += scales['evaluation'] * 1.3 * max((row['_evaluation_relevance'][cell] * row['_evaluation_quality'][cell]
                    for row in subset if cell in row.get('_evaluation_cells', ())), default=0.)
        for term in topic:
            value += .9 * topic_weights[term] / topic_total * max((row['_q'] for row in rows if term in row['_tokens']), default=0.)
        value -= .125 * sum(row['_q'] * max((pair_cost.get((id(row), id(other)), 0.)
                           for other in rows if other is not row), default=0.) for row in rows)
        return value

    chosen, covered, documents, used, topic_coverage, component_coverage = [], {}, set(), 0, {}, {}
    section_coverage = {}
    evaluation_coverage = {}
    maintenance_coverage = {}
    ordered_coverage = {}
    operation_coverage = {}
    while ranked and len(chosen) < 8:
        options = []
        current_value = set_value(chosen)
        for row in ranked:
            if used + row['_bytes'] > 12000:
                continue
            if any(row['source_id'] == prior['source_id'] and row['document_id'] == prior['document_id']
                   for prior in chosen):
                continue  # Preserve one intact window per source, no stitched quote.
            similarity = max((len(row['_tokens'] & prior['_tokens']) /
                              max(1, len(row['_tokens'] | prior['_tokens'])) for prior in chosen), default=0.)
            named_document_not_yet_represented = (row['document_id'] in named_documents
                                                   and row['document_id'] not in documents)
            same_text_new_role = row['document_id'] in named_documents and any(
                row['_q'] * row['_roles'][role] > max(covered.get((row['document_id'], role), [0.]))
                for role in roles)
            same_text_new_cell = (
                any(row['_evaluation_relevance'][cell] * row['_evaluation_quality'][cell] >
                    evaluation_coverage.get((row['document_id'], cell), 0.)
                    for cell in row.get('_evaluation_cells', ()))
                or any(row['_q'] > component_coverage.get((row['document_id'], key), 0.)
                       for key in row['_components'])
                or any(row['_q'] * row['_operation_quality'].get(stage, 0.) > operation_coverage.get((row['document_id'], stage), 0.)
                       for stage in row['_operation_stages'])
                or (row['_ordered_steps'] and row['_q'] > ordered_coverage.get(
                    (row['document_id'], row['_ordered_steps']), 0.))
                or (row['_state_maintenance'] and row['_q'] > maintenance_coverage.get(row['document_id'], 0.)))
            if similarity >= .9 and not (named_document_not_yet_represented or same_text_new_role or same_text_new_cell):
                continue
            # The same bounded set objective drives greedy and the single
            # exchange pass; shape scores never certify factual support.
            utility = round(set_value([*chosen, row]) - current_value, 12)
            options.append((utility, row['_q'], row))
        if not options:
            break
        options.sort(key=lambda item: (-item[0], -item[1], item[2]['_bytes'],
                     *_stable_key(item[2])))
        utility, _, row = options[0]
        if utility < .08 * row['_q']:
            break
        chosen.append(row)
        ranked.remove(row)
        used += row['_bytes']
        documents.add(row['document_id'])
        for role in roles:
            cell = row['document_id'], role
            covered[cell] = sorted([*covered.get(cell, []), row['_q'] * row['_roles'][role]], reverse=True)[:2]
            path = tuple(row.get('outline_path', ()))
            if path and row['document_id'] in named_documents and role in row.get('outline_roles', ()):
                sections = section_coverage.setdefault(cell, {})
                sections[path] = max(sections.get(path, 0.), row['_q'] * row['_roles'][role])
        for term in row['_tokens'] & topic:
            topic_coverage[term] = max(topic_coverage.get(term, 0.), row['_q'])
        for key in row['_components']:
            cell = row['document_id'], key
            component_coverage[cell] = max(component_coverage.get(cell, 0.), row['_q'])
        for stage in row['_operation_stages']:
            key = row['document_id'], stage
            operation_coverage[key] = max(operation_coverage.get(key, 0.), row['_q'] * row['_operation_quality'].get(stage, 0.))
        if row['_ordered_steps']:
            key = row['document_id'], row['_ordered_steps']
            ordered_coverage[key] = max(ordered_coverage.get(key, 0.), row['_q'])
        for cell in row.get('_evaluation_cells', ()):
            key = row['document_id'], cell
            evaluation_coverage[key] = max(evaluation_coverage.get(key, 0.), row['_evaluation_relevance'][cell] * row['_evaluation_quality'][cell])
        if row['_state_maintenance']:
            maintenance_coverage[row['document_id']] = max(maintenance_coverage.get(row['document_id'], 0.), row['_q'])
    # Exactly one deterministic pass over the fixed <=48 candidates. A weak
    # early multi-cell window may be replaced, but no until-success loop or
    # new retrieval/model call is introduced. Quotes remain intact and unique.
    exchanges = 0
    exchange_attempts = 0
    for candidate in sorted(all_ranked, key=_stable_key):
        if candidate in chosen:
            continue
        current_value = set_value(chosen)
        proposals = []
        positions = list(range(len(chosen))) if len(chosen) == 8 else [None, *range(len(chosen))]
        for position in positions:
            if exchange_attempts >= 48 * 8:
                break
            exchange_attempts += 1
            proposal = list(chosen)
            if position is None:
                proposal.append(candidate)
            else:
                proposal[position] = candidate
            if sum(row['_bytes'] for row in proposal) > 12000:
                continue
            if len({(row['document_id'], row['source_id']) for row in proposal}) != len(proposal):
                continue
            difference = round(set_value(proposal) - current_value, 12)
            if difference > (1e-9 if position is not None else .08 * candidate['_q']):
                proposals.append((difference, position, proposal))
        if proposals:
            proposals.sort(key=lambda item: (-item[0], -1 if item[1] is None else item[1]))
            chosen = proposals[0][2]
            exchanges += 1
    # Receipts describe the final set, not the superseded greedy set.
    documents = {row['document_id'] for row in chosen}
    used = sum(row['_bytes'] for row in chosen)
    covered, component_coverage, evaluation_coverage = {}, {}, {}
    maintenance_coverage, ordered_coverage, section_coverage, topic_coverage = {}, {}, {}, {}
    operation_coverage = {}
    for row in chosen:
        doc = row['document_id']
        for role in roles:
            covered.setdefault((doc, role), []).append(row['_q'] * row['_roles'][role])
            path = tuple(row.get('outline_path', ()))
            if path and doc in named_documents and role in row.get('outline_roles', ()):
                sections = section_coverage.setdefault((doc, role), {})
                sections[path] = max(sections.get(path, 0.), row['_q'] * row['_roles'][role])
        for key in row['_components']:
            component_coverage[doc, key] = max(component_coverage.get((doc, key), 0.), row['_q'])
        for stage in row['_operation_stages']:
            operation_coverage[doc, stage] = max(operation_coverage.get((doc, stage), 0.), row['_q'] * row['_operation_quality'].get(stage, 0.))
        for cell in row.get('_evaluation_cells', ()):
            evaluation_coverage[doc, cell] = max(evaluation_coverage.get((doc, cell), 0.), row['_evaluation_relevance'][cell] * row['_evaluation_quality'][cell])
        if row['_ordered_steps']:
            ordered_coverage[doc, row['_ordered_steps']] = max(ordered_coverage.get((doc, row['_ordered_steps']), 0.), row['_q'] * row['_ordered_quality'])
        if row['_state_maintenance']:
            maintenance_coverage[doc] = max(maintenance_coverage.get(doc, 0.), row['_q'])
        for term in row['_tokens'] & topic:
            topic_coverage[term] = max(topic_coverage.get(term, 0.), row['_q'])
    sources = []
    for index, row in enumerate(chosen, 1):
        result = {key: row[key] for key in ('document_id', 'filename', 'content', 'ordinal',
                  'excerpt_start', 'excerpt_end', 'offset_unit', 'original_content_length',
                  'sentence_boundary_preserved', 'truncated')}
        result.update(id=row['source_id'], label=f'S{index}',
                      evidence_scope='excerpt_only' if row['truncated'] else 'source_chunk')
        result['sentence_boundary_preserved'] = row['sentence_boundary_preserved'] and _terminal_boundary_observed(row['content'])
        if 'page' in row:
            result['page'] = row['page']
        sources.append(result)
    return sources, {'objective': 'document_role_and_user_topic_coverage_minus_lexical_redundancy',
                     'requested_roles': sorted(roles), 'covered_document_roles': sum(any(value > 0 for value in values) for values in covered.values()),
                     'covered_user_topic_terms': len(topic_coverage), 'user_topic_terms': len(topic),
                     'selected_documents': len(documents), 'source_bytes': used,
                     'defined_component_cells': len(component_coverage),
                     'ordered_method_cells': sum(min(MAX_ORDERED_METHOD_CELLS_PER_DOCUMENT,
                         sum(1 for document, _ in ordered_coverage if document == doc)) for doc in documents),
                     'located_section_role_cells': sum(min(2, len([value for value in cells.values() if value > 0]))
                                                       for cells in section_coverage.values()),
                     'section_role_priority_is_verified_support': False,
                     'evaluation_cells_covered': len(evaluation_coverage),
                     'evaluation_relevance_shared_documents': len(shared_documents),
                     'evaluation_relevance_is_semantic_support': False,
                     'goal_family_caps': {'mechanism': 1.3, 'evaluation': 1.3},
                     'requested_operation_stages': list(operation_needs),
                     'operation_stage_cells_covered': len(operation_coverage),
                     'operation_stage_cells_unverified': sum((doc, stage) not in operation_coverage
                         for doc, stages in operation_documents.items() for stage in stages),
                     'requested_operation_document_cells': sum(map(len, operation_documents.values())),
                     'operation_family_allocation': {'requested_stages': .75, 'detail': .25} if operation_needs else {},
                     'operation_stages_are_semantic_support': False,
                     'goal_normalization_is_semantic_support': False,
                     'explicit_stage_scope_documents': len(stage_scopes),
                     'stage_scope_is_semantic_support': False,
                     'local_exchange_passes': 1, 'local_exchange_changes': exchanges,
                     'local_exchange_attempts': exchange_attempts, 'local_exchange_attempt_limit': 384,
                     'selection_is_global_optimum': False,
                     'state_maintenance_cells': len(maintenance_coverage),
                     'state_maintenance_is_verified_support': False,
                     'evaluation_coverage_is_verified_support': False,
                     'component_coverage_is_semantic_support': False,
                     'document_scope_is_hard_filter': False, 'selection_proxy_is_verified_support': False}


def select_lexical_coverage(question, baseline_sources, chunks):
    """Inspect at most 4096 chunks and rank at most 48 intact source windows.

    Small packs and literal/metadata tasks retain their existing source geometry.
    For substantive requests larger than the eight-source budget, requested
    facets nominate windows before one document/background can fill the pool.
    Selection uses lexical scores only; it cannot resolve cross-language or
    arbitrary paraphrase matches without a suitable local semantic retriever.
    """
    started = time.perf_counter()
    clean = filter_fallback_sources(question, baseline_sources)
    sources = clean['sources']
    requested_documents = set()
    trace = {'algorithm': 'literal_window_coverage_v1', 'status': 'retained',
             'baseline_source_count': len(baseline_sources),
             'selected_source_count': len(sources),
             'metadata_filter': clean['trace'], 'candidate_limit': 48,
             'scan_chunk_limit': 4096, 'model_calls': 0, 'provider_calls': 0,
             'llm_tokens': 0, 'score_kind': 'lexical_retrieval_utility',
             'semantic_entailment_verified': False}

    def result(reason, *, gaps=()):
        nonlocal sources
        gaps = list(gaps)
        sources = [{**row, 'sentence_boundary_preserved':
                    row.get('sentence_boundary_preserved', True) and _terminal_boundary_observed(row['content'])}
                   for row in sources]
        if (any(not row['sentence_boundary_preserved'] for row in sources)
                and not any(row.get('code') == 'source_boundary_unverified' for row in gaps)):
            gaps.append({'code': 'source_boundary_unverified',
                         'message': '部分来源末尾没有完整句界，可能跨页或截断；未提供的续文和条件仍需核对。'})
        if any(opening_reference_unverified(row['content']) for row in sources):
            gaps.append({'code': 'source_antecedent_unverified',
                         'message': '部分节选开头回指缺少先行对象；不能从下一实验或模型常识补造其对象和条件。'})
        if trace.get('corpus_scan_limited') and not any(row.get('code') == 'literal_coverage_scan_limited' for row in gaps):
            gaps.append({'code': 'literal_coverage_scan_limited',
                         'message': '本地原文筛选达到扫描上限，未覆盖全部资料；没有找到不代表不存在。'})
        if (len(requested_documents) >= 2 and not requested_documents <= {row['document_id'] for row in sources}
                and not any(row.get('code') == 'documents_not_selected' for row in gaps)):
            gaps.append({'code': 'documents_not_selected',
                         'message': '本次证据未覆盖全部指定文档，不能视为完整跨文档比较。'})
        trace.update(reason=reason, total_ms=round((time.perf_counter() - started) * 1000, 3),
                     selected_source_count=len(sources),
                     source_bytes=sum(len(row['content'].encode()) for row in sources))
        return {'sources': sources, 'trace': trace, 'gaps': gaps}

    if clean['trace']['status'] == 'metadata_query_retained':
        return result('explicit_metadata_request')
    if (_EXACT_ONLY.search(question[:8192]) or
            route_context_assist(question, baseline_sources)['reason'] == 'literal_or_editorial_task'):
        return result('literal_or_editorial_request')
    if sources and len(chunks) <= 8 and {row['id'] for row in chunks} <= {row['id'] for row in sources}:
        return result('bounded_pack_sufficient')
    facets = build_query_facets(question, chunks)
    measurement_only = requests_measurement_without_structure(question)
    if measurement_only:
        facets = {**facets, 'facets': [{**facet, 'roles': sorted(
            (set(facet.get('roles', ())) - {'mechanism'}) | {'result', 'experiment_setup'})}
            for facet in facets['facets']]}
    requested_documents = {doc for facet in facets['facets'] for doc in facet.get('document_ids', ())}
    demand = {role for facet in facets['facets'] for role in facet.get('roles', ())}
    operation_needs = set(requested_operation_stages(question)) if 'mechanism' in demand else set()
    operation_documents = operation_stages_by_document(question, chunks) if operation_needs else {}
    facets = {**facets, 'operation_document_stages': operation_documents}
    if not demand & {'mechanism', 'experiment_setup', 'comparison', 'limitation'}:
        return result('no_focused_evidence_requirement')

    query_component_terms = set(matching_tokens(question))

    def scoped_components(row):
        if not demand & {'mechanism', 'comparison'}:
            return []
        markers = functional_definitions(row)
        if row['document_id'] in requested_documents:
            return markers
        # Unnamed corpora do not acquire arbitrary glossary obligations. A
        # component must have an explicit surface anchor in the user's task.
        return [item for item in markers if set(matching_tokens(item['key'])) & query_component_terms]

    def nomination(row):
        if measurement_only:
            return .9 if _DIRECT_FINDING.search(row['content']) and _EXPERIMENT.search(row['content']) else 0.
        prior = _nomination_prior(row)
        if requests_component_explanation(question) and ordered_method_steps(row):
            prior = max(prior, .75)
        return prior

    pool = prepare_evidence_candidates(question, chunks, sources, facets, {},
                                      matching_tokens=matching_tokens, nomination_prior=nomination,
                                      literal_requested_roles=demand,
                                      component_profile=scoped_components if requests_component_explanation(question) else None,
                                      evaluation_profile=evaluation_cells if 'experiment_setup' in demand else None,
                                      operation_profile=(lambda row: {stage: quality for stage, quality in source_operation_qualities(row).items()
                                          if stage in operation_documents.get(row['document_id'], ())}) if operation_needs else None)
    trace['candidate_pool'] = pool['trace']
    trace['corpus_scan_limited'] = len(chunks) > 4096
    if not pool['candidates']:
        # An empty lexical nomination cannot prove the baseline is irrelevant:
        # preserve the separately retrieved pack, already stripped of metadata.
        return result('no_literal_window_candidates')
    selected, selection = _select_windows(question, pool['candidates'], facets, sources)
    if not selected:
        return result('no_window_above_utility_floor')
    sources = selected
    trace.update(status='selected', candidate_count=len(pool['candidates']),
                 selected_document_count=len({row['document_id'] for row in sources}),
                 replacement_count=len({row['id'] for row in baseline_sources} - {row['id'] for row in sources}),
                 selection=selection, corpus_scan_limited=len(chunks) > 4096)
    gaps = ([{'code': 'literal_coverage_scan_limited',
              'message': '本地原文筛选达到扫描上限，未覆盖全部资料；没有找到不代表不存在。'}]
            if len(chunks) > 4096 else [])
    if any(not row.get('sentence_boundary_preserved') for row in sources):
        gaps.append({'code': 'source_boundary_unverified',
                     'message': '部分来源末尾没有完整句界，可能跨页或截断；未提供的续文和条件仍需核对。'})
    return result('literal_window_competition', gaps=gaps)
