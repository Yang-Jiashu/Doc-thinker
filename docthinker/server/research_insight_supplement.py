"""One bounded, incremental local evidence pass after primary retrieval.

Passage shapes are locators, not semantic support or experimental validity.
Only original contiguous windows can be appended; the primary pack is immutable.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import re

from .research_component_coverage import (
    functional_definitions, ordered_method_steps, requests_component_explanation,
    requests_measurement_without_structure, _MEASUREMENT,
)
from .research_evidence_selection import prepare_evidence_candidates
from .research_fallback_sources import source_filter_reason
from .research_lexical_coverage import (
    evaluation_cells, matching_tokens, _evaluation_quality, _role_strength,
    _NOT_REPORTED_OBSERVATION, _terminal_boundary_observed,
)
from .research_query_facets import build_query_facets
from .research_retrieval import MAX_SOURCE_BYTES
from .research_sentence_windows import (
    opening_reference_unverified, section_heading_spans, sentence_spans,
)
from .research_source_contract import MAX_ANSWER_SOURCES
from .research_document_titles import initial_title_hints
from .research_operation_stages import (
    operation_stages_by_document, requested_operation_stages, source_operation_qualities,
)

MAX_SCAN = 256
MAX_CANDIDATES = 48
MAX_ADDED = 2
MAX_ADDED_BYTES = 2400
MAX_PROFILE_MEMO = 512

_GENERIC = set(matching_tokens(
    'a an the these those this that it its we our is are was were of on for from to and or with by '
    'how what which why should would could can do does did more most only same different between '
    'compare compares comparison comparative review synthesis report write writing study papers '
    'paper studies evidence sources source conditions metrics method methods mechanism mechanisms '
    'evaluation evaluate experiment experimental results result findings finding actual including '
    'measurement measurements measured scoring criteria scale '
    'describe explain discuss design first next product team relevant based using used use '
    '对比 比较 综合 整理 写作 报告 研究 论文 方法 机制 评测 实验 指标 条件 结果 来源 证据 '
    '我们 哪些 怎样 如何 说明 介绍 解释 结合 设计 建议 首先 然后 团队 产品'))
_PLANNED = re.compile(
    r'\b(?:proposed (?:experiment|trial|evaluation|study)|prior work|previous studies|earlier work|'
    r'we (?:plan|will|hope)|would|could)\b|拟议|计划|先前研究|以往研究', re.I)
_BACKGROUND_REQUEST = re.compile(r'\b(?:background|historical|history|inspiration)\b|背景|发展史|启发', re.I)
_PLANNED_REQUEST = re.compile(r'\b(?:proposed experiment|planned (?:study|trial)|prior work|earlier work)\b|拟议实验|计划的实验|先前研究', re.I)
_CONDITION_REQUEST = re.compile(r'\b(?:conditions?|limitations?|controls?|assumptions?|constraints?)\b|条件|局限|前提|对照|限制', re.I)
_MEASUREMENT_REQUEST = re.compile(r'\b(?:measurements?|measured|scor\w*|metrics?|accuracy|latency|throughput|precision|recall)\b|测量|测法|评分|指标|准确率|延迟|吞吐|召回率', re.I)
_SETUP_ONLY_REQUEST = re.compile(r'\bonly\b.{0,60}\b(?:setup|participants?|sample size)\b|仅.{0,20}(?:设置|样本|参与者)|只.{0,20}(?:设置|样本|参与者)', re.I)
_EXCLUDED_EVALUATION = re.compile(
    r'\b(?:do not|don\x27t|no need to|avoid)\s+(?:explain|describe|discuss|include|cover|show|report)\w*\s+'
    r'(?:(?:the|any)\s+)?(?P<english>experiments?|evaluations?|benchmarks?|measurements?|metrics?|scoring|results?|findings?|outcomes?)\b|'
    r'(?:不要|无需|不必|不需要)(?:解释|说明|讨论|展开|介绍|讲解|讲|包含)?'
    r'(?P<han>实验|评测|评估|测量|测法|评分|指标|结果|结论)', re.I)
_EXCLUDED_METRIC = re.compile(
    r'(?:(?:\b(?:do not|don\x27t|no need to|avoid)\s+'
    r'(?:explain|describe|discuss|include|cover|show|report)\w*\s+|'
    r'\b(?:except|excluding|exclude|not)\s+)(?:(?:the|any)\s+)?|'
    r'(?:不要|无需|不必|不需要|不)(?:解释|说明|讨论|展开|介绍|讲解|讲|包含)?)'
    rf'\s*(?:{_MEASUREMENT.pattern})', re.I)
_ONLY_METRIC = re.compile(
    rf'(?:\b(?:only|exclusively)\b[^.!?;\n]{{0,120}}|(?:只|仅)[^。！？；\n]{{0,45}})'
    rf'(?:{_MEASUREMENT.pattern})', re.I)
_BACKGROUND_SECTION = re.compile(r'\b(?:background|related works?|references)\b|背景|相关工作|参考文献', re.I)
_OVERVIEW_SECTION = re.compile(r'\b(?:abstract|introduction|conclusions?|discussion|background)\b|摘要|引言|结论|讨论|背景', re.I)
_EXPLICIT_COMPARISON = re.compile(
    r'\b(?:outperform\w*|underperform\w*)\s+(?:both\s+)?(?:the\s+)?'
    r'[A-Za-z][^.!?\n]{2,140}\b(?:and|versus)\s+(?:the\s+)?[A-Za-z][^.!?\n]{2,100}|'
    r'(?:优于|劣于).{2,45}(?:和|与).{2,45}', re.I)
_CONDITIONS = {
    'fixed_controls': r'\b(?:requir\w*|under|using|used|with)\b.{0,65}\b(?:fixed|matched|identical)\s+controls?\b|固定对照',
    'fixed_model': r'\b(?:same|fixed|identical)\s+model\b|固定模型|相同模型',
    'fixed_budget': r'\b(?:same|fixed|identical)\s+(?:output|token|time|input)\s+(?:budget|limit|size)\b|固定(?:输出|输入|时间)预算',
    'failure_handling': r'\b(?:unanswered|failed|timeouts?)\b.{0,90}\b(?:count\w*|scor\w*|exclud\w*|denominator|incorrect)\b|未回答.{0,30}(?:分母|错误|计入)',
    'scope_limit': r'\b(?:not tested|not evaluated|does not establish|cannot generaliz\w*|only under|only after|unless)\b|未测试|未评测|不能推广|仅在|除非',
}


def _scoring_views(row, *, split=sentence_spans):
    """Intact local runs, never concatenate across omitted text or paragraphs.

    This affects shape scoring only. Quotes and offsets are never rewritten.
    A URL's full stop cannot complete the preceding unfinished PDF finding.
    """
    text = re.sub(r'\n\s*\d{1,4}\s*$', '', row.get('content', '')).rstrip()
    footnote = re.search(r'(?m)^\s*(?:\d{1,3}\s*)?https?://', text)
    if footnote:
        text = text[:footnote.start()].rstrip()
    views, run = [], None
    for left, right in split(text):
        part = text[left:right]
        valid = bool(re.search(r'[.!?。！？][\]\)"”’]*\s*$', part)) and not _NOT_REPORTED_OBSERVATION.search(part)
        new_paragraph = bool(re.match(r'\s*\n[ \t]*\n', part))
        if run is not None and (not valid or new_paragraph):
            views.append(text[run[0]:run[1]])
            run = None
        if valid:
            run = (left if run is None else run[0], right)
    if run is not None:
        views.append(text[run[0]:run[1]])
    return [view for view in views if not opening_reference_unverified(view)]


def _stable(row):
    return (row.get('filename', ''), row.get('ordinal', 0), row.get('excerpt_start', 0),
            hashlib.sha256(row.get('content', '').encode()).hexdigest())


def _profile(row, *, demand, structural, topic, explicit_background, explicit_planned,
             topic_subjects=None, evaluation_requested=None, evaluation_subjects=(), _parsers=None):
    if row.get('status') in ('revoked', 'deleted', 'invalidated'):
        return {'goals': {}, 'fallback': set(), 'literal_topic': False}
    text = row.get('content', '')
    section = row.get('section', '')
    background = bool(_BACKGROUND_SECTION.search(section))
    split = _parsers.spans if _parsers else sentence_spans
    tokenize = _parsers.tokens if _parsers else matching_tokens
    evaluate = _parsers.evaluation if _parsers else evaluation_cells
    components_for = _parsers.components if _parsers else functional_definitions
    operations_for = _parsers.operations if _parsers else source_operation_qualities
    # Keep whole-source guards (notably an embedded prompt's role header).
    # The shared parser scores each complete action sentence separately; a
    # later paragraph or another operation's condition cannot upgrade it.
    operations = operations_for(row) if structural and not background else {}
    text_tokens = set(tokenize(text))
    goals = {}
    for body in _scoring_views(row, split=split):
        typed = {**row, 'content': body}
        parts = [body[left:right] for left, right in split(body)]
        cells = set(evaluate(typed)) if not background and demand & {'experiment_setup', 'result', 'comparison'} else set()
        if evaluation_requested is not None:
            cells &= evaluation_requested
        # A narrow metric request cannot borrow another task's measurement or
        # its qualifiers just because both occur in the same document/window.
        # Unknown literal subjects remain conservative locators, not inferred
        # synonyms. Only the intact matching sentence contributes cell quality.
        scoped_parts = [(part, set(evaluate({**typed, 'content': part})) & cells)
                        for part in parts
                        if any(subject <= set(tokenize(part)) for subject in evaluation_subjects)] if evaluation_subjects else []
        if evaluation_subjects:
            cells &= set().union(*(found for _, found in scoped_parts))
        components = components_for(typed) if structural and not background else []
        steps = ordered_method_steps(typed) if structural and not background else ()
        roles = _role_strength({**typed, '_evaluation_cells': list(cells)}, steps)
        local = {('evaluation', cell): (max((_evaluation_quality({**typed, 'content': part}, cell)
                    for part, found in scoped_parts if cell in found), default=0.)
                    if evaluation_subjects else _evaluation_quality(typed, cell)) for cell in cells}
        if ('outcome' in cells and not _OVERVIEW_SECTION.search(section)
                and any(_EXPLICIT_COMPARISON.search(part) for part in
                        ([part for part, found in scoped_parts if 'outcome' in found] if evaluation_subjects else parts))):
            # An explicit reported X-versus-Y construction is a comparator
            # locator even without the words "than" or singular "baseline".
            # This does not establish significance, truth or product transfer.
            local['evaluation', 'outcome'] = max(.75, local['evaluation', 'outcome'])
        subjects = topic if topic_subjects is None else topic_subjects
        if not _OVERVIEW_SECTION.search(section):
            for part in ([part for part, found in scoped_parts if found] if evaluation_subjects else parts):
                for cell in set(evaluate({**typed, 'content': part})) & cells:
                    for token in set(tokenize(part)) & subjects:
                        local['evaluation', cell + ':subject:' + token] = local['evaluation', cell]
        if structural and not background and roles['mechanism'] >= .5:
            local['mechanism', 'operation'] = roles['mechanism']
        if operations:
            for stage, quality in operations_for(typed).items():
                if stage in operations:
                    local['mechanism', 'stage:' + stage] = min(quality, operations[stage])
        for component in components:
            local['mechanism', 'component:' + component['key']] = 1.
        if steps:
            local['mechanism', 'ordered:' + '/'.join(steps)] = 1.
        matched_topic = set(tokenize(body)) & topic
        # Never borrow a condition from a separate task/paragraph to complete
        # the measurement or outcome in this one. Max preserves one local cue.
        if not background and (matched_topic or local) and ('limitation' in demand or cells):
            for key, pattern in _CONDITIONS.items():
                if any(re.search(pattern, part, re.I | re.S) for part in
                       ([part for part, found in scoped_parts if found] if evaluation_subjects else parts)):
                    local['condition', key] = 1.
        if local:
            for token in matched_topic:
                local['topic', token] = 1.
        for key, value in local.items():
            goals[key] = max(goals.get(key, 0.), value)
    # This exception describes what was explicitly requested, not actual
    # experimental evidence. It receives its own unverified, one-slot budget.
    fallback = set()
    if text_tokens & topic:
        if explicit_background and (_BACKGROUND_REQUEST.search(text) or background):
            fallback.add('requested_background')
        if explicit_planned and _PLANNED.search(text):
            fallback.add('requested_planned_or_prior_work')
    return {'goals': goals, 'fallback': fallback, 'literal_topic': bool(text_tokens & topic)}


class _RequestProfiles:
    """Bounded memo for one supplement call, never a user/global LLM cache.

    Text and source-local heading/outline are part of the key. Metadata IDs do
    not imply equivalent evidence. All values are private read-only analysis;
    original quotes, offsets and selected-source projections stay untouched.
    Eviction may recompute a shape; it never filters a candidate.
    """
    def __init__(self, settings):
        self.spans = lru_cache(maxsize=MAX_PROFILE_MEMO)(sentence_spans)
        self.tokens = lru_cache(maxsize=MAX_PROFILE_MEMO)(matching_tokens)

        @lru_cache(maxsize=MAX_PROFILE_MEMO)
        def evaluation(text, section, outline):
            return evaluation_cells({'content': text, 'section': section, 'outline_roles': outline})

        @lru_cache(maxsize=MAX_PROFILE_MEMO)
        def components(text, section):
            return functional_definitions({'content': text, 'section': section})

        @lru_cache(maxsize=MAX_PROFILE_MEMO)
        def operations(text, section, outline):
            return source_operation_qualities({'content': text, 'section': section, 'outline_roles': outline})

        @lru_cache(maxsize=MAX_PROFILE_MEMO)
        def profile(text, section, outline):
            return _profile({'content': text, 'section': section, 'outline_roles': outline},
                            **settings, _parsers=self)

        self._evaluation_cached = evaluation
        self._components_cached = components
        self._operations_cached = operations
        self._profile_cached = profile

    def profile(self, row):
        if row.get('status') in ('revoked', 'deleted', 'invalidated'):
            return {'goals': {}, 'fallback': set(), 'literal_topic': False}
        return self._profile_cached(row.get('content', ''), row.get('section', ''),
                                    tuple(row.get('outline_roles', ())))

    def evaluation(self, row):
        return self._evaluation_cached(row.get('content', ''), row.get('section', ''),
                                       tuple(row.get('outline_roles', ())))

    def components(self, row):
        return self._components_cached(row.get('content', ''), row.get('section', ''))

    def operations(self, row):
        if row.get('status') in ('revoked', 'deleted', 'invalidated'):
            return {}
        return self._operations_cached(row.get('content', ''), row.get('section', ''),
                                       tuple(row.get('outline_roles', ())))


def supplement_evidence(question, sources, chunks, proposal, *, append_sources):
    """Select up to two genuinely incremental grammatical/source-local cues.

    Finite literal coverage is an incomplete proxy; no semantic completeness,
    model call, source rewrite, exhaustive scan, or automatic experiment occurs.
    """
    actions = [item for item in proposal.get('actions', [])
               if item.get('kind', item.get('type')) == 'supplemental_query'][:2]
    receipt = {'status': 'skipped', 'reason': 'no_actions', 'retrieval_passes': 0,
               'model_calls': 0, 'added_source_ids': [], 'action_ids': [],
               'candidate_count': 0, 'source_bytes': 0, 'scanned_chunks': 0,
               'scan_limited': False, 'candidate_limit': MAX_CANDIDATES,
               'candidate_pool_count': 0, 'rejection_counts': {},
               'selection_is_semantic_verification': False,
               'required_coarse_goal_priority': False,
               'requested_operation_cells': 0,
               'operation_quality_is_semantic_support': False,
               'literal_metric_scope_status': 'not_applied',
               'goal_cell_scope': 'retained_supplement_packet_before_prompt_fit',
               'coverage_is_complete': False, 'algorithm': 'source_local_incremental_supplement_v3'}
    if not actions:
        return {'sources': sources, 'receipt': receipt}
    remaining = min(MAX_ADDED_BYTES, MAX_SOURCE_BYTES - sum(len(row['content'].encode()) for row in sources))
    limit = min(MAX_ADDED, MAX_ANSWER_SOURCES - len(sources))
    if remaining <= 0:
        return {'sources': sources, 'receipt': {**receipt, 'reason': 'evidence_budget'}}
    if limit <= 0:
        return {'sources': sources, 'receipt': {**receipt, 'reason': 'addition_limit'}}
    receipt['action_ids'] = [item['id'] for item in actions]
    dimensions = [term for item in actions for term in item.get('payload', {}).get('dimension_terms', [])
                  if isinstance(term, str) and len(term.encode()) <= 64][:32]
    if not dimensions:
        return {'sources': sources, 'receipt': {**receipt, 'reason': 'no_dimension_terms'}}
    documents = {row['document_id'] for row in sources}
    source_keys = {(row['document_id'], row['id']) for row in sources}
    source_ids = {row['id'] for row in sources}
    normalized_texts = {' '.join(row['content'].split()) for row in sources}
    # Ordering inspects metadata only; never hash/tokenize every chunk before
    # the advertised text-scan bound. Window tie-breaking is content-stable.
    scan = sorted(chunks, key=lambda row: (row['document_id'] not in documents,
        row.get('filename', ''), row.get('ordinal', 0), row['document_id'], row['id']))[:MAX_SCAN]
    receipt.update(scanned_chunks=len(scan), scan_limited=len(chunks) > MAX_SCAN, retrieval_passes=1)
    facets = build_query_facets(question, scan)
    demand = {role for facet in facets['facets'] for role in facet.get('roles', ())}
    # This is only the positive metric-locator view. The real question/facets
    # remain untouched, including their exclusions. No translated aliases or
    # synonyms are generated when an exact original-language match is absent.
    metric_question = _EXCLUDED_METRIC.sub(' ', question)
    explicit_metric_scope = bool(_ONLY_METRIC.search(metric_question))
    # Comparison alone, or a card's proposed dimensions, creates no obligation
    # to find an experiment. Required cells derive only from the user's words.
    evaluation_requested = ({'setup', 'measurement', 'outcome'} if 'experiment_setup' in demand else set())
    if 'result' in demand:
        evaluation_requested.add('outcome')
    if _MEASUREMENT_REQUEST.search(metric_question):
        evaluation_requested.add('measurement')
    if _SETUP_ONLY_REQUEST.search(question):
        evaluation_requested = {'setup'}
    for excluded in _EXCLUDED_EVALUATION.finditer(question):
        noun = (excluded['english'] or excluded['han']).casefold()
        if re.match(r'experiment|evaluation|benchmark|实验|评测|评估', noun):
            evaluation_requested.clear()
        elif re.match(r'measurement|metric|scoring|测量|测法|评分|指标', noun):
            evaluation_requested.discard('measurement')
        else:
            evaluation_requested.discard('outcome')
    if evaluation_requested:
        demand.add('experiment_setup')
    condition_requested = bool(_CONDITION_REQUEST.search(question))
    if condition_requested:
        demand.add('limitation')
    # Explicit operations and metric-only exclusions use the same shared
    # bounded request scope as primary retrieval, not a second local grammar.
    requested_stages = requested_operation_stages(question)
    structural = bool(requested_stages or requests_component_explanation(question) or 'mechanism' in demand)
    structural = structural and not requests_measurement_without_structure(question)
    if structural:
        demand.add('mechanism')
    explicit_operations = bool(structural and requested_stages)
    operation_documents = operation_stages_by_document(question, scan) if explicit_operations else {}
    receipt['requested_operation_cells'] = sum(map(len, operation_documents.values()))
    named_documents = {doc for facet in facets['facets'] for doc in facet.get('document_ids', ())}
    # A literal user subject remains a subject even if a card also mentions it;
    # generic method dimensions are removed by the fixed framing vocabulary.
    topic = set(matching_tokens(metric_question)) - _GENERIC
    topic = {token for token in topic if len(token) >= (3 if token.isascii() else 2)}
    document_terms = {token for facet in facets['facets'] if facet.get('kind') == 'document'
                      for token in matching_tokens(facet['text'])}
    evaluation_subjects = ()
    if (evaluation_requested and _MEASUREMENT_REQUEST.search(metric_question) and not structural
            and (explicit_metric_scope or not any('experiment_setup' in facet.get('roles', ()) for facet in facets['facets']))):
        metric_terms = set(matching_tokens(' '.join(match.group() for match in _MEASUREMENT.finditer(metric_question)))) - _GENERIC
        # Known physical/statistical metric names form alternatives (e.g.
        # accuracy AND latency each needs its own sentence). An unfamiliar
        # literal subject uses all its remaining terms, not a guessed synonym.
        evaluation_subjects = (tuple(frozenset({term}) for term in sorted(metric_terms)) if metric_terms
                               else (frozenset(topic - document_terms),) if topic - document_terms else ())
    if evaluation_subjects:
        receipt['literal_metric_scope_status'] = 'limited_unverified'
    required = {('evaluation', cell) for cell in evaluation_requested}
    if structural:
        required.add(('mechanism', 'operation'))
    if condition_requested:
        required.add(('condition', 'constraint'))
    receipt['required_coarse_goal_priority'] = bool(required)
    settings = dict(demand=demand, structural=structural, topic=topic,
                    topic_subjects=topic - document_terms,
                    evaluation_requested=evaluation_requested, evaluation_subjects=evaluation_subjects,
                    explicit_background=bool(_BACKGROUND_REQUEST.search(question)),
                    explicit_planned=bool(_PLANNED_REQUEST.search(question)))
    profiles = _RequestProfiles(settings)

    def scoped_profile(row):
        profile = profiles.profile(row)
        if not explicit_operations:
            return profile
        wanted = operation_documents.get(row['document_id'], ())
        stages = {key[1][6:] for key in profile['goals']
                  if key[0] == 'mechanism' and key[1].startswith('stage:')}
        # A known document cannot inherit an unresolved or explicitly
        # different document's operation request. Nor may an excluded action
        # re-enter under its generic method/component description.
        exclude_method = not wanted or (stages and not stages.intersection(wanted))
        goals = {key: value for key, value in profile['goals'].items()
                 if key[0] != 'mechanism' or (not exclude_method and
                    (not key[1].startswith('stage:') or key[1][6:] in wanted))}
        return {**profile, 'goals': goals}

    def operation_nomination(row):
        # The bounded pool needs quality as well as stage names; converting
        # this mapping to a tuple would erase detail before nomination.
        return {stage: quality for stage, quality in profiles.operations(row).items()
                if not explicit_operations or stage in operation_documents.get(row['document_id'], ())}

    # A schema action authorizes this bounded nomination pass. Its dimensions
    # never establish a missing fact or enter the incremental evidence profile.
    pool = prepare_evidence_candidates(question, scan, sources, facets, {},
        max_candidates=MAX_CANDIDATES, matching_tokens=profiles.tokens,
        literal_requested_roles=demand,
        component_profile=profiles.components if structural else None,
        evaluation_profile=profiles.evaluation if demand & {'experiment_setup', 'result'} else None,
        operation_profile=operation_nomination if structural else None,
        # This callback receives text/section only, not a document identity.
        # It is a shape prior; operation nomination and admission below apply
        # the real document-bound request rather than guessing a missing ID.
        nomination_prior=lambda row: .8 if profiles.profile(row)['goals'] else 0.)
    receipt['candidate_pool_count'] = len(pool['candidates'])
    rejected = Counter()
    candidates = []
    for row in pool['candidates']:
        if row['source_id'] in source_ids:
            rejected['already_selected_source'] += 1
            continue
        if ' '.join(row['content'].split()) in normalized_texts:
            rejected['duplicate_original_text'] += 1
            continue
        if source_filter_reason(question, row['content'], section=row.get('section', '')):
            rejected['source_hygiene'] += 1
            continue
        if opening_reference_unverified(row['content']):
            rejected['source_scope_unverified'] += 1
            continue
        profile = scoped_profile(row)
        # A named/current document can supply an unlabelled local method or
        # measured result; unknown fallback requires the literal topic itself.
        if not profile['literal_topic'] and row['document_id'] not in documents | named_documents:
            rejected['no_literal_topic_or_document_scope'] += 1
            continue
        if not profile['goals'] and not profile['fallback']:
            rejected['no_source_local_goal'] += 1
            continue
        candidates.append((row, profile))
    receipt['candidate_count'] = len(candidates)
    coverage, coarse_coverage, fallback_coverage = {}, {}, set()

    def coarse(profile, doc):
        needed = required | {('mechanism', 'stage:' + stage)
                             for stage in operation_documents.get(doc, ())}
        values = {key: value for key, value in profile['goals'].items()
                  if key in needed}
        if ('condition', 'constraint') in required:
            values['condition', 'constraint'] = max((value for (family, _), value in profile['goals'].items()
                                                     if family == 'condition'), default=0.)
        return values

    def record(row, profile):
        doc = row['document_id']
        for key, value in profile['goals'].items():
            coverage[doc, key] = max(coverage.get((doc, key), 0.), value)
        for key, value in coarse(profile, doc).items():
            coarse_coverage[doc, key] = max(coarse_coverage.get((doc, key), 0.), value)
        fallback_coverage.update((doc, item) for item in profile['fallback'])

    def priority(row, profile):
        doc = row['document_id']
        gains = Counter()
        for goal, value in profile['goals'].items():
            gains[goal[0]] += max(0., value - coverage.get((doc, goal), 0.))
        core = min(1., gains['evaluation']) + min(1., gains['mechanism']) + .5 * min(1., gains['condition'])
        detail = core + (.25 * min(1., gains['topic']) if core > 0 else 0.)
        values = coarse(profile, doc)
        # Requested operation cells share the original mechanism family's
        # single coarse vote/budget with its generic cell. Adding three names
        # must not create three extra votes against measurement or outcomes.
        mechanism_cells = 1 + len(operation_documents.get(doc, ()))
        weight = lambda key: 1. / mechanism_cells if key[0] == 'mechanism' else 1.
        missing = sum(weight(key) for key, value in values.items()
                      if value > 0 and coarse_coverage.get((doc, key), 0.) <= 0)
        quality = sum(weight(key) * max(0., value - coarse_coverage.get((doc, key), 0.))
                      for key, value in values.items())
        fallback = detail <= 0 and not fallback_used and any((doc, item) not in fallback_coverage for item in profile['fallback'])
        # Lexicographic, not an opaque extra weight: first expose a requested
        # missing cell; then improve those required cells; only then add detail
        # to already represented families. Counts never certify completeness.
        return (missing, quality, detail if not fallback else .15), fallback

    # Section hints may be absent from the persisted projection. Recover only
    # an original heading at/before the exact span, never a different quote.
    raw_by_key = {(row['document_id'], row['id']): row for row in scan}
    abstract_starts = {(hint['document_id'], hint['source_id']): hint['abstract_body_start']
                       for hint in initial_title_hints(scan)}
    baseline_profiles = []
    for row in sources:
        raw = raw_by_key.get((row['document_id'], row['id']), {})
        whole = raw.get('content', '')
        start = row.get('excerpt_start', 0)
        headings = [(a, b) for a, b in section_heading_spans(whole) if a <= start]
        hint = whole[headings[-1][0]:headings[-1][1]] if headings else row.get('section', '')
        abstract_start = abstract_starts.get((row['document_id'], row['id']))
        if (abstract_start is not None and start >= abstract_start
                and not any(a >= abstract_start for a, _ in headings)):
            # A title/authors can be the last recognized heading when the
            # plain Abstract label follows an unpunctuated email. The bounded
            # reliable title locator recovers its role without changing quotes.
            hint = 'Abstract'
        profile = scoped_profile({**row, 'section': hint})
        baseline_profiles.append((row, profile))
        record(row, profile)
    initial_cells = len(coverage)
    extras, selected_keys = [], set(source_keys)
    chosen_profiles, accepted_fallback_keys = {}, set()
    fallback_used = False
    for _ in range(MAX_CANDIDATES):
        if len(extras) >= limit:
            break
        ranked = []
        for row, profile in candidates:
            key = row['document_id'], row['source_id']
            if key in selected_keys or ' '.join(row['content'].split()) in normalized_texts:
                continue
            size = len(row['content'].encode())
            if size > remaining:
                continue
            score, fallback = priority(row, profile)
            if score[2] <= 0:
                continue
            ranked.append((score, row, profile, fallback))
        if not ranked:
            break
        ranked.sort(key=lambda item: (*(-value for value in item[0]), -item[1].get('pool_score', 0.), *_stable(item[1])))
        _, row, profile, fallback = ranked[0]
        extra = {key: row[key] for key in (
            'document_id', 'filename', 'content', 'ordinal', 'page', 'excerpt_start',
            'excerpt_end', 'offset_unit', 'original_content_length',
            'sentence_boundary_preserved', 'truncated', 'evidence_scope') if key in row}
        extra['id'] = row['source_id']
        extra['sentence_boundary_preserved'] = (row.get('sentence_boundary_preserved', True)
                                               and _terminal_boundary_observed(row['content']))
        key = row['document_id'], row['source_id']
        selected_keys.add(key)
        accepted = append_sources([*sources, *extras], [extra])
        if len(accepted) != len(sources) + len(extras) + 1:
            rejected['append_contract_rejected'] += 1
            continue
        extra = accepted[-1]
        extras.append(extra)
        normalized_texts.add(' '.join(extra['content'].split()))
        chosen_profiles[key] = profile
        if fallback:
            accepted_fallback_keys.add(key)
        remaining -= len(row['content'].encode())
        fallback_used |= fallback
        record(row, profile)
    selected = append_sources(sources, extras)
    added = selected[len(sources):]
    added_keys = {(row['document_id'], row['id']) for row in added}
    # The public receipt is based on retained evidence, never on nominations
    # that the append contract may reject. Rebuild from the immutable baseline.
    coverage.clear()
    coarse_coverage.clear()
    fallback_coverage.clear()
    for row, profile in baseline_profiles:
        record(row, profile)
    for row in added:
        record(row, chosen_profiles[row['document_id'], row['id']])
    fallback_used = bool(accepted_fallback_keys & added_keys)
    for row, profile in candidates:
        key = row['document_id'], row['source_id']
        if key in added_keys:
            continue
        if key in selected_keys:
            rejected['same_source_alternative'] += 1
        elif priority(row, profile)[0][2] <= 0:
            rejected['no_remaining_incremental_goal'] += 1
        elif len(row['content'].encode()) > remaining:
            rejected['evidence_budget'] += 1
        elif len(added) >= limit:
            rejected['addition_limit'] += 1
        else:
            rejected['append_contract_rejected'] += 1
    receipt.update(status='expanded' if added else 'unchanged',
                   reason='incremental_source_local_coverage' if added else 'no_incremental_evidence',
                   added_source_ids=[row['id'] for row in added],
                   source_bytes=sum(len(row['content'].encode()) for row in added),
                   rejection_counts=dict(sorted(rejected.items())),
                   initial_goal_cells=initial_cells, final_goal_cells=len(coverage),
                   unverified_fallback_used=fallback_used)
    return {'sources': selected, 'receipt': receipt}
