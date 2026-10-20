"""Bounded paired/three-level designs and relative arithmetic, not experiments.

This pure compiler never invents samples, observations, scales or thresholds.
Source binding proves current coordinates, not the truth of a measurement or
that an operator description actually isolates its declared single factor.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math

VERSION = 'paired-validation-design-v1'
MAX_DESIGN_BYTES = 32768
MAX_SAMPLES = 32
MAX_SOURCES = 128
MAX_SOURCE_BYTES = 1048576
LEVELS = ('baseline', 'intermediate', 'high')


class DesignError(ValueError):
    """Fixed safe error codes never echo source or operator contents."""


def _fail(code):
    raise DesignError(code)


def _plain(value):
    stack = [(value, 0)]
    nodes = 0
    while stack:
        part, depth = stack.pop()
        nodes += 1
        if nodes > 4096 or depth > 12:
            _fail('design_limit')
        if type(part) is dict:
            if any(type(key) is not str for key in part):
                _fail('schema_invalid')
            stack.extend((item, depth + 1) for pair in part.items() for item in pair)
        elif type(part) is list:
            stack.extend((item, depth + 1) for item in part)
        elif type(part) is str:
            if len(part) > MAX_DESIGN_BYTES:
                _fail('design_limit')
        elif type(part) is int:
            if part.bit_length() > 128:
                _fail('design_limit')
        elif type(part) is float:
            if not math.isfinite(part):
                _fail('nonfinite_value')
        elif part is not None and type(part) is not bool:
            _fail('schema_invalid')
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (ValueError, TypeError, UnicodeError, RecursionError):
        _fail('schema_invalid')
    if len(raw) > MAX_DESIGN_BYTES:
        _fail('design_limit')
    return json.loads(raw)


def _keys(value, required):
    if type(value) is not dict or set(value) != set(required):
        _fail('schema_invalid')


def _text(value, limit=1024):
    if type(value) is not str or not value.strip() or len(value) > limit:
        _fail('text_invalid')
    return value


def _number(value, *, integer=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or type(value) is int and value.bit_length() > 128
            or type(value) is float and not math.isfinite(value)):
        _fail('observation_value_invalid')
    return Fraction(str(value))


def _sources(sources):
    if type(sources) is not list or not 1 <= len(sources) <= MAX_SOURCES:
        _fail('sources_invalid')
    indexed, labels, size = {}, set(), 0
    for source in sources:
        if type(source) is not dict or not {'source_id', 'label', 'content'} <= set(source) or set(source) - {'source_id', 'label', 'content', 'revoked', 'kind'}:
            _fail('sources_invalid')
        identifier, label, content = source['source_id'], source['label'], source['content']
        _text(identifier, 256)
        _text(label, 64)
        if type(content) is not str or len(content) > MAX_SOURCE_BYTES or type(source.get('revoked', False)) is not bool:
            _fail('sources_invalid')
        if source.get('kind', 'material') not in ('material', 'observation_record'):
            _fail('sources_invalid')
        try:
            size += len(content.encode())
        except UnicodeError:
            _fail('sources_invalid')
        if size > MAX_SOURCE_BYTES or identifier in indexed or label in labels:
            _fail('sources_invalid')
        indexed[identifier] = source
        labels.add(label)
    return indexed


def _ref(ref, sources):
    _keys(ref, ('source_id', 'label', 'span', 'sha256'))
    source = sources.get(ref['source_id']) if type(ref['source_id']) is str else None
    if source is None or source.get('revoked', False):
        _fail('source_unavailable')
    if ref['label'] != source['label']:
        _fail('source_label_mismatch')
    span = ref['span']
    if type(span) is not list or len(span) != 2 or any(type(i) is not int for i in span):
        _fail('source_span_invalid')
    start, end = span
    if not 0 <= start < end <= len(source['content']) or end - start > 4096:
        _fail('source_span_invalid')
    quote = source['content'][start:end]
    digest = hashlib.sha256(quote.encode()).hexdigest()
    if type(ref['sha256']) is not str or ref['sha256'] != digest:
        _fail('source_hash_mismatch')
    return quote


def _compile(spec, sources, trusted_budget_minutes):
    spec = _plain(spec)
    _keys(spec, ('design_type', 'target', 'measurement', 'factor', 'levels', 'fixed_conditions', 'samples',
                 'prediction', 'falsifier', 'missing_policy', 'activity_minutes'))
    indexed = _sources(sources)
    _keys(spec['target'], ('id', 'description'))
    _text(spec['target']['id'], 128)
    _text(spec['target']['description'])
    measurement = spec['measurement']
    _keys(measurement, ('id', 'target_id', 'quantity', 'kind', 'unit', 'method', 'observation_source', 'ordered_values'))
    for key in ('id', 'target_id', 'quantity', 'unit', 'method', 'observation_source'):
        _text(measurement[key])
    if measurement['target_id'] != spec['target']['id']:
        _fail('measurement_target_mismatch')
    if measurement['kind'] not in ('count', 'number', 'ordinal', 'qualitative'):
        _fail('measurement_kind_invalid')
    scale = measurement['ordered_values']
    if measurement['kind'] == 'ordinal':
        if type(scale) is not list or not 2 <= len(scale) <= 16 or any(type(v) is not str or not v.strip() or len(v) > 128 for v in scale) or len(set(scale)) != len(scale):
            _fail('ordinal_scale_invalid')
    elif scale is not None:
        _fail('invented_scale')
    _keys(spec['factor'], ('id', 'description'))
    factor = _text(spec['factor']['id'], 128)
    _text(spec['factor']['description'])
    fixed = spec['fixed_conditions']
    if type(fixed) is not dict or not fixed or len(fixed) > 16 or factor in fixed:
        _fail('fixed_conditions_invalid')
    for key, value in fixed.items():
        _text(key, 128)
        _text(value, 512)
    levels = spec['levels']
    if spec['design_type'] not in ('paired', 'three_level'):
        _fail('design_type_invalid')
    expected = LEVELS if spec['design_type'] == 'three_level' else ('baseline', 'high')
    if type(levels) is not list or len(levels) != len(expected):
        _fail('levels_incomplete')
    values = []
    for level, identifier in zip(levels, expected):
        _keys(level, ('id', 'operation', 'settings'))
        if level['id'] != identifier:
            _fail('levels_incomplete')
        _text(level['operation'])
        settings = level['settings']
        if type(settings) is not dict or set(settings) != set(fixed) | {factor}:
            _fail('confounded_factors')
        if any(type(settings[key]) is not str or settings[key] != value for key, value in fixed.items()):
            _fail('confounded_factors')
        values.append(_text(settings[factor], 512))
    if len(set(values)) != len(values):
        _fail('factor_levels_not_distinct')
    samples = spec['samples']
    if type(samples) is not list or not 1 <= len(samples) <= MAX_SAMPLES:
        _fail('samples_invalid')
    identities, refs = set(), set()
    for sample in samples:
        _keys(sample, ('id', 'task', 'source'))
        identifier = _text(sample['id'], 128)
        _text(sample['task'])
        _ref(sample['source'], indexed)
        ref_key = json.dumps(sample['source'], sort_keys=True)
        if identifier in identities or ref_key in refs:
            _fail('duplicate_sample')
        identities.add(identifier)
        refs.add(ref_key)
    _keys(spec['prediction'], ('pairs', 'description'))
    prediction = spec['prediction']
    pairs = prediction['pairs']
    if type(pairs) is not list or not len(expected) - 1 <= len(pairs) <= 3:
        _fail('prediction_invalid')
    covered, comparisons, order = set(), set(), {level: set() for level in expected}
    for pair in pairs:
        _keys(pair, ('from', 'to', 'direction'))
        left, right, direction = pair['from'], pair['to'], pair['direction']
        if (type(left) is not str or type(right) is not str or left not in expected or right not in expected
                or left == right or direction not in ('lower', 'higher')):
            _fail('prediction_invalid')
        identity = tuple(sorted((left, right)))
        if identity in comparisons:
            _fail('prediction_duplicate_pair')
        comparisons.add(identity)
        covered.update((left, right))
        small, large = (right, left) if direction == 'lower' else (left, right)
        order[small].add(large)
    if covered != set(expected):
        _fail('prediction_levels_incomplete')
    for level in expected:
        pending, seen = list(order[level]), set()
        while pending:
            descendant = pending.pop()
            if descendant == level:
                _fail('prediction_order_conflict')
            if descendant not in seen:
                seen.add(descendant)
                pending.extend(order[descendant])
    _text(prediction['description'])
    _text(spec['falsifier'])
    if spec['missing_policy'] != 'retain_inconclusive':
        _fail('missing_policy_invalid')
    _keys(spec['activity_minutes'], ('setup', 'per_sample_level', 'review'))
    activity = spec['activity_minutes']
    for value in activity.values():
        if _number(value) < 0:
            _fail('activity_budget_invalid')
    if _number(trusted_budget_minutes) < 0:
        _fail('activity_budget_invalid')
    total = _number(activity['setup']) + len(samples) * len(levels) * _number(activity['per_sample_level']) + _number(activity['review'])
    if total > _number(trusted_budget_minutes):
        _fail('activity_budget_exceeded')
    return spec, pairs, float(total)


def compile_design(spec, sources, *, trusted_budget_minutes):
    """Validate a draft; compiled math is not source-truth/scientific approval."""
    try:
        design, pairs, minutes = _compile(spec, sources, trusted_budget_minutes)
    except DesignError as error:
        return {'version': VERSION, 'status': 'draft', 'violations': [str(error)], 'plan': None,
                'execution_authorized': False, 'measurement_verified': False}
    return {'version': VERSION, 'status': 'review_required', 'violations': [],
            'plan': design, 'comparisons': pairs,
            'relative_support_rule': 'every_fixed_sample_every_declared_pair_strictly_matches_direction',
            'relative_refute_rule': 'any_complete_pair_does_not_match_its_declared_direction',
            'activity_minutes': minutes, 'execution_authorized': False,
            'measurement_verified': False, 'semantic_verified': False, 'observations_present': False}


def evaluate_design(plan, observations, sources, *, trusted_budget_minutes):
    """Compare every fixed sample; unavailable cells stay visible, never dropped.

    Observations are caller-supplied records, each tied to an actual live source
    span. Their meaning/value extraction is not independently verified here.
    """
    compiled = compile_design(plan, sources, trusted_budget_minutes=trusted_budget_minutes)
    if compiled['plan'] is None:
        return {**compiled, 'verdict': 'inconclusive', 'samples': []}
    try:
        observations = _plain(observations)
        if type(observations) is not list or len(observations) > MAX_SAMPLES * 3:
            _fail('observations_invalid')
        indexed = _sources(sources)
        design = compiled['plan']
        samples = {row['id']: row for row in design['samples']}
        levels = [row['id'] for row in design['levels']]
        measurement = design['measurement']
        cells = {}
        for record in observations:
            _keys(record, ('sample_id', 'level', 'target_id', 'measurement_id', 'value', 'source'))
            sample, level = record['sample_id'], record['level']
            if type(sample) is not str or sample not in samples or type(level) is not str or level not in levels:
                _fail('unpaired_sample_or_level')
            if record['target_id'] != design['target']['id'] or record['measurement_id'] != measurement['id']:
                _fail('measurement_target_mismatch')
            _ref(record['source'], indexed)
            if indexed[record['source']['source_id']].get('kind', 'material') != 'observation_record':
                _fail('observation_source_is_not_a_record')
            key = (sample, level)
            if key in cells:
                _fail('duplicate_observation')
            value = record['value']
            if value is not None:
                if measurement['kind'] in ('number', 'count'):
                    number = _number(value, integer=measurement['kind'] == 'count')
                    if measurement['kind'] == 'count' and number < 0:
                        _fail('observation_value_invalid')
                elif measurement['kind'] == 'ordinal':
                    if type(value) is not str or value not in measurement['ordered_values']:
                        _fail('observation_value_invalid')
                else:
                    _text(value)
            cells[key] = record
        results, any_refute, any_missing = [], False, False
        for sample in samples:
            complete = all((sample, level) in cells and cells[sample, level]['value'] is not None for level in levels)
            comparisons = []
            for pair in compiled['comparisons']:
                left, right = pair['from'], pair['to']
                first, second = cells.get((sample, left)), cells.get((sample, right))
                outcome = 'inconclusive'
                if first and second and first['value'] is not None and second['value'] is not None and measurement['kind'] != 'qualitative':
                    if measurement['kind'] == 'ordinal':
                        a, b = (measurement['ordered_values'].index(item['value']) for item in (first, second))
                    else:
                        a, b = _number(first['value']), _number(second['value'])
                    supports = b < a if pair['direction'] == 'lower' else b > a
                    outcome = 'support' if supports else 'refute'
                comparisons.append({'from': left, 'to': right, 'verdict': outcome})
            verdict = ('refute' if any(row['verdict'] == 'refute' for row in comparisons) else
                       'support' if complete and comparisons and all(row['verdict'] == 'support' for row in comparisons) else 'inconclusive')
            any_refute |= verdict == 'refute'
            any_missing |= verdict == 'inconclusive'
            results.append({'sample_id': sample, 'source': samples[sample]['source'],
                            'levels': {level: cells.get((sample, level)) for level in levels},
                            'complete': complete, 'comparisons': comparisons, 'verdict': verdict})
        verdict = 'refute' if any_refute else 'inconclusive' if any_missing else 'support'
        return {**compiled, 'status': 'observations_compared', 'observations_present': bool(observations),
                'verdict': verdict, 'samples': results, 'scope': 'declared_paired_observations_only_not_causal_proof'}
    except DesignError as error:
        return {**compiled, 'status': 'draft', 'violations': [str(error)], 'verdict': 'inconclusive', 'samples': []}
