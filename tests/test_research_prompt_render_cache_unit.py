"""Full formatter differential checks; all material is invented and local."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from docthinker.server import research_prompt_budget as current
from docthinker.server.insight_runtime import InsightRuntime
from docthinker.server.knowledge_skills import retrieve_knowledge_skill
from docthinker.server.research_answer_contract import SYSTEM_POLICY, knowledge_skill_system_policy
from docthinker.server.research_retrieval import build_retrieval_plan
from docthinker.server.research_trial_delivery import trial_contract_packet
from docthinker.server.research_retrieval import lexical_tokens
from docthinker.server.research_surface_coverage import surface_term_coverage


REFERENCE = Path(__file__).parent / 'fixtures' / 'reference_research_prompt_20261013.py'
SPEC = importlib.util.spec_from_file_location('frozen_prompt_formatter_20261013', REFERENCE)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)
QUESTION = 'Compare archive memory architectures and their validation conditions.'
LEDGER_GAPS = {'final_need_witness_omitted', 'final_need_witness_unmatched'}
HISTORY_MARKER = '\n近期对话（可能截短；不是原文证据，只作指代上下文）：\n'


def _without_added_metadata(result):
    """Only explicit schema additions; never normalize prompt/message bytes."""
    result = deepcopy(result)
    if result[0] == 'ok':
        packet = result[1]
        packet.pop('need_ledger')
        packet.pop('history_included')
        packet['reasoning']['trace'].pop('final_need_ledger')
        for value in (packet['plan'], packet['reasoning']):
            value['gaps'] = [row for row in value['gaps'] if row.get('code') not in LEDGER_GAPS]
    return result


def _assert_current_packet(packet, value, support, history):
    """Independently check new bookkeeping against the actual dispatch bytes."""
    messages, budget = packet['messages'], packet['budget']
    assert [row['role'] for row in messages] == ['system', 'user']
    assert packet['prompt'] == messages[1]['content']
    assert budget == packet['reasoning']['trace']['context_budget']
    measured = sum(len(row['role'].encode()) + len(row['content'].encode()) for row in messages)
    assert measured == budget['input_bytes'] <= budget['max_input_bytes'] == 24576
    canonical = lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    contract = packet['reasoning']['trace']['answer_contract']
    assert contract['messages_sha256'] == hashlib.sha256(canonical(messages)).hexdigest()
    assert contract['policy_sha256'] == hashlib.sha256(messages[0]['content'].encode()).hexdigest()
    included = packet['history_included']
    assert type(included) is bool
    assert included == (budget['history_bytes'] > 0) == (HISTORY_MARKER in packet['prompt'])
    if included:
        rendered_history = packet['prompt'].split(HISTORY_MARKER, 1)[1]
        decoded, _ = json.JSONDecoder().raw_decode(rendered_history)
        assert decoded == history
        assert budget['history_bytes'] == len(ORACLE._json(decoded).encode())
    originals = value['sources'] + support['sources']
    for source in packet['plan']['sources']:
        assert source in originals  # Every field, whole text and geometry preserved.
        assert source['excerpt_end'] - source['excerpt_start'] == len(source['content'])
    ledger, receipt = packet['need_ledger'], packet['reasoning']['trace']['final_need_ledger']
    assert receipt['ledger_sha256'] == hashlib.sha256(canonical(ledger)).hexdigest()
    assert receipt['semantic_verified'] is False and receipt['whole_document_conditions_verified'] is False
    sources = packet['plan']['sources']
    assert receipt['retained_window_count'] == len(sources)
    assert receipt['invalid_retained_window_count'] == 0
    for source in sources:
        digest = hashlib.sha256(source['content'].encode()).hexdigest()
        matches = [row for row in ledger['bundles'] if row['source_id'] == source['id']
                   and row['document_id'] == source['document_id'] and row['label'] == source['label']
                   and row['span'] == [source['excerpt_start'], source['excerpt_end']]
                   and row['sha256'] == digest]
        assert len(matches) == 1 and matches[0]['retained_exactly'] is True


def _literal_coverage(sources):
    tokens = set(term for row in sources for term in lexical_tokens(row['content']))
    return set(surface_term_coverage(lexical_tokens(QUESTION), tokens)['covered'])


def assert_formatter_compat(actual, expected, value, support, history, options):
    if actual[0] != 'ok' or expected[0] != 'ok':
        assert actual == expected  # Keep error type and fixed diagnostics too.
        return
    _assert_current_packet(actual[1], value, support, history)
    # The frozen oracle predates these explicit non-wire return fields.
    old = deepcopy(expected)
    old[1]['need_ledger'] = None
    old[1]['history_included'] = None
    old[1]['reasoning']['trace']['final_need_ledger'] = None
    if actual[1]['plan']['sources'] == expected[1]['plan']['sources']:
        assert _without_added_metadata(actual) == _without_added_metadata(old)
        return
    # New fallback may choose a different WHOLE existing window. It cannot
    # trade away a literal need already covered by the frozen selection.
    assert _literal_coverage(expected[1]['plan']['sources']) <= _literal_coverage(actual[1]['plan']['sources'])
    fixed = deepcopy(actual[1]['plan'])
    fixed['gaps'] = [row for row in fixed['gaps'] if row.get('code') not in LEDGER_GAPS]
    kept_history = history if actual[1]['history_included'] else []
    kept_support = (support if actual[1]['budget']['assist_bytes'] else
                    ORACLE.empty_assistance(fixed['sources'], reason='prompt_budget'))
    kept_support = {**deepcopy(kept_support), 'sources': fixed['sources']}
    fixed_options = {**options, 'knowledge_support': actual[1]['knowledge_support']
                     if actual[1]['knowledge_skill_included'] else None}
    rebuilt = run(ORACLE.fit_research_prompt, fixed, kept_support, kept_history, fixed_options)
    assert rebuilt[0] == 'ok'
    # Re-render the precise retained packet with the OLD formatter. Hashes,
    # policy, wrappers and budgets remain byte-exact, not merely <= a cap.
    for key in ('messages', 'prompt', 'budget'):
        assert actual[1][key] == rebuilt[1][key]
    reference = deepcopy(rebuilt[1]['reasoning']['trace']['answer_contract'])
    reference['source_count_before_fit'] = len(support['sources'])
    reference['source_count_reduction'] = max(0, len(support['sources']) - len(fixed['sources']))
    assert actual[1]['reasoning']['trace']['answer_contract'] == reference


def fixture(extension=0, *, knowledge=False, trial=False, policy_size=0):
    chunks = [{'id': f'c{i}', 'document_id': f'd{i}', 'filename': f'example-{i}.txt',
               'page': i + 1, 'ordinal': 0,
               'content': f'Archive memory {i} stores records only after validation succeeds. '
                          f'The query cache {i} returns results under fixed conditions.'}
              for i in range(6)]
    plan = build_retrieval_plan(QUESTION, chunks, [
        {'id': 'm1', 'content': 'Preserve every stated condition, not just the claim.', 'status': 'confirmed'},
        {'id': 'm2', 'content': '偏好简洁报告，但不能丢失否定条件。', 'status': 'confirmed'},
    ])
    for row in plan['sources']:
        row['content'] += ' Additional synthetic limitation applies only after review.' * extension
        row['excerpt_end'] = row['excerpt_start'] + len(row['content'])
    support = current.empty_assistance(plan['sources'])
    support['prompt'] = '\nSynthetic auxiliary packet (not new evidence).\n'
    history = [{'role': 'user', 'content': 'Previous archive question. ' * 20},
               {'role': 'assistant', 'content': 'Previous non-evidence response. ' * 20}]
    method = retrieve_knowledge_skill('literature-synthesis', QUESTION) if knowledge else None
    options = {'system_policy': 'p' * policy_size if policy_size else
               knowledge_skill_system_policy(knowledge),
               'knowledge_support': method, 'memory_as_of': 1700000000.0,
               'project_context': 'Fixed dependency handoff: preserve validation conditions.',
               'trial_contract': trial_contract_packet() if trial else None}
    return plan, support, history, options


def run(formatter, value, support, history, options):
    options = deepcopy(options)
    if options['knowledge_support'] is not None:
        options['insight_runtime'] = InsightRuntime(QUESTION, options['knowledge_support'])
    try:
        return ('ok', formatter(QUESTION, value, history, support, **options))
    except ValueError as error:
        return ('error', type(error).__name__, str(error))


def test_frozen_reference_fingerprint():
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == (
        'f5481ea4cd93d9dc4f0b7a0e4d24b326e7ccbfb8e2cfd064ec82fccb868845ee')


@pytest.mark.parametrize('extension', [0, 18, 90])
@pytest.mark.parametrize('policy_size', [0, 9000, 16000, 23500, 26000])
@pytest.mark.parametrize('knowledge', [False, True])
@pytest.mark.parametrize('trial', [False, True])
def test_whole_result_matches_prior_formatter(extension, policy_size, knowledge, trial):
    value, support, history, options = fixture(extension, knowledge=knowledge,
                                               trial=trial, policy_size=policy_size)
    original = deepcopy((value, support, history, options))
    expected = run(ORACLE.fit_research_prompt, value, support, history, options)
    actual = run(current.fit_research_prompt, value, support, history, options)
    assert_formatter_compat(actual, expected, value, support, history, options)
    assert (value, support, history, options) == original


def test_fallback_reuses_only_identical_base_packets(monkeypatch):
    value, support, history, options = fixture(18, knowledge=True, trial=True, policy_size=16000)
    calls = {'old': [], 'new': []}
    old_grounded, new_grounded = ORACLE.grounded_prompt, current.grounded_prompt
    def counted(which, function):
        def render(question, plan, **kwargs):
            calls[which].append(deepcopy((plan, kwargs)))
            return function(question, plan, **kwargs)
        return render
    monkeypatch.setattr(ORACLE, 'grounded_prompt', counted('old', old_grounded))
    monkeypatch.setattr(current, 'grounded_prompt', counted('new', new_grounded))
    expected = run(ORACLE.fit_research_prompt, value, support, history, options)
    actual = run(current.fit_research_prompt, value, support, history, options)
    assert_formatter_compat(actual, expected, value, support, history, options)
    def base_packet(packet):
        plan, kwargs = deepcopy(packet)
        plan['gaps'] = [row for row in plan['gaps'] if row.get('code') not in LEDGER_GAPS]
        return plan, kwargs
    old_packets = [base_packet(packet) for packet in calls['old']]
    new_packets = [base_packet(packet) for packet in calls['new']]
    assert len(calls['old']) > len(calls['new']) >= 1
    assert all(packet in old_packets for packet in new_packets)
    assert all(left != right for i, left in enumerate(calls['new']) for right in calls['new'][i + 1:])


def test_new_request_reformats_same_ids_and_revalidates_current_contents(monkeypatch):
    value, support, history, options = fixture()
    calls = []
    original = current.grounded_prompt
    def render(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(current, 'grounded_prompt', render)
    first = run(current.fit_research_prompt, value, support, history, options)
    value['sources'][0]['content'] = 'Current archive record: the prior condition was withdrawn.'
    value['sources'][0]['excerpt_end'] = value['sources'][0]['excerpt_start'] + len(value['sources'][0]['content'])
    second = run(current.fit_research_prompt, value, support, history, options)
    assert_formatter_compat(second, run(ORACLE.fit_research_prompt, value, support, history, options),
                            value, support, history, options)
    assert len(calls) == 2
    assert first[1]['messages'] != second[1]['messages']
    assert first[1]['reasoning']['trace']['answer_contract']['messages_sha256'] != (
        second[1]['reasoning']['trace']['answer_contract']['messages_sha256'])


@pytest.mark.parametrize('question', ['', 'x' * 9000, '\ud800'])
def test_invalid_question_keeps_original_failure_semantics(question):
    value, support, history, options = fixture(knowledge=True)
    def attempt(formatter):
        with pytest.raises(ValueError) as error:
            formatter(question, value, history, support, **options)
        return str(error.value)
    assert attempt(current.fit_research_prompt) == attempt(ORACLE.fit_research_prompt)


def test_request_local_render_storage_does_not_create_persistent_cache():
    assert not hasattr(current, 'base_prompt_variants')
    assert not hasattr(current, 'conversation_variants')
    assert SYSTEM_POLICY


def test_changed_fallback_keeps_whole_quotes_and_rebuilds_exact_fixed_packet_with_frozen_formatter():
    value, _, _, options = fixture()
    rows = deepcopy(value['sources'][:2])
    rows[0]['content'] = 'Archive memory ordinary background. ' + 'Unrelated introduction background. ' * 100
    rows[1]['content'] = ('Compare archive memory architectures: validation succeeds only under fixed conditions.' * 10)
    for row in rows:
        row.update(excerpt_start=0, excerpt_end=len(row['content']), original_content_length=len(row['content']))
    value.update(sources=rows, selected_chunks=rows, confirmed_memories=[])
    support = current.empty_assistance(rows)
    options.update(knowledge_support=None, memory_as_of=None, project_context=None, system_policy='')
    singleton_sizes = []
    for row in rows:
        one = {**deepcopy(value), 'sources': [row], 'selected_chunks': [row]}
        measured = run(ORACLE.fit_research_prompt, one, current.empty_assistance([row]), [], options)
        assert measured[0] == 'ok'
        singleton_sizes.append(measured[1]['budget']['input_bytes'])
    # Both singletons are legal, but the pair cannot fit. The old prefix loses
    # validation/condition needs that the new complete witness keeps.
    options['system_policy'] = 'p' * (24576 - max(singleton_sizes) - 512)
    expected = run(ORACLE.fit_research_prompt, value, support, [], options)
    actual = run(current.fit_research_prompt, value, support, [], options)
    assert expected[1]['plan']['sources'] == rows[:1]
    assert actual[1]['plan']['sources'] == rows[1:]
    assert _literal_coverage(expected[1]['plan']['sources']) < _literal_coverage(actual[1]['plan']['sources'])
    assert_formatter_compat(actual, expected, value, support, [], options)
