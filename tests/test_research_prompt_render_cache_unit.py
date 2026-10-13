"""Full formatter differential checks; all material is invented and local."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from docthinker.server import research_prompt_budget as current
from docthinker.server.insight_runtime import InsightRuntime
from docthinker.server.knowledge_skills import retrieve_knowledge_skill
from docthinker.server.research_answer_contract import SYSTEM_POLICY, knowledge_skill_system_policy
from docthinker.server.research_retrieval import build_retrieval_plan
from docthinker.server.research_trial_delivery import trial_contract_packet


REFERENCE = Path(__file__).parent / 'fixtures' / 'reference_research_prompt_20261013.py'
SPEC = importlib.util.spec_from_file_location('frozen_prompt_formatter_20261013', REFERENCE)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)
QUESTION = 'Compare archive memory architectures and their validation conditions.'


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
    assert actual == expected  # Messages, references, budgets, gaps and hashes.
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
    assert run(current.fit_research_prompt, value, support, history, options) == expected
    assert len(calls['old']) > len(calls['new']) >= 1
    assert all(packet in calls['old'] for packet in calls['new'])
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
    assert second == run(ORACLE.fit_research_prompt, value, support, history, options)
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
