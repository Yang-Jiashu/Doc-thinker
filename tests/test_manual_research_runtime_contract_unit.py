"""Manual gateway acceptance distinguishes restart persistence from withdrawal."""
from copy import deepcopy

import pytest

from tests.manual_research_runtime_check import assert_restarted_answer


def packets(*, revoked=True):
    before = {'answer_id': 'same-accepted-answer', 'answer': 'Original private generated answer.',
        'sources': [{'id': 'source', 'content': 'PRIVATE_RESEARCH_FIXTURE_QUOTE'}],
        'used_memory_count': 2 if revoked else 0,
        'memory_selection': {'strategy': 'indexed_temporal', 'selected_count': 2 if revoked else 0},
        'citation_validation': {'valid': True, 'warnings': []},
        'retrieval_gaps': [{'code': 'semantic_retrieval_unavailable', 'message': 'Offline'}]}
    replay = deepcopy(before)
    if revoked:
        replay.update(answer='The original answer is withdrawn because its memory changed.', sources=[],
            used_memory_count=0, citation_validation={'valid': False,
                'warnings': [{'code': 'memory_context_revoked'}]})
    history = [{'id': 'user-question', 'role': 'user', 'content': 'A question'},
               {**deepcopy(replay), 'id': replay['answer_id'], 'role': 'assistant', 'content': replay['answer']}]
    return before, replay, history


@pytest.mark.parametrize('revoked', [True, False])
def test_corrected_alice_withdraws_while_unchanged_bob_survives_restart(revoked):
    before, replay, history = packets(revoked=revoked)
    assert_restarted_answer(before, replay, history, memory_revoked=revoked)


@pytest.mark.parametrize('mutation', ['request_identity', 'sources', 'usage', 'warning', 'validity',
                                   'body', 'history_only', 'selection'])
def test_memory_revision_acceptance_cannot_pass_leaked_or_reactivated_old_content(mutation):
    before, replay, history = packets()
    current = history[-1]
    if mutation == 'request_identity':
        replay['answer_id'] = 'another-answer'
    elif mutation == 'sources':
        replay['sources'] = current['sources'] = deepcopy(before['sources'])
    elif mutation == 'usage':
        replay['used_memory_count'] = current['used_memory_count'] = 2
    elif mutation == 'warning':
        replay['citation_validation']['warnings'] = current['citation_validation']['warnings'] = []
    elif mutation == 'validity':
        replay['citation_validation']['valid'] = current['citation_validation']['valid'] = True
    elif mutation == 'body':
        replay['answer'] = current['content'] = before['answer']
    elif mutation == 'history_only':
        current['content'] = before['answer']
    else:
        replay['memory_selection'] = {'strategy': 'another'}
    with pytest.raises(AssertionError):
        assert_restarted_answer(before, replay, history, memory_revoked=True)


@pytest.mark.parametrize('field,value', [('answer', 'Changed answer'), ('sources', []),
                                      ('used_memory_count', 2), ('citation_validation', {'valid': False})])
def test_unrelated_user_replay_must_remain_unchanged(field, value):
    before, replay, history = packets(revoked=False)
    replay[field] = value
    if field == 'answer':
        history[-1]['content'] = value
    elif field in {'sources', 'used_memory_count'}:
        history[-1][field] = value
    with pytest.raises(AssertionError):
        assert_restarted_answer(before, replay, history, memory_revoked=False)


def test_unicode_old_answer_cannot_hide_in_nested_replay_metadata():
    before, replay, history = packets()
    before['answer'] = '旧的中文回答含有已经更正的私人记忆。'
    replay['unexpected_metadata'] = {'old_answer': before['answer']}
    with pytest.raises(AssertionError, match='leaked'):
        assert_restarted_answer(before, replay, history, memory_revoked=True)
