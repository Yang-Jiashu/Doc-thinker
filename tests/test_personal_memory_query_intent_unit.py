import pytest

from docthinker.server.research_memory import classify_memory_query, select_confirmed_memories


@pytest.mark.parametrize('question', ['What did I decide?', 'What have I decided?',
    'What are my decisions?', '我的决定是什么？', 'What did we decide?'])
def test_personal_decision_recall_does_not_require_a_project_we_pronoun(question):
    assert classify_memory_query(question) == 'recent_project_context'
    rows = [{'id': 'decision', 'kind': 'decision', 'text': 'Family budget is 4500 yuan.',
             'confirmed_by_user': True, 'status': 'active', 'revision': 2, 'updated_at': 2000}]
    assert select_confirmed_memories(question, rows)['memories'] == rows


@pytest.mark.parametrize('question', ['How do decision trees work?',
    'I decided to study decision trees.', '帮我分析这个决定的利弊。'])
def test_ordinary_analysis_does_not_enable_unrelated_recent_memory(question):
    assert classify_memory_query(question) == 'lexical_bm25_with_preferences'
