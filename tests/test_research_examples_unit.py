"""Shipped demos execute the same live parser/store/service path as uploads."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from docthinker.server.research_service import ResearchService
from docthinker.server.research_reasoning_context import render_reasoning
from tests.test_research_service_unit import workspace, text_extractor


def test_analogy_summary_keeps_transfer_conditions_visible():
    report = render_reasoning({'mode': 'analogy', 'claims': [], 'conclusions': [{
        'kind': 'analogy_hypothesis', 'text': 'A bounded mapping', 'source_ids': ['one'],
        'mappings': [{'base': 'solar', 'target': 'practice'}], 'candidate_inferences': [{
            'text': 'practice supports performance', 'conditions': ['low load']}]}]},
        [{'id': 'one', 'label': 'S1'}])
    assert '沿用来源条件：low load' in report and '目标场景条件未验证' in report


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,pattern,question', [
    ('deduction', 'energy_chain.txt', '能源安全是否提高？'),
    ('analogy', 'analogy_*.txt', 'Solar 和 Practice 可以如何结构类比？'),
    ('induction', 'induction_*.txt', 'Treatment 对 risk 有什么共同规律？'),
])
async def test_shipped_examples_are_executable_not_mocked(workspace, mode, pattern, question):
    root, manager, sid, _ = workspace
    service = ResearchService(root, extractor=text_extractor)
    for path in sorted((Path(__file__).parents[1] / 'examples/research_reasoning').glob(pattern)):
        document = service.store.accept_document(sid, path.name, path.read_bytes())
        assert await service.process(sid, document['id'])
    calls = []
    async def writing_model(prompt, **options):
        calls.append(prompt)
        return '这是根据资料撰写的自然回答，不是独立算法报告。[S1]'
    answer = await service.answer(SimpleNamespace(session_id=sid, question=question,
        reasoning_mode='auto', use_memory=False, use_conversation_context=False), writing_model, manager)
    reasoning = answer['reasoning']
    assert len(calls) == 1
    assert reasoning['mode'] == 'context_assist'
    assert reasoning['trace']['model_calls'] == 0
    assert reasoning['trace']['answer_model_calls'] == 1
    assert reasoning['trace']['algorithm_modes'] == [mode]
    assert 'supporting_material' in calls[0]
    assert answer['answer'] == '这是根据资料撰写的自然回答，不是独立算法报告。[S1]'
    assert reasoning['status'] == 'derived' and reasoning['conclusions']
    assert answer['sources'] and answer['citation_validation']['valid']
    if mode == 'deduction':
        assert any(item['subject'] == '能源安全' and item['depth'] == 7 for item in reasoning['conclusions'])
    elif mode == 'analogy':
        assert any(item['candidate_inferences'] for item in reasoning['conclusions'])
        assert any(pair['base'] == 'solar' and pair['target'] == 'practice' for item in reasoning['conclusions'] for pair in item['mappings'])
    else:
        assert any(item['support_count'] == 2 and item['counterexample_count'] == 1 for item in reasoning['conclusions'])
        assert any('controlled conditions' in item['text'] for item in reasoning['conclusions'])


@pytest.mark.asyncio
async def test_source_revocation_removes_derived_claim_and_atom_indexes(workspace):
    root, _, sid, other = workspace
    service = ResearchService(root, extractor=text_extractor)
    content = b'Observation: grid ready. If grid ready then supply grows.'
    documents = []
    for project in (sid, other):
        doc = service.store.accept_document(project, 'rules.txt', content)
        assert await service.process(project, doc['id'])
        documents.append(doc)
    with service.store._read() as db:
        assert db.execute('SELECT COUNT(*) FROM research_claim_index WHERE session_id=?', (sid,)).fetchone()[0] == 2
        assert db.execute('SELECT COUNT(*) FROM research_claim_atoms WHERE session_id=?', (sid,)).fetchone()[0] >= 2
    service.store.delete_document(sid, documents[0]['id'])
    with service.store._read() as db:
        assert db.execute('SELECT COUNT(*) FROM research_claim_index WHERE session_id=?', (sid,)).fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM research_claim_atoms WHERE session_id=?', (sid,)).fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM research_claim_index WHERE session_id=?', (other,)).fetchone()[0] == 2
