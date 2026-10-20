"""Real long-answer batches do not consume a memory packet's recursion quota."""
import json

import pytest

from docthinker.server.research_service import WITHDRAWN_ANALYSIS
from tests.test_research_answer_dependencies_unit import opened, personal, work_source
from tests.test_research_service_unit import FakeModel, workspace


@pytest.mark.asyncio
@pytest.mark.parametrize('base', ['document', 'memory_derived_from_document'])
async def test_actual_160_step_batch_and_history_share_memo_without_false_withdrawal(workspace, base):
    root, manager, sid, _ = workspace
    service = opened(root)
    calls = []
    try:
        document, previous = await work_source(service, manager, sid)
        if base != 'document':
            note = service.store.create_memory(sid, 'decision',
                'The work handover deadline remains Thursday; do not send.', source_answer_id=previous['answer_id'])
            model = FakeModel('PRIVATE_VALID_0: a private draft based on the confirmed handover note.')
            previous = await personal(service, manager, sid, model, contexts={note['id']: 1})
            assert previous['answer'] == model.answer
        ids = [previous['answer_id']]
        for index in range(1, 160):
            model = FakeModel(f'PRIVATE_VALID_{index}: a private follow-up, not an executed action.')
            previous = await personal(service, manager, sid, model, parents=[previous['answer_id']],
                context=json.dumps({'previous_analysis': previous['answer'], 'not_original_evidence': True}))
            assert previous['answer'] == model.answer and len(model.prompts) == 1
            assert previous['reasoning']['trace']['context_budget']['input_bytes'] <= 24576
            ids.append(previous['answer_id'])
            calls.append(model)
        # Individually all three have always been readable; batch memo size
        # must not newly revoke the later descendants of a valid old chain.
        for index in [0, 90, 159]:
            assert service.answer_dependencies.states(sid, [ids[index]]) == {ids[index]: True}
        states = service.answer_dependencies.states(sid, ids[-100:])
        assert states == {identifier: True for identifier in ids[-100:]}
        history = service.hydrate_history(sid, manager.get_history(sid))
        tail = next(item for item in history if item.get('id') == ids[-1])
        assert tail['content'] == previous['answer']
        assert not any(warning.get('code') == 'provenance_unavailable'
                       for warning in tail['citation_validation']['warnings'])
        saved = service.store.create_memory(sid, 'finding', 'A confirmed private long-task follow-up.', source_answer_id=ids[-1])
        service.store.delete_document(sid, document['id'])
        await service.stop()
        service = opened(root)
        assert service.answer_dependencies.states(sid, ids[-100:]) == {identifier: False for identifier in ids[-100:]}
        tail = next(item for item in service.hydrate_history(sid, manager.get_history(sid)) if item.get('id') == ids[-1])
        assert tail['content'] == WITHDRAWN_ANALYSIS
        assert service.store.get_memory(sid, saved['id'])['redacted'] is True
        assert sum(len(model.prompts) for model in calls) == 159
    finally:
        await service.stop()
