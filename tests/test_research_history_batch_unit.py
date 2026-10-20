"""Request-local batch performance and unchanged provenance contracts, no provider.

The two legacy functions below preserve the pre-batching read implementation,
with the explicit current citation-namespace projection applied equally. They
are equality/performance oracles, not runtime alternatives.
"""
import copy
import json
import sqlite3
import uuid
from types import MethodType

import pytest

from docthinker.server.research_store import (
    ResearchStoreError, ResearchLimitError, MAX_ANSWER_LOOKUP, _session, _identifier,
)
from docthinker.server.research_service import ResearchService, WITHDRAWN_ANALYSIS
from docthinker.server.research_evidence_index import ResearchEvidenceIndex
from docthinker.server.research_provenance import audit_answer
from docthinker.server.research_reasoning_context import render_evidence, render_reasoning
from docthinker.server.research_trial_delivery import unavailable_trial_receipt
from docthinker.server.research_knowledge import unavailable_knowledge
from docthinker.server.research_retrieval import refresh_citation_validation

def legacy_answer_metadata(self, session_id: str, answer_ids: list[str]) -> dict:
    _session(session_id)
    if not isinstance(answer_ids, list) or len(answer_ids) > MAX_ANSWER_LOOKUP:
        raise ResearchLimitError("Too many answer references")
    for answer_id in answer_ids:
        _identifier(answer_id)
    if not answer_ids:
        return {}
    with self._read() as db:
        if db is None:
            return {}
        # Older empty research stores do not acquire a new schema on reads.
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_answers'").fetchone():
            return {}
        rows = db.execute("SELECT * FROM research_answers WHERE session_id=? AND answer_id IN (" + ",".join("?" for _ in answer_ids) + ")",
                          [session_id, *answer_ids]).fetchall()
        result = {}
        for row in rows:
            refs = json.loads(row["source_refs"])
            for source in refs:
                source["available"] = bool(db.execute("SELECT 1 FROM research_chunks c JOIN research_documents d ON d.document_id=c.document_id "
                                                      "WHERE c.session_id=? AND d.session_id=? AND c.chunk_id=? AND d.state='ready'",
                                                      (session_id, session_id, source["id"])).fetchone())
            result[row["answer_id"]] = {"answer_id": row["answer_id"], "source_refs": refs,
                                        "validation": json.loads(row["validation"]),
                                        "answer_mode": row["answer_mode"], "created_at": row["created_at"],
                                        **self._public_answer_context(self._answer_context_metadata(
                                            json.loads(row["context_metadata_json"]) if "context_metadata_json" in row.keys() else {}))}
        return result


def legacy_hydrate_history(self, sid: str, messages: list[dict]) -> list[dict]:
    ids = [m["id"] for m in messages[-200:] if m.get("role") == "assistant" and isinstance(m.get("id"), str)]
    metadata = self.store.answer_metadata(sid, ids)
    result = []
    for message in messages:
        item = metadata.get(message.get("id"))
        if item:
            sources, unavailable = [], 0
            for ref in item["source_refs"]:
                rows = self.store.source_chunks(sid, [ref["id"]]) if ref["available"] else []
                if rows:
                    row = rows[0]
                    start, end = ref.get("excerpt_start", 0), ref.get("excerpt_end", len(row["content"]))
                    sources.append({**{key: value for key, value in row.items() if key != 'text'}, "content": row["content"][start:end], "label": ref["label"],
                                    'excerpt_start': start, 'excerpt_end': end, 'original_content_length': len(row['content']),
                                    "truncated": ref.get("truncated", False)})
                else:
                    unavailable += 1
            validation = item["validation"]
            if unavailable:
                validation = {**validation, "valid": False, "warnings": [*validation["warnings"],
                              {"code": "source_revoked", "message": "部分原文依据已被删除，无法继续核对。"}]}
            message = {**message, "sources": sources, "answer_mode": item["answer_mode"],
                       "citation_validation": validation, "unavailable_source_count": unavailable,
                       **{key: item[key] for key in ("retrieval_gaps", "used_memory_count", "memory_selection") if key in item},
                       "research_core": True}
            proof_metadata = self.evidence.hydrate(sid, message.get('id'), sources)
            # A request-status replay may carry an old response packet.
            # Fresh proof metadata replaces that packet; a missing record
            # must never leave cached quotes/methods visible in other cards.
            for field in ('reasoning', 'claim_audit', 'knowledge_skill', 'trial_delivery'):
                message.pop(field, None)
            message.update(proof_metadata)
            # Namespace hardening is a deliberate return-time contract change,
            # independent of batched reads. Keep the old read order and apply
            # precisely that projection; body/source/invalidation stay exact.
            knowledge = message.get('knowledge_skill') or {}
            validation = refresh_citation_validation(
                message.get('content', message.get('answer', '')),
                [source['label'] for source in sources], validation,
                knowledge_labels=[row['label'] for row in knowledge.get('sources', [])]
                    if knowledge.get('status') == 'ready' else [])
            message['citation_validation'] = validation
            trial_receipt = message.get('reasoning', {}).get('trace', {}).get('trial_delivery')
            if trial_receipt:
                message['trial_delivery'] = trial_receipt
            withdrawn_literal = unavailable and message.get('reasoning_mode') == 'evidence'
            legacy_proof = (message.get('reasoning', {}).get('status') == 'source_unavailable'
                            and message.get('reasoning', {}).get('mode') != 'context_assist')
            withdrawn_analysis = (message.get('reasoning', {}).get('status') == 'source_unavailable'
                                  and message.get('reasoning', {}).get('mode') == 'context_assist')
            if message.get('research_provenance_required') is True and (unavailable or not proof_metadata):
                withdrawn_analysis = True
                message['citation_validation'] = {**validation, 'valid': False, 'warnings': [*validation['warnings'],
                    {'code': 'provenance_unavailable', 'message': '推断依据不可用，本次分析已收起，不会重新请求模型。'}]}
            if withdrawn_analysis:
                if trial_receipt:
                    message['trial_delivery'] = unavailable_trial_receipt()
                message.update(content=WITHDRAWN_ANALYSIS, claim_audit=audit_answer(WITHDRAWN_ANALYSIS, sources))
                if message.get('knowledge_skill'):
                    message['knowledge_skill'] = unavailable_knowledge(message['knowledge_skill'])
                if 'answer' in message:
                    message['answer'] = WITHDRAWN_ANALYSIS
            elif withdrawn_literal or legacy_proof:
                rewritten = (render_evidence(sources) + '\n\n部分原文已被撤回，已移除对应摘录。' if withdrawn_literal else
                             render_reasoning(message['reasoning'], sources) + '\n\n部分前提已被撤回，相关推导已收起。')
                message.update(content=rewritten, claim_audit=audit_answer(rewritten, sources))
                if 'answer' in message:
                    message['answer'] = rewritten
        elif message.get("research_core") is True or message.get('research_provenance_required') is True:
            message = {**message, "sources": [], "citation_validation": {
                "valid": False, "entailment_verified": False, "warnings": [{"code": "provenance_unavailable", "message": "依据记录不可用，不能继续核对本次引用。"}]}}
            had_trial = bool(message.get('trial_delivery') or message.get('reasoning', {}).get('trace', {}).get('trial_delivery'))
            for field in ('reasoning', 'claim_audit', 'knowledge_skill', 'trial_delivery'):
                message.pop(field, None)
            if had_trial:
                message['trial_delivery'] = unavailable_trial_receipt()
            if message.get('research_provenance_required') is True:
                message.update(content=WITHDRAWN_ANALYSIS, claim_audit=audit_answer(WITHDRAWN_ANALYSIS, []))
                if 'answer' in message:
                    message['answer'] = WITHDRAWN_ANALYSIS
        result.append(message)
    return result


def seeded(tmp_path, *, count=200, unique_chunks=10):
    service = ResearchService(tmp_path, semantic_retriever=None,
                              evidence_reranker=None, query_translator=None)
    sid = '#00001'
    ids = [str(uuid.uuid4()) for _ in range(unique_chunks)]
    aids = [str(uuid.uuid4()) for _ in range(count)]
    text = '前提：条件成立时可以采用方法。若条件不满足则不能推广。'
    document = str(uuid.uuid4())
    with service.store._write() as db:
        ResearchEvidenceIndex._schema(db)
        db.execute('INSERT INTO research_documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   (document, sid, 'synthetic.txt', '.txt', 'a' * 64, 100, 'ready',
                    'synthetic', 1, unique_chunks, len(text) * unique_chunks, None, 1, 1))
        db.executemany('INSERT INTO research_chunks VALUES(?,?,?,?,?,?)',
                       [(key, sid, document, i, text, 1) for i, key in enumerate(ids)])
        for index, aid in enumerate(aids):
            keys = [ids[(index * 10 + i) % unique_chunks] for i in range(min(10, unique_chunks))]
            refs = [{'id': key, 'label': f'S{i + 1}', 'excerpt_start': 3,
                     'excerpt_end': len(text), 'truncated': True} for i, key in enumerate(keys)]
            sources = [{'id': ref['id'], 'label': ref['label'], 'document_id': document,
                        'content': text[3:], 'excerpt_start': 3,
                        'excerpt_end': len(text)} for ref in refs]
            audit = audit_answer(text[3:] + ' [S1]', sources)
            for claim in audit['claims']:
                for ref in claim['source_refs']:
                    ref.pop('quote', None)
            proof = {'reasoning': {'mode': 'context_assist', 'status': 'ready',
                     'claims': [], 'conclusions': [], 'gaps': [],
                     'trace': {'trial_delivery': {'status': 'draft', 'source_ids': keys,
                                                 'semantic_verified': False}}},
                     'claim_audit': audit}
            db.execute('INSERT INTO research_answers(answer_id,session_id,source_refs,validation,answer_mode,created_at,context_metadata_json) VALUES(?,?,?,?,?,?,?)',
                       (aid, sid, json.dumps(refs), json.dumps({'valid': True, 'warnings': []}),
                        'explore', 1, '{}'))
            db.execute('INSERT INTO research_answer_evidence VALUES(?,?,?)', (aid, sid, json.dumps(proof)))
    messages = [{'id': aid, 'role': 'assistant', 'content': text[3:] + ' [S1]',
                 'research_core': True, 'research_provenance_required': True,
                 'reasoning': {'old_private_packet': 'discard'}, 'knowledge_skill': {'old': True}}
                for aid in aids]
    return service, sid, document, ids, messages


def metered(monkeypatch, action):
    original = sqlite3.connect
    count = {'connections': 0, 'selects': 0, 'max_parameters': 0}

    class Connection(sqlite3.Connection):
        def execute(self, sql, parameters=(), /):
            count['max_parameters'] = max(count['max_parameters'], len(parameters))
            return super().execute(sql, parameters)

    def connect(*args, **kwargs):
        count['connections'] += 1
        connection = original(*args, **kwargs, factory=Connection)
        connection.set_trace_callback(lambda sql: count.__setitem__(
            'selects', count['selects'] + int(sql.lstrip().upper().startswith('SELECT'))))
        return connection

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, 'connect', connect)
        result = action()
    return result, count


def legacy(service, sid, messages, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(service.store, 'answer_metadata', MethodType(legacy_answer_metadata, service.store))
        return legacy_hydrate_history(service, sid, copy.deepcopy(messages))


def test_200_answers_with_shared_sources_batch_without_changing_result(tmp_path, monkeypatch):
    service, sid, _, _, messages = seeded(tmp_path)
    expected, old = metered(monkeypatch, lambda: legacy(service, sid, messages, monkeypatch))
    actual, new = metered(monkeypatch, lambda: service.hydrate_history(sid, copy.deepcopy(messages)))
    assert actual == expected
    assert old['connections'] == 2201 and old['selects'] == 6603
    assert new['connections'] <= 3 and new['selects'] <= 10
    assert new['max_parameters'] <= 400
    assert actual[0]['sources'][0]['content'].startswith('条件成立')
    assert actual[0]['sources'][0]['excerpt_start'] == 3
    assert 'knowledge_skill' not in actual[0]
    assert actual[0]['trial_delivery']['status'] == 'draft'


def test_2000_unique_refs_are_chunked_below_sql_parameter_limit(tmp_path, monkeypatch):
    service, sid, _, ids, messages = seeded(tmp_path, unique_chunks=2000)
    result, count = metered(monkeypatch, lambda: service.hydrate_history(sid, messages))
    assert count['connections'] <= 3 and count['selects'] <= 20
    assert count['max_parameters'] <= 400
    assert {ref['id'] for message in result for ref in message['sources']} == set(ids)


@pytest.mark.parametrize('mutation', ['revoke', 'missing_proof', 'changed_text'])
def test_current_batch_matches_legacy_after_source_or_proof_change(tmp_path, monkeypatch, mutation):
    service, sid, document, ids, messages = seeded(tmp_path, count=3)
    first = service.hydrate_history(sid, copy.deepcopy(messages))
    with service.store._write() as db:
        if mutation == 'revoke':
            db.execute("UPDATE research_documents SET state='deleted' WHERE document_id=?", (document,))
        elif mutation == 'missing_proof':
            db.execute('DELETE FROM research_answer_evidence')
        else:
            db.execute('UPDATE research_chunks SET text=? WHERE chunk_id=?', ('x' * 40, ids[0]))
    expected = legacy(service, sid, messages, monkeypatch)
    actual = service.hydrate_history(sid, copy.deepcopy(messages))
    assert actual == expected and actual != first
    if mutation in {'revoke', 'missing_proof'}:
        assert all(item['content'] == WITHDRAWN_ANALYSIS for item in actual)
    else:
        assert actual[0]['claim_audit']['claims'][0]['status'] != 'exact_quote'


def test_batches_never_read_other_sessions_or_keep_cross_request_sources(tmp_path, monkeypatch):
    service, sid, document, ids, messages = seeded(tmp_path, count=2)
    assert service.store.answer_source_chunks('#00002', ids) == {}
    assert service.evidence.hydrate_many('#00002', {m['id']: [] for m in messages}) == {}
    assert not service.store.answer_metadata('#00002', [m['id'] for m in messages])
    service.hydrate_history(sid, copy.deepcopy(messages))
    with service.store._write() as db:
        db.execute("UPDATE research_documents SET state='deleted' WHERE document_id=?", (document,))
    assert service.store.answer_source_chunks(sid, ids) == {}
    assert all(not row['sources'] for row in service.hydrate_history(sid, messages))


def test_empty_bounded_batches_do_not_create_store(tmp_path):
    service = ResearchService(tmp_path, semantic_retriever=None,
                              evidence_reranker=None, query_translator=None)
    assert service.store.answer_source_chunks('#00001', []) == {}
    assert service.evidence.hydrate_many('#00001', {}) == {}
    assert list(tmp_path.iterdir()) == []
    for bad in (None, [True], ['../bad']):
        with pytest.raises((ResearchStoreError, ResearchLimitError)):
            service.store.answer_source_chunks('#00001', bad)
    with pytest.raises(ResearchLimitError):
        service.store.answer_source_chunks('#00001', [str(uuid.uuid4())] * 2001)
    with pytest.raises((ResearchStoreError, ResearchLimitError)):
        service.evidence.hydrate_many('#00001', {str(uuid.uuid4()): [] for _ in range(201)})


def test_proof_packet_cannot_borrow_other_answers_live_sources(tmp_path):
    service, sid, _, _, messages = seeded(tmp_path, count=2, unique_chunks=20)
    with service.store._write() as db:
        first = db.execute('SELECT payload FROM research_answer_evidence WHERE answer_id=?',
                           (messages[0]['id'],)).fetchone()['payload']
        db.execute('UPDATE research_answer_evidence SET payload=? WHERE answer_id=?',
                   (first, messages[1]['id']))
    result = service.hydrate_history(sid, messages)
    assert result[0]['content'] != WITHDRAWN_ANALYSIS
    assert result[1]['content'] == WITHDRAWN_ANALYSIS
    assert result[1]['reasoning']['status'] == 'source_unavailable'
    assert result[1]['trial_delivery']['status'] == 'source_unavailable'


def test_shared_chunk_preserves_each_answers_own_offsets_labels_and_proof(tmp_path, monkeypatch):
    service, sid, _, ids, messages = seeded(tmp_path, count=2, unique_chunks=1)
    full = service.store.source_chunks(sid, ids)[0]['content']
    with service.store._write() as db:
        db.execute('UPDATE research_answers SET source_refs=? WHERE answer_id=?',
                   (json.dumps([{'id': ids[0], 'label': 'S10', 'excerpt_start': 0,
                                 'excerpt_end': len(full), 'truncated': False}]), messages[1]['id']))
    result = service.hydrate_history(sid, copy.deepcopy(messages))
    assert result == legacy(service, sid, messages, monkeypatch)
    assert result[0]['sources'][0]['label'] == 'S1'
    assert result[0]['sources'][0]['excerpt_start'] == 3
    assert result[1]['sources'][0]['label'] == 'S10'
    assert result[1]['sources'][0]['content'] == full


def test_delete_between_metadata_and_source_batch_is_still_revoked(tmp_path, monkeypatch):
    service, sid, document, _, messages = seeded(tmp_path, count=2)
    original = service.store.answer_metadata

    def metadata_then_revoke(*args):
        result = original(*args)
        with service.store._write() as db:
            db.execute("UPDATE research_documents SET state='deleted' WHERE document_id=?", (document,))
        return result

    monkeypatch.setattr(service.store, 'answer_metadata', metadata_then_revoke)
    result = service.hydrate_history(sid, messages)
    assert all(row['content'] == WITHDRAWN_ANALYSIS and row['sources'] == [] for row in result)


def test_missing_proof_table_stays_missing_and_old_packet_is_removed(tmp_path):
    service, sid, _, _, messages = seeded(tmp_path, count=2)
    with service.store._write() as db:
        db.execute('DROP TABLE research_answer_evidence')
    before = service.store.path.read_bytes()
    result = service.hydrate_history(sid, messages)
    assert all(row['content'] == WITHDRAWN_ANALYSIS and 'reasoning' not in row for row in result)
    assert service.store.path.read_bytes() == before

