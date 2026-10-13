"""Replay recorded final answers against frozen public corpora; no provider I/O.

This checks delivery/provenance only, not a new model run or a quality score.
The original benchmark artifacts are read-only; only provider.answer is used.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

os.environ.setdefault('PYTHON_DOTENV_DISABLED', '1')

from docthinker.server.research_service import ResearchService, WITHDRAWN_ANALYSIS
from docthinker.session_manager import SessionManager


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode()).hexdigest()


async def replay(run_dir: Path, name: str, output_dir: Path | None = None) -> dict:
    record = json.loads((run_dir / 'answers' / f'{name}.r1.on.json').read_text())
    final_answer = record['provider']['answer']
    assert isinstance(final_answer, str) and final_answer.strip()
    corpus = run_dir / 'corpora' / record['corpus_digest']
    with tempfile.TemporaryDirectory(prefix='insight-provenance-replay-') as temporary:
        root = Path(temporary).resolve()
        manager = SessionManager(str(root / '_system'), str(root), migrate_legacy=False)
        sid = manager.create_session('Recorded public-paper delivery replay')['id']
        assert sid == '#00001'
        shutil.copytree(corpus / '_system' / 'research', root / '_system' / 'research')
        service = ResearchService(root, context_assistance_enabled=True,
                                  semantic_retriever=None, evidence_reranker=None, query_translator=None)
        persistence_errors = []
        original_record = service.store.record_answer

        def record_answer(*args, **kwargs):
            try:
                return original_record(*args, **kwargs)
            except Exception as error:
                persistence_errors.append(type(error).__name__ + ': ' + str(error))
                raise

        service.store.record_answer = record_answer
        calls = []

        async def recorded_writer(prompt, **kwargs):
            calls.append([{'role': 'system', 'content': kwargs['system_prompt']},
                          {'role': 'user', 'content': prompt}])
            return final_answer

        try:
            await service.prepare_retrieval(sid)
            request = SimpleNamespace(session_id=sid, question=record['question'], reasoning_mode='auto',
                                      use_memory=False, use_conversation_context=False, use_llm_cache=False,
                                      use_self_evolution=False, memory_as_of=None,
                                      insight_mode='auto', knowledge_skill_mode='auto')
            result = await service.answer(request, recorded_writer, manager)
            delivered = result['answer'] == final_answer
            source_payload = calls[0][-1]['content'].split('UNTRUSTED_RESEARCH_DATA_JSON:\n', 1)[1]
            source_quotes = json.JSONDecoder().raw_decode(source_payload)[0]['source_quotes']
            status = {'case': name, 'provider_calls': 0, 'recorded_writer_calls': len(calls),
                      'delivered': delivered, 'source_count': len(result['sources']),
                      'persistence_errors': persistence_errors,
                      'original_prompt_digest': record['provider']['prompt_digest'],
                      'replay_prompt_digest': digest(calls[0]),
                      'prompt_identical_to_paid_call': digest(calls[0]) == record['provider']['prompt_digest'],
                      'replay_source_quotes_digest': digest(source_quotes),
                      'source_comparison_basis': 'full_prompt_hash_includes_exact_source_quotes',
                      'evaluation_scope': 'delivery_and_revocation_replay_not_new_model_quality'}
            if not delivered:
                return status
            assert len(calls) == 1 and len(result['sources']) == 10
            assert status['prompt_identical_to_paid_call'], 'Replay prompt differs from recorded paid call'
            assert [row['label'] for row in result['sources']] == [f'S{i}' for i in range(1, 11)]
            assert sum(len(row['content'].encode()) for row in result['sources']) <= 12000
            for source in result['sources']:
                original = service.store.source_chunks(sid, [source['id']])[0]
                assert original['content'][source['excerpt_start']:source['excerpt_end']] == source['content']
            history = service.hydrate_history(sid, manager.get_history(sid))[-1]
            assert history['content'] == final_answer and len(history['sources']) == 10
            # The tenth source must not be silently discarded on save, restart,
            # or revocation. Delete in this isolated clone, never the corpus.
            tenth = next(row for row in result['sources'] if row['label'] == 'S10')
            service.store.delete_document(sid, tenth['document_id'])
            revoked = service.hydrate_history(sid, manager.get_history(sid))[-1]
            assert revoked['content'] == WITHDRAWN_ANALYSIS
            assert all(row['id'] != tenth['id'] for row in revoked['sources'])
            assert revoked['reasoning']['status'] == 'source_unavailable'
            assert len(calls) == 1
            status['s10_revocation_closed_dependencies'] = True
            if output_dir is not None:
                artifact = {**status, 'source_quotes': source_quotes,
                            'answer': result['answer'], 'sources': result['sources'],
                            'citation_validation': result['citation_validation']}
                (output_dir / f'{name}.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')
            return status
        finally:
            await service.stop()


async def main(run_dir: Path, output_dir: Path | None = None):
    if output_dir is not None:
        assert output_dir != run_dir and run_dir not in output_dir.parents
        output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for name in ('smoke_literature_comparison', 'smoke_testable_hypothesis', 'smoke_assistant_report'):
        result = await replay(run_dir, name, output_dir)
        outcomes.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    assert all(row['delivered'] and row.get('s10_revocation_closed_dependencies') for row in outcomes)
    if output_dir is not None:
        (output_dir / 'summary.json').write_text(json.dumps(outcomes, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    arguments = parser.parse_args()
    asyncio.run(main(arguments.run_dir.resolve(), arguments.output_dir.resolve() if arguments.output_dir else None))
