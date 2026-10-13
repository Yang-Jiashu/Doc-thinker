"""Real subprocess acceptance of durable projects, with a loopback fake model.

Run ``python -m tests.manual_project_task_runtime_check`` after building H5.
``--serve`` keeps this disposable two-user preview alive for browser QA. The
gateway, supervisors, workers, parser, journals and HTTP routes are real; the
model and all documents/accounts are synthetic. No vendor call, Docker, public
hosting, private user data, semantic accuracy or production capacity is tested.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

from docthinker.server.invite_store import InviteStore
from docthinker.server.invite_workers import SHARING_ACK_ENV, SHARING_ACK_VALUE
from tests.manual_invite_runtime_check import FAKE_KEY, ROOT, free_port, port_open
from tests.manual_research_runtime_check import (
    PASSWORD, PREFIX, client, fake_research_model, login, ready_document,
    running, serve_until_stopped, upload,
)
from tests.test_research_service_unit import pdf_with_text

TASKS = PREFIX + '/workbench'
TEXT_A = ('Renewable supply replaces imported energy in this synthetic example. '
          'The renewable supply mechanism increases domestic output before imports decline. '
          'PROJECT_ALPHA_PRIVATE_SOURCE_MARKER. This is not a real scientific result.')
TEXT_B = ('Storage buffers renewable supply during short interruptions. '
          'Storage availability depends on the stored energy reserve. This is a synthetic example.')
TEXT_C = ('Storage reserve testing compares full and empty buffers during interruptions. '
          'The synthetic storage example reports a missing measurement, not a verified benefit.')


def reference():
    return str(uuid.uuid4())


def expect(response, status=200):
    assert response.status_code == status, 'Project HTTP assertion failed: ' + str(response.status_code) + ' ' + response.text[:400]
    return response.json()


def query_usage(store, username='preview'):
    """Read only a count from this disposable registry, not credentials."""
    assert store.root.parent.name.startswith('docthinker-project-check-')
    with store._connect() as db:
        row = db.execute("SELECT COALESCE(SUM(count),0) FROM usage WHERE kind='query' "
                         "AND user_id=(SELECT id FROM users WHERE username=?)", (username,)).fetchone()
        return row[0]


def get_task(browser, identity, sid, task_id):
    return expect(browser.get(TASKS + '/task', headers=identity,
        params={'session_id': sid, 'task_id': task_id}))['task']


def wait_task(browser, identity, sid, task_id, expected):
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline:
        task = get_task(browser, identity, sid, task_id)
        if task['status'] == expected:
            return task
        assert task['status'] not in {'blocked', 'stale', 'cancelled'}, 'Project stopped unexpectedly: ' + json.dumps(task, ensure_ascii=False)[:1200]
        time.sleep(0.1)
    raise AssertionError('Project did not reach ' + expected)


def step(task, key):
    return next(row for row in task['steps'] if row['id'] == key)


def control(browser, identity, sid, task, action):
    return expect(browser.post(TASKS + '/tasks/control', headers=identity, json={
        'session_id': sid, 'task_id': task['id'], 'expected_revision': task['revision'], 'action': action}))['task']


def approve(browser, identity, sid, task, maximum):
    payload = {'session_id': sid, 'task_id': task['id'], 'expected_revision': task['revision'],
               'request_id': reference(), 'max_calls': maximum}
    expect(browser.post(TASKS + '/tasks/run', headers=identity, json=payload))
    return payload


def artifact(browser, identity, sid, task, version, step_id='report'):
    return expect(browser.get(TASKS + '/task/artifact', headers=identity, params={
        'session_id': sid, 'task_id': task['id'], 'step_id': step_id, 'version': version}))['artifact']


def phase_before_restart(origin, invitations, store, calls):
    with client(origin) as browser:
        assert browser.get(TASKS + '/tasks', params={'session_id': '#00001'}).status_code == 401
        identities, sessions = [], []
        for invitation in invitations:
            expect(browser.post('/api/v1/invite/register', json={
                'username': invitation['username'], 'invite_code': invitation['invite_code'], 'password': PASSWORD}))
            identity = login(browser, invitation['username'])
            identities.append(identity)
            created = expect(browser.post(PREFIX + '/sessions', headers=identity,
                                          json={'title': '离线长项目进程验收'}))
            sessions.append(created['session']['id'])
            cap = expect(browser.get(PREFIX + '/capabilities', headers=identity))
            assert cap['workbench']['tasks'] is True
        assert sessions == ['#00001', '#00001']
        alice, bob = identities
        sid = sessions[0]
        a = upload(browser, alice, sid, 'synthetic-renewable-mechanism.txt', TEXT_A.encode())['document']['id']
        b = upload(browser, alice, sid, 'synthetic-storage-mechanism.pdf', pdf_with_text(TEXT_B))['document']['id']
        other = upload(browser, bob, sid, 'bobby-synthetic-only.txt', b'BOBBY_PRIVATE_FIXTURE_ONLY.')['document']['id']
        a_doc = ready_document(browser, alice, sid, a)
        b_doc = ready_document(browser, alice, sid, b)
        ready_document(browser, bob, sid, other)
        assert a_doc['chunks'][0]['content'] == TEXT_A
        assert b_doc['chunks'][0]['page'] == 1
        assert not calls and query_usage(store) == 0
        questions = [
            {'id': 'q1', 'title': 'Renewable mechanism', 'question': 'Explain the renewable supply mechanism and its limits.', 'document_ids': [a]},
            {'id': 'q2', 'title': 'Storage mechanism', 'question': 'Explain the storage reserve mechanism and its limits.', 'document_ids': [b]},
            {'id': 'q3', 'title': 'Comparison', 'question': 'Compare renewable supply and storage mechanisms, using the prior analysis as leads.',
             'document_ids': [a, b], 'depends_on': ['q1', 'q2']},
        ]
        definition = {'session_id': sid, 'goal': 'Produce a reviewable comparison from the uploaded materials.',
                      'constraints': 'Keep source references. Do not invent scientific results or perform external actions.',
                      'questions': questions}
        task = expect(browser.post(TASKS + '/tasks', headers=alice, json=definition))['task']
        assert task['status'] == 'ready' and len(task['steps']) == 5
        assert not calls
        assert expect(browser.get(TASKS + '/tasks', headers=bob, params={'session_id': sid})) == {'items': []}
        assert browser.get(TASKS + '/task', headers=bob,
                           params={'session_id': sid, 'task_id': task['id']}).status_code == 404
        run = approve(browser, alice, sid, task, 1)
        assert query_usage(store) == 1, 'Gateway did not pre-reserve one approved call'
        expect(browser.post(TASKS + '/tasks/run', headers=alice, json=run))
        assert query_usage(store) == 1, 'Run retry charged twice'
        paused = wait_task(browser, alice, sid, task['id'], 'paused')
        assert paused['reason'] == 'budget_exhausted'
        assert step(paused, 'q1')['status'] == 'completed' and step(paused, 'q2')['status'] == 'ready'
        assert step(paused, 'report')['artifact_version'] == 0 and len(calls) == 1
        first = artifact(browser, alice, sid, paused, 1, step_id='q1')
        assert first['available'] is True and first['references']['document_ids'] == [a]
        assert first['references']['handoff'] == {'required': 0, 'used': 0, 'full': 0, 'excerpted': 0, 'omitted': 0}
        assert 'PROJECT_ALPHA_PRIVATE_SOURCE_MARKER' not in json.dumps(expect(browser.get(
            PREFIX + '/history', headers=bob, params={'session_id': sid})).get('history', []))
        return {'sid': sid, 'task': paused, 'first': first, 'definition': definition, 'run': run, 'documents': [a, b]}


def phase_after_restart(origin, saved, store, calls, temporary):
    with client(origin) as browser:
        alice, bob = login(browser, 'preview'), login(browser, 'bobby')
        sid, task_id = saved['sid'], saved['task']['id']
        current = get_task(browser, alice, sid, task_id)
        assert current['status'] == 'paused' and len(calls) == 1
        assert step(current, 'q1')['request_id'] == step(saved['task'], 'q1')['request_id']
        assert artifact(browser, alice, sid, current, 1, 'q1')['references'] == saved['first']['references']
        expect(browser.post(TASKS + '/tasks/run', headers=alice, json=saved['run']))
        assert query_usage(store) == 1 and len(calls) == 1, 'Restart replay dispatched or billed'
        current = control(browser, alice, sid, current, 'resume')
        assert len(calls) == 1, 'Resume without a new allowance dispatched'
        run = approve(browser, alice, sid, current, 2)
        assert query_usage(store) == 3, 'Remaining two calls were not pre-reserved'
        expect(browser.post(TASKS + '/tasks/run', headers=alice, json=run))
        assert query_usage(store) == 3
        done = wait_task(browser, alice, sid, task_id, 'completed')
        assert len(calls) == 3 and step(done, 'q1')['artifact_version'] == 1
        handoff = step(done, 'q3')['handoff']
        assert handoff['required'] == handoff['used'] == 2 and handoff['omitted'] == 0
        assert handoff['full'] + handoff['excerpted'] == 2
        assert artifact(browser, alice, sid, done, 1, 'q1')['references'] == saved['first']['references']
        assert '已完成步骤的分析线索' in str(calls[-1]), 'Later step received no predecessor analysis'
        report = artifact(browser, alice, sid, done, 1)
        assert report['available'] is True and report['content']
        content = report['content']
        assert content.count('### 本节来源定位') == 3 and '块内字符 [' in content
        assert '第 1 页' in content and 'S 标签仅对应本节来源，不跨节合并' in content
        assert '未追加模型写作' in content
        assert '需要 2 项，实际使用 2 项' in content
        downloaded = temporary / 'synthetic-report-v1.md'
        descriptor = os.open(downloaded, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(content)
        assert downloaded.read_text(encoding='utf-8') == content
        assert browser.get(TASKS + '/task/artifact', headers=bob, params={
            'session_id': sid, 'task_id': task_id, 'version': 1}).status_code == 404

        # A new material and one explicitly edited branch must not rebill q1.
        c = upload(browser, alice, sid, 'synthetic-storage-testing.txt', TEXT_C.encode())['document']['id']
        ready_document(browser, alice, sid, c)
        current = get_task(browser, alice, sid, task_id)
        questions = [dict(row) for row in saved['definition']['questions']]
        questions[1] = {**questions[1], 'question': 'Explain the storage reserve mechanism and missing measurement in the added material.',
                        'document_ids': [saved['documents'][1], c]}
        # The user explicitly approves the added material for the dependent
        # comparison too. The host must never silently widen this scope.
        questions[2] = {**questions[2], 'document_ids': [*saved['documents'], c]}
        updated = expect(browser.post(TASKS + '/tasks/update', headers=alice, json={
            **saved['definition'], 'questions': questions, 'task_id': task_id,
            'expected_revision': current['revision']}))['task']
        assert step(updated, 'q1')['status'] == 'completed'
        assert step(updated, 'q2')['status'] == step(updated, 'q3')['status'] == 'stale'
        refreshed = control(browser, alice, sid, updated, 'refresh')
        assert step(refreshed, 'q1')['request_id'] == step(done, 'q1')['request_id']
        assert step(refreshed, 'q2')['request_id'] != step(done, 'q2')['request_id']
        approve(browser, alice, sid, refreshed, 2)
        assert query_usage(store) == 5
        revised = wait_task(browser, alice, sid, task_id, 'completed')
        assert len(calls) == 5
        assert [step(revised, key)['artifact_version'] for key in ('q1', 'q2', 'q3', 'report')] == [1, 2, 2, 2]
        assert step(revised, 'q3')['handoff']['required'] == step(revised, 'q3')['handoff']['used'] == 2
        assert artifact(browser, alice, sid, revised, 1, 'q1')['references'] == saved['first']['references']
        assert not artifact(browser, alice, sid, revised, 1)['available']
        assert artifact(browser, alice, sid, revised, 2)['available']
        assert query_usage(store, 'bobby') == 0

        # Source withdrawal hides the report without deleting unrelated work.
        expect(browser.post(PREFIX + '/research/documents/delete', headers=alice,
            json={'session_id': sid, 'document_id': saved['documents'][0]}))
        revoked = get_task(browser, alice, sid, task_id)
        hidden = artifact(browser, alice, sid, revoked, 2)
        assert hidden['available'] is False and hidden['content'] is None and hidden['references'] is None
        assert step(revoked, 'q1')['status'] == step(revoked, 'q3')['status'] == 'stale'
        assert step(revoked, 'q2')['status'] == 'completed'
        assert artifact(browser, alice, sid, revoked, 2, 'q2')['available']
        assert len(calls) == 5 and query_usage(store) == 5
        assert expect(browser.get(TASKS + '/tasks', headers=bob, params={'session_id': sid})) == {'items': []}
        assert browser.get(PREFIX + '/research/document', headers=bob,
                           params={'session_id': sid, 'document_id': c}).status_code == 404


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true', help='Keep the synthetic loopback preview alive after acceptance')
    args = parser.parse_args(argv)
    build_dir = (ROOT / 'clients/mobile/dist/build/h5').resolve()
    assert build_dir.is_dir(), 'Build mobile H5 before running this check'
    with tempfile.TemporaryDirectory(prefix='docthinker-project-check-') as directory:
        temporary = Path(directory).resolve()
        temporary.chmod(0o700)
        home, tmp = temporary / 'home', temporary / 'tmp'
        home.mkdir(mode=0o700)
        tmp.mkdir(mode=0o700)
        store = InviteStore.initialize(temporary / 'registry')
        used = set()
        invitations = [store.add_user(name, free_port(used), allow_uploads=True) for name in ('preview', 'bobby')]
        gateway_port = free_port(used)
        owned_ports = [gateway_port, *(int(row['upstream_url'].rsplit(':', 1)[1]) for row in store.list_worker_bindings())]
        origin = f'http://127.0.0.1:{gateway_port}'
        secret = temporary / 'synthetic-model-key'
        descriptor = os.open(secret, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(FAKE_KEY + '\n')
        with fake_research_model(used) as (provider_port, calls):
            environment = {'PATH': os.defpath, 'LANG': 'en_US.UTF-8', 'HOME': str(home), 'TMPDIR': str(tmp),
                'PYTHONPATH': str(ROOT), 'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1',
                'PYTHONDONTWRITEBYTECODE': '1', 'PYTHON_DOTENV_DISABLED': '1',
                SHARING_ACK_ENV: SHARING_ACK_VALUE, 'DOCTHINKER_INVITE_DATA_DIR': str(store.root),
                'DOCTHINKER_INVITE_ORIGIN': origin, 'DOCTHINKER_INVITE_BIND_HOST': '127.0.0.1',
                'DOCTHINKER_INVITE_PORT': str(gateway_port), 'DOCTHINKER_INVITE_WEB_DIR': str(build_dir),
                'DOCTHINKER_RESEARCH_ENABLED': 'true', 'DOCTHINKER_RESEARCH_CONTEXT_ASSIST': 'true',
                'DOCTHINKER_RESEARCH_SEMANTIC': 'off', 'LLM_BINDING_API_KEY_FILE': str(secret),
                'LLM_VLM_HOST': f'http://127.0.0.1:{provider_port}/v1', 'LLM_MODEL': 'offline-runtime-check'}
            checked = subprocess.run([sys.executable, '-m', 'docthinker.server.invite_runtime', '--check'],
                env=environment, cwd=temporary, capture_output=True, timeout=20)
            assert checked.returncode == 0, 'Private synthetic runtime configuration check failed'
            assert not calls and not any(port_open(port) for port in owned_ports)
            with running(environment, temporary, origin, store, owned_ports):
                saved = phase_before_restart(origin, invitations, store, calls)
            assert not any(port_open(port) for port in owned_ports)
            with running(environment, temporary, origin, store, owned_ports):
                phase_after_restart(origin, saved, store, calls, temporary)
                print('PASS: real invitation gateway + two worker subprocesses; ready TXT/PDF inputs; '
                      'three scoped steps; budget-one pause; clean process stop/restart; explicit remaining-two approval; '
                      'first answer not recomputed; direct predecessor references; report v1/v2 with source locators; '
                      'changed branch only regenerated; source withdrawal hides report; idempotent upfront quota; '
                      'same-session-id tenant isolation; five loopback FAKE model calls, zero vendor calls. '
                      'Not semantic-quality, Docker, public-hosting or production-capacity acceptance.', flush=True)
                if args.serve:
                    serve_until_stopped(origin)
        assert not any(port_open(port) for port in used), 'Synthetic listeners leaked'


if __name__ == '__main__':
    main()
