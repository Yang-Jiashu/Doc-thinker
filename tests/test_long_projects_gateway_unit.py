"""Offline invitation-boundary checks; upstream is a MockTransport, not a model."""
import json

import httpx
import pytest

from docthinker.server.invite_app import JSON_FIELDS, MOBILE_ROUTES, create_invite_app
from docthinker.server.invite_store import InviteStore

ORIGIN = 'http://127.0.0.1:8120'
PROJECT = 'a039ad02-a3c6-4794-94da-f1a0a087b424'
PHASE = 'b039ad02-a3c6-4794-94da-f1a0a087b424'
TASK = 'c039ad02-a3c6-4794-94da-f1a0a087b424'
REQUEST = 'd039ad02-a3c6-4794-94da-f1a0a087b424'
BASE = {'session_id': '#00001', 'project_id': PROJECT}
GETS = [
    ('workbench/long-projects', {'session_id': '#00001', 'limit': '20', 'cursor': 'synthetic-cursor'}),
    ('workbench/long-project', BASE),
    ('workbench/long-project/phases', {**BASE, 'limit': '20', 'cursor': 'synthetic-cursor'}),
]
POSTS = [
    ('workbench/long-projects', {'session_id': '#00001', 'request_id': REQUEST,
                               'goal': 'Synthetic household planning', 'acceptance_criteria': ['Owner reviews the result']}),
    ('workbench/long-project/phases', {**BASE, 'expected_revision': 1, 'request_id': REQUEST,
        'title': 'Synthetic phase', 'acceptance_criteria': ['Keep unresolved items visible'],
        'questions': [{'id': 'q1', 'question': 'Draft a checklist, do not execute it',
                       'input_kind': 'personal', 'memory_ids': []}], 'parent_refs': []}),
    ('workbench/long-project/phases/review', {**BASE, 'phase_id': PHASE, 'expected_revision': 2,
        'request_id': REQUEST, 'report_version': 1, 'decision': 'accepted'}),
    ('workbench/long-project/review', {**BASE, 'expected_revision': 3, 'request_id': REQUEST,
        'phase_reports': [{'phase_id': PHASE, 'report_version': 1}], 'decision': 'rejected'}),
    ('workbench/tasks/replan/preview', {'session_id': '#00001', 'task_id': TASK,
        'expected_revision': 1, 'goal': 'Synthetic revised goal', 'constraints': '',
        'questions': [{'id': 'q1', 'question': 'Review the new requirements'}]}),
    ('workbench/tasks/replan', {'session_id': '#00001', 'task_id': TASK,
        'expected_revision': 1, 'goal': 'Synthetic revised goal', 'constraints': '',
        'questions': [{'id': 'q1', 'question': 'Review the new requirements'}],
        'proposal_sha256': 'a' * 64}),
]


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    store = InviteStore.initialize((tmp_path / 'registry').resolve())
    auth = {}
    for name, port in [('alice', 8101), ('bobby', 8102)]:
        invitation = store.add_user(name, port)
        auth[name] = store.register(invitation['invite_code'], name, 'offline-test-password-123')
    calls = []

    async def upstream(request):
        calls.append(request)
        if request.url.path.endswith('/capabilities'):
            return httpx.Response(200, json={'workbench': {'tasks': True, 'long_projects': True}})
        return httpx.Response(200, json={'synthetic_gateway_only': True})

    def create(enabled=True):
        monkeypatch.setenv('DOCTHINKER_RESEARCH_ENABLED', 'true' if enabled else 'false')
        return create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(upstream))

    return create, store, auth, calls


def browser(app, token=None):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN,
        headers={'Origin': ORIGIN, **({'Authorization': 'Bearer ' + token} if token else {})})


async def test_all_project_and_replan_routes_are_explicitly_allowlisted_and_forwarded(gateway):
    create, _, auth, calls = gateway
    app = create()
    async with browser(app, auth['alice']['access_token']) as client:
        for route, params in GETS:
            assert MOBILE_ROUTES['GET', route] == set(params)
            response = await client.get('/api/v1/mobile/' + route, params=params)
            assert response.status_code == 200
            assert dict(calls[-1].url.params) == params
        for route, payload in POSTS:
            assert MOBILE_ROUTES['POST', route] == set()
            assert set(payload) <= JSON_FIELDS[route]
            response = await client.post('/api/v1/mobile/' + route, json=payload)
            assert response.status_code == 200
            assert json.loads(calls[-1].content) == payload
    assert len(calls) == len(GETS) + len(POSTS)
    assert all(call.url.port == 8101 for call in calls)


async def test_project_requests_use_current_worker_identity_not_session_number_or_client_overrides(gateway):
    create, store, auth, calls = gateway
    app = create()
    for name, port in [('alice', 8101), ('bobby', 8102)]:
        async with browser(app, auth[name]['access_token']) as client:
            response = await client.get('/api/v1/mobile/workbench/long-project', params=BASE,
                                        headers={'Cookie': 'private=never-forward'})
        assert response.status_code == 200
        call = calls[-1]
        binding = next(row for row in store.list_worker_bindings() if row['username'] == name)
        assert call.url.port == port
        assert call.headers['authorization'] == 'Bearer ' + binding['upstream_token']
        assert 'cookie' not in call.headers
    count = len(calls)
    async with browser(app, auth['alice']['access_token']) as client:
        assert (await client.get('/api/v1/mobile/workbench/long-project',
            params={**BASE, 'user_id': 'bobby'})).status_code == 400
        route, payload = POSTS[0]
        assert (await client.post('/api/v1/mobile/' + route,
            json={**payload, 'worker_port': 8102})).status_code == 422
    assert len(calls) == count


async def test_closed_profile_masks_capability_and_denies_all_new_project_operations(gateway):
    create, _, auth, calls = gateway
    app = create(False)
    async with browser(app, auth['alice']['access_token']) as client:
        capabilities = await client.get('/api/v1/mobile/capabilities')
        assert capabilities.json()['workbench'] == {'tasks': False, 'long_projects': False, 'artifacts': False}
        for route, params in GETS:
            assert (await client.get('/api/v1/mobile/' + route, params=params)).status_code == 403
        for route, payload in POSTS:
            assert (await client.post('/api/v1/mobile/' + route, json=payload)).status_code == 403
    assert len(calls) == 1


async def test_anonymous_or_identity_injected_requests_do_not_reach_a_worker(gateway):
    create, _, auth, calls = gateway
    app = create()
    async with browser(app) as client:
        for route, params in GETS:
            assert (await client.get('/api/v1/mobile/' + route, params=params)).status_code == 401
    async with browser(app, auth['alice']['access_token']) as client:
        for route, payload in POSTS:
            assert (await client.post('/api/v1/mobile/' + route,
                json={**payload, 'user_id': 'another-user'})).status_code == 422
        assert (await client.get('/api/v1/mobile/workbench/long-projects',
            params=[('session_id', '#00001'), ('session_id', '#00002')])).status_code == 400
    assert not calls


async def test_metadata_creation_and_human_reviews_do_not_consume_model_query_allowance(gateway):
    create, store, auth, calls = gateway
    app = create()
    uid = auth['alice']['user']['id']
    for _ in range(50):
        store.reserve_action(uid, 'query')
    async with browser(app, auth['alice']['access_token']) as client:
        for route, payload in POSTS:
            assert (await client.post('/api/v1/mobile/' + route, json=payload)).status_code == 200
    with store._connect() as db:
        usage = {row['kind']: row['count'] for row in db.execute(
            'SELECT kind,count FROM usage WHERE user_id=?', (uid,))}
    assert usage == {'query': 50, 'workbench_write': len(POSTS)}
    assert len(calls) == len(POSTS)
    assert not any(call.url.path.endswith('/run') or call.url.path.endswith('/query') for call in calls)


async def test_explicit_domain_events_also_require_auth_and_the_research_profile(gateway):
    create, _, auth, calls = gateway
    payload = {'session_id': '#00001', 'confirmed': True, 'event_type': 'decision',
               'action': 'Synthetic decision, not execution',
               'origins': [{'kind': 'memory', 'memory_id': PHASE, 'revision': 1}]}
    closed = create(False)
    async with browser(closed, auth['alice']['access_token']) as client:
        assert (await client.post('/api/v1/mobile/research/events', json=payload)).status_code == 403
    opened = create()
    async with browser(opened, auth['alice']['access_token']) as client:
        assert (await client.post('/api/v1/mobile/research/events', json=payload)).status_code == 200
        assert (await client.get('/api/v1/mobile/research/events',
            params={'session_id': '#00001', 'limit': '20'})).status_code == 200
        assert (await client.post('/api/v1/mobile/research/events',
            json={**payload, 'semantic_verified': True})).status_code == 422
    assert len(calls) == 2
