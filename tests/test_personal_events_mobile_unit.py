"""Real authenticated ASGI event routes with private SQLite, zero providers."""
import hashlib
import socket
from copy import deepcopy

import pytest

from tests.test_research_mobile_unit import mobile, mobile_client, upload_ready, PREFIX, SID, OTHER_SID

BASE = PREFIX + '/research'


@pytest.fixture(autouse=True)
def offline_events(monkeypatch):
    monkeypatch.setenv('DOCTHINKER_RESEARCH_SEMANTIC', 'off')
    monkeypatch.setenv('DOCTHINKER_RESEARCH_MEMORY_SEMANTIC', 'off')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    def denied(*args, **kwargs):
        raise AssertionError('No network or provider belongs to event editing')
    monkeypatch.setattr(socket, 'getaddrinfo', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


async def memory_origin(browser, text='User explicitly confirmed the current plan.', sid=SID):
    response = await browser.post(BASE + '/memory', json={'session_id': sid, 'kind': 'decision', 'text': text})
    assert response.status_code == 200, response.text
    row = response.json()['memory']
    return row, {'kind': 'memory', 'memory_id': row['id'], 'revision': row['revision']}


def event_body(origin, *, action='User confirmed a private event.', event_type='decision', sid=SID):
    return {'session_id': sid, 'confirmed': True, 'event_type': event_type, 'action': action,
            'origins': [origin], 'conditions': ['Only within the user-confirmed context.']}


async def create_event(browser, origin, **kwargs):
    response = await browser.post(BASE + '/events', json=event_body(origin, **kwargs))
    assert response.status_code == 200, response.text
    assert response.headers['cache-control'] == 'no-store'
    return response.json()['event']


@pytest.mark.parametrize('event_type,action', [
    ('decision', 'Work delivery review moved to Friday.'),
    ('meeting', 'Household appointment moved to the afternoon.'),
    ('preference', 'Every code execution needs fresh explicit permission.'),
])
async def test_normal_mobile_mount_supports_three_personal_domains(mobile, event_type, action):
    async with mobile_client(mobile) as browser:
        # Exercise the normal app factory, including lazy FastAPI router
        # mounting, rather than injecting a test-only route or inspecting it.
        mounted = await browser.get(BASE + '/events', params={'session_id': SID})
        assert mounted.status_code == 200 and mounted.json()['items'] == []
        _, origin = await memory_origin(browser, action)
        registered = await browser.post(BASE + '/entities', json={'session_id': SID, 'confirmed': True,
            'kind': 'person', 'name': 'User', 'aliases': ['Me']})
        assert registered.status_code == 200
        actor = registered.json()['entity']
        body = event_body(origin, action=action, event_type=event_type)
        body['actor_id'] = actor['id']
        response = await browser.post(BASE + '/events', json=body)
        assert response.status_code == 200, response.text
        row = response.json()['event']
        assert row['actor'] == actor and row['occurred_from'] is None
        assert row['semantic_verified'] is False and row['execution_authorized'] is False
        found = await browser.get(BASE + '/events', params={'session_id': SID, 'entity_id': actor['id']})
        assert found.json()['items'] == [row]
        resolved = await browser.get(BASE + '/entities/resolve', params={'session_id': SID, 'kind': 'person', 'name': 'me'})
        assert resolved.json()['status'] == 'resolved'
    mobile.model.assert_not_awaited()


async def test_auth_precedes_schema_validation_and_no_store_is_created(mobile):
    async with mobile_client(mobile, token=None) as browser:
        response = await browser.post(BASE + '/events', content=b'{bad json', headers={'content-type': 'application/json'})
        assert response.status_code == 401
    assert not mobile.service.store.path.exists()
    mobile.model.assert_not_awaited()


async def test_wrong_project_cannot_read_entities_events_or_bind_foreign_origins(mobile):
    async with mobile_client(mobile) as browser:
        _, origin = await memory_origin(browser)
        row = await create_event(browser, origin)
        assert (await browser.get(BASE + '/event', params={'session_id': OTHER_SID, 'event_id': row['id']})).status_code == 404
        assert (await browser.get(BASE + '/events', params={'session_id': OTHER_SID})).json()['items'] == []
        assert (await browser.post(BASE + '/events', json=event_body(origin, sid=OTHER_SID))).status_code == 422
        assert (await browser.post(BASE + '/events/snapshots', json={'session_id': OTHER_SID, 'event_ids': [row['id']]})).status_code == 404
    mobile.model.assert_not_awaited()


async def test_origin_withdrawal_redacts_action_and_conditions_on_every_public_read(mobile):
    secret = 'PRIVATE-EVENT-NOT-RETURNED-AFTER-WITHDRAWAL'
    async with mobile_client(mobile) as browser:
        document = await upload_ready(mobile, browser, text='Confirmed private schedule; execution still needs consent.', filename='schedule.txt')
        chunk = mobile.service.store.source_chunks(SID)[0]
        origin = {'kind': 'source', 'source_id': chunk['id'], 'span': [0, len(chunk['text'])],
                  'sha256': hashlib.sha256(chunk['text'].encode()).hexdigest()}
        row = await create_event(browser, origin, action=secret)
        other = await create_event(browser, origin, action='A related follow-up.')
        relation = await browser.post(BASE + '/event-relations', json={'session_id': SID, 'confirmed': True,
            'source_id': row['id'], 'target_id': other['id'], 'relation': 'reported_cause', 'conditions': [secret]})
        assert relation.status_code == 200
        assert (await browser.post(BASE + '/documents/delete', json={'session_id': SID, 'document_id': document})).status_code == 200
        requests = [(BASE + '/event', {'session_id': SID, 'event_id': row['id']}),
                    (BASE + '/events', {'session_id': SID}),
                    (BASE + '/event/history', {'session_id': SID, 'event_id': row['id']}),
                    (BASE + '/event-relations', {'session_id': SID})]
        for path, params in requests:
            response = await browser.get(path, params=params)
            assert response.status_code == 200 and secret not in response.text
            assert 'source_unavailable' in response.text
        assert (await browser.post(BASE + '/events', json=event_body(origin))).status_code == 422
    mobile.model.assert_not_awaited()


async def test_revision_history_snapshots_and_restarted_store_keep_actual_binding(mobile, monkeypatch):
    from docthinker.server.research_service import ResearchService
    async with mobile_client(mobile) as browser:
        _, origin = await memory_origin(browser)
        row = await create_event(browser, origin, action='Original appointment.')
        revision = {**event_body(origin, action='Corrected appointment.'), 'event_id': row['id'], 'expected_revision': 1,
                    'occurred_from': 100, 'occurred_until': 120}
        result = await browser.post(BASE + '/events/revise', json=revision)
        assert result.status_code == 200, result.text
        assert result.json()['event']['revision'] == 2
        assert (await browser.post(BASE + '/events/revise', json=revision)).status_code == 409
        restarted = ResearchService(mobile.root)
        monkeypatch.setattr(mobile.state, 'research_service', restarted)
        current = await browser.post(BASE + '/events/snapshots', json={'session_id': SID, 'event_ids': [row['id']]})
        assert current.json()['events'][row['id']]['revision'] == 2
        history = await browser.get(BASE + '/event/history', params={'session_id': SID, 'event_id': row['id']})
        assert [item['historical'] for item in history.json()['items']] == [False, True]
    mobile.model.assert_not_awaited()


async def test_reported_cause_and_hypothesis_cannot_be_submitted_as_verified(mobile):
    async with mobile_client(mobile) as browser:
        _, origin = await memory_origin(browser)
        a, b = await create_event(browser, origin), await create_event(browser, origin)
        for kind in ('reported_cause', 'hypothesis', 'related_to'):
            body = {'session_id': SID, 'confirmed': True, 'source_id': a['id'], 'target_id': b['id'], 'relation': kind}
            response = await browser.post(BASE + '/event-relations', json=body)
            assert response.status_code == 200
            assert response.json()['relation']['relation'] == kind
            assert response.json()['relation']['causality_verified'] is False
            assert (await browser.post(BASE + '/event-relations', json={**body, 'causality_verified': True})).status_code == 422
        assert (await browser.post(BASE + '/event-relations', json={**body, 'relation': 'verified_cause'})).status_code == 422
    mobile.model.assert_not_awaited()


async def test_strict_confirmation_origin_and_operator_fields_fail_closed(mobile):
    async with mobile_client(mobile) as browser:
        _, origin = await memory_origin(browser)
        base = event_body(origin)
        variants = [{**base, 'confirmed': value} for value in (False, 1, 'true', None)]
        variants += [{key: value for key, value in base.items() if key != 'confirmed'},
                     {**base, 'occurred_from': True}, {**base, 'execute': True},
                     {**base, 'origins': []}, {**base, 'origins': [{**origin, 'revision': True}]},
                     {**base, 'conditions': ['too many'] * 17}]
        for body in variants:
            response = await browser.post(BASE + '/events', json=deepcopy(body))
            assert response.status_code == 422, response.text
            assert 'current plan' not in response.text
        assert (await browser.get(BASE + '/events', params={'session_id': SID})).json()['items'] == []
    mobile.model.assert_not_awaited()


async def test_ambiguous_names_are_visible_candidates_not_auto_merged(mobile):
    async with mobile_client(mobile) as browser:
        for name in ('Alex Chen', 'Alex Wang'):
            response = await browser.post(BASE + '/entities', json={'session_id': SID, 'confirmed': True,
                'kind': 'person', 'name': name, 'aliases': ['Alex']})
            assert response.status_code == 200
        response = await browser.get(BASE + '/entities/resolve', params={'session_id': SID, 'kind': 'person', 'name': 'alex'})
        assert response.json()['status'] == 'ambiguous' and response.json()['auto_merged'] is False
        assert len(response.json()['candidates']) == 2
    mobile.model.assert_not_awaited()


async def test_worker_store_is_resolved_per_request_without_cross_owner_cache(mobile, tmp_path, monkeypatch):
    from docthinker.server.research_service import ResearchService
    async with mobile_client(mobile) as browser:
        _, origin = await memory_origin(browser)
        row = await create_event(browser, origin, action='Private to worker one.')
        other_root = tmp_path / 'other-private-worker'
        other_root.mkdir(mode=0o700)
        other = ResearchService(other_root)
        monkeypatch.setattr(mobile.state, 'research_service', other)
        assert (await browser.get(BASE + '/events', params={'session_id': SID})).json()['items'] == []
        hidden = await browser.get(BASE + '/event', params={'session_id': SID, 'event_id': row['id']})
        assert hidden.status_code == 404 and 'Private to worker one' not in hidden.text
        denied = await browser.post(BASE + '/events', json=event_body(origin))
        assert denied.status_code == 422
    mobile.model.assert_not_awaited()
