"""Embedding retry lifecycle only: fake SDK, fake sleep, no network or cost."""
import asyncio
import importlib
import socket
from types import SimpleNamespace

import numpy as np
import pytest

server = importlib.import_module('docthinker.server.app')
BACKOFF = [3, 6, 12, 24, 48, 60, 60]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('Embedding lifecycle tests must not make any network request')
    for owner, name in ((socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                        (socket, 'create_connection'), (socket, 'getaddrinfo')):
        monkeypatch.setattr(owner, name, forbidden)


def fake_factory(monkeypatch, create, sleeps):
    settings = SimpleNamespace(embed_model='synthetic', embed_base_url='https://provider.example.invalid/v1', embed_dim=2)
    monkeypatch.setattr(server.state, 'settings', settings)
    monkeypatch.setattr(server, 'get_embed_client',
        lambda _: SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    async def fake_sleep(delay):
        sleeps.append(delay)
    monkeypatch.setattr(server.asyncio, 'sleep', fake_sleep)
    return server._get_embedding_func()


@pytest.mark.asyncio
async def test_eighth_rate_limit_raises_same_error_without_useless_terminal_sleep(monkeypatch):
    calls, sleeps = [], []
    failure = RuntimeError('synthetic 429 rate limit')
    async def create(**kwargs):
        calls.append(kwargs)
        raise failure
    embedding = await fake_factory(monkeypatch, create, sleeps)
    with pytest.raises(RuntimeError) as result:
        await embedding(['unchanged input'])
    assert result.value is failure
    assert len(calls) == 8 and sleeps == BACKOFF
    assert all(call == {'model': 'synthetic', 'input': ['unchanged input']} for call in calls)
    assert sum(sleeps) == 213  # Same first seven backoffs, no eighth 60s wait.


@pytest.mark.asyncio
@pytest.mark.parametrize('failures', [0, 1, 3, 7])
async def test_early_or_eighth_success_preserves_vectors_request_count_and_prior_backoff(monkeypatch, failures):
    calls, sleeps = [], []
    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) <= failures:
            raise RuntimeError('synthetic 429 quota limit')
        return SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[3.5, 4.25]),
                                     SimpleNamespace(index=0, embedding=[1.0, 2.0])])
    embedding = await fake_factory(monkeypatch, create, sleeps)
    result = await embedding(['first', 'second'])
    np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.5, 4.25]], dtype=np.float32))
    assert result.dtype == np.float32 and len(calls) == failures + 1
    assert sleeps == BACKOFF[:failures]
    assert all(call == {'model': 'synthetic', 'input': ['first', 'second']} for call in calls)


@pytest.mark.asyncio
async def test_provider_cancellation_propagates_without_sleep_or_another_attempt(monkeypatch):
    calls, sleeps = [], []
    cancelled = asyncio.CancelledError('synthetic cancellation')
    async def create(**kwargs):
        calls.append(kwargs)
        raise cancelled
    embedding = await fake_factory(monkeypatch, create, sleeps)
    with pytest.raises(asyncio.CancelledError) as result:
        await embedding(['unchanged input'])
    assert result.value is cancelled and len(calls) == 1 and sleeps == []


@pytest.mark.asyncio
async def test_cancellation_during_backoff_never_enters_next_paid_attempt(monkeypatch):
    calls, sleeps = [], []
    async def create(**kwargs):
        calls.append(kwargs)
        raise RuntimeError('synthetic 429 rate limit')
    embedding = await fake_factory(monkeypatch, create, sleeps)
    async def cancelled_sleep(delay):
        sleeps.append(delay)
        raise asyncio.CancelledError()
    monkeypatch.setattr(server.asyncio, 'sleep', cancelled_sleep)
    with pytest.raises(asyncio.CancelledError):
        await embedding(['unchanged input'])
    assert len(calls) == 1 and sleeps == [3]


@pytest.mark.asyncio
async def test_non_rate_failure_does_not_change_into_retry_or_delay(monkeypatch):
    calls, sleeps = [], []
    failure = RuntimeError('synthetic unrelated failure')
    async def create(**kwargs):
        calls.append(kwargs)
        raise failure
    embedding = await fake_factory(monkeypatch, create, sleeps)
    with pytest.raises(RuntimeError) as result:
        await embedding(['unchanged input'])
    assert result.value is failure and len(calls) == 1 and sleeps == []
