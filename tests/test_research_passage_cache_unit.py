"""Private derived vectors are exact, bounded, isolated, and never source authority."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from docthinker.server import research_passage_cache as cache
from docthinker.server import research_semantic as semantic


def vectors(rows=2):
    matrix = np.arange(1, rows * 384 + 1, dtype=np.float32).reshape(rows, 384)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def fetch(root, identity=None, *, namespace='tenant-a/#00001', matrix=None, **options):
    calls = []
    matrix = vectors() if matrix is None else matrix
    def encode():
        calls.append(True)
        return matrix
    result, status = cache.PrivatePassageCache(root, **options).matrix(
        namespace, {'model': 'manifest-a', 'text': 'digest-a'} if identity is None else identity,
        rows=len(matrix), dimension=384, encode=encode)
    return result, status, len(calls)


def test_exact_bits_new_instance_private_files_and_no_plaintext(tmp_path):
    root = tmp_path / 'private'
    first, status, calls = fetch(root)
    assert (status, calls) == ('stored', 1)
    second, status, calls = fetch(root)
    assert (status, calls) == ('hit', 0)
    assert first.tobytes() == second.tobytes()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    for path in root.rglob('*'):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
        if path.is_file():
            assert b'manifest-a' not in path.read_bytes()
            assert b'digest-a' not in path.read_bytes()


@pytest.mark.parametrize('identity,namespace', [
    ({'model': 'manifest-b', 'text': 'digest-a'}, 'tenant-a/#00001'),
    ({'model': 'manifest-a', 'text': 'digest-b'}, 'tenant-a/#00001'),
    ({'model': 'manifest-a', 'text': 'digest-a'}, 'tenant-b/#00001'),
    ({'model': 'manifest-a', 'text': 'digest-a'}, 'tenant-a/#00002'),
    ({'model': 'manifest-a', 'text': 'digest-a', 'order': [2, 1]}, 'tenant-a/#00001'),
])
def test_identity_and_scope_changes_miss(tmp_path, identity, namespace):
    fetch(tmp_path / 'private')
    assert fetch(tmp_path / 'private', identity, namespace=namespace)[1:] == ('stored', 1)


@pytest.mark.parametrize('poison', ['truncated', 'payload', 'header', 'symlink', 'hardlink', 'mode', 'fifo'])
def test_untrusted_entry_never_supplies_vectors(tmp_path, poison):
    root = tmp_path / 'private'
    fetch(root)
    entry = next(root.rglob('*.pv'))
    if poison == 'symlink':
        other = tmp_path / 'outside'
        other.write_bytes(entry.read_bytes())
        entry.unlink()
        entry.symlink_to(other)
    elif poison == 'hardlink':
        os.link(entry, tmp_path / 'other-link')
    elif poison == 'mode':
        entry.chmod(0o644)
    elif poison == 'fifo':
        entry.unlink()
        os.mkfifo(entry, 0o600)
    else:
        data = bytearray(entry.read_bytes())
        if poison == 'truncated':
            data = data[:20]
        else:
            data[100 if poison == 'header' else -40] ^= 1
        entry.write_bytes(data)
    fresh = vectors() * np.float32(-1)
    matrix, _, calls = fetch(root, matrix=fresh)
    assert calls == 1
    assert matrix.tobytes() == fresh.tobytes()


@pytest.mark.parametrize('target', ['root', 'scope', 'auth', 'lock', 'parent'])
def test_symlinks_are_not_followed(tmp_path, target):
    root = tmp_path / 'private'
    if target in ('scope', 'auth', 'lock'):
        fetch(root)
        path = {'scope': root / hashlib.sha256(b'tenant-a/#00001').hexdigest(),
                'auth': root / '.auth', 'lock': root / '.lock'}[target]
        other = tmp_path / 'outside'
        if path.is_dir():
            path.rename(other)
        else:
            other.write_bytes(path.read_bytes())
            other.chmod(0o600)
            path.unlink()
        path.symlink_to(other, target_is_directory=other.is_dir())
    elif target == 'parent':
        outside = tmp_path / 'outside'
        outside.mkdir(mode=0o700)
        (tmp_path / 'linked').symlink_to(outside, target_is_directory=True)
        root = tmp_path / 'linked' / 'private'
    else:
        outside = tmp_path / 'outside'
        outside.mkdir(mode=0o700)
        root.symlink_to(outside, target_is_directory=True)
    # A lock is irrelevant to an authenticated read; exercise its cold path.
    matrix, status, calls = fetch(root, {'cold': True} if target == 'lock' else None)
    assert (status, calls) == ('unavailable', 1)
    assert matrix.tobytes() == vectors().tobytes()


def test_write_and_maintenance_failure_never_encode_twice(tmp_path, monkeypatch):
    root = tmp_path / 'private'
    fetch(root)
    monkeypatch.setattr(cache, '_trim', lambda *_: (_ for _ in ()).throw(OSError('disk')))
    assert fetch(root)[1:] == ('hit', 0)
    assert fetch(root, {'changed': True})[1:] == ('unavailable', 1)
    monkeypatch.setattr(cache, '_store', lambda *_: (_ for _ in ()).throw(OSError('disk')))
    assert fetch(root, {'changed': 2})[1:] == ('write_unavailable', 1)


@pytest.mark.parametrize('error', [ValueError('encoder failure'), asyncio.CancelledError()])
def test_encoder_failure_or_cancellation_propagates_once_and_releases_lock(tmp_path, error):
    root, calls = tmp_path / 'private', []
    def encode():
        calls.append(True)
        raise error
    with pytest.raises(type(error)):
        cache.PrivatePassageCache(root).matrix('p', {}, rows=2, dimension=384, encode=encode)
    assert len(calls) == 1
    assert not list(root.rglob('*.pv'))
    assert fetch(root)[1:] == ('stored', 1)


def test_atomic_interrupted_write_does_not_leave_partial_cache(tmp_path, monkeypatch):
    root = tmp_path / 'private'
    original = cache.os.replace
    monkeypatch.setattr(cache.os, 'replace', lambda *a, **k: (_ for _ in ()).throw(asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        fetch(root)
    assert not list(root.rglob('*.pv'))
    assert not list(root.rglob('*.tmp'))
    monkeypatch.setattr(cache.os, 'replace', original)
    assert fetch(root)[1:] == ('stored', 1)


def test_capacity_lru_and_orphan_cleanup(tmp_path):
    root = tmp_path / 'private'
    for i in range(4):
        assert fetch(root, {'i': i}, max_entries=2)[1:] == ('stored', 1)
    assert len(list(root.rglob('*.pv'))) == 2
    assert fetch(root, {'i': 3}, max_entries=2)[1:] == ('hit', 0)
    assert fetch(root, {'i': 0}, max_entries=2)[1:] == ('stored', 1)
    # Path.glob trailing-slash filtering differs across supported Python
    # versions; never mistake the root's .lock/.auth files for a scope folder.
    scope = next(path for path in root.iterdir() if path.is_dir())
    orphan = scope / '.abandoned.tmp'
    orphan.write_bytes(b'abandoned')
    orphan.chmod(0o600)
    fetch(root, {'i': 5}, max_entries=2)
    assert not orphan.exists()
    fetch(root, {'large': True}, max_bytes=1024)
    assert sum(p.stat().st_size for p in root.rglob('*.pv')) <= 1024


def test_expired_entry_reencodes_current_batch(tmp_path):
    root = tmp_path / 'private'
    fetch(root)
    entry = next(root.rglob('*.pv'))
    old = time.time() - cache.MAX_IDLE_SECONDS - 1
    os.utime(entry, (old, old))
    assert fetch(root)[1:] == ('stored', 1)


def test_warm_session_does_not_wait_for_other_sessions_cold_encoder(tmp_path):
    root = tmp_path / 'private'
    expected, _, _ = fetch(root)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    outcomes = []
    def encode():
        entered.set()
        assert release.wait(5)
        return vectors()
    def cold():
        outcomes.append(cache.PrivatePassageCache(root).matrix('tenant-b/session',
            {'different': True}, rows=2, dimension=384, encode=encode))
    worker = threading.Thread(target=cold)
    worker.start()
    try:
        assert entered.wait(3)
        def warm():
            outcomes.append(fetch(root))
            finished.set()
        reader = threading.Thread(target=warm)
        reader.start()
        assert finished.wait(1), 'warm hit queued behind unrelated cold encoder'
        assert not release.is_set()
        matrix, status, count = outcomes[0]
        assert (status, count) == ('hit', 0)
        assert matrix.tobytes() == expected.tobytes()
        reader.join(1)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()


PROCESS = r'''
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from docthinker.server.research_passage_cache import PrivatePassageCache
root, counter = map(Path, sys.argv[1:3])
def encode():
    fd = os.open(counter, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.write(fd, b'encoded\n'); os.close(fd)
    time.sleep(float(sys.argv[3]))
    x=np.arange(1,769,dtype=np.float32).reshape(2,384)
    return x/np.linalg.norm(x,axis=1,keepdims=True)
x,status=PrivatePassageCache(root).matrix('tenant/session',{'batch':'same'},rows=2,dimension=384,encode=encode)
print(json.dumps({'status':status,'sha':hashlib.sha256(x.tobytes()).hexdigest()}))
'''


def process(root, counter, delay='0'):
    return subprocess.Popen([sys.executable, '-B', '-c', PROCESS, str(root), str(counter), delay],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(child):
    out, err = child.communicate(timeout=20)
    assert child.returncode == 0, err
    return json.loads(out)


def test_new_process_reuses_exact_bits_and_concurrent_processes_encode_once(tmp_path):
    root, counter = tmp_path / 'private', tmp_path / 'count'
    first, second = process(root, counter, '.2'), process(root, counter, '.2')
    a, b = finish(first), finish(second)
    assert {a['status'], b['status']} == {'stored', 'hit'}
    assert a['sha'] == b['sha']
    third = finish(process(root, counter))
    assert third == {'status': 'hit', 'sha': a['sha']}
    assert counter.read_bytes() == b'encoded\n'


class Tokenizer:
    def encode(self, text, **kwargs):
        return SimpleNamespace(offsets=[(i, i + 1) for i in range(len(text))])


class Encoder:
    def __init__(self):
        self.calls = []
    def embed(self, texts, **kwargs):
        self.calls.append(list(texts))
        for text in texts:
            x = np.zeros(384, dtype=np.float32)
            x[0 if 'target' in text else 1] = 1
            x[2] = .1
            yield x


def clear():
    semantic._MODELS.clear()
    semantic._VECTORS.clear()
    semantic._LOGICAL_VECTORS.clear()


@pytest.fixture
def retrievers(tmp_path, monkeypatch):
    model_path = tmp_path / 'model'
    model_path.mkdir()
    (model_path / 'docthinker-model.json').write_text('{}')
    monkeypatch.setattr(semantic, '_model_manifest', lambda _: ({}, 'manifest-a'))
    models = []
    def load(_):
        model = Encoder()
        models.append(model)
        return model, Tokenizer()
    monkeypatch.setattr(semantic, '_load_model', load)
    def new(root=True):
        clear()
        return semantic.LocalSemanticRetriever(model_path,
            private_cache_root=tmp_path / 'private' if root else None)
    yield new, models, model_path
    clear()


def snapshot():
    return {k: (v[0].tobytes(), v[1], v[2]) for k, v in semantic._VECTORS.items() if k[2] == 'passage'}


def test_live_semantic_exact_equivalence_no_withdrawn_or_generated_authority(retrievers):
    new, models, _ = retrievers
    rows = [{'id': 'a', 'document_id': 'doc', 'content': 'background ' * 40 + 'target evidence'},
            {'id': 'b', 'document_id': 'doc', 'content': 'unrelated'}]
    original = new(False).semantic_scores('target', rows, session_id='tenant/session')
    before = snapshot()
    cold = new().semantic_scores('target', rows, session_id='tenant/session')
    assert cold['scores'] == original['scores'] and cold['spans'] == original['spans']
    assert snapshot() == before
    warm = new().semantic_scores('target', rows, session_id='tenant/session')
    assert warm['scores'] == original['scores'] and warm['spans'] == original['spans']
    assert snapshot() == before
    assert warm['trace']['private_passage_cache']['status'] == 'hit'
    assert warm['trace']['passage_windows_encoded'] == 0
    assert models[-1].calls == [['query: target']]
    withdrawn = new().semantic_scores('target', rows[1:], session_id='tenant/session')
    assert 'a' not in withdrawn['scores'] and 'a' not in withdrawn['spans']
    generated = [{**rows[0], 'is_generated': True}]
    empty = new().semantic_scores('target', generated, session_id='tenant/session')
    assert empty['scores'] == {} and empty['spans'] == {}


@pytest.mark.parametrize('change', ['tenant', 'text', 'model', 'manifest', 'order', 'windows', 'normalization'])
def test_semantic_exact_identity_changes_do_not_reuse_batch(retrievers, monkeypatch, change):
    new, models, path = retrievers
    rows = [{'id': 'a', 'document_id': 'doc', 'content': 'target ' * 60},
            {'id': 'b', 'document_id': 'doc', 'content': 'unrelated'}]
    new().prepare_documents(rows, session_id='tenant/session')
    if change == 'text': rows[0]['content'] += '!'
    if change == 'manifest': (path / 'docthinker-model.json').write_text('{"updated":true}')
    if change == 'order': rows.reverse()
    if change == 'windows': monkeypatch.setattr(semantic, 'PASSAGE_WINDOW_STRIDE', semantic.PASSAGE_WINDOW_STRIDE - 1)
    if change == 'normalization':
        original = semantic._normalized
        def updated(*args):
            return original(*args)
        monkeypatch.setattr(semantic, '_normalized', updated)
    if change == 'model': monkeypatch.setattr(semantic, '_model_manifest', lambda _: ({}, 'manifest-b'))
    result = new().prepare_documents(rows, session_id='other/session' if change == 'tenant' else 'tenant/session')
    assert result['trace']['private_passage_cache']['status'] == 'stored'
    assert result['trace']['passage_windows_encoded'] > 0
    assert models[-1].calls


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['auto','local'])
async def test_service_factory_keeps_cache_private_to_workspace_and_semantic_namespace(tmp_path,monkeypatch,mode):
    from docthinker.server import research_service as service_module
    made=[]
    class Prepared:
        def __init__(self,*,private_cache_root):
            self.root=private_cache_root
            self.namespaces=[]
            made.append(self)
        def prepare_documents(self,rows,*,session_id):
            self.namespaces.append(session_id)
            return {'scores':{},'spans':{},'trace':{'status':'ready','provider_calls':0,'llm_tokens':0}}
    monkeypatch.setenv('DOCTHINKER_RESEARCH_SEMANTIC',mode)
    monkeypatch.setattr(service_module,'LocalSemanticRetriever',Prepared)
    roots=[tmp_path/'tenant-a',tmp_path/'tenant-b']
    for root in roots:root.mkdir(mode=0o700)
    services=[service_module.ResearchService(root,evidence_reranker=None,query_translator=None)
              for root in roots]
    try:
        for service in services:
            await service._semantic_operation('#00001',[{'id':'one','document_id':'doc','content':'Current evidence.'}])
        assert [model.root for model in made]==[root.resolve()/'_system'/'research'/'passage-vectors' for root in roots]
        assert all(model.namespaces and '#00001' in model.namespaces[0] for model in made)
        assert made[0].namespaces[0]!=made[1].namespaces[0]
        assert not any(model.root.exists() for model in made), 'factory must not encode, read, or write cache entries'
    finally:
        for service in services:await service.stop()


@pytest.mark.asyncio
async def test_service_off_or_injected_retriever_does_not_create_disk_cache(tmp_path,monkeypatch):
    from docthinker.server import research_service as service_module
    monkeypatch.setattr(service_module,'LocalSemanticRetriever',
        lambda **kwargs:pytest.fail('Off/injected retriever must not build a cache factory'))
    (tmp_path/'off').mkdir(mode=0o700)
    (tmp_path/'injected').mkdir(mode=0o700)
    monkeypatch.setenv('DOCTHINKER_RESEARCH_SEMANTIC','off')
    off=service_module.ResearchService(tmp_path/'off',evidence_reranker=None,query_translator=None)
    custom=object()
    monkeypatch.setenv('DOCTHINKER_RESEARCH_SEMANTIC','local')
    injected=service_module.ResearchService(tmp_path/'injected',semantic_retriever=custom,
        evidence_reranker=None,query_translator=None)
    try:
        assert off._semantic is None and injected._semantic is custom
        assert not (tmp_path/'off'/'_system'/'research'/'passage-vectors').exists()
        assert not (tmp_path/'injected'/'_system'/'research'/'passage-vectors').exists()
    finally:
        await off.stop()
        await injected.stop()
