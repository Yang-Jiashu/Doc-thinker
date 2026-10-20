import test, {afterEach} from 'node:test';
import assert from 'node:assert/strict';
import {
  FIRST_VALUE_PROGRESS_LIMITS as limits,
  loadFirstValueProgress as load,
  saveFirstValueProgress as save,
  clearFirstValueProgress as clear,
} from '../src/services/first-value-progress.js';

const KEY = 'docthinker:first-value-sample:v1';
const scope = {caseId: 'more-context-is-not-enough', version: 1};
const sample = {...scope, sourceId: 'rag-paper', stepId: 'check-source'};
const originalUni = Object.getOwnPropertyDescriptor(globalThis, 'uni');
const originalNow = Date.now;
const NOW = 1791676800000;

afterEach(() => {
  Date.now = originalNow;
  if (originalUni) Object.defineProperty(globalThis, 'uni', originalUni);
  else delete globalThis.uni;
});

function mockStorage(initial = []) {
  Date.now = () => NOW;
  const values = new Map(initial), calls = [];
  globalThis.uni = {
    getStorageSync(key) { calls.push(['get', key]); return values.get(key) ?? ''; },
    setStorageSync(key, value) { calls.push(['set', key]); values.set(key, value); },
    removeStorageSync(key) { calls.push(['remove', key]); values.delete(key); },
  };
  return {values, calls};
}

function stored(overrides = {}) {
  return JSON.stringify({schemaVersion: 1, ...sample, savedAt: NOW, ...overrides});
}

test('an empty read never opts into saving or creates memory', () => {
  const {values, calls} = mockStorage();
  assert.equal(load(scope), null);
  assert.equal(values.size, 0);
  assert.deepEqual(calls, [['get', KEY]]);
});

test('an explicit save persists only bounded sample IDs and returns a bookmark', () => {
  const {values} = mockStorage();
  const result = save(sample);
  assert.equal(result.ok, true);
  assert.deepEqual(result.record, {schemaVersion: 1, ...sample, savedAt: NOW});
  assert.deepEqual(load(scope), result.record);
  assert.equal(values.size, 1);
  assert.equal(typeof values.get(KEY), 'string');
  assert.ok(values.get(KEY).length < limits.serializedChars);
  assert.equal(Object.isFrozen(result.record), true);
  assert.equal(Object.isFrozen(load(scope)), true);
});

test('the returned object cannot mutate a stored sample', () => {
  mockStorage();
  const result = save(sample);
  assert.throws(() => { result.record.sourceId = 'changed'; }, TypeError);
  assert.equal(load(scope).sourceId, sample.sourceId);
});

test('case and version must both match and mismatches never read through', () => {
  const {calls} = mockStorage([[KEY, stored()]]);
  assert.equal(load({...scope, caseId: 'other-case'}), null);
  assert.equal(load({...scope, version: 'next-version'}), null);
  assert.equal(load(scope).sourceId, sample.sourceId);
  assert.ok(calls.every(([method]) => method === 'get'));
});

test('one slot bounds growth across examples and versions', () => {
  const {values} = mockStorage();
  for (let i = 0; i < 100; i += 1) assert.equal(save({...sample, caseId: `case-${i}`}).ok, true);
  assert.equal(values.size, 1);
  assert.equal(load(scope), null);
  assert.equal(load({...scope, caseId: 'case-99'}).caseId, 'case-99');
});

test('save never touches credentials, preferences, sessions or project memory', () => {
  const otherKeys = ['credentials', 'preferences', 'session-123', 'project-memory'];
  const {values, calls} = mockStorage(otherKeys.map(key => [key, `private:${key}`]));
  assert.equal(save(sample).ok, true);
  assert.deepEqual(clear(scope), {ok: true});
  for (const key of otherKeys) assert.equal(values.get(key), `private:${key}`);
  assert.ok(calls.every(([, key]) => key === KEY));
});

test('an explicit clear removes the matching saved sample', () => {
  const {values} = mockStorage([[KEY, stored()]]);
  assert.deepEqual(clear(scope), {ok: true});
  assert.equal(values.has(KEY), false);
  assert.equal(load(scope), null);
  assert.deepEqual(clear(scope), {ok: true});
});

test('clear preserves a sample owned by a different case or version', () => {
  const {values, calls} = mockStorage([[KEY, stored()]]);
  assert.deepEqual(clear({...scope, caseId: 'other-case'}), {ok: true});
  assert.deepEqual(clear({...scope, version: 'future-version'}), {ok: true});
  assert.equal(values.get(KEY), stored());
  assert.ok(calls.every(([method]) => method === 'get'));
});

test('an explicit clear may clean a corrupt dedicated slot', () => {
  const {values} = mockStorage([[KEY, 'broken-json'], ['project-memory', 'keep']]);
  assert.deepEqual(clear(scope), {ok: true});
  assert.equal(values.has(KEY), false);
  assert.equal(values.get('project-memory'), 'keep');
});

test('expired bookmarks are ignored without a background write or deletion', () => {
  const {values, calls} = mockStorage([[KEY, stored({savedAt: NOW - limits.maxAgeMs - 1})]]);
  assert.equal(load(scope), null);
  assert.equal(values.size, 1);
  assert.deepEqual(calls, [['get', KEY]]);
  values.set(KEY, stored({savedAt: NOW - limits.maxAgeMs}));
  assert.ok(load(scope), 'the exact expiry boundary remains valid');
});

test('far-future timestamps are rejected while small clock skew is tolerated', () => {
  const {values} = mockStorage([[KEY, stored({savedAt: NOW + limits.futureClockToleranceMs + 1})]]);
  assert.equal(load(scope), null);
  values.set(KEY, stored({savedAt: NOW + limits.futureClockToleranceMs}));
  assert.ok(load(scope));
});

test('invalid stored timestamps and schema versions fail closed', () => {
  const {values} = mockStorage();
  for (const savedAt of [0, -1, NOW + 0.5, Number.MAX_SAFE_INTEGER + 1, null, '1791676800000']) {
    values.set(KEY, stored({savedAt})); assert.equal(load(scope), null, String(savedAt));
  }
  for (const schemaVersion of [undefined, '1', 0, 2, null]) {
    values.set(KEY, stored({schemaVersion})); assert.equal(load(scope), null);
  }
});

test('malformed and oversized stored payloads are ignored', () => {
  const {values} = mockStorage();
  for (const raw of ['', '{', 'null', '[]', '42', '{}', 'x'.repeat(limits.serializedChars + 1), {schemaVersion: 1, ...sample, savedAt: NOW}]) {
    values.set(KEY, raw); assert.equal(load(scope), null);
  }
});

test('stored records must contain exactly the known fields', () => {
  const {values} = mockStorage();
  for (const extra of [{question: 'private text'}, {accessToken: 'secret'}, {constructor: 'x'}]) {
    values.set(KEY, stored(extra)); assert.equal(load(scope), null);
  }
  values.set(KEY, stored().replace('"schemaVersion":1', '"__proto__":{"polluted":true},"schemaVersion":1'));
  assert.equal(load(scope), null);
  assert.equal({}.polluted, undefined);
});

test('save rejects missing, extra, invalid and overlong identifier fields without writing', () => {
  const {calls} = mockStorage();
  for (const input of [undefined, null, [], {}, {...sample, question: 'private'}, {...sample, sourceId: undefined}]) {
    assert.deepEqual(save(input), {ok: false, reason: 'invalid_input'});
  }
  for (const field of Object.keys(sample)) {
    for (const value of ['', ' private ', 'my question?', '中文内容', 'https://example.test', '../secret', 'a/b', 'a'.repeat(81), '__proto__', 'constructor', 'prototype', 1, null, true, {}]) {
      if (field === 'version' && value === 1) continue;
      assert.deepEqual(save({...sample, [field]: value}), {ok: false, reason: 'invalid_input'}, `${field}:${String(value)}`);
    }
  }
  assert.deepEqual(calls, []);
});

test('the full permitted ID length and plain null-prototype options are supported', () => {
  mockStorage();
  assert.equal(save({...sample, sourceId: 'a'.repeat(80)}).ok, true);
  assert.equal(load(scope).sourceId.length, 80);
  assert.equal(save(Object.assign(Object.create(null), sample)).ok, true);
  assert.equal(load(Object.assign(Object.create(null), scope)).sourceId, sample.sourceId);
});

test('numeric versions are positive safe integers and retain their exact type', () => {
  mockStorage();
  assert.equal(save({...sample, version: 1}).ok, true);
  assert.equal(load(scope).version, 1);
  assert.equal(load({...scope, version: '1'}), null, 'a string version cannot silently match a different version type');
  for (const version of [0, -1, 1.1, Infinity, NaN, Number.MAX_SAFE_INTEGER + 1]) {
    assert.deepEqual(save({...sample, version}), {ok: false, reason: 'invalid_input'});
  }
  assert.equal(save({...sample, version: '2026-10-11'}).ok, true);
  assert.equal(load({...scope, version: '2026-10-11'}).version, '2026-10-11');
});

test('inherited fields and getters are rejected without executing supplied getters', () => {
  const {calls} = mockStorage();
  let executed = false;
  const getter = {...sample};
  Object.defineProperty(getter, 'sourceId', {get() { executed = true; return 'paper'; }, enumerable: true});
  for (const input of [Object.create(sample), getter, {...sample, [Symbol('private')]: 'text'}]) {
    assert.deepEqual(save(input), {ok: false, reason: 'invalid_input'});
  }
  assert.equal(executed, false);
  assert.deepEqual(calls, []);
});

test('invalid read and clear scopes never access storage', () => {
  const {calls} = mockStorage();
  for (const input of [undefined, null, {}, Object.create(scope), {...scope, sourceId: 'unexpected'}, {...scope, version: '__proto__'}]) {
    assert.equal(load(input), null);
    assert.deepEqual(clear(input), {ok: false, reason: 'invalid_input'});
  }
  assert.deepEqual(calls, []);
});

test('unavailable storage is a reported failure, not a successful save', () => {
  delete globalThis.uni;
  assert.equal(load(scope), null);
  assert.deepEqual(save(sample), {ok: false, reason: 'storage_unavailable'});
  assert.deepEqual(clear(scope), {ok: false, reason: 'storage_unavailable'});
  globalThis.uni = {};
  assert.deepEqual(save(sample), {ok: false, reason: 'storage_unavailable'});
});

test('blocked or quota-exhausted storage does not throw or pretend to persist', () => {
  mockStorage();
  globalThis.uni.getStorageSync = () => { throw new Error('blocked'); };
  globalThis.uni.setStorageSync = () => { throw new Error('quota'); };
  assert.equal(load(scope), null);
  assert.deepEqual(save(sample), {ok: false, reason: 'storage_error'});
  assert.deepEqual(clear(scope), {ok: false, reason: 'storage_error'});
});

test('a failed clear is reported so the UI cannot claim the bookmark was removed', () => {
  const {values} = mockStorage([[KEY, stored()]]);
  globalThis.uni.removeStorageSync = () => { throw new Error('blocked'); };
  assert.deepEqual(clear(scope), {ok: false, reason: 'storage_error'});
  assert.equal(values.has(KEY), true);
});

test('malformed platform access and invalid clocks fail safely', () => {
  mockStorage();
  Object.defineProperty(globalThis, 'uni', {configurable: true, get() { throw new Error('unavailable'); }});
  assert.equal(load(scope), null);
  assert.deepEqual(save(sample), {ok: false, reason: 'storage_unavailable'});
  assert.deepEqual(clear(scope), {ok: false, reason: 'storage_unavailable'});
  Object.defineProperty(globalThis, 'uni', {configurable: true, writable: true, value: {setStorageSync() { throw new Error('must not write'); }}});
  Date.now = () => NaN;
  assert.deepEqual(save(sample), {ok: false, reason: 'invalid_input'});
});
