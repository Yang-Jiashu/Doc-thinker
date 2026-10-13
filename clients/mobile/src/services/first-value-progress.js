// An explicit bookmark for a curated example, not project or account memory.
// Call save only after the user chooses to keep their place. Pass fixed sample
// identifiers, never a question, source text, access code, key, or session ID.
const STORAGE_KEY = 'docthinker:first-value-sample:v1';
const SCHEMA_VERSION = 1;
export const FIRST_VALUE_PROGRESS_LIMITS = Object.freeze({
  idChars: 80,
  serializedChars: 1024,
  maxAgeMs: 30 * 24 * 60 * 60 * 1000,
  futureClockToleranceMs: 5 * 60 * 1000,
});

const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/;
const RESERVED_IDS = new Set(['__proto__', 'prototype', 'constructor']);
const INPUT_KEYS = ['caseId', 'version'];
const SAVE_KEYS = [...INPUT_KEYS, 'sourceId', 'stepId'];
const RECORD_KEYS = ['schemaVersion', ...SAVE_KEYS, 'savedAt'];

// Reject inherited fields, getters and extra data instead of serializing an
// arbitrary object. In particular, a caller cannot accidentally save chat data.
function ownDataFields(value, keys) {
  try {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return null;
    const names = Reflect.ownKeys(value);
    if (names.length !== keys.length || names.some(key => !keys.includes(key))) return null;
    const fields = {};
    for (const key of keys) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (!descriptor || !Object.hasOwn(descriptor, 'value')) return null;
      fields[key] = descriptor.value;
    }
    return fields;
  } catch { return null; }
}

const validId = value => typeof value === 'string'
  && value.length <= FIRST_VALUE_PROGRESS_LIMITS.idChars
  && ID_PATTERN.test(value) && !RESERVED_IDS.has(value);
const validField = (key, value) => key === 'version' && Number.isSafeInteger(value) && value > 0
  ? true : validId(value);

function identifiers(options, keys = INPUT_KEYS) {
  const fields = ownDataFields(options, keys);
  return fields && keys.every(key => validField(key, fields[key])) ? fields : null;
}

function storage(methods) {
  try {
    const api = globalThis.uni;
    return api && methods.every(method => typeof api[method] === 'function') ? api : null;
  } catch { return null; }
}

function decode(raw) {
  // We write a bounded JSON string so platform-dependent object/prototype
  // behavior cannot change validation, and never parse a large stored payload.
  if (typeof raw !== 'string' || !raw || raw.length > FIRST_VALUE_PROGRESS_LIMITS.serializedChars) return null;
  try {
    const record = ownDataFields(JSON.parse(raw), RECORD_KEYS);
    if (!record || record.schemaVersion !== SCHEMA_VERSION
      || !SAVE_KEYS.every(key => validField(key, record[key]))
      || !Number.isSafeInteger(record.savedAt) || record.savedAt <= 0) return null;
    return record;
  } catch { return null; }
}

const matches = (record, scope) => record.caseId === scope.caseId && record.version === scope.version;
const failure = reason => ({ok: false, reason});

/** Read a matching, unexpired sample bookmark. This function never writes. */
export function loadFirstValueProgress(options) {
  const scope = identifiers(options), api = storage(['getStorageSync']);
  if (!scope || !api) return null;
  try {
    const record = decode(api.getStorageSync(STORAGE_KEY));
    if (!record || !matches(record, scope)) return null;
    const age = Date.now() - record.savedAt;
    if (!Number.isFinite(age) || age > FIRST_VALUE_PROGRESS_LIMITS.maxAgeMs
      || age < -FIRST_VALUE_PROGRESS_LIMITS.futureClockToleranceMs) return null;
    return Object.freeze({...record});
  } catch { return null; }
}

/** Explicit opt-in only. One sample slot bounds storage; later saves replace it. */
export function saveFirstValueProgress(options) {
  const fields = identifiers(options, SAVE_KEYS);
  if (!fields) return failure('invalid_input');
  const api = storage(['setStorageSync']);
  if (!api) return failure('storage_unavailable');
  try {
    const savedAt = Date.now();
    if (!Number.isSafeInteger(savedAt) || savedAt <= 0) return failure('invalid_input');
    const record = {schemaVersion: SCHEMA_VERSION, ...fields, savedAt};
    api.setStorageSync(STORAGE_KEY, JSON.stringify(record));
    return {ok: true, record: Object.freeze({...record})};
  } catch { return failure('storage_error'); }
}

/** Explicit reset; never delete a valid bookmark belonging to another sample. */
export function clearFirstValueProgress(options) {
  const scope = identifiers(options);
  if (!scope) return failure('invalid_input');
  const api = storage(['getStorageSync', 'removeStorageSync']);
  if (!api) return failure('storage_unavailable');
  try {
    const raw = api.getStorageSync(STORAGE_KEY);
    if (raw === '' || raw === null || raw === undefined) return {ok: true};
    const record = decode(raw);
    if (record && !matches(record, scope)) return {ok: true};
    // A user-requested reset can remove corrupt data, but only this dedicated
    // sample slot, never application preferences, credentials or real memory.
    api.removeStorageSync(STORAGE_KEY);
    return {ok: true};
  } catch { return failure('storage_error'); }
}
