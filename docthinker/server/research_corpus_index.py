"""Query-independent, bounded lexical features; never answer/source caching.

Compact immutable posting buffers avoid a Python object pair for each term hit.
Every query still supplies live source rows. Full content and presentation
metadata bind features to that view; permissions are not inferred from a cache.
"""
from __future__ import annotations

from array import array
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import sys
import threading
from types import MappingProxyType

MAX_FEATURE_BYTES = 32 * 1024 * 1024
MAX_SNAPSHOTS = 4
FEATURE_VERSION = 'lexical-postings-v1'


class CorpusIndexTooLarge(ValueError):
    """Use the unchanged full scan, not a shortened corpus."""


def canonical_source_view(chunks):
    from .research_retrieval import _source_row, _stable_key, MAX_SCAN_CHARS
    entries = [(row, _stable_key(row)) for raw in chunks if (row := _source_row(raw)) is not None]
    entries.sort(key=lambda entry: entry[1])
    fingerprint = hashlib.sha256(json.dumps([FEATURE_VERSION, MAX_SCAN_CHARS,
        [(key, row['filename'], row.get('page')) for row, key in entries]],
        ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    return entries, fingerprint


def _size(values):
    seen, stack, total = set(), list(values), 0
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        total += sys.getsizeof(value)
        if isinstance(value, dict) or isinstance(value, MappingProxyType):
            stack.extend(value.keys())
            stack.extend(value.values())
            if isinstance(value, MappingProxyType):
                # Include the underlying dict's table storage, not a second
                # copy of its keys/values. This is retained-feature accounting.
                total += sys.getsizeof(dict(value))
        elif isinstance(value, (tuple, list, set, frozenset)):
            stack.extend(value)
    return total


@dataclass(frozen=True)
class PreparedResearchCorpus:
    signature: str
    source_indices: tuple
    stable_keys: tuple
    lengths: tuple
    total_length: int
    document_ids: frozenset
    duplicate_count: int
    postings: object = field(repr=False)
    feature_bytes: int

    def hits(self, term):
        buffer = self.postings.get(term, b'')
        values = memoryview(buffer).cast('I')
        return zip(values[::2], values[1::2])


def _prepare(entries, signature, max_bytes):
    from .research_retrieval import lexical_tokens, MAX_SCAN_CHARS
    if array('I').itemsize != 4:
        raise CorpusIndexTooLarge('Compact index is unavailable on this platform')
    identities, texts, documents = set(), set(), set()
    indices, keys, lengths, postings = [], [], [], {}
    duplicates, total_length = 0, 0
    # A conservative incremental allocation bound prevents building a huge
    # Python posting dictionary merely to decide not to retain it afterwards.
    allocated = 4096 + len(entries) * 400
    for source_index, (row, key) in enumerate(entries):
        if allocated > max_bytes:
            raise CorpusIndexTooLarge('Lexical features exceed the memory budget')
        documents.add(row['document_id'])
        identity = (row['document_id'], row['id'])
        normalized = hashlib.sha256(' '.join(row['content'].split()).encode('utf-8')).hexdigest()
        if identity in identities or normalized in texts:
            duplicates += 1
            continue
        identities.add(identity)
        texts.add(normalized)
        index = len(indices)
        indices.append(source_index)
        keys.append(key)
        counter = Counter(lexical_tokens(row['content'][:MAX_SCAN_CHARS]))
        length = sum(counter.values())
        lengths.append(max(1, length))
        total_length += length
        for term, frequency in counter.items():
            if term not in postings:
                postings[term] = array('I')
                allocated += 256 + sys.getsizeof(term)
            postings[term].extend((index, frequency))
            allocated += 12  # eight actual bytes + amortized buffer allowance
        if allocated > max_bytes:
            raise CorpusIndexTooLarge('Lexical features exceed the memory budget')
    buffers = MappingProxyType({term: values.tobytes() for term, values in postings.items()})
    indices, keys, lengths, documents = tuple(indices), tuple(keys), tuple(lengths), frozenset(documents)
    feature_bytes = _size((signature, indices, keys, lengths, total_length, documents, duplicates, buffers)) + 256
    if feature_bytes > max_bytes:
        raise CorpusIndexTooLarge('Lexical features exceed the memory budget')
    return PreparedResearchCorpus(signature, indices, keys, lengths, total_length,
                                  documents, duplicates, buffers, feature_bytes)


def prepare_research_corpus(chunks, *, max_bytes=MAX_FEATURE_BYTES):
    if type(max_bytes) is not int or max_bytes < 1 or max_bytes > MAX_FEATURE_BYTES:
        raise ValueError('Invalid lexical feature budget')
    entries, signature = canonical_source_view(chunks)
    return _prepare(entries, signature, max_bytes)


class PreparedCorpusCache:
    """One service/worker owns the LRU. No global/session-name-only cache.

    Namespace is explicit and the full live corpus signature is mandatory.
    Empty/missing/withdrawn inputs do not inherit previous source membership.
    No question, generated answer, source quotation, or disk file is retained.
    """
    def __init__(self, *, max_bytes=MAX_FEATURE_BYTES, max_snapshots=MAX_SNAPSHOTS):
        if (type(max_bytes) is not int or not 1 <= max_bytes <= MAX_FEATURE_BYTES
                or type(max_snapshots) is not int or not 1 <= max_snapshots <= MAX_SNAPSHOTS):
            raise ValueError('Invalid lexical cache budget')
        self.max_bytes, self.max_snapshots = max_bytes, max_snapshots
        self._items, self._bytes = OrderedDict(), 0
        self._oversized = OrderedDict()
        self._closed = False
        self._lock = threading.Lock()

    def get(self, chunks, *, namespace):
        if not isinstance(namespace, str) or not namespace or len(namespace) > 512:
            raise ValueError('Invalid lexical feature namespace')
        entries, signature = canonical_source_view(chunks)
        key = (namespace, signature)
        with self._lock:
            if self._closed:
                raise ValueError('Lexical feature cache is closed')
            if key in self._oversized:
                self._oversized.move_to_end(key)
                raise CorpusIndexTooLarge('Lexical features exceed the memory budget')
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            # Building under this per-service lock prevents duplicate cold
            # work and multiple oversized pending indexes. Caller uses to_thread.
            try:
                prepared = _prepare(entries, signature, self.max_bytes)
            except CorpusIndexTooLarge:
                # Avoid repeated failed construction for the same verified
                # live snapshot. Only a bounded hash hint is retained, no data.
                self._oversized[key] = None
                while len(self._oversized) > self.max_snapshots:
                    self._oversized.popitem(last=False)
                raise
            while self._items and (len(self._items) >= self.max_snapshots
                                   or self._bytes + prepared.feature_bytes > self.max_bytes):
                _, removed = self._items.popitem(last=False)
                self._bytes -= removed.feature_bytes
            self._items[key] = prepared
            self._bytes += prepared.feature_bytes
            return prepared

    def lookup(self, chunks, *, namespace):
        """Read prepared features only; never wait behind background building.

        A miss uses unchanged full ranking. Busy locks also return a miss so a
        background preparation cannot turn into a foreground wait queue.
        """
        if not isinstance(namespace, str) or not namespace or len(namespace) > 512:
            raise ValueError('Invalid lexical feature namespace')
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if self._closed:
                raise ValueError('Lexical feature cache is closed')
            if not any(key[0] == namespace for key in (*self._items, *self._oversized)):
                return None
        finally:
            self._lock.release()
        _, signature = canonical_source_view(chunks)
        key = (namespace, signature)
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if self._closed:
                raise ValueError('Lexical feature cache is closed')
            if key in self._oversized:
                raise CorpusIndexTooLarge('Lexical features exceed the memory budget')
            prepared = self._items.get(key)
            if prepared is not None:
                self._items.move_to_end(key)
            return prepared
        finally:
            self._lock.release()

    def clear(self):
        with self._lock:
            self._items.clear()
            self._oversized.clear()
            self._bytes = 0

    def close(self):
        # Wait for the one real builder under the same lock, then prevent any
        # queued/late get from republishing features after service shutdown.
        with self._lock:
            self._closed = True
            self._items.clear()
            self._oversized.clear()
            self._bytes = 0

    def info(self):
        with self._lock:
            return {'snapshots': len(self._items), 'feature_bytes': self._bytes,
                    'max_feature_bytes': self.max_bytes, 'max_snapshots': self.max_snapshots}
