"""Independent cache boundaries; invented corpora, no models or private files.

The byte budget below is retained immutable-feature accounting, not total RSS
or a promise that live caller text / cold-build temporaries take no memory.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields
import inspect
import threading

import pytest

from docthinker.server import research_corpus_index as corpus
from docthinker.server.research_retrieval import MAX_SCAN_CHARS, lexical_tokens


def row(content="The archive retains approved revisions.", *, identifier="c1",
        document="d1", **metadata):
    return {"id": identifier, "document_id": document, "content": content,
            "filename": "invented.pdf", "ordinal": 0, "page": 1, **metadata}


def test_same_visible_session_id_different_private_namespaces_do_not_share():
    cache = corpus.PreparedCorpusCache()
    rows = [row()]
    first = cache.get(rows, namespace="tenant-a::#00001")
    second = cache.get(rows, namespace="tenant-b::#00001")
    assert first == second and first is not second
    assert cache.get(rows, namespace="tenant-a::#00001") is first
    assert cache.info()["snapshots"] == 2


def test_same_worker_different_projects_do_not_share_snapshot_objects():
    cache = corpus.PreparedCorpusCache()
    assert cache.get([row()], namespace="worker::#00001") is not cache.get(
        [row()], namespace="worker::#00002")


def test_full_current_scope_binds_membership_and_cannot_revive_removed_rows():
    cache = corpus.PreparedCorpusCache()
    a, b = row(), row("Inventory records require approval.", identifier="c2", document="d2")
    both = cache.get([a, b], namespace="worker::#00001")
    scoped = cache.get([a], namespace="worker::#00001")
    empty = cache.get([], namespace="worker::#00001")
    assert both.signature != scoped.signature != empty.signature
    assert both.document_ids == {"d1", "d2"}
    assert scoped.document_ids == {"d1"}
    assert empty.source_indices == () and not empty.document_ids
    assert list(empty.hits("archive")) == []
    assert cache.get([], namespace="worker::#00001") is empty


@pytest.mark.parametrize("change", [
    {"is_generated": True}, {"is_generated": "true"}, {"is_discovered": "1"},
    {"provenance": "llm"}, {"provenance": "self_evolution"},
    {"provenance": "hypothesis"}, {"provenance": "memory"},
])
def test_generated_or_non_source_conversion_invalidates_old_features(change):
    cache = corpus.PreparedCorpusCache()
    raw = row()
    original = cache.get([raw], namespace="worker::#00001")
    current = cache.get([{**raw, **change}], namespace="worker::#00001")
    assert current is not original
    assert current.source_indices == () and current.document_ids == frozenset()
    assert list(current.hits("archive")) == []


@pytest.mark.parametrize("change", [
    {"content": "The archive does not retain unapproved revisions."},
    {"filename": "renamed.pdf"}, {"page": 3}, {"ordinal": 7},
    {"id": "replacement"}, {"document_id": "other-document"},
])
def test_content_and_presented_locator_updates_invalidate_snapshot(change):
    cache = corpus.PreparedCorpusCache()
    old = row()
    first = cache.get([old], namespace="worker::#00001")
    revised = {**old, **change}
    second = cache.get([revised], namespace="worker::#00001")
    assert first is not second and first.signature != second.signature
    assert cache.get([revised], namespace="worker::#00001") is second


def test_full_content_tail_beyond_token_scan_still_invalidates():
    prefix = ("archive retained. " * (MAX_SCAN_CHARS // 17 + 2))[:MAX_SCAN_CHARS]
    first = corpus.prepare_research_corpus([row(prefix + " Tail one.")])
    second = corpus.prepare_research_corpus([row(prefix + " Tail two.")])
    assert first.signature != second.signature
    assert dict(first.postings) == dict(second.postings)
    assert first.lengths == second.lengths


def test_postings_preserve_term_frequency_document_frequency_and_zero_lengths():
    rows = [row("Archive archive records.", identifier="a"),
            row("Archive records records.", identifier="b", document="d2"),
            row("!!!", identifier="c", document="d3")]
    entries, _ = corpus.canonical_source_view(rows)
    prepared = corpus.prepare_research_corpus(rows)
    counters = [Counter(lexical_tokens(item["content"][:MAX_SCAN_CHARS]))
                for item, _ in entries]
    terms = set().union(*counters)
    assert set(prepared.postings) == terms
    for term in terms:
        expected = [(index, counter[term]) for index, counter in enumerate(counters) if counter[term]]
        assert list(prepared.hits(term)) == expected
        assert len(list(prepared.hits(term))) == sum(term in counter for counter in counters)
    assert prepared.lengths == tuple(max(1, sum(counter.values())) for counter in counters)
    assert prepared.total_length == sum(sum(counter.values()) for counter in counters)
    assert list(prepared.hits("not-in-this-corpus")) == []


def test_normalized_text_and_same_identity_duplicates_match_canonical_scan():
    rows = [row("Archive revisions stay.", identifier="one", document="a"),
            row("Archive  revisions\nstay. ", identifier="two", document="b"),
            row("Different record survives.", identifier="same", document="c"),
            row("Another record survives.", identifier="same", document="c")]
    entries, _ = corpus.canonical_source_view(rows)
    identities, texts, retained, duplicate_count = set(), set(), [], 0
    for index, (source, _) in enumerate(entries):
        identity = (source["document_id"], source["id"])
        text = " ".join(source["content"].split())
        if identity in identities or text in texts:
            duplicate_count += 1
            continue
        identities.add(identity)
        texts.add(text)
        retained.append(index)
    prepared = corpus.prepare_research_corpus(rows)
    assert prepared.source_indices == tuple(retained)
    assert prepared.duplicate_count == duplicate_count == 2
    # Existing retrieval diagnostics count documents before de-duplication.
    assert prepared.document_ids == {"a", "b", "c"}


def test_input_mutation_cannot_mutate_prepared_features_or_change_canonical_order():
    rows = [row("Revision beta remains.", identifier="b", document="b"), row()]
    cache = corpus.PreparedCorpusCache()
    prepared = cache.get(rows, namespace="tenant")
    assert cache.get(list(reversed(rows)), namespace="tenant") is prepared
    before = dict(prepared.postings)
    rows[0]["content"] = "New material not in the old index."
    assert dict(prepared.postings) == before
    assert cache.get(rows, namespace="tenant") is not prepared
    with pytest.raises(FrozenInstanceError):
        prepared.signature = "poison"
    with pytest.raises(TypeError):
        prepared.postings["poison"] = b""
    assert all(type(value) is bytes for value in prepared.postings.values())


def test_lru_hits_refresh_recency_and_clear_drops_retained_membership():
    cache = corpus.PreparedCorpusCache(max_snapshots=2)
    first = cache.get([row()], namespace="a")
    second = cache.get([row()], namespace="b")
    assert cache.get([row()], namespace="a") is first
    cache.get([row()], namespace="c")
    assert cache.get([row()], namespace="a") is first
    assert cache.get([row()], namespace="b") is not second
    assert cache.info()["snapshots"] == 2
    cache.clear()
    assert cache.info()["snapshots"] == cache.info()["feature_bytes"] == 0
    assert cache.get([row()], namespace="a") is not first


def test_aggregate_retained_byte_limit_evicts_before_snapshot_count_limit():
    rows = [row(" ".join("archiveword" + str(i) for i in range(200)))]
    single = corpus.prepare_research_corpus(rows)
    budget = single.feature_bytes * 3 - 1
    cache = corpus.PreparedCorpusCache(max_bytes=budget, max_snapshots=4)
    first = cache.get(rows, namespace="a")
    cache.get(rows, namespace="b")
    cache.get(rows, namespace="c")
    info = cache.info()
    assert info["snapshots"] == 2 < info["max_snapshots"]
    assert info["feature_bytes"] <= info["max_feature_bytes"] == budget
    assert cache.get(rows, namespace="a") is not first


def test_too_small_budget_raises_without_partial_index_or_shortened_corpus():
    rows = [row("Archive preserves a complete source.")]
    with pytest.raises(corpus.CorpusIndexTooLarge):
        corpus.prepare_research_corpus(rows, max_bytes=1)
    cache = corpus.PreparedCorpusCache(max_bytes=1)
    with pytest.raises(corpus.CorpusIndexTooLarge):
        cache.get(rows, namespace="tenant")
    assert cache.info()["snapshots"] == cache.info()["feature_bytes"] == 0


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, corpus.MAX_FEATURE_BYTES + 1])
def test_invalid_memory_budget_rejected(budget):
    with pytest.raises(ValueError):
        corpus.PreparedCorpusCache(max_bytes=budget)
    with pytest.raises(ValueError):
        corpus.prepare_research_corpus([row()], max_bytes=budget)


@pytest.mark.parametrize("namespace", [None, "", "x" * 513, 1, []])
def test_namespace_is_explicit_and_bounded(namespace):
    cache = corpus.PreparedCorpusCache()
    with pytest.raises(ValueError):
        cache.get([row()], namespace=namespace)


def test_cache_retains_no_answer_question_raw_quote_or_arbitrary_row_metadata():
    content = "A unique sentence with internal spaces and a qualifier only after approval."
    cache = corpus.PreparedCorpusCache()
    value = cache.get([row(content, question="private-query", answer="private-answer",
                           proof={"text": "private-proof"})], namespace="tenant")
    assert set(inspect.signature(cache.get).parameters) == {"chunks", "namespace"}
    names = {field.name for field in fields(value)}
    assert names == {"signature", "source_indices", "stable_keys", "lengths", "total_length",
                     "document_ids", "duplicate_count", "postings", "feature_bytes"}
    rendered = repr(value)
    for excluded in (content, "private-query", "private-answer", "private-proof", "invented.pdf"):
        assert excluded not in rendered
    # A normal query word is a lexical symbol, not an instruction or cache API.
    assert list(value.hits("unknown symbol [S1]")) == []


def test_concurrent_same_snapshot_builds_once_and_different_cold_builds_do_not_overlap(monkeypatch):
    cache = corpus.PreparedCorpusCache()
    original = corpus._prepare
    entered, release = threading.Event(), threading.Event()
    state_lock = threading.Lock()
    counts = {"active": 0, "maximum": 0, "calls": 0}

    def bounded(*args, **kwargs):
        with state_lock:
            counts["active"] += 1
            counts["maximum"] = max(counts["maximum"], counts["active"])
            counts["calls"] += 1
            first = counts["calls"] == 1
        try:
            if first:
                entered.set()
                assert release.wait(3), "test must release its synthetic cold builder"
            return original(*args, **kwargs)
        finally:
            with state_lock:
                counts["active"] -= 1

    monkeypatch.setattr(corpus, "_prepare", bounded)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(cache.get, [row()], namespace="tenant")
        try:
            assert entered.wait(3)
            same = pool.submit(cache.get, [row()], namespace="tenant")
            other = pool.submit(cache.get, [row()], namespace="other")
        finally:
            release.set()
        a, b, c = first.result(timeout=3), same.result(timeout=3), other.result(timeout=3)
    assert a is b and a is not c
    assert counts == {"active": 0, "maximum": 1, "calls": 2}
    assert cache.info()["snapshots"] == 2
