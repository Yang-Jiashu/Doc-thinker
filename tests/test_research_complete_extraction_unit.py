"""Source ingestion must not turn PDF line wraps into missing conditions."""
import pytest

from docthinker.server.research_extract import CHUNK_CHARS, ExtractionError, extract_chunks


def test_ingestion_preserves_a_long_wrapped_statement_and_its_qualification():
    claim = "The intervention " + "improves measured performance under the tested setup " * 20 + "."
    limit = " However, the finding does not establish an effect outside this dataset."
    text = "Prior background. " + claim[:500] + "\n" + claim[500:] + limit + " A separate observation follows."
    rows = extract_chunks(text.encode(), ".txt")
    statement = next(row["content"] for row in rows if "The intervention" in row["content"])
    assert "\n" in statement
    assert limit.strip() in statement
    assert len(statement) > 1000
    assert statement in text
    assert all(row["content"] in text for row in rows)


def test_clause_boundaries_do_not_drop_negation_after_chunk_target():
    prefix = "Earlier measurements provide background. " * 23
    claim = "The treatment improves recall. However, it does not improve accuracy."
    rows = extract_chunks((prefix + claim).encode(), ".md")
    containing = [row["content"] for row in rows if "treatment improves" in row["content"]]
    assert len(containing) == 1
    assert "does not improve accuracy" in containing[0]


def test_unbounded_unpunctuated_statement_fails_explicitly_instead_of_cutting_it():
    with pytest.raises(ExtractionError, match="statement_size_limit"):
        extract_chunks(("x" * 8001).encode(), ".txt")


def test_storage_target_retains_context_but_answer_excerpt_limit_remains_1000():
    from docthinker.server.research_retrieval import MAX_CHUNK_CHARS, build_retrieval_plan

    assert CHUNK_CHARS == 2000
    assert MAX_CHUNK_CHARS == 1000
    text = "Solar performance improved under controlled conditions. " * 45
    chunks = extract_chunks(text.encode(), ".txt")
    assert 1000 < len(chunks[0]["content"]) <= CHUNK_CHARS
    rows = [{**row, "id": f"c{index}", "document_id": "d1"} for index, row in enumerate(chunks)]
    plan = build_retrieval_plan("solar performance", rows)
    assert plan["sources"]
    assert all(len(source["content"]) <= 1000 for source in plan["sources"])


def test_generic_procedural_antecedent_is_kept_with_following_statement():
    prefix = "Prior background observation. " * 64
    method = "A known approach uses a separate index. Queries are then processed by an external retriever."
    chunks = extract_chunks((prefix + method).encode(), ".txt")
    found = next(row["content"] for row in chunks if "Queries are then" in row["content"])
    assert method in found


@pytest.mark.parametrize("heading", ["2. New Method", "## New Method", "NEW METHOD", "New Method"])
def test_antecedent_binding_does_not_cross_a_section_heading(heading):
    from docthinker.server.research_sentence_windows import evidence_units, sentence_spans

    previous = "A known approach uses a separate index."
    following = "Queries are then processed by a newly proposed model."
    text = previous + "\n\n" + heading + "\n" + following
    units = [text[start:end] for start, end in evidence_units(text)]
    assert not any(previous in unit and following in unit for unit in units)
    assert any(unit.strip() == following for unit in units)
    spans = sentence_spans(text)
    assert "".join(text[start:end] for start, end in spans) == text
