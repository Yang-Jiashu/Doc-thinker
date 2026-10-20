"""Real upload/index/background-assist path, then one captured writing call.

Source algorithms use no model. Normal chat/writing receives their material in
one fake model invocation; no external provider or network is involved here.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from docthinker.server.research_evidence_index import hydrate_reasoning
from docthinker.server.research_extract import extract_chunks
from docthinker.server.research_reasoning import extract_claims, extract_document_claims, reason
from docthinker.server.research_reasoning_context import pack_reasoning, render_reasoning
from docthinker.server.research_provenance import audit_answer
from docthinker.server.research_service import ResearchService

with patch("dotenv.load_dotenv", return_value=False):
    from docthinker.session_manager import SessionManager


async def offline_parser(content, filename):
    return {"chunks": extract_chunks(content, ".txt")}


WRITTEN_ANSWER = "这是按用户需求撰写的自然分析，相关原文可供核对。[S1]"


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    manager = SessionManager(str(tmp_path / "_system"), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session("Reasoning acceptance")["id"]
    return ResearchService(tmp_path, extractor=offline_parser), manager, sid


async def upload(service, sid, text, filename="source.txt"):
    accepted = service.store.accept_document(sid, filename, text.encode("utf-8"))
    assert await service.process(sid, accepted["id"])
    assert service.store.get_document(sid, accepted["id"])["status"] == "ready"
    return accepted


async def answer(project, question, mode="auto", **kwargs):
    service, manager, sid = project
    request = SimpleNamespace(session_id=sid, question=question, reasoning_mode=mode,
                              use_memory=kwargs.get("use_memory", False),
                              memory_as_of=kwargs.get("memory_as_of"),
                              use_conversation_context=False, use_llm_cache=False, use_self_evolution=False)
    calls = []
    async def writing_model(prompt, **options):
        calls.append((prompt, options))
        return WRITTEN_ANSWER
    result = await service.answer(request, writing_model, manager)
    assert len(calls) <= 1
    assert result["reasoning"]["trace"]["answer_model_calls"] == len(calls)
    if calls:
        assert result["answer"] == WRITTEN_ANSWER
        assert result["answer_mode"] not in {"deduction", "analogy", "induction"}
    result["_captured_prompt"] = calls[0][0] if calls else ""
    result["_model_calls"] = len(calls)
    return result


def gap_codes(reasoning):
    return {row["code"] for row in reasoning["gaps"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,question,texts,kind", [
    ("deduction", "请分析 grid ready 对 supply 的影响，写成一段说明", ["Observation: grid ready. If grid ready then supply grows."], "deduced"),
    ("analogy", "写一段关于 redundancy outage risk availability 的分析，借鉴其他领域的案例", [
        "Redundancy reduces outage risk. Outage risk reduces availability.",
        "Supplier diversity reduces concentration risk. Concentration risk reduces energy security.",
    ], "analogy_hypothesis"),
    ("induction", "综合论文中 solar imports 的共同发现，写一段综述", [
        "Case: north\nSolar reduces imports.", "Case: south\nSolar reduces imports.",
        "Case: winter\nSolar does not reduce imports.",
    ], "inductive_candidate"),
])
async def test_real_upload_to_background_material_and_one_natural_writing_call(project, mode, question, texts, kind):
    service, manager, sid = project
    for index, text in enumerate(texts):
        await upload(service, sid, text, f"source-{index}.txt")
    result = await answer(project, question)
    assert result["_model_calls"] == 1
    assert result["reasoning"]["mode"] == "context_assist"
    assert result["reasoning"]["trace"]["algorithm_modes"] == [mode]
    assert result["reasoning"]["trace"]["model_calls"] == 0
    assert "后台整理的写作参考" in result["_captured_prompt"]
    assert "supporting_material" in result["_captured_prompt"]
    assert result["reasoning"]["conclusions"]
    assert result["reasoning"]["conclusions"][0]["kind"] == kind
    assert result["sources"]
    assert not any(row["code"] == "provenance_unavailable" for row in result["citation_validation"]["warnings"])
    restarted = ResearchService(service.store.root, extractor=offline_parser)
    history = restarted.hydrate_history(sid, manager.get_history(sid))
    saved = next(row for row in history if row["id"] == result["answer_id"])
    assert saved["reasoning"]["conclusions"] == result["reasoning"]["conclusions"]
    live = {row["id"]: row for row in service.all_chunks(sid)}
    for claim in saved["reasoning"]["claims"]:
        source = live[claim["source_ids"][0]]["content"]
        assert source[claim["excerpt_start"]:claim["excerpt_end"]] == claim["quote"]


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_mode", ["auto", "analogy", "deduction", "induction"])
async def test_legacy_algorithm_flag_cannot_override_natural_question_or_writer(project, legacy_mode):
    service, _, sid = project
    await upload(service, sid, "Observation: grid ready. If grid ready then security improves.")
    result = await answer(project, "Why does security improve? Write a short explanation.", legacy_mode)
    assert result["_model_calls"] == 1
    assert result["reasoning"]["trace"]["algorithm_modes"] == ["deduction"]
    assert result["reasoning"]["conclusions"]
    assert result["answer"] == WRITTEN_ANSWER


@pytest.mark.asyncio
async def test_simple_source_question_does_not_schedule_algorithms_due_to_legacy_flag(project):
    service, _, sid = project
    await upload(service, sid, "Solar reduces imports.")
    result = await answer(project, "What is solar?", "analogy")
    assert result["_model_calls"] == 1
    assert result["reasoning"]["trace"]["algorithm_modes"] == []
    assert result["reasoning"]["conclusions"] == []
    assert "后台整理的写作参考" not in result["_captured_prompt"]


@pytest.mark.asyncio
async def test_one_background_candidate_load_and_no_provider_repair_retry(project, monkeypatch):
    service, manager, sid = project
    await upload(service, sid, "Observation: grid ready. If grid ready then security improves.")
    candidate_calls, provider_calls = [], []
    original = service.evidence.candidates
    def counted(*args):
        candidate_calls.append(args)
        return original(*args)
    monkeypatch.setattr(service.evidence, "candidates", counted)
    async def failed_writer(prompt, **kwargs):
        provider_calls.append(prompt)
        raise RuntimeError("synthetic provider failure")
    request = SimpleNamespace(session_id=sid, question="Explain why security improves",
                              reasoning_mode="auto", use_memory=False, use_conversation_context=False)
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        await service.answer(request, failed_writer, manager)
    assert len(candidate_calls) == len(provider_calls) == 1
    assert "supporting_material" in provider_calls[0]


@pytest.mark.asyncio
async def test_backward_dependency_lookup_finds_nonlexical_fact_document(project):
    service, _, sid = project
    fact = await upload(service, sid, "Observation: grid ready.", "observations.txt")
    rule = await upload(service, sid, "If grid ready then security improves.", "rules.txt")
    result = await answer(project, "请演绎 security 能否成立", "deduction")
    assert result["reasoning"]["conclusions"]
    assert {row["document_id"] for row in result["sources"]} == {fact["id"], rule["id"]}


@pytest.mark.asyncio
async def test_runtime_acceptance_two_step_chain_survives_current_writing_pipeline(project):
    service, _, sid = project
    fact = await upload(service, sid, "Observation: grid ready.", "observation.txt")
    rules = await upload(service, sid,
                         "If grid ready then supply grows. If supply grows then security improves.",
                         "rules.txt")
    result = await answer(project, "请参考资料写一段分析，解释 security improves 为什么成立。", "auto")
    assert any(row["kind"] == "deduced" and row["depth"] == 2 and "security improves" in row["text"]
               for row in result["reasoning"]["conclusions"])
    assert {row["document_id"] for row in result["sources"]} == {fact["id"], rules["id"]}
    assert result["_model_calls"] == 1
    assert result["reasoning"]["trace"]["model_calls"] == 0
    assert "supporting_material" in result["_captured_prompt"]


@pytest.mark.asyncio
async def test_deduction_dependency_closure_includes_nonlexical_inverse_conflict(project):
    service, _, sid = project
    await upload(service, sid, "Solar reduces imports.", "positive.txt")
    await upload(service, sid, "Solar increases imports.", "opposing.txt")
    rule = await upload(service, sid, "If solar reduces imports then security improves.", "rules.txt")
    claims, _ = service.evidence.candidates(sid, service.store.source_chunks(sid, document_id=rule["id"]), "deduction")
    result = reason("演绎 security", claims, "deduction")
    assert result["conclusions"] == []
    assert "contradictory_premises" in gap_codes(result)


@pytest.mark.asyncio
async def test_missing_premise_constrains_single_writing_call_without_inventing_chain(project):
    service, _, sid = project
    await upload(service, sid, "Observation: grid ready. If grid ready and imports decrease then security improves.")
    result = await answer(project, "推导 security", "deduction")
    assert result["reasoning"]["conclusions"] == []
    assert "missing_premises" in gap_codes(result["reasoning"])
    assert result["_model_calls"] == 1
    assert "缺少原文前提" in result["_captured_prompt"]


ENERGY_CHAIN = [
    "观察：气候目标达成。",
    "如果气候目标达成，那么政策工具实施。",
    "如果政策工具实施，那么可再生能源投资增加。",
    "如果可再生能源投资增加，那么国内供给增加。",
    "如果国内供给增加，那么进口依赖下降。",
    "如果进口依赖下降，那么能源多样化提高。",
    "如果能源多样化提高，那么供应风险下降。",
    "如果供应风险下降，那么能源安全提高。",
]


@pytest.mark.asyncio
async def test_user_energy_chain_is_really_derived_across_eight_uploaded_sources(project):
    service, _, sid = project
    for index, text in enumerate(ENERGY_CHAIN):
        await upload(service, sid, text, f"energy-step-{index}.txt")
    result = await answer(project, "请推导能源安全是否提高，并给出完整前提", "deduction")
    target = result["reasoning"]["conclusions"][0]
    assert target["subject"] == "能源安全"
    assert target["relation"] == "state_increase"
    assert target["depth"] == 7
    assert len(target["source_ids"]) == 8
    assert len(result["sources"]) == 8
    assert len(result["reasoning"]["claims"]) == 8
    assert len({node["id"] for node in target["proof"] if node["kind"] != "deduced"}) == 8
    assert result["_model_calls"] == 1
    assert "能源安全" in result["_captured_prompt"]
    assert result["answer"] == WRITTEN_ANSWER


@pytest.mark.asyncio
async def test_missing_energy_bridge_does_not_answer_final_goal_with_earlier_intermediates(project):
    service, _, sid = project
    text = "".join(statement for index, statement in enumerate(ENERGY_CHAIN) if index != 4)
    await upload(service, sid, text)
    result = await answer(project, "请推导能源安全是否提高", "deduction")
    assert result["reasoning"]["conclusions"] == []
    assert "requested_target_not_proved" in gap_codes(result["reasoning"])
    assert result["_model_calls"] == 1
    assert "不能替代目标结论" in result["_captured_prompt"]


@pytest.mark.asyncio
async def test_crossdomain_cases_are_found_by_relation_index_without_name_overlap(project):
    service, _, sid = project
    await upload(service, sid, "Redundancy reduces outage risk. Outage risk reduces availability.", "cloud.txt")
    foreign = await upload(service, sid, "Supplier diversity reduces concentration risk. Concentration risk reduces energy security.", "energy.txt")
    result = await answer(project, "类比 redundancy outage risk availability", "analogy")
    assert result["reasoning"]["conclusions"]
    assert foreign["id"] in {row["document_id"] for row in result["sources"]}
    assert "supplier diversity" in result["_captured_prompt"]
    assert '"mappings"' in result["_captured_prompt"]
    assert result["answer"] == WRITTEN_ANSWER


@pytest.mark.asyncio
async def test_induction_candidate_lookup_includes_opposing_predicate(project):
    service, _, sid = project
    one = await upload(service, sid, "Case: north\nSolar reduces imports.", "north.txt")
    await upload(service, sid, "Case: south\nSolar reduces imports.", "south.txt")
    await upload(service, sid, "Case: rebound\nSolar increases imports.", "rebound.txt")
    seeds = service.store.source_chunks(sid, document_id=one["id"])
    claims, _ = service.evidence.candidates(sid, seeds, "induction")
    result = reason("归纳", claims, "induction")
    assert result["conclusions"][0]["support_count"] == 2
    assert result["conclusions"][0]["counterexample_count"] == 1


@pytest.mark.asyncio
async def test_same_source_reupload_is_not_independent_induction_support(project):
    service, _, sid = project
    first = await upload(service, sid, "Case: north\nSolar reduces imports.", "north.txt")
    duplicate = service.store.accept_document(sid, "renamed.txt", b"Case: north\nSolar reduces imports.")
    assert duplicate["id"] == first["id"]
    result = await answer(project, "归纳 solar imports", "induction")
    assert result["reasoning"]["conclusions"] == []


@pytest.mark.asyncio
async def test_structural_index_does_not_cross_project_boundary(project):
    service, manager, sid = project
    await upload(service, sid, "Redundancy reduces outage risk. Outage risk reduces availability.")
    other = manager.create_session("Other user project")["id"]
    await upload(service, other, "Supplier diversity reduces concentration risk. Concentration risk reduces energy security.")
    result = await answer(project, "类比 redundancy availability", "analogy")
    assert result["reasoning"]["conclusions"] == []


@pytest.mark.asyncio
async def test_source_withdrawal_masks_proof_and_quotes_after_restart(project):
    service, manager, sid = project
    doc = await upload(service, sid, "Observation: grid ready. If grid ready then supply grows.")
    result = await answer(project, "演绎 grid ready supply", "deduction")
    assert result["reasoning"]["conclusions"]
    service.store.delete_document(sid, doc["id"])
    restarted = ResearchService(service.store.root, extractor=offline_parser)
    saved = next(row for row in restarted.hydrate_history(sid, manager.get_history(sid)) if row["id"] == result["answer_id"])
    assert saved["reasoning"]["conclusions"] == []
    assert saved["reasoning"]["claims"] == []
    assert saved["reasoning"]["status"] == "source_unavailable"
    assert saved["sources"] == []
    assert saved["content"] != WRITTEN_ANSWER
    assert "撤回" in saved["content"]
    assert all(not claim["source_refs"] for claim in saved["claim_audit"]["claims"])
    claims, _ = restarted.evidence.candidates(sid, [{"id": result["sources"][0]["id"]}], "deduction")
    assert claims == []


@pytest.mark.asyncio
async def test_confirmed_memory_can_point_to_live_source_but_never_supply_an_axiom(project):
    service, _, sid = project
    doc = await upload(service, sid, "Observation: grid ready. If grid ready then supply grows.")
    original = service.store.source_chunks(sid, document_id=doc["id"])[0]
    service.store.create_memory(sid, "finding", "ZXQ portfolio case to revisit", source_chunk_ids=[original["id"]])
    on = await answer(project, "演绎 ZXQ portfolio", "deduction", use_memory=True)
    off = await answer(project, "演绎 ZXQ portfolio", "deduction", use_memory=False)
    assert on["reasoning"]["conclusions"]
    assert off["reasoning"]["conclusions"] == []
    assert all("ZXQ" not in claim["quote"] for claim in on["reasoning"]["claims"])
    service.store.delete_document(sid, doc["id"])
    revoked = await answer(project, "演绎 ZXQ portfolio", "deduction", use_memory=True)
    assert revoked["reasoning"]["conclusions"] == []


@pytest.mark.asyncio
async def test_unbacked_confirmed_memory_does_not_prove_rule_premise(project):
    service, _, sid = project
    await upload(service, sid, "If grid ready then supply grows.")
    service.store.create_memory(sid, "finding", "Observation: grid ready.")
    result = await answer(project, "演绎 grid ready supply", "deduction", use_memory=True)
    assert result["reasoning"]["conclusions"] == []


@pytest.mark.asyncio
async def test_uploaded_named_case_scope_survives_real_extractor_chunk_boundary(project):
    service, _, sid = project
    from docthinker.server.research_extract import CHUNK_CHARS

    background = "Background context without a supported predicate. "
    text = "Case: north\n" + background * (CHUNK_CHARS // len(background) + 10)
    text += "\nObservation: grid ready.\nCase: south\nIf grid ready then supply grows."
    doc = await upload(service, sid, text)
    assert service.store.get_document(sid, doc["id"])["chunk_count"] > 1
    result = await answer(project, "演绎 supply grid ready", "deduction")
    # Extractive organization can now expose the literal case-scoped quotes;
    # it must not turn north's observation into a proof of south's premise.
    assert all(item["kind"] == "source_evidence" for item in result["reasoning"]["conclusions"])
    claims, _ = service.evidence.candidates(sid, service.store.source_chunks(sid, document_id=doc["id"]), "deduction")
    scoped = reason("演绎 supply grid ready", claims, "deduction")
    assert scoped["conclusions"] == []
    assert "missing_premises" in gap_codes(scoped)


@pytest.mark.asyncio
async def test_index_never_offers_partial_document_past_structured_claim_window(project):
    service, _, sid = project
    text = "Case: north\n" + "Solar reduces imports. " * 1600 + "Solar does not reduce imports."
    doc = await upload(service, sid, text)
    claims, gaps = service.evidence.candidates(sid, service.store.source_chunks(sid, document_id=doc["id"])[:1], "induction")
    assert claims == []
    assert "index_window_reached" in {row["code"] for row in gaps}


def proof_fixture():
    source = {"id": "c1", "document_id": "d1", "content": "Observation: grid ready. If grid ready then supply grows."}
    claims = extract_claims([source])
    result = reason("演绎", claims, "deduction")
    return source, claims, result


def test_atomic_packer_rejects_missing_rule_even_when_other_premise_shares_source():
    source, claims, result = proof_fixture()
    incomplete = [row for row in claims if row["kind"] != "rule"]
    packed, refs = pack_reasoning(result, incomplete, [source])
    assert packed["conclusions"] == []
    assert refs == []


def test_atomic_packer_rejects_forged_literal_quote_or_offsets():
    source, claims, result = proof_fixture()
    bad = copy.deepcopy(claims)
    bad[0]["quote"] = "Observation: source asserts the opposite."
    packed, refs = pack_reasoning(result, bad, [source])
    assert packed["conclusions"] == []
    assert refs == []


def test_atomic_packer_handles_missing_premise_source_without_keyerror():
    sources = [
        {"id": "c1", "document_id": "d1", "content": "Observation: grid ready."},
        {"id": "c2", "document_id": "d2", "content": "If grid ready then supply grows."},
    ]
    claims = extract_claims(sources)
    result = reason("演绎", claims, "deduction")
    # Corrupt provenance metadata must fail closed, not crash or borrow a quote
    # from another premise whose source happens to still be present.
    result["conclusions"][0]["source_ids"] = ["c1"]
    packed, refs = pack_reasoning(result, claims, sources[:1])
    assert packed["conclusions"] == []
    assert refs == []


def test_atomic_packer_omits_whole_proof_exceeding_source_count_budget():
    chunks = [
        {"id": "c1", "document_id": "d1", "content": "Observation: grid ready."},
        {"id": "c2", "document_id": "d2", "content": "If grid ready then supply grows."},
    ]
    claims = extract_claims(chunks)
    packed, refs = pack_reasoning(reason("演绎", claims, "deduction"), claims, chunks, max_sources=1)
    assert packed["conclusions"] == []
    assert refs == []
    assert "proof_budget_reached" in gap_codes(packed)


def test_hydration_omits_conclusion_if_source_excerpt_no_longer_covers_its_rule():
    source, claims, result = proof_fixture()
    packed, refs = pack_reasoning(result, claims, [source])
    partial = {**refs[0], "content": "Observation: grid ready.", "excerpt_start": 0, "excerpt_end": 24}
    hydrated = hydrate_reasoning(packed, [partial])
    assert hydrated["conclusions"] == []


def test_analogy_renderer_shows_roles_and_verification_conditions():
    sources = [
        {"id": "c1", "document_id": "d1", "content": "Redundancy reduces outage risk. Outage risk reduces availability. Redundancy increases availability under low load."},
        {"id": "c2", "document_id": "d2", "content": "Supplier diversity reduces concentration risk. Concentration risk reduces energy security."},
    ]
    claims = extract_claims(sources)
    packed, refs = pack_reasoning(reason("类比", claims, "analogy"), claims, sources)
    output = render_reasoning(packed, refs)
    assert "角色对应" in output
    assert "redundancy" in output and "supplier diversity" in output
    assert "low load" in output


def test_document_parser_carries_case_across_ordered_chunks():
    sources = [
        {"id": "later", "document_id": "d1", "ordinal": 1, "content": "Observation: grid ready."},
        {"id": "earlier", "document_id": "d1", "ordinal": 0, "content": "Case: north\nSolar reduces imports."},
    ]
    parsed = extract_document_claims(sources)
    assert parsed["limited"] is False
    assert {row["case_label"] for row in parsed["claims"]} == {"north"}
    assert len({row["source_fingerprint"] for row in parsed["claims"]}) == 1


def test_document_parser_never_returns_partial_case_when_claim_window_fills():
    sources = [{"id": f"c{letter}", "document_id": "d1", "ordinal": index,
                "content": "Case: north\nSolar reduces imports."} for index, letter in enumerate("abcdef")]
    parsed = extract_document_claims(sources, max_claims=4)
    assert parsed == {"claims": [], "limited": True}


def test_oversized_case_cannot_hide_a_later_opposite_relation():
    base = "Alpha reduces beta. Beta increases gamma."
    filler = " ".join(f"Extra{name} reduces spare{name}." for name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron"])
    source = {"id": "c1", "document_id": "d1", "content": base + " " + filler + " Alpha does not reduce beta."}
    target = {"id": "c2", "document_id": "d2", "content": "Delta reduces epsilon. Epsilon increases zeta."}
    result = reason("类比", extract_claims([source, target]), "analogy")
    assert result["conclusions"] == []
    assert "case_omitted_as_whole" in gap_codes(result)


@pytest.mark.parametrize("content", [
    "There is no evidence that solar reduces imports.",
    "Only under low demand, solar reduces imports.",
])
def test_literal_substring_without_its_negation_or_condition_is_not_clean_quote(content):
    result = audit_answer("solar reduces imports. [S1]", [
        {"id": "c1", "document_id": "d1", "label": "S1", "content": content},
    ])
    assert result["status"] == "review_needed"
    assert result["claims"][0]["status"] != "exact_quote"
