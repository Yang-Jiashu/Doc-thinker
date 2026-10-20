"""Local query translation is a bounded search aid, never answer evidence."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from docthinker.server import research_faceted_assist as assist
from docthinker.server.research_query_facets import build_query_facets


QUESTION = "请解释 Crimson Atlas 的 coral-lock dispatch 怎样处理队列？实验如何设置？"
TRANSLATED = "Explain how Crimson Atlas coral-lock dispatch handles the queue. How is the experiment designed?"
SOURCE = "Coral-lock dispatch computes a parity flag before selecting a queue. The experiment samples independent chambers."


def chunks(count=1):
    return [{"id": f"source-{index}", "document_id": "atlas", "filename": "Crimson_Atlas.pdf",
             "ordinal": index, "content": SOURCE + f" Queue index {index}."} for index in range(count)]


class Translator:
    def __init__(self, result=None, *, error=None):
        self.result = result if result is not None else {"text": TRANSLATED, "trace": {"status": "ready"}}
        self.error = error
        self.calls = []

    def translate(self, question, *, session_id):
        self.calls.append((question, session_id))
        if self.error:
            raise RuntimeError(self.error)
        return self.result


class Semantic:
    def __init__(self):
        self.calls = []

    def semantic_scores(self, question, rows, *, session_id):
        self.calls.append((question, [row["document_id"] for row in rows], session_id))
        return {"scores": {row["id"]: .86 for row in rows},
                "spans": {row["id"]: [0, len(row["content"])] for row in rows}, "trace": {"status": "ready"}}


class Reranker:
    def __init__(self):
        self.calls = []

    def rank_pairs(self, candidates, *, session_id):
        self.calls.append((list(candidates), session_id))
        return {"scores": {row["id"]: 1. for row in candidates}, "trace": {"status": "ready", "provider_calls": 0}}


def alias_plan(question, translated):
    return assist._with_query_translation(question, [], build_query_facets(question),
                                         Translator({"text": translated, "trace": {"status": "ready"}}), "s")[0]


def test_invariance_rejection_remains_visible_without_leaking_translation():
    question = "改变照明后如何保持温度不变？"
    original = build_query_facets(question)
    subject = Translator({"text": question, "trace": {
        "status": "lexical_guard", "guard_reasons": ["missing_invariance", "private rejected text"]}})
    plan, receipt = assist._with_query_translation(question, [], original, subject, "private-session")
    assert plan == original
    assert receipt["guard_reasons"] == ["missing_invariance"]
    assert receipt["status"] == "lexical_guard" and not receipt["applied"]
    assert "private" not in json.dumps(receipt)


def test_unique_translated_requirement_is_nomination_hint_not_new_requirement():
    plan = alias_plan("Atlas如何合并输入？实验中是否已经验证泛化？",
                      "How does Atlas combine inputs? Whether the experiment has verified generalization?")
    aliases = [row for row in plan["facets"] if row.get("query_alias_of")]
    assert len(aliases) == 1
    alias = aliases[0]
    literal = next(row for row in plan["facets"] if row["id"] == alias["query_alias_of"])
    assert literal["text"] == "实验中是否已经验证泛化"
    assert alias["derived_translation"] and alias["query_only"]
    assert not alias["translation_full_query"]
    assert alias["semantic_equivalence_verified"] is False
    assert len(plan["facets"]) <= 8


@pytest.mark.parametrize("question,translated", [
    ("Atlas如何合并输入？实验中是否已经验证泛化？实验中是否已经测试延迟？",
     "How does Atlas combine inputs? Whether the experiment verified generalization? Whether the study tested latency?"),
    ("Atlas如何合并输入？是否已经在12组中验证泛化？",
     "How does Atlas combine inputs? Whether the study verified generalization in 13 groups?"),
    ("Atlas如何合并输入？是否已经证明不依赖监督？",
     "How does Atlas combine inputs? Whether the study demonstrated dependence on supervision?"),
    ("Atlas如何合并输入？是否已经验证Zephyr机制？",
     "How does Atlas combine inputs? Whether the study verified Boreal mechanism?"),
    ("Atlas如何合并输入？是否已经验证泛化？",
     "How does Atlas combine inputs? Whether the study has not verified generalization?"),
])
def test_ambiguous_or_surface_unfaithful_alias_never_aligns(question, translated):
    assert not any(row.get("query_alias_of") for row in alias_plan(question, translated)["facets"])


def test_alias_guard_does_not_shift_a_preserved_number_from_another_clause():
    # The full query retains both numbers, but the verification view changed.
    plan = alias_plan("Atlas为何使用12组？是否已经在24组中验证泛化？",
                      "Why does Atlas use 24 groups? Whether the study verified generalization in 12 groups?")
    assert not any(row.get("query_alias_of") for row in plan["facets"])


def test_alias_nomination_is_reranked_by_literal_need_without_forging_literal_support():
    question = "Atlas如何合并输入？实验中是否已经验证泛化？"
    plan = alias_plan(question, "How does Atlas combine inputs? Whether the experiment has verified generalization?")
    alias = next(row for row in plan["facets"] if row.get("query_alias_of"))
    need = next(row for row in plan["facets"] if row["id"] == alias["query_alias_of"])
    full = next(row for row in plan["facets"] if row.get("translation_full_query"))
    candidates = [dict(id=f"w{i}", source_id=f"s{i}", document_id="d", content="SOURCE_ONLY",
                       facet_scores={alias["id"]: .8-i/100, full["id"]: .6}) for i in range(48)]
    pairs = assist._rerank_pairs(question, candidates, plan["facets"])
    focused = [row for row in pairs if row.get("rerank_branch") == "focused"]
    assert len(pairs) == 48 and len(focused) == 24
    assert {row["query"] for row in focused} == {need["text"]}
    assert {row["rerank_facet_id"] for row in focused} == {need["id"]}
    assert all(need["id"] not in row["facet_scores"] for row in focused)
    assert all(row["nomination_facet_ids"] == [alias["id"]] for row in focused)
    assert all(row["nomination_alignment"] == "unique_query_only_requirement_kind" for row in focused)
    assert all(row["retrieval_support"] == row["facet_scores"][alias["id"]] > 0 for row in focused)
    # A full English view must never be broadcast as support for each need.
    unsupported = [{**row, "facet_scores": {full["id"]:.9}} for row in candidates]
    assert all(row.get("rerank_branch") != "focused" for row in assist._rerank_pairs(question, unsupported, plan["facets"]))


def language_focus_pairs(contents):
    question = "Atlas如何合并输入？实验中是否已经验证泛化？"
    plan = alias_plan(question, "How does Atlas combine inputs? Whether the experiment has verified generalization?")
    alias = next(row for row in plan["facets"] if row.get("query_alias_of"))
    literal = next(row for row in plan["facets"] if row["id"] == alias["query_alias_of"])
    rows = [dict(id=f"w{i}", source_id=f"s{i}", document_id="d", content=contents[i % len(contents)],
                 facet_scores={alias["id"]:.8-i/100}) for i in range(48)]
    return assist._rerank_pairs(question, rows, plan["facets"]), alias, literal


def test_english_material_uses_one_guarded_alias_view_not_two_query_quotas():
    pairs, alias, literal = language_focus_pairs(["The independent sample groups are evaluated with a fixed control procedure."])
    focused = [row for row in pairs if row.get("rerank_branch") == "focused"]
    assert len(pairs) == 48 and len(focused) == 24
    assert len({row["query"] for row in focused}) == 1
    assert all(row["query"] == alias["text"] and row["rerank_facet_id"] == alias["id"] for row in focused)
    assert all(row["rerank_view_of"] == literal["id"] and row["rerank_view"] == "guarded_alias" for row in focused)
    assert all(literal["id"] not in row["facet_scores"] for row in focused)


@pytest.mark.parametrize("contents", [
    ["独立样本组使用固定控制程序评估，实验条件与所有评价设置应保持一致。"],
    ["The independent groups use a fixed control procedure.", "独立样本组使用固定控制程序评估。"],
    ["12 0.3 N=40 1.2"],
    ["The independent groups use a fixed control procedure.", "数据"],
    ["Les groupes indépendants utilisent une procédure expérimentale stable et répétable."],
    ["Alpha Beta Gamma Zeta Lambda Control Result Experiment Dataset Benchmark"],
])
def test_mixed_or_unknown_material_keeps_literal_scoring_view(contents):
    pairs, _, literal = language_focus_pairs(contents)
    focused = [row for row in pairs if row.get("rerank_branch") == "focused"]
    assert len(pairs) == 48 and len(focused) == 24
    assert all(row["query"] == literal["text"] and row["rerank_view"] == "literal" for row in focused)
    assert all(row["rerank_facet_id"] == literal["id"] for row in focused)


def test_scoring_language_selection_does_not_depend_on_nominee_order():
    pairs, alias, literal = language_focus_pairs(["The independent sample groups are evaluated with a fixed control procedure."])
    nominees = [row for row in pairs if row.get("rerank_branch") == "focused"]
    assert assist._focused_scoring_view(literal, [alias], nominees) == assist._focused_scoring_view(literal, [alias], nominees[::-1]) == alias


def test_unscored_proposal_does_not_choose_the_language_of_scored_query_group():
    question = "Atlas如何合并输入？它为何能扩展？实验中是否已经验证泛化？"
    plan = alias_plan(question, "How does Atlas combine inputs? Why does it scale? Whether the experiment verified generalization?")
    alias = next(row for row in plan["facets"] if row.get("query_alias_of"))
    rows = [dict(id=f"w{i:02}", source_id=f"s{i}", document_id="d",
                 content="The independent sample groups are evaluated with a fixed control procedure." if i < 12 else "X = 12",
                 facet_scores={row["id"]: .9-i/100 for row in plan["facets"]}) for i in range(48)]
    pairs = assist._rerank_pairs(question, rows, plan["facets"])
    focus = [row for row in pairs if row.get("rerank_view_of") == alias["query_alias_of"]]
    assert len(pairs) == 48 and len(focus) == 12
    assert all(row["rerank_view"] == "guarded_alias" for row in focus)
    assert all(row["content"] != "X = 12" for row in focus)


def run(question=QUESTION, *, translator=None, rows=None, primary=None):
    semantic, reranker = Semantic(), Reranker()
    with patch.object(assist, "prepare_evidence_candidates", wraps=assist.prepare_evidence_candidates) as capture:
        result = assist.build_faceted_assistance(question, [], chunks() if rows is None else rows,
                                                semantic, reranker, session_id="tenant::session",
                                                primary_semantic=primary, query_translator=translator)
    return result, semantic, reranker, capture


def test_original_question_primary_semantics_anchors_and_document_scope_are_preserved():
    rows = chunks()
    original = build_query_facets(QUESTION, rows)
    primary = {"scores": {}, "spans": {}, "trace": {"status": "ready"}}
    translator = Translator()
    result, semantic, reranker, capture = run(translator=translator, rows=rows, primary=primary)
    passed_question, _, _, plan, semantics = capture.call_args.args
    assert passed_question == QUESTION
    assert plan["facets"][0] == original["facets"][0]
    assert semantics[original["facets"][0]["id"]] is primary
    assert not any(question == QUESTION for question, _, _ in semantic.calls)
    english = [row for row in plan["facets"] if row.get("derived_translation")]
    assert 1 <= len(english) <= 4
    assert all(row["kind"] == "clause" and row["query_only"] for row in english)
    assert all(row["document_ids"] == ["atlas"] for row in english)
    assert all("anchors" not in row for row in english)
    assert translator.calls == [(QUESTION, "tenant::session")]
    assert result["trace"]["local_query_translation"]["applied"] is True
    assert len(reranker.calls) == 1
    assert result["sources"] and all(source["content"] in rows[0]["content"] for source in result["sources"])
    assert result["prompt"] == ""


def test_many_translated_and_literal_facets_stay_within_eight_queries_and_48_pairs():
    question = "；".join(f"coral-{index} routing 如何运行" for index in range(14))
    translation = "? ".join(f"How does coral-{index} routing process the queue" for index in range(14)) + "?"
    translator = Translator({"text": translation, "trace": {"status": "ready"}})
    rows = chunks(80)
    result, semantic, reranker, capture = run(question, translator=translator, rows=rows)
    plan = capture.call_args.args[3]
    assert len(plan["facets"]) <= 8
    assert len([row for row in plan["facets"] if row.get("derived_translation")]) <= 4
    assert plan["facets"][0] == build_query_facets(question, rows)["facets"][0]
    assert len(semantic.calls) <= 8
    assert len(reranker.calls) == 1 and len(reranker.calls[0][0]) <= 48
    assert result["trace"]["model_calls"] == result["trace"]["provider_calls"] == result["trace"]["llm_tokens"] == 0


def test_four_literal_content_needs_survive_translation_alias_budget():
    question = 'Atlas在何处合并输入？它如何选择样本？它为何能扩展？实验中是否已经验证跨域泛化？'
    translated = 'Where does Atlas combine inputs? How does it select examples? Why can it scale? Whether the experiments verified transfer?'
    original = build_query_facets(question, chunks())
    combined, receipt = assist._with_query_translation(question, chunks(), original,
        Translator({'text':translated,'trace':{'status':'ready'}}),'synthetic::literal-priority')
    literal = [row for row in combined['facets'] if row['kind']=='clause' and not row.get('derived_translation')]
    assert len(literal)==4
    assert any(row['text']=='实验中是否已经验证跨域泛化' for row in literal)
    assert len(combined['facets'])<=8 and receipt['english_facet_count']<=3
    assert any(row.get('translation_full_query') and row['text']==translated for row in combined['facets'])


@pytest.mark.parametrize("translation", [
    {"text": TRANSLATED, "trace": {"status": "busy"}},
    {"text": TRANSLATED, "trace": {"status": "model_unavailable"}},
    {"text": "", "trace": {"status": "ready"}},
    {"text": QUESTION, "trace": {"status": "ready"}},
    {"text": "x" * 9000, "trace": {"status": "ready"}},
    {"text": "\ud800", "trace": {"status": "ready"}},
    {"text": None, "trace": {"status": "ready"}},
    {"text": TRANSLATED, "trace": None},
    ["unexpected response"],
])
def test_unavailable_invalid_and_unchanged_translation_keep_literal_channel(translation):
    result, semantic, reranker, capture = run(translator=Translator(translation))
    assert capture.call_args.args[3] == build_query_facets(QUESTION, chunks())
    assert result["trace"]["local_query_translation"]["applied"] is False
    assert semantic.calls and len(reranker.calls) == 1


def test_translator_errors_and_untrusted_trace_values_never_expose_queries_or_paths():
    secret = "secret-query /private/user/config"
    for translator in (Translator(error=secret), Translator({"text": TRANSLATED, "trace": {
            "status": secret, "error": secret, "path": secret, "question": secret, "total_ms": secret,
            "inference_ms": float("inf"), "cache_hit": secret}})):
        result, _, _, _ = run(translator=translator)
        trace = json.dumps(result["trace"])
        assert secret not in trace
        assert QUESTION not in trace and TRANSLATED not in trace
        assert result["trace"]["local_query_translation"]["applied"] is False


@pytest.mark.parametrize("question,rows", [(QUESTION, []), ("", chunks())])
def test_translator_not_called_without_live_chunks_or_a_query(question, rows):
    translator = Translator()
    result, semantic, reranker, _ = run(question, translator=translator, rows=rows)
    assert translator.calls == semantic.calls == reranker.calls == []
    assert result["trace"]["skipped_reason"] == "no_evidence_query"


def test_translated_clause_can_win_rerank_without_becoming_a_source_or_conclusion():
    translation = "How does coral-lock dispatch compute the parity flag?"
    result, _, reranker, capture = run("珊瑚锁如何分配队列？", translator=Translator({
        "text": translation, "trace": {"status": "ready", "inference_ms": 1.25, "cache_hit": True}}))
    pairs, _ = reranker.calls[0]
    assert any(pair["query"] == translation for pair in pairs)
    assert all(pair["content"] in chunks()[0]["content"] for pair in pairs)
    assert capture.call_args.args[0] == "珊瑚锁如何分配队列？"
    assert not result["reasoning"].get("conclusions")
    assert translation not in result["prompt"]
    assert result["trace"]["local_query_translation"]["query_only"] is True


def test_translated_document_name_cannot_introduce_a_new_document_scope():
    rows = chunks() + [{"id": "foreign", "document_id": "foreign", "filename": "Frost_Ledger.pdf", "content": SOURCE, "ordinal": 0}]
    translation = "How does Frost Ledger implement coral-lock dispatch?"
    _, semantic, _, capture = run(translator=Translator({"text": translation, "trace": {"status": "ready"}}), rows=rows)
    translated_facets = [row for row in capture.call_args.args[3]["facets"] if row.get("derived_translation")]
    assert all(row["document_ids"] == ["atlas"] for row in translated_facets)
    translated_queries = {row["text"] for row in translated_facets}
    assert all(docs == ["atlas"] for query, docs, _ in semantic.calls if query in translated_queries)
