"""Self-study work limits tested across real P1-P6 control flow, without APIs."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from docthinker.kg_self_study.orchestrator import SelfStudyConfig, SelfStudyOrchestrator
from docthinker.kg_self_study.question_generator import QuestionGenerator
from docthinker.kg_self_study.work_budget import StudyBudgetStop, StudyWorkBudget


RESPONSES = [
    {},
    [{"question": "How are A and B related?", "question_type": "bridge"}],
    {"answer": "A relates to B", "answerable": True, "confidence": 0.9},
    {
        "new_edges": [
            {
                "source": "A",
                "target": "B",
                "confidence": 0.9,
                "evidence_chain": ["c1", "c2"],
            }
        ]
    },
    {},
]


def study(*, model=None, **limits):
    calls = []

    async def fake(prompt):
        index = len(calls)
        calls.append(prompt)
        return json.dumps(RESPONSES[index % len(RESPONSES)])

    nodes = [{"id": name, "description": name, "source_id": "c1"} for name in "ABC"]
    edges = [
        {
            "source": "A",
            "target": "B",
            "description": "A relates to B",
            "source_id": "c1",
        }
    ]
    read_nodes, read_edges = (
        AsyncMock(return_value=nodes),
        AsyncMock(return_value=edges),
    )
    write = AsyncMock()
    retrieval = AsyncMock(
        return_value={"chunks": [{"id": "c1", "content": "A relates to B"}]}
    )
    config = {
        "max_tokens": 1_000_000,
        "max_llm_calls": 50,
        "max_rounds": 1,
        "questions_per_round": 1,
    }
    config.update(limits)
    orchestrator = SelfStudyOrchestrator(
        llm_func=model or fake,
        kg_query_func=retrieval,
        kg_write_func=write,
        kg_read_nodes_func=read_nodes,
        kg_read_edges_func=read_edges,
        config=SelfStudyConfig(**config),
    )
    orchestrator.selector.select = Mock(
        return_value={
            "entities": nodes,
            "relations": edges,
            "strategy": "bridge_entity",
        }
    )
    return orchestrator, calls, read_nodes, read_edges, write, retrieval


@pytest.mark.parametrize("allowed", range(6))
async def test_call_budget_stops_at_each_pipeline_stage_without_extra_calls(allowed):
    orchestrator, calls, _, _, write, retrieval = study(
        max_llm_calls=allowed, max_rounds=2
    )
    result = await orchestrator.run_session()
    assert result.stopped_reason == "max_llm_calls"
    assert len(calls) == result.llm_usage["llm_calls"] == allowed
    if allowed <= 2:
        retrieval.assert_not_awaited()
    if allowed < 4:
        write.assert_not_awaited()
    else:
        write.assert_awaited_once()
        assert result.total_new_edges == 1  # P4 completed before P5/next round stopped.


@pytest.mark.parametrize("allowed", range(5))
async def test_token_budget_rejects_whole_next_prompt_in_every_stage(allowed):
    probe, prompts, *_ = study()
    await probe.run_session()
    reservation = probe.config.max_output_tokens_per_call
    costs = [len(prompt.encode("utf-8")) + reservation for prompt in prompts]
    token_limit = sum(costs[: allowed + 1]) - 1
    orchestrator, calls, *_ = study(max_tokens=token_limit)
    result = await orchestrator.run_session()
    assert result.stopped_reason == "token_budget_exhausted"
    assert len(calls) == allowed
    assert result.llm_usage["charged_tokens"] == sum(costs[:allowed])
    assert result.llm_usage["charged_tokens"] <= token_limit
    assert (
        result.llm_usage["token_count_method"]
        == "utf8_input_bytes_upper_bound_plus_reserved_output"
    )


@pytest.mark.parametrize("failed_stage", range(1, 6))
async def test_model_error_ends_session_instead_of_consuming_remaining_candidates(
    failed_stage,
):
    calls = []

    async def model(prompt):
        calls.append(prompt)
        if len(calls) == failed_stage:
            raise RuntimeError("provider unavailable")
        return json.dumps(RESPONSES[len(calls) - 1])

    orchestrator, *_ = study(model=model)
    result = await orchestrator.run_session()
    assert result.stopped_reason == "llm_error"
    assert result.llm_usage["llm_calls"] == failed_stage
    assert (
        result.llm_usage["reserved_output_tokens"]
        == failed_stage * orchestrator.config.max_output_tokens_per_call
    )


async def test_timeout_cancels_model_work_and_stops_pipeline():
    cancelled = asyncio.Event()

    async def model(_prompt):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    orchestrator, _, _, _, write, retrieval = study(
        model=model, llm_timeout_seconds=0.01
    )
    result = await orchestrator.run_session()
    assert result.stopped_reason == "llm_timeout"
    assert result.llm_usage["llm_calls"] == 1
    assert cancelled.is_set()
    write.assert_not_awaited()
    retrieval.assert_not_awaited()


async def test_external_cancellation_propagates_and_keeps_attempt_accounted():
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def model(_prompt):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    orchestrator, *_ = study(model=model)
    task = asyncio.create_task(orchestrator.run_session())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert orchestrator.last_llm_usage["llm_calls"] == 1
    assert orchestrator.last_llm_usage["stop_reason"] == "cancelled"


async def test_sequential_and_concurrent_sessions_get_independent_budgets():
    calls = []

    async def model(prompt):
        calls.append(prompt)
        await asyncio.sleep(0)
        return "{}"

    orchestrator, *_ = study(model=model, max_llm_calls=2)
    first = await orchestrator.run_session()
    second = await orchestrator.run_session()
    concurrent = await asyncio.gather(
        orchestrator.run_session(), orchestrator.run_session()
    )
    assert [r.llm_usage["llm_calls"] for r in [first, second, *concurrent]] == [
        2,
        2,
        2,
        2,
    ]
    assert len(calls) == 8


@pytest.mark.parametrize("audit_only,expected_reads", [(True, 1), (False, 3)])
async def test_only_audit_writers_reuse_graph_snapshot(audit_only, expected_reads):
    orchestrator, calls, nodes, edges, *_ = study(max_rounds=3, audit_only=audit_only)
    result = await orchestrator.run_session()
    assert result.stopped_reason == "max_rounds_reached"
    assert len(calls) == 15
    assert nodes.await_count == edges.await_count == expected_reads


async def test_question_cap_stops_additional_bridge_and_contradiction_generation():
    model = AsyncMock(
        return_value=json.dumps([{"question": f"Q{index}"} for index in range(100)])
    )
    generator = QuestionGenerator(model, questions_per_strategy=2)
    questions = await generator.generate_questions(
        "bridge_entity",
        {
            "bridge_candidates": [{"entity": "A"}, {"entity": "B"}],
            "potential_contradictions": [{}],
        },
        [{"id": "A"}, {"id": "B"}],
        [],
    )
    assert [q["question"] for q in questions] == ["Q0", "Q1"]
    model.assert_awaited_once()


async def test_source_description_is_not_silently_cut_to_fit_prompt():
    model = AsyncMock(return_value="{}")
    generator = QuestionGenerator(model)
    description = "evidence " * 500 + "critical final condition"
    await generator.analyze_subgraph([{"id": "A", "description": description}], [])
    assert description in model.await_args.args[0]


async def test_synthesis_and_experience_prompts_keep_complete_json_evidence():
    model = AsyncMock(return_value="{}")
    orchestrator, *_ = study(model=model)
    budget = orchestrator._new_budget()
    token = orchestrator._active_budget.set(budget)
    complete_record = {"answer": "x" * 20000 + " critical final condition"}
    try:
        await orchestrator._synthesize_knowledge([complete_record])
        await orchestrator.experience_mgr.extract_experiences(
            complete_record, orchestrator._call_llm
        )
    finally:
        orchestrator._active_budget.reset(token)
    assert model.await_count == 2
    for call in model.await_args_list:
        assert "critical final condition" in call.args[0]
    assert budget.calls == 2


async def test_p6_stops_between_experiences_and_reports_partial_work():
    model = AsyncMock(return_value='{"action":"deprecate"}')
    orchestrator, *_ = study(model=model, max_llm_calls=1)
    orchestrator.experience_mgr._experiences = {
        name: {"experience_id": name, "times_retrieved": 10, "status": "active"}
        for name in ("first", "second")
    }
    assert await orchestrator.run_refinement() == 1
    assert orchestrator.last_llm_usage["llm_calls"] == 1
    assert orchestrator.last_llm_usage["stop_reason"] == "max_llm_calls"
    assert orchestrator.experience_mgr._experiences["first"]["status"] == "deprecated"
    assert orchestrator.experience_mgr._experiences["second"]["status"] == "active"
    assert await orchestrator.run_refinement() == 1
    assert orchestrator.last_llm_usage["llm_calls"] == 1
    assert orchestrator.last_llm_usage["stop_reason"] == "completed"


async def test_p6_token_budget_refuses_prompt_without_calling_or_modifying_experience():
    model = AsyncMock(return_value='{"action":"deprecate"}')
    orchestrator, *_ = study(model=model, max_tokens=100)
    experience = {"experience_id": "first", "times_retrieved": 10, "status": "active"}
    orchestrator.experience_mgr._experiences = {"first": experience}
    assert await orchestrator.run_refinement() == 0
    assert orchestrator.last_llm_usage["stop_reason"] == "token_budget_exhausted"
    assert experience["status"] == "active"
    model.assert_not_awaited()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_budget_configuration_fails_before_work(value):
    with pytest.raises(ValueError):
        StudyWorkBudget(
            max_tokens=value, max_calls=3, output_tokens_per_call=32, timeout_seconds=1
        )


async def test_provider_usage_is_optional_and_separate_from_estimated_budget():
    model = AsyncMock(
        return_value={
            "content": "{}",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )
    budget = StudyWorkBudget(
        max_tokens=1000, max_calls=2, output_tokens_per_call=32, timeout_seconds=1
    )
    assert await budget.call(model, "问") == "{}"
    model.assert_awaited_once_with("问", max_tokens=32)
    report = budget.report()
    assert report["charged_tokens"] == 35
    assert report["provider_usage"]["total_tokens"] == 5
    assert report["provider_reported_calls"] == 1


async def test_reported_provider_overrun_stops_further_calls():
    model = AsyncMock(return_value={"content": "{}", "usage": {"total_tokens": 1001}})
    budget = StudyWorkBudget(
        max_tokens=1000, max_calls=3, output_tokens_per_call=32, timeout_seconds=1
    )
    with pytest.raises(StudyBudgetStop, match="provider_usage_exceeded_budget"):
        await budget.call(model, "input")
    with pytest.raises(StudyBudgetStop, match="provider_usage_exceeded_budget"):
        await budget.call(model, "again")
    model.assert_awaited_once()
    assert budget.report()["charged_tokens"] == 1001


@pytest.mark.parametrize(
    "usage,expected_charge",
    [
        ({"input_tokens": 1001}, 1033),
        ({"prompt_tokens": 1001}, 1033),
        ({"input_tokens": 3, "prompt_tokens": 1001}, 1033),
        ({"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}, 42),
        ({"total_tokens": 10}, 33),
    ],
)
async def test_partial_usage_keeps_missing_reservations_without_double_counting(
    usage, expected_charge
):
    model = AsyncMock(return_value={"content": "{}", "usage": usage})
    budget = StudyWorkBudget(
        max_tokens=1000, max_calls=3, output_tokens_per_call=32, timeout_seconds=1
    )
    if expected_charge > 1000:
        with pytest.raises(StudyBudgetStop, match="provider_usage_exceeded_budget"):
            await budget.call(model, "x")
        with pytest.raises(StudyBudgetStop):
            await budget.call(model, "again")
    else:
        assert await budget.call(model, "x") == "{}"
    assert budget.report()["charged_tokens"] == expected_charge
    model.assert_awaited_once()


async def test_oversized_response_usage_is_charged_before_terminal_signal():
    model = AsyncMock(
        return_value={
            "content": "x" * 300,
            "usage": {"output_tokens": 300, "total_tokens": 301},
        }
    )
    budget = StudyWorkBudget(
        max_tokens=1000, max_calls=3, output_tokens_per_call=32, timeout_seconds=1
    )
    with pytest.raises(StudyBudgetStop, match="output_budget_exceeded"):
        await budget.call(model, "x")
    assert budget.report()["charged_tokens"] == 301
    assert budget.report()["provider_usage"]["total_tokens"] == 301


async def test_provider_that_swallows_timeout_is_accounted_but_not_accepted():
    async def late(_prompt, **_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return {"content": "{}", "usage": {"total_tokens": 100}}

    budget = StudyWorkBudget(
        max_tokens=1000, max_calls=3, output_tokens_per_call=32, timeout_seconds=0.01
    )
    with pytest.raises(StudyBudgetStop, match="llm_timeout"):
        await budget.call(late, "x")
    assert budget.report()["charged_tokens"] == 100
    assert budget.report()["output_limit_requested_calls"] == 1
    assert budget.report()["provider_output_limit_verified"] is False


@pytest.mark.parametrize(
    "analysis", ["解释文字", [], 1, True, None, {"bridge_candidates": "bad"}]
)
async def test_p1_invalid_json_shape_uses_safe_defaults(analysis):
    outputs = [analysis, *RESPONSES[1:]]
    model = AsyncMock(side_effect=[json.dumps(value) for value in outputs])
    orchestrator, *_ = study(model=model)
    result = await orchestrator.run_session()
    assert result.stopped_reason == "max_rounds_reached"
    assert result.llm_usage["llm_calls"] == 5
    assert orchestrator.last_llm_usage["stop_reason"]


@pytest.mark.parametrize(
    "invalid",
    [
        {"questions_per_round": "2"},
        {"max_rounds": -2},
        {"questions_per_round": 0},
        {"max_entities_per_round": False},
        {"llm_timeout_seconds": float("nan")},
        {"audit_only": "true"},
    ],
)
def test_invalid_config_is_rejected_before_any_model_call(invalid):
    model = AsyncMock()
    with pytest.raises(ValueError):
        study(model=model, **invalid)
    model.assert_not_called()


async def test_mutated_config_is_revalidated_before_next_run():
    model = AsyncMock()
    orchestrator, *_ = study(model=model)
    orchestrator.config.questions_per_round = "2"
    with pytest.raises(ValueError):
        await orchestrator.run_session()
    model.assert_not_called()
    assert orchestrator.last_llm_usage["stop_reason"] == "invalid_config"


@pytest.mark.parametrize("phase", ["session", "refinement"])
async def test_unexpected_pipeline_error_is_not_reported_as_completion(phase):
    orchestrator, *_ = study()
    if phase == "session":
        orchestrator.selector.select.side_effect = RuntimeError("selection failed")
        operation = orchestrator.run_session
    else:
        orchestrator.experience_mgr.refine_experiences = AsyncMock(
            side_effect=RuntimeError("refinement failed")
        )
        operation = orchestrator.run_refinement
    with pytest.raises(RuntimeError):
        await operation()
    assert orchestrator.last_llm_usage["stop_reason"] == "pipeline_error"
    assert orchestrator._active_budget.get() is None


@pytest.mark.parametrize(
    "proposal",
    [
        {"action": "refine", "refined_experience": ["bad"]},
        {"action": "merge", "merge_with": ["bad"], "merged_result": {}},
    ],
)
async def test_p6_malformed_updates_are_skipped_without_mutating_experience(proposal):
    model = AsyncMock(return_value=json.dumps(proposal))
    orchestrator, *_ = study(model=model)
    experience = {"experience_id": "first", "times_retrieved": 10, "status": "active"}
    orchestrator.experience_mgr._experiences = {"first": dict(experience)}
    assert await orchestrator.run_refinement() == 0
    assert orchestrator.experience_mgr._experiences["first"] == experience
    assert orchestrator.last_llm_usage["refined_experiences"] == 0


async def test_p6_cancellation_persists_and_reports_completed_items(tmp_path):
    started = asyncio.Event()
    calls = 0

    async def model(_prompt, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return '{"action":"deprecate"}'
        started.set()
        await asyncio.Event().wait()

    orchestrator, *_ = study(model=model)
    orchestrator.experience_mgr._store_path = tmp_path / "experiences.json"
    orchestrator.experience_mgr._experiences = {
        name: {"experience_id": name, "times_retrieved": 10, "status": "active"}
        for name in ("first", "second")
    }
    task = asyncio.create_task(orchestrator.run_refinement())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    persisted = json.loads((tmp_path / "experiences.json").read_text())
    assert persisted["first"]["status"] == "deprecated"
    assert persisted["second"]["status"] == "active"
    assert orchestrator.last_llm_usage["refined_experiences"] == 1
    assert orchestrator.last_llm_usage["llm_calls"] == 2
    assert orchestrator.last_llm_usage["stop_reason"] == "cancelled"


async def test_oversized_output_stops_without_truncation_or_further_calls():
    calls = []

    async def model(prompt):
        calls.append(prompt)
        return "x" * 100

    budget = StudyWorkBudget(
        max_tokens=1000, max_calls=4, output_tokens_per_call=32, timeout_seconds=1
    )
    with pytest.raises(StudyBudgetStop, match="output_budget_exceeded"):
        await budget.call(model, "input")
    with pytest.raises(StudyBudgetStop, match="output_budget_exceeded"):
        await budget.call(model, "again")
    assert len(calls) == 1
    assert budget.report()["observed_output_bytes"] == 100


@pytest.mark.parametrize(
    "strategy,analysis",
    [
        ("two_hop_completion", {"two_hop_gaps": [{}]}),
        ("comparison_alignment", {"same_type_pairs": [{}]}),
        ("weak_component", {"isolated_clusters": [{}]}),
        ("evidence_chain", {"weak_edges": [{}]}),
        ("bridge_entity", {"bridge_candidates": [{"entity": "A"}]}),
        ("two_hop_completion", {"potential_contradictions": [{}]}),
    ],
)
async def test_all_p2_handlers_propagate_terminal_budget_signal(strategy, analysis):
    model = AsyncMock(side_effect=StudyBudgetStop("token_budget_exhausted"))
    generator = QuestionGenerator(model)
    with pytest.raises(StudyBudgetStop, match="token_budget_exhausted"):
        await generator.generate_questions(strategy, analysis, [{"id": "A"}], [])
    model.assert_awaited_once()
