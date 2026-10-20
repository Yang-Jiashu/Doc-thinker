"""The real-model acceptance declares each paid stage before dispatch."""
import ast
from pathlib import Path

from scripts import benchmark_goal_evidence_staged as staged


def test_staged_registration_declares_eight_distinct_compose_and_verify_calls():
    assert len(staged.CALL_IDS) == len(set(staged.CALL_IDS)) == 8
    for round_id in ('work.initial', 'work.updated', 'home.current', 'paper.current'):
        offset = staged.CALL_IDS.index(round_id + '.compose')
        assert staged.CALL_IDS[offset + 1] == round_id + '.verify'
    assert staged.BASE == 'https://api.stepfun.com/step_plan/v1'
    assert staged.MODEL == 'step-5-preview'
    assert len(staged.PAPER_SHA) == 64
    assert len(staged.rubric()['work.updated.compose']) == 4


def test_staged_verification_uses_the_shared_production_target_builder_and_no_generation_override():
    # A small structural guard complements the actual zero-API preparation;
    # it does not claim model quality or replace the live acceptance.
    source = Path(staged.__file__).read_text()
    parsed = ast.parse(source)
    assert 'from docthinker.server.research_review_stage import build_review_target' in source
    model_calls = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == 'RealStepModel']
    assert len(model_calls) == 1
    assert 'max_output_tokens' not in {kw.arg for kw in model_calls[0].keywords}
    assert 'temperature' not in {kw.arg for kw in model_calls[0].keywords}
    assert "'automatic_retry': False" in source
    assert "'draft_preserved': draft is not None" in source
