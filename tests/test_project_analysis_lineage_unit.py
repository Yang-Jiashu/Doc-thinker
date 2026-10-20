"""Replacement projection is explicit, bounded and independent of model text."""
import pytest

from docthinker.server.project_analysis_lineage import superseded_step_ids


def step(key, **kwargs):
    return {'id': key, 'kind': 'analysis', 'question': 'Same question', **kwargs}


def test_refinement_of_review_retires_old_body_and_all_its_audit_copies():
    rows = [step('a'), step('av', purpose='verify', review_of='a'),
            step('av2', purpose='verify', review_of='av'),
            step('b', refine_of='av', status='ready'),
            step('bv', purpose='verify', review_of='b'), step('ordinary', depends_on=['a'])]
    assert superseded_step_ids(rows) == {'a', 'av', 'av2'}
    rows.extend([step('c', refine_of='bv'), step('cv', purpose='verify', review_of='c')])
    assert superseded_step_ids(rows) == {'a', 'av', 'av2', 'b', 'bv'}


def test_same_question_normal_dependencies_and_model_receipts_do_not_replace():
    assert superseded_step_ids([step('a'), step('b', depends_on=['a']),
        step('review', purpose='verify', review_of='a', model_says_replaced=True)]) == set()


def test_no_unbounded_scan_or_duplicate_identity():
    with pytest.raises(ValueError):
        superseded_step_ids([step(str(i)) for i in range(13)])
    with pytest.raises(ValueError):
        superseded_step_ids([step('a'), step('a')])
