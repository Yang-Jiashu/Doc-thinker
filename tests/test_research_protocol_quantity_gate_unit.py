"""No second model call or paper-fact laundering through a proposal field."""
from copy import deepcopy

import pytest

from docthinker.server import research_decision_protocol as protocol
from tests.test_research_decision_protocol_unit import fixture, parse


@pytest.mark.parametrize('field', ['findings', 'do_now', 'hypothesis', 'baseline'])
def test_paper_section_number_cannot_ground_invented_option_or_document_limits(field):
    data, sources = fixture()
    sources[0]['content'] = 'Section 3: 1. Keep actions simple with few options and concise documentation. 2. Consolidate actions.'
    data[field] = '论文要求命令选项≤3、文档≤2行。[S1]'
    result = parse(data, sources)
    assert result['validation']['status'] == 'draft'
    assert field in result['quantity_grounding']['withheld_fields']
    assert data[field] not in protocol.render(result)
    assert '数值归因，已收起' in protocol.render(result)
    assert result['quantity_grounding']['extra_model_calls'] == 0
    assert not result['quantity_grounding']['semantic_verified']


def test_explicit_our_choice_is_not_a_paper_fact_or_a_semantic_certificate():
    data, sources = fixture()
    data['do_now'] = '我们拟定选项≤3[S1]，文档保持简明；这是待测设计，不是论文给的数值。'
    result = parse(data, sources)
    assert result['validation']['status'] == 'review_required'
    check = result['quantity_grounding']['checks']['do_now']
    assert check['status'] == 'proposed_not_source_claim' and not check['semantic_verified']
    assert data['do_now'] in protocol.render(result)


def test_exact_same_unit_is_only_a_numeric_match_and_wrong_unit_is_withheld():
    data, sources = fixture()
    sources[0]['content'] = 'SWE-bench resolved 12.5%; ablation increased by 10.7 percentage points.'
    data['findings'] = '原文报告12.5%，不是本产品结果。[S1]'
    result = parse(data, sources)
    assert result['quantity_grounding']['checks']['findings']['status'] == 'direct_numeric_match'
    assert not result['quantity_grounding']['semantic_verified']
    data['findings'] = '原文报告10.7%，不是本产品结果。[S1]'
    assert parse(data, sources)['validation']['status'] == 'draft'


def test_quantity_receipt_is_detached_and_tampering_cannot_unhide_the_field():
    data, _ = fixture(); data['findings'] = '论文要求3个选项。[S1]'
    result = parse(data)
    receipt = protocol.receipt(result, ['source-a'], requested_contract=protocol.contract())
    receipt['quantity_grounding']['withheld_fields'].clear()
    assert result['quantity_grounding']['withheld_fields'] == ['findings']
    changed = deepcopy(result); changed['quantity_grounding']['withheld_fields'].clear()
    with pytest.raises(ValueError, match='binding_changed'):
        protocol.render(changed)


def test_invalid_protocol_does_not_bypass_findings_quantity_gate():
    data, _ = fixture(); data['findings'] = '论文要求3个选项。[S1]'; del data['baseline']
    result = parse(data)
    assert result['brief'] is None and data['findings'] not in protocol.render(result)
