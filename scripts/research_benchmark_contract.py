"""Pure delivery eligibility checks shared by generation and offline analysis.

No provider client, model imports, configuration loading, or network access.
These checks reject failed delivery, not ordinary answer-quality weaknesses.
"""

DELIVERY_GATE_VERSION = 1
PROVENANCE_FAILURE_CODES = frozenset({'provenance_unavailable'})


def delivery_failure_reason(result, *, require_sources=True):
    if not isinstance(result, dict):
        return 'delivery_invalid_result'
    answer = result.get('answer')
    if not isinstance(answer, str) or not answer.strip():
        return 'delivery_missing_answer'
    reasoning = result.get('reasoning')
    if isinstance(reasoning, dict) and reasoning.get('status') == 'source_unavailable':
        return 'delivery_source_unavailable'
    validation = result.get('citation_validation')
    warnings = validation.get('warnings', []) if isinstance(validation, dict) else []
    if isinstance(warnings, list) and any(isinstance(row, dict) and row.get('code') in
                                         PROVENANCE_FAILURE_CODES for row in warnings):
        return 'delivery_provenance_unavailable'
    knowledge = result.get('knowledge_skill')
    if (result.get('research_provenance_required') is True and isinstance(knowledge, dict)
            and knowledge.get('status') == 'unavailable'):
        return 'delivery_dependency_unavailable'
    sources = result.get('sources')
    if not isinstance(sources, list):
        return 'delivery_invalid_sources'
    if not sources and (require_sources or result.get('research_provenance_required') is True):
        return 'delivery_missing_sources'
    labels = set()
    for source in sources:
        if (not isinstance(source, dict) or not isinstance(source.get('label'), str)
                or not source['label'] or source['label'] in labels
                or not isinstance(source.get('content'), str) or not source['content'].strip()
                or source.get('available') is False):
            return 'delivery_invalid_sources'
        labels.add(source['label'])
    # A partial source_revoked warning with remaining usable sources and an
    # ordinary answer is reviewable; unavailable dependencies above are not.
    # Missing/unknown citations and weak entailment belong to quality scoring.
    # Do not hide them by requiring citation_validation.valid == True here.
    return None
