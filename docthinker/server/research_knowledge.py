"""Task-knowledge orchestration and receipts, separate from document evidence.

No model, network, memory/index writes, or caller-selected resource paths.
Only the current task routes; historical topics cannot silently activate a pack.
"""
from __future__ import annotations

import re

from .knowledge_skills import MAX_PROMPT_BYTES, _payload, route_knowledge_skill, retrieve_knowledge_skill
from .method_planner import compile_method_plan
from .research_citation_syntax import parse_citation_groups


def select_task_knowledge(question, mode='auto', *, literal=False):
    routed = route_knowledge_skill(question, mode)
    if literal:
        routed = {**routed, 'skill_id': None,
                  'trace': {**routed['trace'], 'status': 'not_applicable', 'task_type': None}}
    selected = retrieve_knowledge_skill(routed['skill_id'], question) if routed['skill_id'] else None
    return selected, routed['trace']


def compile_task_knowledge(support, sources):
    """Bind methods to the actual excerpts, retaining whole cards and boundaries.

    The planner checks excerpt structure only. Natural-language prerequisites
    stay pending; an S label or a matching word cannot discharge them. This
    runs on a fitted source variant, never on history or a different source set.
    """
    if not support or not support.get('prompt') or support['trace']['status'] != 'ready':
        return support
    methods = list(support['sources'])
    omitted = 0
    while methods:
        method_plan = compile_method_plan(methods, sources, skill_id=support['skill']['id'])
        prompt = _payload(support['skill'], methods, method_plan=method_plan)
        if len(prompt.encode('utf-8')) <= MAX_PROMPT_BYTES:
            return {**support, 'sources': methods, 'prompt': prompt, 'method_plan': method_plan,
                    'trace': {**support['trace'], 'selected_count': len(methods),
                              'prompt_bytes': len(prompt.encode('utf-8')),
                              'budget_omitted_count': support['trace']['budget_omitted_count'] + omitted,
                              'method_planner': method_plan['version']}}
        # Keep the highest-ranked complete methods. Counterexamples, required
        # facts and stop conditions never get truncated to fit a plan receipt.
        methods.pop()
        omitted += 1
    return {**support, 'sources': [], 'prompt': '',
            'trace': {**support['trace'], 'status': 'budget_omitted', 'selected_count': 0,
                      'prompt_bytes': 0,
                      'budget_omitted_count': support['trace']['budget_omitted_count'] + omitted}}


def knowledge_receipt(support, *, included, answer, model_called):
    if not support or not support.get('skill'):
        return None
    status = support['trace']['status']
    if status == 'ready' and not model_called:
        status = 'not_applicable'
    elif status == 'ready' and not included:
        status = 'budget_omitted'
    sources = support['sources'] if status == 'ready' else []
    labels = {row['label'] for row in sources}
    citations = parse_citation_groups(answer)
    used = {label for label in citations['labels'] if label.startswith('K')}
    return {**support['skill'], 'status': status, 'sources': sources,
            **({'method_plan': support['method_plan']} if status == 'ready' and support.get('method_plan') else {}),
            'authorship': 'project_original_method_design',
            'validation_status': 'design_unvalidated',
            'citation_validation': {
                'unknown_labels': sorted(used - labels)[:32],
                'missing_citations': bool(labels and not used.intersection(labels)),
                **({'metadata_truncated': True} if citations['truncated'] else {}),
                'entailment_verified': False}}


def unavailable_knowledge(receipt):
    """An inactive card must not retain methods from a hidden derived answer."""
    return {**{key: value for key, value in receipt.items() if key != 'method_plan'},
            'status': 'unavailable', 'sources': [],
            'citation_validation': {'unknown_labels': [], 'missing_citations': False,
                                    'entailment_verified': False}}
