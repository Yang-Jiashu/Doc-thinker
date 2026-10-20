"""Compose bounded graph lookup and idea scaffolding, never a second answer.

This is a local, optional pass on the current project's live sources. It does
not promote hypotheses to evidence, create memories, or call a provider.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from .research_hypotheses import build_hypothesis_support, detect_hypothesis_intent
from .research_prompt_budget import empty_assistance
from .research_retrieval import lexical_tokens


def build_discovery_context(question, plan, assistance, chunks, graph, *, search_question=None):
    """Keep source order/labels stable for already-built proofs and prompts."""
    original = assistance['sources']
    expanded = graph.expand(search_question or question, original, chunks,
                            protected_source_ids=[row['id'] for row in original])
    sources = expanded['sources']
    # A future graph implementation must not silently invalidate the labels or
    # exact source geometry referenced by a previously packed proof.
    if (sources[:len(original)] != original or len(sources) > 8 or
            sum(len(row['content'].encode()) for row in sources) > 12000):
        raise ValueError('Graph expansion violated the source contract')
    live = {row['id']: row for row in chunks}
    for row in sources[len(original):]:
        raw = live.get(row['id'])
        if (raw is None or raw['document_id'] != row['document_id'] or
                raw['content'][row['excerpt_start']:row['excerpt_end']] != row['content']):
            raise ValueError('Graph expansion has no current original source')
    result = deepcopy(assistance)
    result['sources'] = sources
    trace = result['reasoning']['trace']
    trace['graph_retrieval'] = expanded['trace']
    trace['graph_retrieval']['kept_source_ids'] = [row['id'] for row in sources]
    trace['graph_retrieval']['model_calls'] = 0
    updated_plan = deepcopy(plan)
    # Graph additions are retrieval results, not hint-only invented text. Keep
    # these quotes when optional analysis is shed by the prompt budget.
    if len(sources) > len(original):
        updated_plan['sources'] = updated_plan['selected_chunks'] = sources
        updated_plan['source_bytes'] = sum(len(row['content'].encode()) for row in sources)
        updated_plan['evidence_ids'] = [row['id'] for row in sources]
        updated_plan['context'] = '\n\n'.join(f"[{row['label']}] {row['filename']}\n{row['content']}" for row in sources)
        covered = set(lexical_tokens('\n'.join(row['content'] for row in sources)))
        query = set(plan.get('query_terms', []))
        updated_plan['covered_query_terms'] = sorted(query & covered)
        updated_plan['gaps'] = [row for row in plan['gaps']
                                if row['code'] not in {'uncovered_query_terms', 'documents_not_selected', 'insufficient_evidence'}]
        if query - covered:
            updated_plan['gaps'].append({'code': 'uncovered_query_terms', 'terms': sorted(query - covered),
                'message': '部分问题用词仍没有字面匹配；图关联不代表找齐了证据。'})
        missing_docs = {row['document_id'] for row in chunks} - {row['document_id'] for row in sources}
        if missing_docs:
            updated_plan['gaps'].append({'code': 'documents_not_selected', 'count': len(missing_docs),
                'message': '部分资料未进入本次上下文；未选中不等于不存在相关证据。'})
        updated_plan['digest'] = hashlib.sha256(json.dumps({
            'parent': plan.get('digest'), 'sources': sources}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    updated_plan['gaps'].extend(expanded['gaps'])
    if detect_hypothesis_intent(question):
        # The three operators run on the same selected excerpts. Replace the
        # older single-operator optional block, not the original evidence.
        hypothesis = build_hypothesis_support(question, sources)
        if hypothesis['prompt']:
            saved_trace = deepcopy(trace)
            result = empty_assistance(sources)
            result['prompt'] = hypothesis['prompt']
            result['reasoning']['status'] = 'hypothesis_support'
            result['reasoning']['trace'].update(saved_trace)
            result['reasoning']['trace'].update(
                hypothesis_support=hypothesis['trace'],
                algorithm_modes=hypothesis['trace']['algorithm_modes'],
                assist_bytes=len(hypothesis['prompt'].encode()), skipped_reason=None)
        else:
            trace['hypothesis_support'] = hypothesis['trace']
    result['trace'] = result['reasoning']['trace']
    return {'plan': updated_plan, 'assistance': result}


def fit_graph_receipt(trace, sources):
    """Describe the final context, not candidates later removed for budget."""
    result = deepcopy(trace)
    kept = {row['id'] for row in sources}
    bridges = [item for item in result.get('bridge_source_ids', []) if item in kept]
    result['bridge_source_ids'] = bridges
    result['kept_source_ids'] = [row['id'] for row in sources]
    by_id = {row['id']: row for row in sources}
    def window_kept(window):
        row = by_id.get(window.get('source_id'))
        if row is None:
            return False
        left, right = window.get('excerpt_start'), window.get('excerpt_end')
        base = row.get('excerpt_start', 0)
        if (type(left) is not int or type(right) is not int or type(base) is not int
                or not base <= left < right <= base + len(row['content'])):
            return False
        quote = row['content'][left - base:right - base]
        return hashlib.sha256(quote.encode()).hexdigest() == window.get('content_sha256')
    result['paths'] = [path for path in result.get('paths', [])
                       if set(path.get('source_ids', [])) <= kept
                       and all(window_kept(window) for window in path.get('source_windows', []))]
    if len(result['paths']) < len(trace.get('paths', [])):
        result['path_window_omitted'] = True
    if result.get('status') == 'expanded' and not bridges:
        result['status'] = 'unchanged'
        result['budget_omitted_bridges'] = True
    result['evidence_scope'] = 'final_context_not_full_corpus'
    return result
