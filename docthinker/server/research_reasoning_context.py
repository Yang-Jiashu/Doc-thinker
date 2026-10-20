"""Atomic proof packing: an incomplete proof is omitted, never abbreviated."""
from __future__ import annotations


def render_evidence(sources):
    return ('## 原文摘录\n\n' + '\n\n'.join(f"{source['content']} [{source['label']}]" for source in sources)
            if sources else '原文已撤回或不可用，没有可以继续展示的摘录。')


def proof_claim_ids(conclusion, claims):
    """Resolve every DAG leaf, not merely one quote from the same source."""
    nodes = {node['id']: node for node in conclusion.get('proof', [])}
    resolved, visiting = set(), set()
    def visit(identifier, depth=0):
        if identifier in claims:
            resolved.add(identifier)
            return True
        if identifier in visiting or depth > 8:
            return False
        node = nodes.get(identifier)
        if not node or node.get('kind') != 'deduced' or not node.get('premise_ids'):
            return False
        visiting.add(identifier)
        complete = all(visit(parent, depth + 1) for parent in node['premise_ids'])
        visiting.remove(identifier)
        return complete
    roots = [*conclusion.get('premise_ids', []), *nodes,
             *[item['base_claim_id'] for item in conclusion.get('candidate_inferences', [])]]
    if not roots or not all(visit(identifier) for identifier in roots):
        return None
    covered = {source for identifier in resolved for source in claims[identifier]['source_ids']}
    if not set(conclusion.get('source_ids', [])) <= covered:
        return None
    return resolved


def pack_reasoning(reasoning, claims, chunks, *, max_sources=8, max_bytes=12000):
    live = {row['id']: row for row in chunks}
    accepted, needed, gaps = [], {}, list(reasoning.get('gaps', []))
    claim_ids = {row['id']: row for row in claims}
    for conclusion in reasoning['conclusions']:
        ids = proof_claim_ids(conclusion, claim_ids)
        proof_claims = [claim_ids[item] for item in sorted(ids)] if ids else []
        covered = {source for claim in proof_claims for source in claim['source_ids']}
        source_ids = set(conclusion['source_ids'])
        if (ids is None or source_ids != covered or not covered <= live.keys()
                or any(not 0 <= claim['excerpt_start'] < claim['excerpt_end'] <= len(live[claim['source_ids'][0]]['content'])
                       or live[claim['source_ids'][0]]['content'][claim['excerpt_start']:claim['excerpt_end']] != claim['quote'] for claim in proof_claims)):
            gaps.append({'code': 'incomplete_proof_sources', 'message': '部分前提无法绑定当前原文，已省略该结论。'})
            continue
        proposed = {**needed}
        for claim in proof_claims:
            for source in claim['source_ids']:
                left, right = proposed.get(source, (claim['excerpt_start'], claim['excerpt_end']))
                proposed[source] = (min(left, claim['excerpt_start']), max(right, claim['excerpt_end']))
        if (len(proposed) > max_sources or any(right - left > 1000 for left, right in proposed.values())
                or sum(len(live[source]['content'][left:right].encode()) for source, (left, right) in proposed.items()) > max_bytes):
            gaps.append({'code': 'proof_budget_reached', 'message': '完整前提超过本次证据预算，已整体省略该结论；不会只给半条证明。'})
            continue
        needed = proposed
        accepted.append(conclusion)
    sources = []
    for source, (left, right) in sorted(needed.items()):
        raw = live[source]
        sources.append({**{key: raw[key] for key in ('id', 'document_id', 'filename', 'page', 'ordinal') if key in raw}, 'label': f'S{len(sources) + 1}', 'content': raw['content'][left:right],
                        'excerpt_start': left, 'excerpt_end': right,
                        'original_content_length': len(raw['content']), 'truncated': left > 0 or right < len(raw['content']),
                        'offset_unit': 'characters_in_source_chunk', 'evidence_scope': 'excerpt_only'})
    kept = [row for row in claims if row['source_ids'][0] in needed
            and needed[row['source_ids'][0]][0] <= row['excerpt_start'] < row['excerpt_end'] <= needed[row['source_ids'][0]][1]]
    return {**reasoning, 'conclusions': accepted, 'claims': kept,
            'status': 'derived' if accepted else 'insufficient_evidence',
            'gaps': list({row['code']: row for row in gaps}.values())[:8]}, sources


def render_reasoning(reasoning, sources):
    labels = {row['id']: row['label'] for row in sources}
    names = {'deduction': '演绎推导', 'analogy': '结构类比', 'induction': '样本归纳'}
    lines = [f"## {names[reasoning['mode']]}", '']
    if not reasoning['conclusions']:
        return '\n'.join(lines + ['在本次可解析的原文与预算范围内，没有找到前提完整的结果。缺少证据不等于结论一定错误；请补充明确的规则、观察或可比较案例。'])
    lines += ['以下是有界算法结果，不是对原文真实性的认证；不会自动写入长期记忆。', '']
    for index, item in enumerate(reasoning['conclusions'], 1):
        cites = ' '.join(f'[{labels[source]}]' for source in item['source_ids'])
        lines += [f"{index}. {item['text']} {cites}"]
        if item['kind'] == 'deduced':
            lines.append('成立条件：原文规则与所有前提成立；仅表示规则系统内可推导，不新增现实因果事实。')
            if item.get('scope') and item['scope'] != 'project':
                lines.append('适用情境：' + item['scope'])
        if item['kind'] == 'analogy_hypothesis':
            mapping = item.get('mappings', item.get('mapping', {}))
            if isinstance(mapping, dict):
                lines.append('角色对应：' + '；'.join(f'{a} → {b}' for a, b in mapping.items()))
            elif isinstance(mapping, list):
                lines.append('角色对应：' + '；'.join(f"{pair['base']} → {pair['target']}" for pair in mapping))
        for candidate in item.get('candidate_inferences', []):
            conditions = '、'.join(candidate.get('conditions', [])) or '来源未明确给出'
            lines.append(f"待验证迁移假设：{candidate['text']}；沿用来源条件：{conditions}；目标场景条件未验证。")
    lines += ['', '## 原文前提']
    for claim in reasoning.get('claims', [])[:24]:
        lines.append(f"- {claim['quote']} [{labels[claim['source_ids'][0]]}]")
    if len(reasoning.get('claims', [])) > 24:
        lines.append('更多前提可在推理与来源卡片核对；此处只展示前 24 条。')
    return '\n'.join(lines)
