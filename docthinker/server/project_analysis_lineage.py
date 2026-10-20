"""Bounded host-owned replacement projection over an approved analysis DAG.

Only typed ``refine_of`` replaces a body. A normal dependency, same question,
title or model receipt cannot supersede a task. Pending refinements also retire
old goal contributions; a failed new attempt must not resurrect old proof.
Historical artifacts and the complete provenance manifest remain untouched.
"""
from __future__ import annotations


def superseded_step_ids(steps):
    rows = [dict(row) for row in steps if row['kind'] == 'analysis']
    if len(rows) > 12:
        raise ValueError('Analysis replacement projection exceeds the approved DAG capacity')
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate analysis identities')
    superseded = set()
    for row in rows:
        target, visited = row.get('refine_of'), set()
        while target in by_id and target not in visited:
            visited.add(target)
            superseded.add(target)
            prior = by_id[target]
            target = prior.get('review_of') if prior.get('purpose') == 'verify' else prior.get('refine_of')
    for row in rows:
        target, visited = row['id'], set()
        while target in by_id and target not in visited:
            visited.add(target)
            if target in superseded:
                superseded.add(row['id'])
                break
            prior = by_id[target]
            # A review copies its target body, but a refinement writes a NEW
            # body. Do not follow refine_of here and retire the new result too.
            target = prior.get('review_of') if prior.get('purpose') == 'verify' else None
    return frozenset(superseded)
