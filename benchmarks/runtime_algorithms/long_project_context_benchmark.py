"""External real-SQLite ancestor traversal benchmark; no providers or network.

Reuses the production unit test's phase/task construction helpers.  The bridge
deliberately returns True: this isolates TaskStore traversal/connection cost,
not ResearchStore freshness, semantic correctness, model quality, or release
acceptance.  Production validity must also pass independent real-store tests.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import runpy
import sqlite3
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Copied verbatim from the pre-optimization LongProjectContext.valid method on
# 2026-10-20, with only its name changed and method indentation removed. Its bound is
# frozen separately so changing production constants cannot rewrite baseline.
OLD_MAX_ANCESTORS_PER_CHECK = 128
OLD_VALID_SOURCE = '''def old_valid(self, task_id):
    """Bounded BFS with O(1) pending/visited membership; no provider calls."""
    try:
        queue, pending, visited = deque([task_id]), {task_id}, set()
        while queue:
            current = queue.popleft(); pending.discard(current)
            if current in visited:
                continue
            visited.add(current)
            if len(visited) > MAX_ANCESTORS_PER_CHECK:
                return False  # Explicit bounded inspection, never silent skipping.
            for ref in self.projects.parent_dependencies(current):
                parent = self.tasks.get_task(ref['task_id'])
                selected = [row for row in parent['steps'] if row['id'] in ref['step_versions']]
                if len(selected) != len(ref['step_versions']) or not self.bridge.validate_inputs(selected):
                    return False
                if ref['task_id'] not in pending and ref['task_id'] not in visited:
                    pending.add(ref['task_id']); queue.append(ref['task_id'])
        return True
    except (ValueError, sqlite3.Error, KeyError, TypeError):
        return False
'''
_old_namespace = {'deque': deque, 'sqlite3': sqlite3,
                  'MAX_ANCESTORS_PER_CHECK': OLD_MAX_ANCESTORS_PER_CHECK}
exec(compile(OLD_VALID_SOURCE, '<frozen-pre-optimization-valid>', 'exec'), _old_namespace)
old_valid = _old_namespace['old_valid']

from docthinker.server.long_project_context import LongProjectContext


def input_group(rows):
    """Exactly the input fields batched by the optimized implementation."""
    return [{key: row.get(key) for key in (
        'input_kind', 'document_ids', 'input_snapshot', 'context_snapshot')}
        for row in rows]


class FakeBridge:
    """No API: count both old per-parent and new batched freshness calls."""
    def __init__(self, tasks):
        self.session_id, self.instance_id = tasks.session_id, tasks.instance_id
        self.manager = self.service = self
        self.reset()

    def reset(self):
        self.validate_inputs_calls = 0
        self.task_input_groups_live_calls = 0
        self.manager_get_session_calls = 0
        self.groups_received = 0
        self.unique_groups = set()

    def get_session(self, session_id):
        self.manager_get_session_calls += 1
        assert session_id == self.session_id
        return {'instance_id': self.instance_id}

    def validate_inputs(self, rows):
        self.validate_inputs_calls += 1
        self.record([input_group(rows)])
        return True

    def task_input_groups_live(self, session_id, groups, *, deadline=None):
        assert session_id == self.session_id
        assert deadline is None or time.monotonic() < deadline
        self.task_input_groups_live_calls += 1
        self.record(groups)
        return True

    def record(self, groups):
        self.groups_received += len(groups)
        for group in groups:
            self.unique_groups.add(json.dumps(group, sort_keys=True,
                                             separators=(',', ':'), allow_nan=False))


def measured(tasks, bridge, check, task_id):
    bridge.reset()
    original_db = tasks._db
    counts = {'taskstore_db_calls': 0, 'taskstore_write_db_calls': 0}

    def counted_db(*args, **kwargs):
        counts['taskstore_db_calls'] += 1
        counts['taskstore_write_db_calls'] += int(kwargs.get('write', False))
        return original_db(*args, **kwargs)

    tasks._db = counted_db
    try:
        started = time.perf_counter()
        valid = check(task_id)
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        tasks._db = original_db
    return {'valid': valid, 'elapsed_ms': elapsed_ms, **counts,
            'bridge_validate_inputs_calls': bridge.validate_inputs_calls,
            'bridge_batch_calls': bridge.task_input_groups_live_calls,
            'bridge_groups_received': bridge.groups_received,
            'unique_selected_input_groups': len(bridge.unique_groups),
            'manager_get_session_calls': bridge.manager_get_session_calls}


def benchmark(size, repeats, helpers):
    with tempfile.TemporaryDirectory(prefix='docthinker-phase-benchmark-') as directory:
        started = time.perf_counter()
        tasks, store = helpers['opened'](Path(directory))
        root, prior = helpers['project'](store), None
        for index in range(size):
            parents = [{'phase_id': prior['id'], 'report_version': 1,
                        'step_ids': ['draft']}] if prior else []
            phase = helpers['add'](store, root, parents=parents,
                                   title='Benchmark phase ' + str(index + 1))
            helpers['complete'](tasks, phase['task_id'])
            prior = store.get_phase(root['id'], phase['id'])
        setup_seconds = time.perf_counter() - started
        bridge = FakeBridge(tasks)
        context = LongProjectContext(tasks, bridge)
        results = {'old': [], 'new': []}
        # Alternate order to avoid assigning every first-touch read to one arm.
        for iteration in range(repeats):
            for name in ('old', 'new') if iteration % 2 == 0 else ('new', 'old'):
                check = (lambda task_id: old_valid(context, task_id)) if name == 'old' else context.valid
                results[name].append(measured(tasks, bridge, check, prior['task_id']))
        summaries = {}
        for name, samples in results.items():
            summaries[name] = {'median_elapsed_ms': statistics.median(
                row['elapsed_ms'] for row in samples), 'min_elapsed_ms': min(
                row['elapsed_ms'] for row in samples), 'max_elapsed_ms': max(
                row['elapsed_ms'] for row in samples), 'samples': samples}
        return {'completed_phases': size, 'parent_edges': size - 1,
                'setup_seconds': setup_seconds, **summaries,
                'median_speedup_old_div_new': summaries['old']['median_elapsed_ms'] /
                    summaries['new']['median_elapsed_ms']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[30, 150])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20 or any(not 2 <= size <= 300 for size in args.sizes):
        parser.error('Use repeats 1..20 and sizes 2..300.')
    helpers = runpy.run_path(str(ROOT / 'tests/test_long_project_store_unit.py'))
    production = ROOT / 'docthinker/server/long_project_context.py'
    output = {'benchmark': 'real SQLite completed linear phase DAG; fake input freshness bridge',
        'scope': 'TaskStore algorithm and connection overhead only; no model/provider/network; '
                 'not real ResearchStore freshness or answer-quality acceptance',
        'frozen_baseline_function_sha256': hashlib.sha256(OLD_VALID_SOURCE.encode()).hexdigest(),
        'frozen_baseline_max_ancestors': OLD_MAX_ANCESTORS_PER_CHECK,
        'production_context_sha256': hashlib.sha256(production.read_bytes()).hexdigest(),
        'sizes': [benchmark(size, args.repeats, helpers) for size in args.sizes]}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
