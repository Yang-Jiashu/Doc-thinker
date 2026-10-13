#!/usr/bin/env python3
"""Offline invented formatter benchmark, not model latency or answer quality.

Requires the full development checkout and test fixtures. Prints only counts,
timings and hashes; no keys, actual user files or network/model operations.
Every arm compares the complete formatter result against a frozen reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ['PYTHON_DOTENV_DISABLED'] = '1'
os.environ['DOCTHINKER_RESEARCH_SEMANTIC'] = 'off'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_research_prompt_render_cache_unit import ORACLE, current, fixture, run


def summary(samples):
    ordered = sorted(samples)
    return {'samples_ms': [round(value, 3) for value in samples],
            'median_ms': round(statistics.median(samples), 3),
            'p95_ms': round(ordered[math.ceil(.95 * len(ordered)) - 1], 3),
            'p95_method': 'nearest_rank_observed_not_production_tail', 'n': len(samples)}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure_case(name, options, repeats):
    value, support, history, fit_options = fixture(**options)
    expected = run(ORACLE.fit_research_prompt, value, support, history, fit_options)
    if run(current.fit_research_prompt, value, support, history, fit_options) != expected:
        raise AssertionError('Formatter changed; no timing accepted')
    elapsed = {'before': [], 'after': []}
    functions = {'before': ORACLE.fit_research_prompt, 'after': current.fit_research_prompt}
    for repeat in range(repeats):
        order = ('before', 'after') if repeat % 2 == 0 else ('after', 'before')
        for arm in order:
            started = time.perf_counter()
            actual = run(functions[arm], value, support, history, fit_options)
            duration = (time.perf_counter() - started) * 1000
            if actual != expected:
                raise AssertionError('Whole result changed; no timing accepted')
            elapsed[arm].append(duration)
    render_counts = {}
    for arm, module in (('before', ORACLE), ('after', current)):
        function, calls = module.grounded_prompt, []
        def counted(*args, **kwargs):
            calls.append(1)
            return function(*args, **kwargs)
        module.grounded_prompt = counted
        try:
            if run(module.fit_research_prompt, value, support, history, fit_options) != expected:
                raise AssertionError('Instrumented result changed')
            render_counts[arm] = len(calls)
        finally:
            module.grounded_prompt = function
    result_hash = hashlib.sha256(json.dumps(expected, ensure_ascii=True, sort_keys=True,
                                           separators=(',', ':')).encode()).hexdigest()
    return {'case': name, 'whole_result_equal': True, 'result_sha256': result_hash,
            'source_count_input': len(value['sources']), 'result_status': expected[0],
            'kept_sources': len(expected[1]['plan']['sources']) if expected[0] == 'ok' else None,
            'base_render_calls': render_counts, **{arm: summary(values) for arm, values in elapsed.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5, choices=range(1, 10))
    args = parser.parse_args()
    paths = (ROOT / 'docthinker/server/research_prompt_budget.py',
             ROOT / 'tests/fixtures/reference_research_prompt_20261013.py')
    before = {str(path.relative_to(ROOT)): fingerprint(path) for path in paths}
    cases = [('ordinary', {'extension': 0, 'knowledge': True}),
             ('many_budget_variants', {'extension': 18, 'knowledge': True, 'trial': True, 'policy_size': 16000}),
             ('over_budget_failure', {'extension': 90, 'knowledge': True, 'trial': True, 'policy_size': 26000})]
    result = {'measurement_scope': 'pure complete formatter; no retrieval, IO, event-loop scheduling or provider',
              'data': 'invented six sources, full developer fixtures', 'model_calls': 0,
              'cases': [measure_case(name, options, args.repeats) for name, options in cases]}
    after = {str(path.relative_to(ROOT)): fingerprint(path) for path in paths}
    if after != before:
        raise AssertionError('Code changed during benchmark; timings rejected')
    result['fingerprints'] = before
    result['fingerprints_unchanged'] = True
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
