"""Offline hygiene audit of frozen source packets, not generated-answer quality.

No source is fetched and no provider/model is invoked. This measures only
whole-source removals and exact preservation using existing public-paper runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('PYTHON_DOTENV_DISABLED', '1')

from docthinker.server.research_fallback_sources import filter_fallback_sources, source_filter_reason


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def audit(run_dir):
    cases = []
    for path in sorted((run_dir / 'answers').glob('*.off.json')):
        record = json.loads(path.read_text())
        sources = record['result']['sources']
        if not sources:
            continue
        snapshot = json.loads(json.dumps(sources))
        result = filter_fallback_sources(record['question'], sources)
        kept = result['sources']
        removed = [row for row in sources if not any(row is current for current in kept)]
        assert sources == snapshot
        assert all(any(row is original for original in sources) for row in kept)
        assert filter_fallback_sources('List the references and author affiliations.', sources)['sources'] == sources
        # Content cannot be thrown away just because it follows a bibliography
        # or a generic introduction. Added uncertainty preserves the whole row.
        assert all(source_filter_reason(record['question'], row['content'] +
                                        '\nHowever, these results do not establish the comparison.') is None
                   for row in removed)
        cases.append({
            'case': path.stem, 'question': record['question'], 'provider_calls': 0, 'model_calls': 0,
            'scope': 'frozen_source_packet_hygiene_not_new_retrieval_or_answer_quality',
            'input_source_count': len(sources), 'retained_source_count': len(kept),
            'input_source_bytes': sum(len(row['content'].encode()) for row in sources),
            'retained_source_bytes': sum(len(row['content'].encode()) for row in kept),
            'reason_counts': result['trace']['reason_counts'],
            'removed': [{'label': row['label'], 'filename': row['filename'], 'page': row.get('page'),
                         'reason': source_filter_reason(record['question'], row['content']),
                         'source_digest': digest(row)} for row in removed],
            'retained_labels': [row['label'] for row in kept],
            'retained_sources_digest': digest(kept),
            'exact_content_labels_offsets_preserved': True,
            'reference_list_query_retained_all': True,
            'mixed_negated_statement_retained_whole': True,
        })
    assert cases, 'No recorded OFF source packets found'
    return {'schema': 'source_hygiene_offline_v1', 'input_directory': str(run_dir),
            'provider_calls': 0, 'quality_score': None, 'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root, output = args.run_dir.resolve(), args.output.resolve()
    assert output != root and root not in output.parents, 'Keep original benchmark artifacts unchanged'
    result = audit(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
