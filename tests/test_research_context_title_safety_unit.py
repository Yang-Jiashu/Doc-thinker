"""The real static-assistance entry must not resurrect excluded source metadata."""
from copy import deepcopy

from docthinker.server.research_context_assist import build_context_assist, _replacement_loss_reason


def test_actual_static_assistance_preserves_body_seed_without_restoring_header():
    rows, seeds = [], []
    for i in range(6):
        text = f'CedarFlow recall retrieves record group {i} using an index. It does not expose private records.'
        raw = dict(id=f'c{i}', document_id=f'd{i}', filename=f'f{i}.pdf', ordinal=0, page=1, content=text)
        rows.append(raw)
        seeds.append(dict(raw, label=f'S{i + 1}', excerpt_start=0, excerpt_end=len(text)))
    header = 'CedarFlow: Recall Architecture for Record Retrieval\nAlice Example\nalice@example.org\nAbstract\n'
    body = ('CedarFlow recall indexes stored records and retrieves the selected records. '
            'It does not preserve deleted records, and access requires permission.')
    raw = dict(id='head', document_id='dhead', filename='upload.pdf', ordinal=0, page=1, content=header + body)
    rows.append(raw)
    seeds.append(dict(raw, label='S7', content=body, excerpt_start=len(header), excerpt_end=len(header + body)))
    snapshot = deepcopy((rows, seeds))
    result = build_context_assist('Explain CedarFlow recall architecture and limitations.',
                                 seeds, rows, lambda *args: ([], []))
    delivered = next(source for source in result['sources'] if source['id'] == 'head')
    assert delivered['content'] == body
    assert delivered['excerpt_start'] == len(header)
    assert raw['content'][delivered['excerpt_start']:delivered['excerpt_end']] == delivered['content']
    assert result['trace']['model_calls'] == 0
    assert (rows, seeds) == snapshot


def test_same_source_id_does_not_hide_loss_of_negative_condition():
    before = dict(id='same', document_id='doc', content='The archive retrieves records only with owner permission.',
                  excerpt_start=100, excerpt_end=157)
    after = dict(id='same', document_id='doc', content='The archive retrieves records.', excerpt_start=0, excerpt_end=30)
    assert _replacement_loss_reason('Explain archive retrieval.', [before], [after], None) is not None


def test_safe_same_source_expansion_is_not_treated_as_replacement():
    before = dict(id='same', document_id='doc', content='The archive retrieves records.', excerpt_start=100, excerpt_end=130)
    after = dict(id='same', document_id='doc', content='Prefix. The archive retrieves records. Tail.', excerpt_start=92, excerpt_end=140)
    assert _replacement_loss_reason('Explain archive retrieval.', [before], [after], None) is None
