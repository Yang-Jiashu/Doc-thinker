"""Conservative sentence windows with exact offsets in untouched source text.

These boundaries are a lexical heuristic, not a claim that a PDF chunk contains
the complete statement or all its qualifications. In particular a source-chunk
boundary may itself be a cut made by the document parser.
"""

from __future__ import annotations

from bisect import bisect_right
import re

_ENDING = re.compile(r'''[.!?]+["'”’»)\]}]*(?=\s|$)|[。！？]+["'”’»)\]}]*''')
_FIGURE_CAPTION = re.compile(
    r"^[ \t]*(?:Figure|Fig\.)[ \t]*(?:[A-Za-z]\.?[ \t]*)?\d+(?:\.\d+)*[ \t]*(?::|\.(?=[ \t]|$))", re.I | re.M,
)
_ABBREVIATIONS = frozenset({
    "al", "approx", "cf", "dr", "eq", "eqs", "etc", "fig", "figs",
    "inc", "jr", "mr", "mrs", "ms", "no", "nos", "pp", "prof", "ref",
    "refs", "sec", "secs", "sr", "st", "vs", "vol",
})
_FOLLOWING_QUALIFIER = re.compile(
    r"^(?:however\b|but\b|yet\b|nevertheless\b|nonetheless\b|except\b|unless\b|"
    r"only\b|provided\b|subject to\b|in contrast\b|by contrast\b|"
    r"if\b|unless\b|under these\b|under those\b|"
    r"(?:this|these|that|those|it|such|the (?:results?|effects?|gains?|improvements?|"
    r"findings?|benefits?|methods?|claims?|conclusions?|experiments?))\b.{0,110}\b(?:only|not|never|cannot|"
    r"limited|limitation|unless|except|condition|assum|however)\b|"
    r"但|然而|不过|否则|除非|仅当|仅在|只有|需要注意|值得注意|"
    r"(?:该|这|此).{0,35}(?:仅|不|未|限于|前提|条件))", re.I | re.S,
)
_PRECEDING_CONDITION = re.compile(
    r"^(?:if\b|unless\b|provided\b|under the assumption\b|assuming\b|"
    r"given that\b|in the case\b|在.{0,35}(?:条件|前提)|如果|仅当|假设)", re.I,
)
_FOLLOWING_ANTECEDENT = re.compile(
    r"^(?:to\s+(?:overcome|address|resolve|handle|avoid|mitigate)\s+(?:this|these|that|those)\b|"
    r"(?:this|these|those|such)\s+(?:approach|method|technique|system|model|procedure|"
    r"strategy|representation|step|result)s?\b|(?:it|they)\s+(?:then|also|can|is|are|was|were)\b|"
    r"then\b|next\b|subsequently\b|in this (?:approach|method|setting)\b|"
    r"(?:[a-z][a-z-]*\s+){0,5}(?:are|is|were|was|can be|will be)\s+(?:then|subsequently|next)\b|"
    r"(?:为了解决|为了避免|为处理|为此)|随后|接着|该方法|该模型|该系统|这些步骤|这一过程)", re.I,
)
_OPENING_REFERENCE = re.compile(
    r"^(?:(?:this|these|that|those|such)\s+(?:datasets?|test sets?|tasks?|settings?|experiments?|"
    r"conditions?|approaches?|methods?|models?|results?)\b|(?:it|they|its|their)\b|"
    r"(?:we|the authors?)\b[^.!?。！？]{0,100}\b(?:both|these|those|the two|the former|the latter)\s+"
    r"(?:test sets?|datasets?|approaches?|methods?|models?|tasks?|settings?|experiments?|conditions?)\b|"
    r"(?:该|这(?:个|些)?|上述|前述)(?:数据集|测试集|任务|方法|模型|设置|实验|条件)|"
    r"(?:我们|作者).{0,35}(?:这两个|上述|这两种|两种|两个)(?:数据集|测试集|方法|任务|实验))", re.I,
)
_DEMONSTRATIVE_NOMINAL = re.compile(
    r'^(?:this|these|that|those|such)\s+'
    r'(?!(?:is|are|was|were|be|been|being|has|have|had|can|could|may|might|'
    r'will|would|should|must|does|do|did|means|shows|suggests|indicates|'
    r'implies|demonstrates|as|that)\b)[A-Za-z][A-Za-z-]{1,39}\b', re.I)
_LOCAL_DOCUMENT_REFERENCE = re.compile(
    r'^this\s+(?:paper|article|document|report|section|chapter|appendix)\b', re.I)


def _demonstrative_reference(text: str):
    # A bounded nominal-shape proxy, not a POS tagger or a resolution model.
    # Do not enumerate task/domain nouns: unfamiliar participants, objects and
    # records need their actual preceding sentence just as known model names do.
    if _LOCAL_DOCUMENT_REFERENCE.match(text):
        return None
    return _DEMONSTRATIVE_NOMINAL.match(text)


def opening_reference_unverified(text: str) -> bool:
    """Flag visible opening references, never resolve them or infer an owner.

    Full sentence punctuation does not verify an antecedent. Even when a
    different excerpt contains plausible context, this source alone has not
    established the binding. Headings inside a later section cannot supply it.
    """
    if not isinstance(text, str):
        return False
    opening = text.lstrip()
    if re.match(r"^it\s+(?:is|was|may be|can be)\s+(?:not\s+)?(?:important|possible|necessary|useful|clear|"
                r"well known|worth noting)\s+(?:to|that)\b", opening, re.I):
        return False  # Expletive "it", not a reference to an omitted subject.
    reference = _OPENING_REFERENCE.match(opening) or _demonstrative_reference(opening)
    if reference:
        suffix = opening[reference.end():]
        if (re.match(r"\s*\([A-Z][A-Za-z0-9_-]{1,40}\)", suffix)
                or re.match(r"\s*[:,—-]?\s*(?:(?:named|called)\s+)?[A-Z][A-Za-z0-9_-]{1,40}\s+"
                            r"(?:and|versus|or)\s+[A-Z][A-Za-z0-9_-]{1,40}\b", suffix)):
            return False  # Explicit local names, not an inferred cross-chunk link.
    return bool(_FOLLOWING_ANTECEDENT.match(opening) or reference)
_HEADING_MINOR = frozenset({"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"})
_HEADING_SINGLE = frozenset({"abstract", "introduction", "background", "methods", "methodology", "results", "discussion",
                             "conclusion", "conclusions", "limitations", "references", "appendix", "evaluation", "experiments"})
_HEADING_NOUNS = _HEADING_SINGLE | frozenset({
    "analysis", "comparison", "performance", "implementation", "setup", "training", "inference",
    "architecture", "ablation", "design",
})
_HEADING_NOUN_PHRASES = frozenset({
    "experimental setup", "experimental results", "experimental evaluation", "experimental design",
    "implementation details", "training details", "evaluation setup", "evaluation protocol", "ablation study",
})
_QUANTITY_UNITS = frozenset({
    "cpu", "cpus", "gpu", "gpus", "ram", "kb", "mb", "gb", "tb", "kib", "mib", "gib", "tib",
    "hz", "khz", "mhz", "ghz", "ns", "us", "ms", "s", "sec", "secs", "seconds", "minutes", "hours",
    "bytes", "bits", "flops", "tflops", "joules", "watts", "kg", "mg", "km", "cm", "mm",
})


def _sentence_boundary(text: str, match: re.Match[str]) -> bool:
    """Share abbreviation/decimal guards with both typography and sentences."""
    position = match.start()
    punctuation = match.group(0).rstrip('"\'”’»)]}')
    if punctuation.startswith("."):
        if len(punctuation) > 1:
            return False
        # A line-leading list/section marker is not itself a sentence. When
        # soft-wrap context has rejected a putative heading, splitting at its
        # marker would let downstream code rediscover and discard the title
        # fragment without that context.
        line_start = text.rfind("\n", 0, position) + 1
        if _FIGURE_CAPTION.fullmatch(text[line_start:position + 1]):
            return False  # "Figure 2." belongs to the caption, not a tiny unit.
        if re.fullmatch(r"[ \t]*(?:\d+(?:\.\d+){0,3}|[IVXLC]{1,6})", text[line_start:position]):
            return False
        before = text[max(0, position - 32):position]
        word_match = re.search(r"([A-Za-z]+)$", before)
        word = word_match.group(1) if word_match else ""
        if word.lower() in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
            return False
        if re.search(r"(?:[A-Za-z]\.)+[A-Za-z]$", before):
            return False
        if position and text[position - 1].isdigit() and position + 1 < len(text) and text[position + 1].isdigit():
            return False
    return True


def is_section_heading(text: str) -> bool:
    """Recognize short structural lines, not arbitrary PDF line wraps."""
    value = text.strip()
    if not value or is_figure_caption_start(value) or len(value) > 120 or "\n" in value or re.search(r"[.!?。！？;；]$", value):
        return False
    if re.match(r"^#{1,6}\s+\S", value):
        return True
    numbered = re.match(r"^(?:\d+(?:\.\d+){0,3}|[IVXLC]{1,6})[.)]?\s+(\S.*)", value)
    if numbered:
        # A wrapped quantity ("12 hours ...") is not a section number.
        # Require heading-like text after the marker, not just any nonspace.
        # Retain ambiguous prose/axis labels instead of deleting it as a
        # structural boundary. This is typography, not syntactic validation.
        body = numbered.group(1)
        if (re.search(r"[;；\[\]]", body)
                or any(_sentence_boundary(body, match) for match in _ENDING.finditer(body))):
            return False
        words = re.findall(r"[A-Za-z][A-Za-z-]*", body)
        if not words or len(words) > 10:
            return False
        if words[0].casefold() in _QUANTITY_UNITS:
            return False
        if body.casefold() in _HEADING_SINGLE:
            return True
        if not words[0][0].isupper():
            return False
        if all(word[0].isupper() or word.casefold() in _HEADING_MINOR for word in words):
            return True
        subject = re.fullmatch(r'([A-Z][A-Za-z0-9-]{2,63})\s+(?:methods?|results?|evaluation|experiments?)', body)
        if subject and (subject[1].isupper() or '-' in subject[1]
                        or re.search(r'[a-z][A-Z]', subject[1])):
            # A compact technical subject followed by a conventional section
            # noun is still a title in sentence case. No verbs, values,
            # citations or arbitrary numbered prose are accepted here.
            return True
        # Accept limited noun-shaped sentence-case titles, not any capitalized
        # clause ("Patients received placebo"). This is not a syntactic proof;
        # uncertain forms remain evidence rather than expanding a verb list.
        normalized = " ".join(body.casefold().split())
        heads = (*_HEADING_NOUNS, *_HEADING_NOUN_PHRASES)
        return any(normalized == head or any(normalized.startswith(head + " " + connector + " ")
                   for connector in ("of", "on", "with", "for", "in", "under", "across"))
                   for head in heads) and not re.search(
                       r"\b(?:is|are|was|were|has|have|had|will|would|can|could|should|must)\b", body, re.I)
    words = re.findall(r"[A-Za-z][A-Za-z-]*", value)
    if not words or len(words) > 10:
        return False
    if value.lower() in _HEADING_SINGLE:
        return True
    if value.isupper():
        return True
    return len(words) >= 2 and all(word[0].isupper() or word.lower() in _HEADING_MINOR for word in words)


def section_heading_spans(text: str) -> list[tuple[int, int]]:
    """Require a structural boundary, not a soft wrap inside unfinished prose.

    Explicit Markdown, blank lines, and already accepted headings establish
    structure. At a chunk start no prior context exists; the conservative
    isolated-line test applies. Uncertain wraps stay as source evidence.
    """
    spans, cursor = [], 0
    previous = ""
    previous_heading = False
    for line in text.splitlines(keepends=True):
        end = cursor + len(line)
        previous_value = previous.rstrip()
        completed = any(match.end() == len(previous_value) and _sentence_boundary(previous_value, match)
                        for match in _ENDING.finditer(previous_value))
        boundary = (cursor == 0 or not previous_value or previous_heading or completed
                    or bool(re.match(r"^\s*#{1,6}\s+\S", line)))
        heading = boundary and is_section_heading(line)
        if heading:
            spans.append((cursor, end))
        previous, previous_heading = line, heading
        cursor = end
    return spans


def is_figure_caption_start(text: str) -> bool:
    """An explicit line-leading figure marker, not an inferred caption."""
    return bool(_FIGURE_CAPTION.match(text))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return contiguous source spans; PDF line wraps are not sentence ends.

    Common scholarly abbreviations, initials, decimals and ellipses are kept
    inside a span. The final unpunctuated source span is retained intact.
    """
    # PDF axes often have no sentence punctuation. The explicit next caption
    # starts a new unit without deleting either side or changing source bytes.
    boundaries = [0, *(match.start() for match in _FIGURE_CAPTION.finditer(text) if match.start() > 0), len(text)]
    return [(left + start, left + end)
            for left, right in zip(boundaries, boundaries[1:])
            for start, end in _structural_sentence_spans(text[left:right])]


def _structural_sentence_spans(text: str) -> list[tuple[int, int]]:
    headings = section_heading_spans(text)
    if headings:
        spans, cursor = [], 0
        for left, right in headings:
            spans.extend((cursor + a, cursor + b) for a, b in _punctuation_spans(text[cursor:left]))
            spans.append((left, right))
            cursor = right
        spans.extend((cursor + a, cursor + b) for a, b in _punctuation_spans(text[cursor:]))
        # Preserve whitespace between adjacent headings as well: downstream
        # source windows count exact bytes of a single continuous slice.
        if spans:
            ends = [end for _, end in spans]
            ends[-1] = len(text)
            return [(0 if index == 0 else ends[index - 1], end) for index, end in enumerate(ends)]
        return []
    return _punctuation_spans(text)


def _punctuation_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for match in _ENDING.finditer(text):
        if not _sentence_boundary(text, match):
            continue
        end = match.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    elif spans:
        spans[-1] = (spans[-1][0], len(text))
    return spans


def evidence_units(text: str) -> list[tuple[int, int]]:
    """Bind adjacent explicit qualifications so a budget cannot remove them.

    Only contiguous spans are merged. This is deliberately conservative: when
    a claim and its visible qualification do not fit together, neither is a
    safe standalone evidence window for that claim.
    """
    units: list[tuple[int, int]] = []
    previous = ""
    for start, end in sentence_spans(text):
        current = text[start:end].strip()
        boundary = (is_section_heading(current) or is_section_heading(previous)
                    or is_figure_caption_start(current) or is_figure_caption_start(previous))
        if units and not boundary and (_FOLLOWING_QUALIFIER.match(current) or _PRECEDING_CONDITION.match(previous)
                                       or _FOLLOWING_ANTECEDENT.match(current)
                                       or (_demonstrative_reference(current) and opening_reference_unverified(current))):
            units[-1] = (units[-1][0], end)
        else:
            units.append((start, end))
        previous = current
    return units


def bounded_evidence_extension(text: str, retained: tuple[int, int],
                               supplement: tuple[int, int], max_chars: int) -> tuple[int, int] | None:
    """Extend an existing quote only through whole, adjacent evidence units.

    A supplement can overlap the retained quote while its full union exceeds
    the character budget. In that case a complete prefix/suffix of its units
    is usable context; arbitrary character truncation is not. The returned
    geometry conveys no reranker score or factual-support credit for a partly
    included supplement. Callers must separately check section/source safety.
    """
    if (not isinstance(text, str) or type(max_chars) is not int or max_chars <= 0
            or not all(type(value) is int for value in (*retained, *supplement))):
        return None
    a, b = retained
    c, d = supplement
    if not (0 <= a < b <= len(text) and 0 <= c < d <= len(text) and b - a <= max_chars):
        return None
    left, right = min(a, c), max(b, d)
    if left == a and right == b:
        return None
    units = evidence_units(text)
    # Retention must not begin/end inside a bound claim + condition unit.
    if any(start < boundary < end and text[start:boundary].strip() and text[boundary:end].strip()
           for start, end in units for boundary in (a, b)):
        return None
    starts = [a, *(start for start, _ in units if left <= start < a)]
    ends = [b, *(end for _, end in units if b < end <= right)]
    # A maximal valid end is sufficient for each start. Avoid a quadratic
    # cross-product for dense short Chinese sentences in a long source chunk.
    options = []
    for start in starts:
        position = bisect_right(ends, start + max_chars) - 1
        if position >= 0:
            end = ends[position]
            if start < a or end > b:
                options.append((start, end))
    if not options:
        return None
    # Maximal complete context with a stable geometry tie-break; no query/gold
    # terms are consulted and no sentence from a disconnected chunk is copied.
    return min(options, key=lambda span: (-(span[1] - span[0]), span[0], span[1]))
