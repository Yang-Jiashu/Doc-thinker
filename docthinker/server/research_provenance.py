"""Literal, claim-level provenance checks with no model calls.

Only an exact contiguous quotation can receive ``exact_quote``. Paraphrases
remain unverified: lexical overlap, a plausible citation, and numeric agreement
are not semantic entailment. Inconsistency checks are review signals, not a
general-purpose fact checker.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re

from .research_sentence_windows import evidence_units
from .research_source_contract import MAX_ANSWER_SOURCES
from .research_citation_syntax import CITATION_GROUP_PATTERN, parse_citation_groups, strip_citation_groups

MAX_CLAIMS = 12
MAX_CLAIM_CHARS = 800
MAX_SOURCE_REFS = 3
MAX_REVIEW_QUOTE_CHARS = 1000
MAX_REVIEW_SCAN_CHARS = 12000
_LABEL = CITATION_GROUP_PATTERN
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:\s*[%％])?(?!\w)")
_ANCHOR_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[+-]?\d+(?:[.,]\d+)*(?:\s*[%％])?(?![A-Za-z0-9_]|\.\d)")
_WORD = re.compile(r"[A-Za-z][A-Za-z_-]+|[\u3400-\u9fff]{2,}")
_STOP_WORDS = frozenset("a an and are as at be been being but by can could did do does for from had has have how if in into is it its may more most not of on only or our should than that the their these they this those to was we were what when where which while who will with would you your".split())
_STOP_HAN = frozenset({"以及", "因此", "所以", "可以", "这个", "这些", "那个", "那些", "进行", "其中", "对于", "根据", "表明", "显示", "结果"})
_NEGATIVE = re.compile(r"\b(?:not|never|cannot|no|fails?\s+to|\w+n['’]t)\b|不能|并非|未能|不支持|无证据|没有证据|未证明", re.I)
_CONDITIONAL = re.compile(r"\b(?:if|unless|only|provided\s+that|under|assuming|may|might)\b|如果|若|仅|只有|在.{1,30}条件下|可能|或许", re.I)
# Surface cues only: these raise review priority, never classify a result as
# false or supported. Negated/conditional and unfamiliar claims remain eligible.
_PERFORMANCE_SCOPE_CUE = re.compile(
    r"\b(?:proven|guaranteed?|excelling|excels?|superior|outperform(?:s|ed|ing)?|"
    r"state[- ]of[- ]the[- ]art|sota|best[- ]performing)\b|"
    r"效果(?:优秀|更好|最好|已证明)|优于|超过(?:所有|其他|基线)|"
    r"显著(?:提高|提升|改善)|已(?:经)?(?:证实|证明)|最先进|保证(?:准确|有效|成功)", re.I)
_SENTENCE_END = re.compile(r'''[.!?。！？]["'”’»)\]}]*\s*$''')
_LITERAL_ANCHOR = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*")
_PDF_WORD_WRAP = re.compile(r"(?<=[A-Za-z])-[ \t]*\r?\n[ \t]*(?=[A-Za-z])")
_ANCHOR_DISCOURSE = frozenset(
    'experimental experiment experiments result results finding findings '
    'clearly demonstrate demonstrates demonstrated demonstrating '
    'indicate indicates indicated suggest suggests suggested '
    'report reports reported describe describes described '
    'conclude concludes conclusion conclusions evidence study studies '
    'evaluate evaluates evaluated evaluation evaluations performance '
    'compare compares compared comparison comparisons approach approaches '
    'propose proposes proposed proposal proposals advantage advantages '
    'additional furthermore therefore however according'.split())
_LAYOUT_TITLES = frozenset({
    'scope', 'scope and evidence limits', 'evidence limits', 'demonstrated results',
    'results', 'methods', 'architecture', 'architecture and training', 'limitations',
    'proposed transfer', 'missing direct comparisons', 'conclusions', 'further experiments',
    'conclusions requiring further experiments', 'conclusions that require further experiments',
    'summary', 'bottom line', 'privacy', 'references',
    '范围', '证据范围', '证据与限制', '已验证结果', '实验结果', '方法', '架构', '局限',
    '拟议迁移', '缺少的直接比较', '结论', '后续实验', '总结', '隐私', '参考文献',
})
_TABLE_LABELS = _LAYOUT_TITLES | frozenset({
    'system', 'method', 'model', 'metric', 'metrics', 'conditions', 'evidence', 'source',
    'task', 'key conditions / metrics', 'what is directly demonstrated',
    '系统', '方法', '模型', '指标', '条件', '来源', '任务', '说明',
})


def _literal_anchors(text):
    """Exact English surface markers only, not translated/stemmed concepts.

    PDF soft wraps are normalized only for this diagnostic, never for quotes
    or source coordinates. Three distinct, reasonably long markers are needed
    so a shared acronym or dataset name alone cannot produce a warning.
    """
    # Horizontal whitespace only: two overlapping \s* ranges would cause
    # quadratic backtracking on a long run of newlines without a next letter.
    text = _PDF_WORD_WRAP.sub('', text)
    return {match.group().casefold() for match in _LITERAL_ANCHOR.finditer(text)
            if len(match.group()) >= 5 and match.group().casefold() not in _STOP_WORDS
            and match.group().casefold() not in _ANCHOR_DISCOURSE}


def _citation_anchor_drift(text, labels, anchors):
    """Flag displaced literal markers; never fix a citation or judge support.

    Comparing the union of all cited excerpts avoids a false warning when a
    sentence legitimately cites several sources. Uncited material is merely a
    review signal, not a replacement source, independent evidence or proof.
    """
    cited = set().union(*(anchors.get(label, set()) for label in labels))
    absent = _literal_anchors(text[:MAX_CLAIM_CHARS]) - cited
    return any(len(absent & terms) >= 3 for label, terms in anchors.items()
               if label not in labels)


def _presentation_only(line, following=""):
    """Skip positive Markdown furniture, not claims written in bold.

    This is a layout rule, not a factual/semantic classifier. Ambiguous prose,
    numbers, negations and citations stay eligible for the bounded review.
    """
    if re.fullmatch(r"(?:[-*_]\s*){3,}", line):
        return True
    if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?", line):
        return True
    if '[' in line or _NEGATIVE.search(line):
        return False
    if (line.startswith('|') and
            re.fullmatch(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?", following)):
        cells = [cell.strip().strip('*').casefold() for cell in line.strip('|').split('|')]
        return bool(cells) and all(cell in _TABLE_LABELS for cell in cells)
    if line.startswith('#'):
        value = re.sub(r'^#{1,6}\s+', '', line)
    elif ((line.startswith('**') and line.endswith('**')) or
          (line.startswith('*') and line.endswith('*') and line.count('*') == 2)):
        value = line.strip('*').strip()
    else:
        return False
    value = re.sub(r'^\d+[.)、]\s*', '', value)
    # Only known layout labels are omitted. Bold/heading formatting alone
    # cannot make arbitrary prose nonfactual; an open-ended verb denylist
    # would silently discard assertions using unseen predicates/languages.
    return value.casefold() in _LAYOUT_TITLES


def _sentences(answer):
    # A citation often follows punctuation. Keep it attached to its sentence.
    lines = answer.splitlines()
    for index, line in enumerate(lines):
        line = line.strip()
        following = lines[index + 1].strip() if index + 1 < len(lines) else ''
        if not line or _presentation_only(line, following):
            continue
        # Consume each whitespace run once. A nested whitespace lookahead here
        # used to backtrack quadratically before a citation after punctuation.
        pieces, start = [], 0
        for match in re.finditer(r"[。！？!?]\s*|\.\s+", line):
            end = match.end()
            if _LABEL.match(line, end):
                continue
            pieces.append(line[start:match.start() + 1])
            start = end
        pieces.append(line[start:])
        current = ""
        for piece in pieces:
            if not piece:
                continue
            if (not strip_citation_groups(piece, prefixes='SK').strip(' \t。.!')
                    and _LABEL.search(piece) and current):
                current += " " + piece
                continue
            if current:
                yield current
            current = piece
        if current:
            yield current


def _plain(statement):
    text = strip_citation_groups(statement, prefixes='S').strip()
    text = re.sub(r"^\s*(?:[-*>]|\d+[.)、])\s*", "", text).strip()
    # Strip presentation wrappers, not negations, clauses, units or wording.
    text = text.strip(' \t“”\"「」『』')
    if len(text) > 1 and text[-1] in "。.!" and text[-2] in '”\"」』':
        text = text[:-1].strip('“”\"「」『』')
    return text


def _sample_statements(answer):
    """Spread twelve review opportunities across substantive output positions.

    Each positional band contributes at most one statement. A literal citation
    is a locator, not a truth vote. Within a band, explicit effect/guarantee
    wording receives review priority before citation, bounded length and
    position. These incomplete surface cues are not a semantic risk classifier.
    No model, support verdict, answer key or source-result union is used.
    All unselected eligible statements remain in the omitted count.
    """
    count = sum(1 for _ in _sentences(answer))
    if count <= MAX_CLAIMS:
        return list(enumerate(_sentences(answer))), count
    bands = {}
    for ordinal, statement in enumerate(_sentences(answer)):
        band = ordinal * MAX_CLAIMS // count
        center = (band + .5) * count / MAX_CLAIMS - .5
        cited = bool(_LABEL.search(statement))
        plain = _plain(statement)
        performance_cue = bool(_PERFORMANCE_SCOPE_CUE.search(plain[:MAX_CLAIM_CHARS]))
        rank = (performance_cue, cited, min(len(plain), MAX_CLAIM_CHARS), -abs(ordinal - center), -ordinal)
        if band not in bands or rank > bands[band][0]:
            bands[band] = (rank, ordinal, statement)
    return [(ordinal, statement) for _, ordinal, statement in sorted(bands.values(), key=lambda item: item[1])], count


def _locate_literal(text, source):
    if len(text.strip()) < 4:
        return None
    position = source.find(text)
    if position >= 0:
        return position, position + len(text)
    # The terminal sentence mark can be outside the quotation in Markdown.
    if text.endswith(("。", ".", "!", "！")):
        shortened = text[:-1]
        position = source.find(shortened)
        if len(shortened) >= 4 and position >= 0:
            return position, position + len(shortened)
    return None


def _source_fingerprint(row):
    """Bind persisted pointers to the bounded source window, without text copies.

    No exact status is granted when this window excludes part of an excerpt.
    Hashing the same bounded window also keeps history hydration bounded.
    """
    content = row['content']
    value = f"{row.get('excerpt_start', 0)}:{len(content)}:" + content[:MAX_REVIEW_SCAN_CHARS]
    return hashlib.sha256(value.encode()).hexdigest()


def _quote_context_needs_review(content, location):
    """Check the bound unit, not a PDF line or just the literal substring."""
    left, right = location
    units = evidence_units(content)
    overlapping = [index for index, (a, b) in enumerate(units) if a < right and left < b]
    if not overlapping:
        return True
    first, last = overlapping[0], overlapping[-1]
    # Heading heuristics must not turn an unpunctuated PDF wrap into a sentence
    # boundary (including title-cased/all-caps continuations).
    while first and not _SENTENCE_END.search(content[units[first - 1][0]:units[first - 1][1]]):
        first -= 1
    while last + 1 < len(units) and not _SENTENCE_END.search(content[units[last][0]:units[last][1]]):
        last += 1
    before, after = units[first][0], units[last][1]
    surrounding = content[before:after]
    # Each visible qualification must be retained, not merely one occurrence
    # of a negation/condition somewhere in an otherwise truncated quote.
    return any(not (left <= before + match.start() and before + match.end() <= right)
               for pattern in (_NEGATIVE, _CONDITIONAL) for match in pattern.finditer(surrounding))


def _review_anchors(text):
    # Normalization is for location ranking only; returned quotes are always
    # untouched continuous slices with offsets into the original excerpt.
    value = _PDF_WORD_WRAP.sub('', text)
    words = set()
    for match in _WORD.finditer(value):
        word = match.group().lower()
        if '\u3400' <= word[0] <= '\u9fff':
            words.update(word[index:index + 2] for index in range(len(word) - 1)
                         if word[index:index + 2] not in _STOP_HAN)
        elif word not in _STOP_WORDS:
            words.add(word)
    numbers = {re.sub(r"\s+", "", match.group()).replace('％', '%')
               for match in _ANCHOR_NUMBER.finditer(value)}
    return words, numbers


def _review_locations(text, content, max_spans):
    """Locate review material, never classify support or infer experiments.

    At most three separate references share the old 1000-character quote
    allowance per cited source. Complete, explicitly bound evidence units are
    retained or omitted; disconnected hits are never spliced into one quote.
    Numeric-only and tied locations are disclosed, not promoted to entailment.
    """
    issues = []
    scanned = content[:MAX_REVIEW_SCAN_CHARS]
    if len(scanned) < len(content):
        issues.append({'code': 'source_location_windowed', 'message': '来源定位达到扫描上限，未核对该节选的全部位置。'})
    words, numbers = _review_anchors(text[:MAX_CLAIM_CHARS])
    candidates = []
    for left, right in evidence_units(scanned):
        # A scan cap must not manufacture a complete final sentence.
        if len(scanned) < len(content) and right == len(scanned):
            continue
        unit = scanned[left:right]
        left += len(unit) - len(unit.lstrip())
        right -= len(unit) - len(unit.rstrip())
        unit_words, unit_numbers = _review_anchors(scanned[left:right])
        shared_words, shared_numbers = words & unit_words, numbers & unit_numbers
        if not shared_words and not shared_numbers:
            continue
        candidates.append((left, right, shared_words, shared_numbers))
    if not candidates:
        return [], [*issues, {'code': 'source_location_unresolved', 'message': '未能定位与这段改写相关的原文范围；引用来源存在，但不等于支持该表述。'}]
    frequencies = Counter(word for item in candidates for word in item[2])
    def score(item):
        lexical = sum(1 / frequencies[word] for word in item[2])
        return 4 * len(item[3]) / max(1, len(numbers)) + lexical
    ranked = sorted(candidates, key=lambda item: (-score(item), -len(item[2]), -len(item[3]), item[0]))
    best = score(ranked[0])
    tied = [item for item in ranked if abs(score(item) - best) < 1e-9]
    tied_locations = {(item[0], item[1]) for item in tied}
    if len(tied) > 1:
        issues.append({'code': 'source_location_ambiguous', 'message': '原文中有多个同等匹配位置，不能仅凭重复词项或数值确定所指实验。'})
    selected, seen_words, seen_numbers, size = [], set(), set(), 0
    omitted = False
    for item in ranked:
        left, right, matched_words, matched_numbers = item
        # Cover distinct assertion anchors and disclose equally ranked
        # alternatives; adding adjacent text is not itself extra evidence.
        if selected and not (matched_words - seen_words or matched_numbers - seen_numbers or (left, right) in tied_locations):
            continue
        if len(selected) >= max_spans or size + right - left > MAX_REVIEW_QUOTE_CHARS:
            omitted = True
            continue
        selected.append(item)
        size += right - left
        seen_words.update(matched_words)
        seen_numbers.update(matched_numbers)
    if omitted:
        issues.append({'code': 'source_location_budget', 'message': '部分相关原文完整句超出核对展示预算，未将其截断或视作已展示。'})
    if selected and not seen_words:
        issues.append({'code': 'source_location_numeric_only', 'message': '这里只按数值定位核对材料；相同数值可能属于不同实验，尚未验证其含义。'})
    if not selected:
        issues.append({'code': 'source_location_unresolved', 'message': '相关原文完整句无法放入核对窗口，未用无关的开头片段代替。'})
    return [(item[0], item[1]) for item in sorted(selected)], issues


def audit_answer(answer: str, sources: list[dict]) -> dict:
    """Bind every inspected statement to literal live source offsets.

    The bounded audit explicitly reports statements beyond its window. It never
    treats a generated quotation or a filename as an original source.
    """
    available = {row.get("label"): row for row in sources[:MAX_ANSWER_SOURCES] if isinstance(row, dict)
                 and isinstance(row.get("content"), str) and row.get("id")}
    anchors = {label: _literal_anchors(row['content'][:MAX_REVIEW_SCAN_CHARS])
               for label, row in available.items()}
    statements, statement_count = _sample_statements(str(answer))
    claims = []
    for ordinal, statement in statements:
        text = _plain(statement)
        parsed_citations = parse_citation_groups(statement)
        labels = [label for label in parsed_citations['labels'] if label.startswith('S')]
        refs, issues, exact_labels = [], [], 0
        if parsed_citations['truncated']:
            issues.append({'code': 'citation_metadata_window',
                           'message': '引用语法核对达到上限，未把截流后的部分引用视作完整或已核对。'})
        quote_sizes = Counter()
        if not labels:
            issues.append({"code": "uncited_statement", "message": "这段表述没有绑定原文，不能据此认定为资料事实。"})
        elif all(label in available for label in labels) and _citation_anchor_drift(text, labels, anchors):
            issues.append({'code': 'citation_anchor_drift', 'message':
                '多个英文原词未在所引节选出现，却在本次其他节选出现；请核对具体引用。这仅是词面信号，不是语义判定。'})
        if len(text) > MAX_CLAIM_CHARS:
            issues.append({"code": "statement_window", "message": "表述较长，只展示核对窗口；未将全文标记为已核对。"})
        live_labels = 0
        for label_index, label in enumerate(labels[:MAX_SOURCE_REFS]):
            row = available.get(label)
            if row is None:
                issues.append({"code": "source_unavailable", "message": "引用来源不在当前可核对的资料中。"})
                continue
            live_labels += 1
            content = row["content"]
            scanned = content[:MAX_REVIEW_SCAN_CHARS]
            if len(scanned) < len(content):
                issues.append({'code': 'source_location_windowed', 'message': '来源定位达到扫描上限，未核对该节选的全部位置。'})
            location = _locate_literal(text, scanned) if len(text) <= MAX_CLAIM_CHARS else None
            if location:
                locations = [location]
                exact_labels += 1
                if _quote_context_needs_review(scanned, location):
                    issues.append({'code': 'quote_context_review', 'message': '虽有逐字匹配，但原文的完整语境还有否定或条件措辞，请结合完整语境核对。'})
            else:
                # Reserve one reference for each remaining cited label. This
                # keeps the existing API's total reference count unchanged.
                remaining_labels = len(labels[:MAX_SOURCE_REFS]) - label_index - 1
                locations, location_issues = _review_locations(text, content, MAX_SOURCE_REFS - len(refs) - remaining_labels)
                issues.extend(location_issues)
                inspected_text = text[:MAX_CLAIM_CHARS]
                numbers = set(_NUMBER.findall(inspected_text))
                if numbers - set(_NUMBER.findall(scanned)):
                    issues.append({"code": "number_not_in_source", "message": "表述中的部分数值未出现在所引节选中，请核对数值、单位和范围。"})
                if bool(_NEGATIVE.search(inspected_text)) != bool(_NEGATIVE.search(scanned)):
                    issues.append({"code": "polarity_review", "message": "表述与原文的否定措辞不同，需要核对结论方向。"})
                if _CONDITIONAL.search(scanned) and not _CONDITIONAL.search(inspected_text):
                    issues.append({"code": "condition_review", "message": "原文含有条件或不确定措辞，表述可能遗漏适用范围。"})
            offset = row.get("excerpt_start", 0)
            for left, right in locations:
                if quote_sizes[row['id']] + right - left > MAX_REVIEW_QUOTE_CHARS:
                    issues.append({'code': 'source_location_budget', 'message': '部分相关原文完整句超出核对展示预算，未将其截断或视作已展示。'})
                    continue
                quote_sizes[row['id']] += right - left
                refs.append({"label": label, "source_id": row["id"], "quote": content[left:right],
                             "start": offset + left, "end": offset + right,
                             "source_window_sha256": _source_fingerprint(row)})
        if len(labels) > MAX_SOURCE_REFS:
            issues.append({"code": "citation_window", "message": "本段引用超过核对窗口，未逐项验证全部引用。"})
        # A statement citing a missing source is never marked clean, even when
        # another of its citations happens to contain the same text.
        status = "source_unavailable" if labels and not live_labels else "exact_quote" if labels and exact_labels == len(labels) and not issues else "unverified"
        if status == 'unverified' and _PERFORMANCE_SCOPE_CUE.search(text[:MAX_CLAIM_CHARS]):
            issues.append({'code': 'performance_scope_review', 'message':
                '表述含有效果、优越性或已证实措辞；请逐项核对任务、指标、比较对象及条件。这里只定位核对重点，不判定表述正确或错误。'})
        if status == "unverified" and not issues:
            issues.append({"code": "semantic_review", "message": "这是改写或推断，尚未验证原文是否完整支持。"})
        issues = list({item["code"]: item for item in issues}.values())
        claims.append({"id": hashlib.sha256(f"{ordinal}:{statement}".encode()).hexdigest()[:24],
                       "text": text[:MAX_CLAIM_CHARS], "status": status, "source_refs": refs, "issues": issues})
    omitted = max(0, statement_count - len(statements))
    exact_count = sum(row["status"] == "exact_quote" for row in claims)
    review_count = len(claims) - exact_count
    return {"status": "review_needed" if review_count or omitted else "quoted_only" if claims else "no_claims",
            "claims": claims, "summary": {"exact_quote_count": exact_count, "unverified_count": review_count,
                                          "issue_count": sum(len(row["issues"]) for row in claims),
                                          "omitted_statement_count": omitted},
            "scope": "literal_and_consistency_checks", "entailment_verified": False}


def hydrate_audit(audit: dict, sources: list[dict]) -> dict:
    """Rebuild identity-bound quotes; legacy unbound pointers stay unverified."""
    by_id = {source["id"]: source for source in sources if isinstance(source, dict)
             and source.get('id') and isinstance(source.get('content'), str)}
    result = {**audit, "claims": []}
    for claim in audit.get("claims", [])[:MAX_CLAIMS]:
        refs, revoked = [], False
        issues = list(claim.get('issues', []))
        needs_review = False
        quote_sizes = Counter()
        saved_refs = claim.get('source_refs', [])
        if len(saved_refs) > MAX_SOURCE_REFS:
            needs_review = True
            issues.append({'code': 'citation_window', 'message': '本段引用超过核对窗口，未逐项验证全部引用。'})
        for ref in saved_refs[:MAX_SOURCE_REFS]:
            source = by_id.get(ref.get("source_id"))
            if source is None:
                revoked = True
                continue
            offset = source.get('excerpt_start', 0)
            if any(type(value) is not int for value in (ref.get('start'), ref.get('end'), offset)):
                revoked = True
                continue
            start = ref['start'] - offset
            end = ref['end'] - offset
            if not 0 <= start < end <= len(source["content"]):
                revoked = True
                continue
            fingerprint = ref.get('source_window_sha256')
            if fingerprint is not None and fingerprint != _source_fingerprint(source):
                revoked = True
                continue
            if fingerprint is None and not isinstance(ref.get('quote'), str):
                # Pre-fingerprint history cannot prove that today's text is
                # the text seen at generation time. Do not silently fill it in.
                needs_review = True
                issues.append({'code': 'historical_source_unverified', 'message': '旧记录仅有位置，没有可复核的原文或内容指纹，未用当前文字替换历史引用。'})
                continue
            if end > MAX_REVIEW_SCAN_CHARS:
                needs_review = True
                issues.append({'code': 'source_location_windowed', 'message': '来源定位达到扫描上限，未核对该节选的全部位置。'})
                continue
            if quote_sizes[source['id']] + end - start > MAX_REVIEW_QUOTE_CHARS:
                needs_review = True
                issues.append({'code': 'source_location_budget', 'message': '历史引用超出核对展示预算，未将其截断或视作已展示。'})
                continue
            quote = source["content"][start:end]
            if isinstance(ref.get('quote'), str) and ref['quote'] != quote:
                revoked = True
                continue
            if claim.get('status') == 'exact_quote' and fingerprint is None and (start != 0 or end != len(source['content'])):
                needs_review = True
                issues.append({'code': 'historical_context_unverified', 'message': '旧引用未保存完整语境或内容指纹，无法确认当前语境与生成时相同。'})
            if len(source['content']) > MAX_REVIEW_SCAN_CHARS:
                needs_review = True
                issues.append({'code': 'source_location_windowed', 'message': '来源定位达到扫描上限，未核对该节选的全部位置。'})
            if claim.get('status') == 'exact_quote':
                text = claim.get('text', '')
                location = _locate_literal(text, quote) if isinstance(text, str) and len(text) <= MAX_CLAIM_CHARS else None
                if location is None:
                    # A legacy any-source exact badge was not proof that every
                    # reference was literal. This is not a source revocation.
                    needs_review = True
                    issues.append({'code': 'semantic_review', 'message': '这是改写或推断，尚未验证原文是否完整支持。'})
                elif _quote_context_needs_review(source['content'][:MAX_REVIEW_SCAN_CHARS],
                                                (start + location[0], start + location[1])):
                    needs_review = True
                    issues.append({'code': 'quote_context_review', 'message': '虽有逐字匹配，但原文的完整语境还有否定或条件措辞，请结合完整语境核对。'})
            quote_sizes[source['id']] += end - start
            refs.append({**ref, "quote": quote})
        if claim.get('status') == 'exact_quote' and not refs and not revoked:
            needs_review = True
            issues.append({'code': 'historical_source_unverified', 'message': '旧记录没有可复核的来源引用，不能继续标为逐字核对。'})
        item = {**claim, "source_refs": refs, 'issues': list({issue['code']: issue for issue in issues}.values())}
        if revoked:
            item.update(status="source_unavailable", issues=[{"code": "source_unavailable", "message": "来源已撤回或不再可用，无法继续核对这段表述。"}])
        elif needs_review:
            item['status'] = 'unverified'
        result["claims"].append(item)
    exact = sum(row["status"] == "exact_quote" for row in result["claims"])
    review = len(result["claims"]) - exact
    result["summary"] = {**audit.get("summary", {}), "exact_quote_count": exact,
                         "unverified_count": review, "issue_count": sum(len(row["issues"]) for row in result["claims"])}
    result["status"] = "review_needed" if review or result["summary"].get("omitted_statement_count") else "quoted_only" if result["claims"] else "no_claims"
    return result
