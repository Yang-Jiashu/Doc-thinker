"""Conservative whole-window evidence hygiene, independent of local models.

No retrieval, ranking, cropping, model calls, or semantic claims. Unknown and
mixed material is deliberately retained. The public predicate can screen a
bounded candidate pool; the pack wrapper never expands its existing sources.
"""
from __future__ import annotations

import re
from collections import Counter

from .research_sentence_windows import sentence_spans

MAX_SOURCES = 8
MAX_SOURCE_CHARS = 16000
MAX_QUERY_CHARS = 8192

_METADATA_QUERY = re.compile(
    r"\b(?:bibliograph\w*|reference\s+lists?|list\s+(?:the\s+)?references|"
    r"list\s+of\s+references|(?:list|show|give|provide)\b.{0,40}\b(?:references|citations)|"
    r"which\s+(?:references|citations)|affiliat\w*|author\s+(?:names?|list|institutions?)|"
    r"who\s+(?:are|is|wrote|authored)|e-?mails?|contact\s+(?:details?|information|address)|"
    r"corresponding\s+author)\b|参考文献|引用清单|文献列表|作者名单|作者是谁|哪些作者|"
    r"作者.{0,12}(?:机构|单位|是谁)|列出.{0,12}(?:引用|文献)|哪些文献|所属机构|隶属|联系方式|联系地址|邮箱|电子邮件|著者|所属機関|"
    r"参考文献一覧|連絡先|소속|참고문헌|이메일", re.I)
_REFERENCE_NUMBER = re.compile(r"(?m)^\s*\[(\d{1,4})\]\s*")
_AUTHOR = re.compile(
    r"^(?:[^\W\d_][\w'’−-]+,\s*(?:[A-Z]\.?\s*){1,4}|"
    r"(?:[A-Z][\w'’−-]*\s+){1,4}[A-Z][\w'’−-]*\.)", re.UNICODE)
_BIBLIOGRAPHIC_END = re.compile(
    r"(?:\b\d{1,4}\s*,\s*\d{1,7}(?:\s*[–−-]\s*\d{1,7})?\s*"
    r"\((?:18|19|20)\d{2}\)|\barXiv\s*(?:preprint\s*)?(?::|arXiv:)?\s*"
    r"\d{4}\.\d{4,5}(?:v\d+)?\s*\((?:19|20)\d{2}\))\s*[.]?\s*$", re.I)
_AFFILIATION_NUMBER = re.compile(r"(?m)^\s*(\d{1,3})\s+(?=\S)")
_INSTITUTION = re.compile(
    r"\b(?:university|department|institute|school|faculty|laboratory|laboratories|college)\b|"
    r"大学|学院|研究所|研究院|科学院|大学院|대학교|연구소", re.I)
_INSTITUTION_SHAPE = re.compile(
    r"\b(?:department|school|faculty|institute|university|college)\s+(?:of|for)\b|"
    r"\b(?:university|college|laboratory|laboratories)\s*$", re.I)
_AFFILIATION_WORD = re.compile(r"^[A-ZÀ-ÖØ-Þ][\w’'−-]*$", re.UNICODE)
_AFFILIATION_PREDICATE = re.compile(
    r"\b(?:reports?|reported|reporting|proves?|proved|proving|results?|positive|negative|"
    r"shows?|showed|demonstrates?|demonstrated|finds?|found|discovers?|discovered|"
    r"measures?|measured|uses?|used|adopts?|adopted|releases?|released|develops?|developed|"
    r"achieves?|achieved|increases?|increased|reduces?|reduced|produces?|produced|"
    r"confirms?|confirmed|rejects?|rejected|supports?|supported|tests?|tested|"
    r"launches?|launched|sells?|sold|owns?|owned|is|are|was|were|has|have|had)\b", re.I)
_EMAIL = r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
_CONTACT = re.compile(
    rf"^(?:(?:e-?mail|mailto|contact|correspondence|邮箱|电子邮件)\s*[:：]\s*)?"
    rf"(?:{_EMAIL})(?:\s*[;,，；]\s*{_EMAIL})*\s*[.]?$", re.I)
_AUTHOR_YEAR = re.compile(r"^(.{3,360}?)\.\s*((?:18|19|20)\d{2}[a-z]?)\.\s+", re.S)
_NAME_PARTICLES = frozenset({'and', 'de', 'del', 'der', 'di', 'da', 'dos', 'la', 'le', 'van', 'von', 'al'})
_VENUE_END = re.compile(
    r"(?:\barxiv\s*:\s*\d{4}\.\d{4,5}(?:v\d+)?|\bdoi\s*:\s*10\.\S+|"
    r"\bIn\s+(?:Proc\.(?:\s+of)?|Proceedings(?:\s+of)?)\s+[^.!?]{2,180}|"
    r"\bJournal\s+[^.!?]{2,140}\d[\d():,–−\-\s]*)\.\s*$", re.I)
_NARRATIVE_IN_REFERENCES = re.compile(
    r"\b(?:we|our|however|unless|therefore|because|cannot|does not|did not|"
    r"these (?:results|findings)|the (?:method|experiment|result) (?:is|was|uses|shows)|"
    r"(?:results|findings) (?:show|establish|indicate)|improves?|measured|computed)\b|"
    r"但是|然而|除非|我们|结果表明|不支持", re.I)
_BACKGROUND_HEADING = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?(?:introduction|background|motivation|引言|背景)\s*$")
_DETAIL_QUERY = re.compile(
    r"\b(?:methods?|mechanisms?|algorithms?|evaluat\w*|experiments?|results?|"
    r"limitations?|compar\w*|benchmarks?)\b|方法|机制|算法|评测|实验|结果|局限|比较|对比", re.I)
_BACKGROUND_QUERY = re.compile(
    r"\b(?:background|introduction|motivation|history|historical|definition|define|"
    r"availability|release|publication status|preprint|implementation status)\b|"
    r"背景|引言|动机|发展史|定义|发布时间|发布状态|开源|下载", re.I)
_QUALIFICATION = re.compile(
    r"\b(?:not|no|never|cannot|unless|except|however|although|only|if|assuming|"
    r"provided|conditional|condition|conditions|limitation|limitations|fail\w*)\b|"
    r"不|未|否|无|仅|若|如果|除非|然而|但是|条件|局限", re.I)
_EVIDENCE_ACTION = re.compile(
    r"\b(?:comput(?:e[sd]?|ing)|calculat\w*|measur\w*|sampl\w*|evaluat\w*|compar\w*|"
    r"train\w*|partition\w*|allocat\w*|retrieves?|retrieved|retrieving|"
    r"updates?|updated|construct\w*|uses?|used|using|selects?|selected|selecting|"
    r"outperform\w*|improv\w*|decreas\w*|increas\w*|observ\w*|demonstrat\w*|"
    r"causes?|caused|affects?|affected|depends?|depended|requires?|required|"
    r"produces?|produced|outputs?|returns?|extracts?|extracted|reports?|reported|"
    r"shows?|showed|finds?|found|prevents?|prevented|enables?|enabled)\b|"
    r"\b(?:defined as|denotes|means)\s+\S|"
    r"\d+(?:\.\d+)?\s*(?:%|(?:tokens?|samples?|participants?|seconds?|ms|GB|MB)\b)|"
    r"计算|采样|评测|划分|构建|检索|使用|训练|观察|提升|降低|表明", re.I)
_GENERIC_FRAMING = re.compile(
    r"\b(?:across (?:a |the )?(?:range|variety)|wide(?:ly)?|many|various|complex|broad)\b"
    r".{0,100}\b(?:domains?|fields?|areas?|applications?)\b|"
    r"\b(?:with (?:the )?(?:emergence|rise|development)|growing (?:interest|importance)|"
    r"recent advances|we are (?:already )?witnessing|in recent years)\b|"
    r"\b(?:has|have) (?:received|attracted) (?:much |considerable |increasing )?(?:attention|interest)\b|"
    r"\b(?:important|fundamental|essential|critical) (?:role|problem|challenge)\b", re.I | re.S)


def _edge_page_numbers(text):
    lines = text.strip().splitlines()
    while lines and re.fullmatch(r"\s*\d{1,4}\s*", lines[0]):
        lines.pop(0)
    while lines and re.fullmatch(r"\s*\d{1,4}\s*", lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def _pure_references(text):
    text = _edge_page_numbers(text)
    text = re.sub(r"^(?:References|Bibliography|参考文献)\s*\n", "", text, flags=re.I)
    matches = list(_REFERENCE_NUMBER.finditer(text))
    if len(matches) < 2 or text[:matches[0].start()].strip():
        return False
    numbers = [int(match.group(1)) for match in matches]
    if numbers != sorted(set(numbers)):
        return False
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        entry = text[match.end():end].strip()
        # A partial last entry or trailing prose is unknown, not pure metadata.
        if not _AUTHOR.search(entry) or not _BIBLIOGRAPHIC_END.search(entry):
            return False
    return True


def _pure_author_year_references(text):
    """Recognize repeated author/year/title/venue entries without a heading.

    PDF soft wraps and a clipped first author list are allowed, but a naked
    year/name, an unfinished last entry, or substantive prose is not enough.
    This deliberately misses unfamiliar citation styles rather than deleting
    a mixed evidence paragraph. It never changes the returned source string.
    """
    text = _edge_page_numbers(text)
    text = re.sub(r"^(?:References|Bibliography|参考文献)\s*\n", '', text, flags=re.I)
    if _NARRATIVE_IN_REFERENCES.search(text):
        return False
    starts = []
    for line in re.finditer(r"(?m)^\s*(?=\S)", text):
        if starts and line.start() < starts[-1][1]:
            continue  # A wrapped author line belongs to the same entry.
        match = _AUTHOR_YEAR.match(text[line.end():line.end() + 400])
        if match is None:
            continue
        authors = re.sub(r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2", match.group(1))
        # Names may contain initials, particles and diacritics, but a title
        # sentence preceding the next authors cannot become part of the list.
        words = re.findall(r"[^\W\d_]+(?:[’'−-][^\W\d_]+)*", authors, re.UNICODE)
        if (len(words) < 2 or any(not word[0].isupper() and word.casefold() not in _NAME_PARTICLES for word in words)
                or re.search(r"[\d]|[^\w\s.,’'−&-]", authors, re.UNICODE)):
            continue
        starts.append((line.start(), line.end() + match.end()))
    if len(starts) < 2 or text[:starts[0][0]].strip():
        return False
    for index, (_, title_start) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
        entry = text[title_start:end].strip()
        venue = _VENUE_END.search(entry)
        if venue is None or not entry[:venue.start()].strip() or len(entry) > 700:
            return False
    return True


def _pure_generic_background(question, text, section=''):
    """Only explicit, all-generic framing for a requested technical comparison.

    An Introduction heading alone is never a removal reason. Results, methods,
    definitions, numbers, conditions and negations protect the whole window.
    """
    if (not _DETAIL_QUERY.search(question) or _BACKGROUND_QUERY.search(question)
            or not (_BACKGROUND_HEADING.search(text) or _BACKGROUND_HEADING.fullmatch(section.strip()))):
        return False
    if _QUALIFICATION.search(text):
        return False
    body = _BACKGROUND_HEADING.sub('', text)
    body = re.sub(r"(?im)^\s*(?:Preprint\.?(?:\s+Under review\.?)?|arXiv:\d{4}\.\d{4,5}[^\n]*)\s*$", '', body)
    framing = 0
    for start, end in sentence_spans(body):
        value = ' '.join(body[start:end].split())
        if not value:
            continue
        # Availability announcements are metadata, but a sentence containing
        # method/results is still kept. Ignore only its generic release form.
        availability = re.search(r'\b(?:implementation|code|software)\b.{0,180}\b(?:forthcoming|available)\b', value, re.I)
        if availability and not _EVIDENCE_ACTION.search(value):
            continue
        if _EVIDENCE_ACTION.search(value) or not _GENERIC_FRAMING.search(value):
            return False
        framing += 1
    return framing > 0


def source_filter_reason(question, text, *, section=''):
    """Explain a whole-window exclusion; ``None`` means retain, not verified.

    Callers keep exact original offsets/text. Explicit metadata questions are
    exempt. Bounds fail open because classification cannot inspect the text.
    """
    if (not isinstance(question, str) or not isinstance(text, str)
            or len(question) > MAX_QUERY_CHARS or len(text) > MAX_SOURCE_CHARS):
        return None
    if _METADATA_QUERY.search(re.sub(r'\s+', ' ', question)):
        return None
    if _CONTACT.fullmatch(text.strip()):
        return 'isolated_contact'
    if _pure_references(text):
        return 'numbered_reference_list'
    if _pure_author_year_references(text):
        return 'author_year_reference_list'
    if _pure_affiliations(text):
        return 'numbered_affiliation_list'
    if _pure_generic_background(question, text, section if isinstance(section, str) else ''):
        return 'generic_background'
    return None


def _pure_affiliations(text):
    text = _edge_page_numbers(text)
    matches = list(_AFFILIATION_NUMBER.finditer(text))
    if len(matches) < 3 or text[:matches[0].start()].strip():
        return False
    numbers = [int(match.group(1)) for match in matches]
    if numbers != sorted(set(numbers)):
        return False
    anchored = 0
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        entry = text[match.end():end].strip()
        if (len(entry) > 280 or not entry or re.search(r"[.!?。！？;；:=<>@]", entry)
                or _AFFILIATION_PREDICATE.search(entry)):
            return False
        if re.search(r"[\u3400-\u9fff]", entry):
            # CJK affiliation-only lines need an institutional ending; prose
            # and unknown scripts are retained rather than language-guessed.
            if not re.fullmatch(r"[\u3400-\u9fffA-Za-z ,，()（）−-]+(?:大学|学院|研究所|研究院|科学院|系|学部)", entry):
                return False
            anchored += bool(_INSTITUTION.search(entry))
        else:
            # An institution word somewhere in a Title Case sentence does not
            # establish affiliation. Require at least two named-institution
            # shapes in the list; ambiguous short organization lines alone do
            # not qualify the pack.
            anchored += bool(_INSTITUTION_SHAPE.search(entry))
            words = re.split(r"[\s,()]+", entry)
            if any(word and word not in {"of", "and", "the", "for", "de", "la", "der", "und"}
                   and not _AFFILIATION_WORD.fullmatch(word) for word in words):
                return False
    return anchored >= 2


def filter_fallback_sources(question, sources):
    """Return original retained dictionaries, in order, and a bounded receipt."""
    trace = {"algorithm": "whole_window_evidence_hygiene_v2", "input_source_count": len(sources),
             "retained_source_count": len(sources),
             "filtered_count": 0, "reason_counts": {}, "provider_calls": 0,
             "llm_tokens": 0, "semantic_entailment_verified": False}
    if len(sources) > MAX_SOURCES or len(question) > MAX_QUERY_CHARS:
        return {"sources": list(sources), "trace": {**trace, "status": "bounds_exceeded_retained"}}
    if _METADATA_QUERY.search(re.sub(r"\s+", " ", question)):
        return {"sources": list(sources), "trace": {**trace, "status": "metadata_query_retained"}}
    kept, reasons = [], Counter()
    for source in sources:
        reason = source_filter_reason(question, source.get('content'), section=source.get('section', ''))
        if reason:
            reasons[reason] += 1
        else:
            kept.append(source)
    trace.update(status="filtered" if reasons else "unchanged", filtered_count=sum(reasons.values()),
                 reason_counts=dict(reasons), retained_source_count=len(kept))
    return {"sources": kept, "trace": trace}
