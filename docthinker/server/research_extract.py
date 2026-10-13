"""Offline text extraction child. No provider, media renderer or external URL.

Run as a short-lived process, with CPU/address-space limits where supported.
These limits are resource containment, not a hostile-code sandbox. PDF reading
does not execute embedded JavaScript, attachments or document instructions.
"""
from __future__ import annotations

import io
import json
import logging
import re
import sys

from .research_sentence_windows import evidence_units

MAX_BYTES = 10 * 1024 * 1024
MAX_TEXT = 300_000
MAX_PAGES = 300
MAX_CHUNKS = 600
CHUNK_CHARS = 2_000
MAX_STATEMENT_CHARS = 8_000


class ExtractionError(ValueError):
    pass


def extract_chunks(content: bytes, extension: str) -> list[dict]:
    if not content or len(content) > MAX_BYTES:
        raise ExtractionError("file_size_limit")
    pages = []
    if extension in {".txt", ".md"}:
        try:
            pages = [(None, content.decode("utf-8-sig", errors="strict"))]
        except UnicodeError:
            raise ExtractionError("utf8_required") from None
    elif extension == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise ExtractionError("invalid_pdf")
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content), strict=True)
        if reader.is_encrypted:
            raise ExtractionError("encrypted_pdf")
        if len(reader.pages) > MAX_PAGES:
            raise ExtractionError("page_limit")
        total = 0
        for index, page in enumerate(reader.pages):
            stream = page.get_contents()
            if stream is not None and len(stream.get_data()) > MAX_BYTES:
                raise ExtractionError("page_stream_limit")
            text = page.extract_text() or ""
            total += len(text)
            if total > MAX_TEXT:
                raise ExtractionError("text_limit")
            pages.append((index + 1, text))
    else:
        raise ExtractionError("unsupported_format")
    if sum(len(text) for _, text in pages) > MAX_TEXT:
        raise ExtractionError("text_limit")
    chunks = []
    for page_number, text in pages:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text).strip()
        # PDF line wraps are layout, not statement boundaries. Keep a claim
        # and its visible adjacent qualification in the same source chunk.
        # A long statement may exceed the target but never the store's limit.
        start = end = None
        def append_part(left, right):
            part = text[left:right].strip()
            if part:
                chunks.append({"content": part, "page": page_number, "ordinal": len(chunks)})
            if len(chunks) > MAX_CHUNKS:
                raise ExtractionError("chunk_limit")
        for left, right in evidence_units(text):
            if len(text[left:right].strip()) > MAX_STATEMENT_CHARS:
                raise ExtractionError("statement_size_limit")
            if start is not None and right - start > CHUNK_CHARS:
                append_part(start, end)
                start = None
            if start is None:
                start = left
            end = right
        if start is not None:
            append_part(start, end)
    if not chunks:
        raise ExtractionError("no_extractable_text")
    return chunks


def main() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (15, 16))
        # RLIMIT_AS is useful on Linux; macOS may not implement its enforcement.
        try:
            resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
        except (ValueError, OSError):
            pass
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    except ImportError:
        pass
    logging.disable(logging.CRITICAL)
    try:
        extension = sys.argv[1] if len(sys.argv) == 2 else ""
        result = {"chunks": extract_chunks(sys.stdin.buffer.read(MAX_BYTES + 1), extension)}
    except ExtractionError as error:
        result = {"error_code": str(error)}
    except Exception:
        result = {"error_code": "extraction_failed"}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
