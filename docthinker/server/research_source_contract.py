"""Final research evidence packet contract, shared by producers and consumers.

Base retrieval remains capped at eight excerpts. Insight may append two using
the same 12 KB evidence budget; persistence must retain both supplements.
Citation scanning is intentionally broader so invalid labels remain warnings.
"""
import re

MAX_ANSWER_SOURCES = 10
SOURCE_LABEL_PATTERN = re.compile(r"S(?:[1-9]|10)\Z")
