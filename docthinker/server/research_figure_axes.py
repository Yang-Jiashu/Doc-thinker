"""High-confidence PDF axis detritus, never general numeric-data filtering.

Only a dense, almost entirely numeric unit immediately before an explicit
figure caption is eligible. Unknown mixed text stays intact. These bounded
typographic tests do not interpret a figure, table or paper's scientific claim.
"""
from __future__ import annotations

import math
import re


_NUMBER = re.compile(r"[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?")
_AXIS_LABEL = re.compile(r"[A-Za-z][A-Za-z \t()\[\]/%_-]{0,47}")
_GRAPH_DATA_REQUEST = re.compile(
    r"\b(?:axes|axis|ticks?|coordinates?|abscissa|ordinate)\b|"
    r"\b(?:figure|fig\.|plot|graph|chart)\b.{0,64}\b(?:data|values?|readings?|numbers?|scale|points?)\b|"
    r"\b(?:data|values?|readings?|numbers?|scale|points?)\b.{0,64}\b(?:figure|fig\.|plot|graph|chart)\b|"
    r"坐标|刻度|横轴|纵轴|图.{0,16}(?:读数|数据|数值|取值)|(?:读数|数据|数值).{0,16}图", re.I,
)


def requests_graph_data(question: str) -> bool:
    """Explicit coordinate/figure-data requests retain their numeric units."""
    return bool(_GRAPH_DATA_REQUEST.search(question))


def pure_axis_ticks(text: str, *, immediately_before_caption: bool) -> bool:
    """Reject ambiguous numeric tables, variables, quantities and prose.

    The figure context is mandatory. Eight single-number lines, or a long
    strictly monotonic all-number line, plus at most three short alphabetic
    labels are the only accepted forms. Mixed number+unit/label lines and
    multi-column rows are deliberately retained, even when they might be axes.
    """
    if not immediately_before_caption or not isinstance(text, str) or len(text) > 8_000:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or len(lines) > 256:
        return False
    short_ticks = long_ticks = numeric_lines = label_lines = 0
    for line in lines:
        if len(line) <= 24 and _NUMBER.fullmatch(line):
            short_ticks += 1
            numeric_lines += 1
            continue
        pieces = line.split()
        if len(pieces) >= 8 and all(_NUMBER.fullmatch(piece) for piece in pieces):
            values = [float(piece.replace("−", "-")) for piece in pieces]
            if (not all(math.isfinite(value) for value in values)
                    or not (all(a < b for a, b in zip(values, values[1:]))
                            or all(a > b for a, b in zip(values, values[1:])))):
                return False
            long_ticks += 1
            numeric_lines += 1
            continue
        if _AXIS_LABEL.fullmatch(line) and len(re.findall(r"[A-Za-z]+", line)) <= 3:
            label_lines += 1
            if label_lines > 3:
                return False
            continue
        return False
    # Multiple long numeric rows could just as easily be a matrix/table.
    return (long_ticks <= 1 and (short_ticks >= 8 or long_ticks == 1)
            and (long_ticks == 1 or numeric_lines >= 2 * label_lines))
