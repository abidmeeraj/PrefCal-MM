"""Parsers for judge outputs: Likert integers and batched fact verdicts."""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

_INT_RE = re.compile(r"-?\d+")

# A line that opens with the fact index ("1.", "1)", "1:", "1 -") followed by
# the rest of the line, in which the verdict word is searched.
_LINE_RE = re.compile(r"^\s*(\d+)\s*[.):\-]?\s+(.*)$", re.MULTILINE)
# "not supported" is tried before "supported", so the longer match wins.
_VERDICT_RE = re.compile(r"\b(not[ _]?supported|supported)\b", re.IGNORECASE)


def extract_first_int(text: str, allowed: Optional[Iterable[int]] = None) -> Optional[int]:
    """First integer in ``text`` (restricted to ``allowed`` if given), else None."""
    if not text:
        return None
    if allowed is not None:
        allowed = set(int(a) for a in allowed)
    for m in _INT_RE.finditer(text):
        try:
            v = int(m.group())
        except ValueError:
            continue
        if allowed is None or v in allowed:
            return v
    return None


def strip_chat_artifacts(text: str) -> str:
    """Drop prompt delimiters that some chat templates echo into the output."""
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1]
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1]
    return text.strip()


def parse_batched_verdicts(raw: str, n: int, *, strip_artifacts: bool = False) -> List[str]:
    """Parse ``n`` numbered verdicts into {'supported', 'not_supported', 'unk'}.

    Two passes:
      1. Strict: the printed line index must fall in 1..n; the first verdict
         seen for an index wins.
      2. Positional fallback: if the strict pass leaves every slot 'unk' but
         the response contains exactly ``n`` verdict-bearing numbered lines,
         assign them by position. This recovers outputs whose numbering is
         offset (e.g. a judge that continues counting from a numbered list
         earlier in the prompt).

    Slots that remain 'unk' are excluded from the factual-score denominator.
    """
    verdicts: List[str] = ["unk"] * n
    text = strip_chat_artifacts(raw) if strip_artifacts else raw
    for m in _LINE_RE.finditer(text):
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        if idx < 1 or idx > n:
            continue
        if verdicts[idx - 1] != "unk":
            continue
        v = _VERDICT_RE.search(m.group(2))
        if v is None:
            continue
        verdicts[idx - 1] = (
            "not_supported" if "not" in v.group(1).lower() else "supported"
        )

    if all(v == "unk" for v in verdicts):
        positional: List[str] = []
        for m in _LINE_RE.finditer(text):
            v = _VERDICT_RE.search(m.group(2))
            if v is None:
                continue
            positional.append(
                "not_supported" if "not" in v.group(1).lower() else "supported"
            )
        if len(positional) == n:
            verdicts = positional

    return verdicts
