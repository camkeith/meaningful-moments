"""
Parse Qwen3-VL output into a validated top-5 list of SSv2 labels.

Expected input: a JSON array of 5 strings, each an exact copy of a taxonomy
template. Normalization layer handles the common cleanly-reversible variants
(case, whitespace, punctuation) so we don't waste a retry on a trivially
fixable output. Anything beyond that (abbreviations, paraphrases, invented
templates) is rejected — we prefer a clean parse failure over a wrong label.

Maps: normalized-form -> canonical taxonomy string. Canonical string maps to
an id via its index in SSV2_LABELS.
"""

import json
import re
from typing import List, Optional, Tuple

from .prompt import SSV2_LABELS, SSV2_LABEL_SET


PARSE_FAILED = object()  # sentinel


# ---- normalization -----------------------------------------------------------
# Reversible transforms only: lowercase, collapse whitespace, strip common
# trailing punctuation, and drop wrapping quotes/parens. No fuzzy matching.

_PUNCT_STRIP_RE = re.compile(r'[.,;!?\'"`]+$')  # trailing punctuation only
_WRAPPING_RE = re.compile(r'^[\[\("\']+|[\]\)"\']+$')
_WS_RE = re.compile(r"\s+")


def _norm_key(s: str) -> str:
    s = s.strip()
    s = _WRAPPING_RE.sub("", s).strip()
    s = _PUNCT_STRIP_RE.sub("", s).strip()
    s = s.lower()
    s = _WS_RE.sub(" ", s)
    return s


_NORMAL_MAP = {_norm_key(label): label for label in SSV2_LABELS}
assert len(_NORMAL_MAP) == len(SSV2_LABELS), "Normalization keys collided for the taxonomy"


def _normalize(s: str) -> Optional[str]:
    """Return canonical label if `s` maps to one under the reversible-normalization layer, else None."""
    if s in SSV2_LABEL_SET:
        return s
    return _NORMAL_MAP.get(_norm_key(s))


# ---- JSON extraction ---------------------------------------------------------

_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw


_TOP5_KEYS = ("top5_label_ids", "top5_ids", "top5")  # accepted schemas in priority order


def _arr_from_dict(v) -> Optional[list]:
    if not isinstance(v, dict):
        return None
    for k in _TOP5_KEYS:
        if k in v and isinstance(v[k], list):
            return v[k]
    return None


def _extract_top5_array(raw: str) -> Optional[list]:
    """Parse model output. Accepts (in priority order):
       {"top5_label_ids":[int,...]}  — new ID-based schema
       {"top5_ids":[int,...]}
       {"top5":[str,...]}            — legacy string schema
       bare [ ... ]                  — legacy
    """
    raw = _strip_fences(raw)

    # 1) Whole-string JSON
    try:
        v = json.loads(raw)
        arr = _arr_from_dict(v)
        if arr is not None:
            return arr
        if isinstance(v, list):
            return v
    except json.JSONDecodeError:
        pass

    # 2) {...} object anywhere in string
    m = _JSON_OBJECT_RE.search(raw)
    if m:
        try:
            v = json.loads(m.group(0))
            arr = _arr_from_dict(v)
            if arr is not None:
                return arr
        except json.JSONDecodeError:
            pass

    # 3) bare [...] array
    m = _JSON_ARRAY_RE.search(raw)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    return None


def _coerce_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return int(s)
            except ValueError:
                return None
    return None


# ---- public API --------------------------------------------------------------

def parse_top5(raw_output: str) -> Tuple[List[str], bool]:
    top5, norm, _reason = parse_top5_diagnostic(raw_output)
    return top5, norm


def parse_top5_diagnostic(raw_output: str) -> Tuple[List[str], bool, str]:
    """Parse model output into (top5_canonical_labels, normalization_applied, reason).

    Primary schema: {"top5_label_ids": [int, int, int, int, int]}  with ints in [1, 174].
    Fallback schemas: {"top5": [str, ...]} or bare [str, ...] with exact-match strings.

    Returns ([], False, <reason>) on failure.
    """
    arr = _extract_top5_array(raw_output)
    if arr is None:
        return [], False, "no JSON array found"
    if len(arr) != 5:
        return [], False, f"expected 5 items, got {len(arr)}"

    # Try integer-id mode first (0-indexed: range [0, 173]).
    maybe_ids = [_coerce_int(x) for x in arr]
    if all(i is not None for i in maybe_ids):
        max_id = len(SSV2_LABELS) - 1  # 173
        for pos, i in enumerate(maybe_ids):
            if not (0 <= i <= max_id):
                return [], False, f"item[{pos}] id={i} out of range [0, {max_id}]"
        labels = [SSV2_LABELS[i] for i in maybe_ids]
        if len(set(labels)) != 5:
            return [], False, "duplicate ids"
        return labels, False, ""

    # Fallback: string mode (legacy).
    out: List[str] = []
    normalized_any = False
    for i, item in enumerate(arr):
        if not isinstance(item, str):
            return [], False, f"item[{i}] neither int nor string ({type(item).__name__}): {item!r}"
        canon = _normalize(item)
        if canon is None:
            return [], False, f"item[{i}] {item!r} is not in taxonomy"
        if canon != item:
            normalized_any = True
        out.append(canon)
    if len(set(out)) != 5:
        return [], False, "duplicate labels"
    return out, normalized_any, ""


def label_to_id(label: str) -> int:
    """Canonical-label -> SSv2 class id (0-indexed, matching labels.json ordering)."""
    return SSV2_LABELS.index(label)
