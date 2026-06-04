"""
Closed-set SSv2 classification prompt.

Loads the 174 SSv2 class templates from data/SSv2/labels.json and builds the
system + user messages that ask Qwen3-VL for a ranked JSON array of the top 5
most likely labels, chosen verbatim from the taxonomy.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LABELS_PATH = _REPO_ROOT / "data" / "SSv2" / "labels.json"


def _load_labels(path: Path = _DEFAULT_LABELS_PATH) -> List[str]:
    with open(path, "r") as f:
        d = json.load(f)
    # labels.json is {template_string: "class_id_as_str"}; sort by class id
    return [k for k, _ in sorted(d.items(), key=lambda kv: int(kv[1]))]


SSV2_LABELS: List[str] = _load_labels()
SSV2_LABEL_SET = set(SSV2_LABELS)
assert len(SSV2_LABELS) == 174, f"Expected 174 SSv2 labels, got {len(SSV2_LABELS)}"


_NUMBERED_LIST = "\n".join(f"{i}: {label}" for i, label in enumerate(SSV2_LABELS))
_MAX_ID = len(SSV2_LABELS) - 1  # 173 for standard SSv2

_SYSTEM_PROMPT = """You are a strict closed-set classifier for the Something-Something V2 (SSv2) dataset.

Use only the allowed classes in the user prompt.
Return exactly 5 unique class IDs ranked from most likely to least likely.
Do not output class names.
Do not paraphrase.
Do not invent actions.
Do not output explanations.

Return only valid JSON in exactly this format:
{"top5_label_ids":[ID1,ID2,ID3,ID4,ID5]}
"""


_USER_PROMPT = f"""Classify this video into the 5 most likely Something-Something V2 classes.

Focus on:
- the main object interaction,
- the direction or relation,
- actual vs pretending,
- any important qualifier such as "but pulling it right out", "but missing", "revealing", or "not tearable".

Requirements:
- Return exactly 5 unique IDs.
- Use only IDs from the allowed list below (valid range: 0 to {_MAX_ID}).
- The ID is the integer before the colon on each allowed-class line.
- Do not output class names.
- Do not output explanations.
- Do not output any text besides the JSON.

Before responding, verify that:
1. all 5 IDs are integers,
2. all 5 IDs are unique,
3. every ID is between 0 and {_MAX_ID},
4. the response contains only the JSON object.
If any of those checks fail, fix the response before sending it.

Return only:
{{"top5_label_ids":[ID1,ID2,ID3,ID4,ID5]}}

Allowed classes:
{_NUMBERED_LIST}
"""


def build_classification_messages(pil_frames, video_fps: float) -> list:
    """Build the chat messages for Qwen3-VL classification (official typed-content format)."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": pil_frames, "fps": video_fps},
                {"type": "text", "text": _USER_PROMPT},
            ],
        },
    ]


def prompt_hash() -> str:
    """Stable hash of the (system + user) prompt content. Used for run-metadata parity checks."""
    h = hashlib.sha256()
    h.update(_SYSTEM_PROMPT.encode("utf-8"))
    h.update(b"\n---\n")
    h.update(_USER_PROMPT.encode("utf-8"))
    return h.hexdigest()[:16]
