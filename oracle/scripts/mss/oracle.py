"""
Oracle interface for MSS verification.

Provides YES/NO/SKIP decision interface for verifying whether
a (possibly masked) video contains sufficient evidence for a given action label.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .masking import MaskConfig, compute_mask_pattern_hash
from .prompts import get_prompt

logger = logging.getLogger(__name__)


class OracleDecision(Enum):
    """Oracle decision types."""

    YES = "YES"
    NO = "NO"
    SKIP = "SKIP"


@dataclass
class OracleResponse:
    """Response from oracle verification query.

    Attributes:
        decision: YES/NO/SKIP decision
        confidence: Confidence score 0.0-1.0
        evidence: Evidence text (brief description, max 20 words)
        rationale: Reasoning for the decision (2-3 sentences)
        raw_output: Raw model output for debugging
        logit_yes: Logit for YES token (if available)
        logit_no: Logit for NO token (if available)
        logit_skip: Logit for SKIP token (if available)
        logit_confidence: Binary P(YES|not SKIP) from YES/NO logits only.
            Uses 2-way softmax: P(YES) = exp(yes) / (exp(yes) + exp(no))
            SKIP is handled separately via logit_p_skip.
        logit_p_skip: P(SKIP) from 3-way softmax over YES/NO/SKIP (if available).
            If P(SKIP) > skip_threshold, the video should be flagged as
            corrupted rather than used for YES/NO decisions.
    """

    decision: OracleDecision
    confidence: float
    evidence: str
    rationale: str = ""
    raw_output: str = ""
    logit_yes: Optional[float] = None
    logit_no: Optional[float] = None
    logit_skip: Optional[float] = None
    logit_confidence: Optional[float] = None
    logit_p_skip: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "decision": self.decision.value,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "rationale": self.rationale,
            "raw_output": self.raw_output,
        }
        if self.logit_yes is not None:
            result["logit_yes"] = round(self.logit_yes, 4)
        if self.logit_no is not None:
            result["logit_no"] = round(self.logit_no, 4)
        if self.logit_skip is not None:
            result["logit_skip"] = round(self.logit_skip, 4)
        if self.logit_confidence is not None:
            result["logit_confidence"] = round(self.logit_confidence, 4)
        if self.logit_p_skip is not None:
            result["logit_p_skip"] = round(self.logit_p_skip, 4)
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "OracleResponse":
        """Create from dictionary."""
        return cls(
            decision=OracleDecision(d["decision"]),
            confidence=d["confidence"],
            evidence=d["evidence"],
            rationale=d.get("rationale", ""),
            raw_output=d.get("raw_output", ""),
            logit_yes=d.get("logit_yes"),
            logit_no=d.get("logit_no"),
            logit_skip=d.get("logit_skip"),
            logit_confidence=d.get("logit_confidence"),
            logit_p_skip=d.get("logit_p_skip"),
        )


@dataclass
class OracleCacheEntry:
    """Cache entry for oracle queries."""

    response: OracleResponse
    mask_pattern_hash: str
    prompt_template_id: str
    timestamp: str


@dataclass
class OracleConfig:
    """Oracle configuration.

    Attributes:
        prompt_template_id: ID of prompt template to use
        max_evidence_words: Maximum words in evidence field
    """

    prompt_template_id: str = "mss_verification"
    max_evidence_words: int = 30

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prompt_template_id": self.prompt_template_id,
            "max_evidence_words": self.max_evidence_words,
        }


def get_verification_prompt(prompt_template_id: str = "mss_verification_v1") -> str:
    """Get verification prompt template by ID.

    Args:
        prompt_template_id: Prompt template identifier

    Returns:
        Prompt template string
    """
    return get_prompt(prompt_template_id)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*(\{.*?\})\s*```", re.DOTALL
)


def _strip_preamble(raw: str) -> str:
    """Strip common wrappers Qwen sometimes emits before its JSON.

    Handles:
      - <think>...</think> blocks (Qwen3-VL thinking models)
      - ```json ... ``` markdown code fences
    """
    # Remove thinking blocks first; they may contain JSON-looking text
    # that would otherwise confuse downstream extractors.
    raw = _THINK_BLOCK_RE.sub("", raw)

    # If the model wrapped its JSON in a code fence, prefer that span.
    fence = _CODE_FENCE_RE.search(raw)
    if fence:
        return fence.group(1)
    return raw


def _extract_balanced(raw: str, start: int) -> Optional[str]:
    """Return the substring from `start` through the matching close brace.

    Walks bracket depth, treating string contents as opaque (with proper
    escape handling). Returns None if no balanced span exists.
    """
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(raw)):
        c = raw[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            if in_string:
                escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None


_ORACLE_TOP_LEVEL_KEYS = {
    "decision",
    "confidence",
    "action_summary",
    "evidence",
    "segments",
    "minimum_sufficient_set",
}


def _looks_like_oracle_response(data: Any) -> bool:
    """A parsed dict only counts if it has at least one expected top-level key.

    Without this filter, _try_balanced_parse would happily return inner
    objects (like {"segment_id": 1, ...}) when the outer object is unbalanced.
    """
    return isinstance(data, dict) and bool(
        set(data.keys()) & _ORACLE_TOP_LEVEL_KEYS
    )


def _try_balanced_parse(raw: str) -> Optional[Dict[str, Any]]:
    """Try every '{' position; first balanced substring that parses AND looks
    like an oracle response wins."""
    pos = 0
    while True:
        start = raw.find("{", pos)
        if start == -1:
            return None
        candidate = _extract_balanced(raw, start)
        if candidate is not None:
            try:
                data = json.loads(candidate)
                if _looks_like_oracle_response(data):
                    return data
            except json.JSONDecodeError:
                pass
        pos = start + 1


def _try_autoclose(raw: str) -> Optional[Dict[str, Any]]:
    """Recover a truncated JSON object by trimming and closing brackets.

    For genuinely cut-off output (e.g., max_new_tokens hit mid-segment),
    truncates at the last comma found OUTSIDE any string, then appends
    the closing brackets needed to balance the bracket stack.
    """
    start = raw.find("{")
    if start == -1:
        return None

    # First pass: find the last comma at non-string scope.
    in_string = False
    escape_next = False
    last_safe_comma = -1
    for i in range(start, len(raw)):
        c = raw[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            if in_string:
                escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == ",":
            last_safe_comma = i

    snippet = raw[start : last_safe_comma] if last_safe_comma > start else raw[start:]

    # Second pass: compute the bracket stack at the truncation point so we
    # know exactly which closers to append.
    stack: List[str] = []
    in_string = False
    escape_next = False
    for c in snippet:
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            if in_string:
                escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c in "{[":
            stack.append(c)
        elif c == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif c == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    # If we're inside a string at the truncation point, close it first.
    if in_string:
        snippet += '"'

    closing = "".join("}" if b == "{" else "]" for b in reversed(stack))
    snippet += closing

    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    return data if _looks_like_oracle_response(data) else None


_REGEX_DECISION = re.compile(r'"decision"\s*:\s*"([^"]+)"', re.IGNORECASE)
_REGEX_CONFIDENCE = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)')
_REGEX_SUMMARY = re.compile(
    r'"(?:action_summary|evidence)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"',
    re.DOTALL,
)


def _try_regex_extract(raw: str) -> Optional[Dict[str, Any]]:
    """Last-resort extraction of decision/confidence/summary via regex.

    Returns None unless at least the decision field is recoverable. The
    caller can still construct a usable OracleResponse from this.
    """
    decision_match = _REGEX_DECISION.search(raw)
    if not decision_match:
        return None
    data: Dict[str, Any] = {"decision": decision_match.group(1)}

    conf_match = _REGEX_CONFIDENCE.search(raw)
    if conf_match:
        try:
            data["confidence"] = float(conf_match.group(1))
        except ValueError:
            pass

    summary_match = _REGEX_SUMMARY.search(raw)
    if summary_match:
        data["evidence"] = summary_match.group(1)

    data["_recovered_via_regex"] = True
    return data


def parse_oracle_response(raw_output: str) -> OracleResponse:
    """Parse oracle response from raw model output.

    Robust to:
      - <think>...</think> preambles (Qwen3-VL thinking models)
      - ```json ... ``` markdown code fences
      - Leading prose before the JSON
      - Trailing prose / extra braces after the JSON
      - Genuinely truncated output (auto-closes brackets)
      - Falls back to regex extraction of `decision`, `confidence`,
        `action_summary`/`evidence` if structured parsing fails.

    Args:
        raw_output: Raw text output from model

    Returns:
        Parsed OracleResponse

    Raises:
        ValueError: If even regex fallback finds no decision token.
    """
    cleaned = _strip_preamble(raw_output)

    data: Optional[Dict[str, Any]] = None
    recovery_path = "strict"

    # Step 1: strict balanced parse over every candidate '{' span.
    data = _try_balanced_parse(cleaned)

    # Step 2: auto-close brackets for genuinely truncated output.
    if data is None:
        data = _try_autoclose(cleaned)
        if data is not None:
            recovery_path = "autoclose"
            logger.warning(
                "Oracle JSON parse: recovered via auto-close fallback "
                "(response was truncated or had unbalanced brackets)"
            )

    # Step 3: regex extraction of key fields as last resort.
    if data is None:
        data = _try_regex_extract(cleaned)
        if data is not None:
            recovery_path = "regex"
            logger.warning(
                "Oracle JSON parse: recovered via regex fallback "
                "(structured parsing failed; only top-level fields extracted)"
            )

    if data is None:
        head = raw_output[:200]
        tail = raw_output[-200:] if len(raw_output) > 400 else ""
        raise ValueError(
            f"No valid JSON found in oracle response "
            f"(len={len(raw_output)}, recovery=failed). "
            f"Head: {head!r} | Tail: {tail!r}"
        )

    if recovery_path != "strict":
        logger.debug(f"Recovery path: {recovery_path}")

    # Parse decision
    decision_str = data.get("decision", "").upper()
    if decision_str == "YES":
        decision = OracleDecision.YES
    elif decision_str == "NO":
        decision = OracleDecision.NO
    elif decision_str in ("SKIP", "INSUFFICIENT", "UNKNOWN", "UNCERTAIN"):
        decision = OracleDecision.SKIP
    else:
        raise ValueError(f"Unknown decision: {decision_str}")

    # Parse confidence
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    # Parse evidence (v4 prompts use "action_summary" instead of "evidence")
    evidence = str(data.get("evidence", "") or data.get("action_summary", ""))

    # Parse rationale
    rationale = str(data.get("rationale", ""))

    return OracleResponse(
        decision=decision,
        confidence=confidence,
        evidence=evidence,
        rationale=rationale,
        raw_output=raw_output,
    )


def get_logit_confidence(response: OracleResponse) -> float:
    """Get P(YES | not SKIP) from a single oracle response.

    Uses logit_confidence (binary softmax of YES/NO logits) for scoring.
    Falls back to text-based confidence if logits unavailable.

    Note: This returns the binary YES/NO confidence. Callers should check
    response.logit_p_skip separately to detect corrupted videos.

    Args:
        response: Single oracle response

    Returns:
        P(YES | not SKIP) confidence (0.0 to 1.0)
    """
    if response.logit_confidence is not None:
        return response.logit_confidence
    # Fallback: use text confidence if YES, else 0
    return response.confidence if response.decision == OracleDecision.YES else 0.0


def is_skip_flagged(
    response: OracleResponse, skip_threshold: float = 0.5
) -> bool:
    """Check if a response should be treated as SKIP based on logits.

    A response is flagged as SKIP if P(SKIP) from the 3-way softmax
    exceeds the threshold. This means the model considers the video
    corrupted/unreadable, separate from the YES/NO decision.

    Args:
        response: Oracle response to check
        skip_threshold: P(SKIP) threshold (default 0.5)

    Returns:
        True if the video should be flagged as corrupted
    """
    if response.logit_p_skip is not None:
        return response.logit_p_skip > skip_threshold
    # Fallback: check text decision
    return response.decision == OracleDecision.SKIP


def logits_to_confidence(
    logit_yes: float,
    logit_no: float,
    logit_skip: Optional[float] = None,
    skip_threshold: float = 0.5,
) -> Tuple[float, Optional[float]]:
    """Two-stage logit evaluation: check SKIP first, then binary YES/NO.

    Stage 1: If SKIP logit is available, compute P(SKIP) via 3-way softmax.
        If P(SKIP) > skip_threshold, the video is likely corrupted/unreadable.

    Stage 2: Compute P(YES | not SKIP) using only YES and NO logits:
        P(YES) = exp(yes) / (exp(yes) + exp(no))
        This gives a clean binary signal without SKIP stealing probability mass.

    Args:
        logit_yes: Logit for YES token
        logit_no: Logit for NO token
        logit_skip: Logit for SKIP token (optional)
        skip_threshold: P(SKIP) threshold for flagging corrupted videos

    Returns:
        Tuple of (p_yes, p_skip):
        - p_yes: P(YES | not SKIP) from binary softmax of YES/NO only
        - p_skip: P(SKIP) from 3-way softmax, or None if SKIP logit unavailable
    """
    import math
    import os

    # Binary P(YES | not SKIP) from YES and NO only (default / production behavior)
    diff = logit_yes - logit_no
    diff = max(-100.0, min(100.0, diff))
    p_yes_2way = 1.0 / (1.0 + math.exp(-diff))

    # Compute P(SKIP) and 3-way P(YES) from 3-way softmax if SKIP logit available
    p_skip = None
    p_yes_3way = None
    if logit_skip is not None:
        max_logit = max(logit_yes, logit_no, logit_skip)
        exp_yes = math.exp(logit_yes - max_logit)
        exp_no = math.exp(logit_no - max_logit)
        exp_skip = math.exp(logit_skip - max_logit)
        z = exp_yes + exp_no + exp_skip
        p_skip = exp_skip / z
        p_yes_3way = exp_yes / z

    # Opt-in legacy 3-way confidence (SKIP steals probability mass), gated by env var.
    # Default is the 2-way P(YES|not SKIP). Set MSS_CONFIDENCE_3WAY=1 to reproduce the
    # pre-b7ecf45 greedy behavior for apples-to-apples analysis.
    if os.environ.get("MSS_CONFIDENCE_3WAY") == "1" and p_yes_3way is not None:
        p_yes = p_yes_3way
    else:
        p_yes = p_yes_2way

    return p_yes, p_skip


class OracleCache:
    """Cache for oracle query results.

    Caches results by (video_id, mask_pattern_hash, prompt_template_id).
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache.

        Args:
            cache_dir: Directory for persistent cache (optional)
        """
        self._memory_cache: Dict[str, List[OracleResponse]] = {}
        self._cache_dir = cache_dir
        self._hits = 0
        self._misses = 0

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(
        self,
        video_id: str,
        mask_pattern_hash: str,
        prompt_template_id: str,
    ) -> str:
        """Create cache key."""
        return f"{video_id}:{mask_pattern_hash}:{prompt_template_id}"

    def get(
        self,
        video_id: str,
        mask_pattern_hash: str,
        prompt_template_id: str,
    ) -> Optional[List[OracleResponse]]:
        """Get cached responses if available.

        Args:
            video_id: Video identifier
            mask_pattern_hash: Hash of mask pattern
            prompt_template_id: Prompt template ID

        Returns:
            List of cached responses or None
        """
        key = self._make_key(video_id, mask_pattern_hash, prompt_template_id)

        if key in self._memory_cache:
            self._hits += 1
            return self._memory_cache[key]

        # Try file cache
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                    responses = [OracleResponse.from_dict(d) for d in data]
                    self._memory_cache[key] = responses
                    self._hits += 1
                    return responses
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")

        self._misses += 1
        return None

    def put(
        self,
        video_id: str,
        mask_pattern_hash: str,
        prompt_template_id: str,
        responses: List[OracleResponse],
    ) -> None:
        """Store responses in cache.

        Args:
            video_id: Video identifier
            mask_pattern_hash: Hash of mask pattern
            prompt_template_id: Prompt template ID
            responses: Oracle responses to cache
        """
        key = self._make_key(video_id, mask_pattern_hash, prompt_template_id)
        self._memory_cache[key] = responses

        # Write to file cache
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key}.json"
            try:
                with open(cache_file, "w") as f:
                    json.dump([r.to_dict() for r in responses], f)
            except OSError as e:
                logger.warning(f"Failed to write cache file {cache_file}: {e}")

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
            "memory_entries": len(self._memory_cache),
        }


class OracleInterface:
    """Interface for querying VLM oracle.

    This is an abstract interface - concrete implementations will use
    specific VLM backends (Qwen, etc.).
    """

    def __init__(
        self,
        config: OracleConfig,
        cache: Optional[OracleCache] = None,
    ):
        """Initialize oracle interface.

        Args:
            config: Oracle configuration
            cache: Optional cache for responses
        """
        self.config = config
        self.cache = cache or OracleCache()
        self._call_count = 0

    def query_single(
        self,
        frames: List[np.ndarray],
        action_label: str,
        video_id: str,
        removal_note: Optional[str] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> OracleResponse:
        """Query oracle with a single video (frames).

        This method must be implemented by concrete subclasses.

        Args:
            frames: List of video frames (RGB numpy arrays)
            action_label: Action label to verify
            video_id: Video identifier for logging
            removal_note: Optional note about removed segments (for CUT operator)
            prompt_context: Optional dict with additional prompt variables
                           (e.g., action_template, placeholders for SSv2)

        Returns:
            Oracle response
        """
        raise NotImplementedError("Subclasses must implement query_single")

    def query_batch(
        self,
        batch: List[Dict[str, Any]],
        batch_size: int = 4,
    ) -> List[OracleResponse]:
        """Query oracle with a batch of videos.

        Default implementation calls query_single sequentially.
        Subclasses can override for true batch processing.

        Args:
            batch: List of dicts with keys:
                - frames: List of video frames
                - action_label: Action label to verify
                - video_id: Video identifier
                - mask_pattern_hash: Hash for caching
                - removal_note: Optional removal note
                - prompt_context: Optional prompt context
            batch_size: Number of videos per forward pass (used by subclasses)

        Returns:
            List of oracle responses (same order as batch)
        """
        responses = []
        for item in batch:
            response = self.query_search(
                frames=item["frames"],
                action_label=item["action_label"],
                video_id=item["video_id"],
                mask_pattern_hash=item["mask_pattern_hash"],
                removal_note=item.get("removal_note"),
                prompt_context=item.get("prompt_context"),
            )
            responses.append(response)
        return responses

    def _get_effective_template_id(
        self, prompt_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get the effective prompt template ID from context or config."""
        if prompt_context and "prompt_template_id" in prompt_context:
            return prompt_context["prompt_template_id"]
        return self.config.prompt_template_id

    def query_search(
        self,
        frames: List[np.ndarray],
        action_label: str,
        video_id: str,
        mask_pattern_hash: str,
        removal_note: Optional[str] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> OracleResponse:
        """Query oracle for search-time screening.

        Uses single logit-based query with P(YES) > 0.5 threshold.

        Args:
            frames: List of video frames
            action_label: Action label to verify
            video_id: Video identifier
            mask_pattern_hash: Hash of mask pattern for caching
            removal_note: Optional note about removed segments (for CUT operator)
            prompt_context: Optional dict with additional prompt variables

        Returns:
            Single oracle response
        """
        # Get effective template ID (from context or config)
        template_id = self._get_effective_template_id(prompt_context)

        # Check cache
        cached = self.cache.get(video_id, mask_pattern_hash, template_id)
        if cached:
            return cached[0]

        # Query oracle
        response = self.query_single(
            frames, action_label, video_id, removal_note, prompt_context
        )
        self._call_count += 1

        # Cache result
        self.cache.put(video_id, mask_pattern_hash, template_id, [response])

        return response

    def query_final(
        self,
        frames: List[np.ndarray],
        action_label: str,
        video_id: str,
        mask_pattern_hash: str,
        removal_note: Optional[str] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[OracleResponse, float, bool]:
        """Query oracle for pre-check verification (single query).

        Two-stage evaluation:
        1. Check P(SKIP) — if high, video is corrupted → fail
        2. Check P(YES | not SKIP) using binary YES/NO softmax → pass if > 0.5

        Args:
            frames: List of video frames
            action_label: Action label to verify
            video_id: Video identifier
            mask_pattern_hash: Hash of mask pattern for caching
            removal_note: Optional note about removed segments (for CUT operator)
            prompt_context: Optional dict with additional prompt variables

        Returns:
            Tuple of (response, p_yes, ok_final)
            - response: Single oracle response
            - p_yes: P(YES | not SKIP) from binary logits
            - ok_final: True if not SKIP-flagged and P(YES) > 0.5
        """
        # Get effective template ID (from context or config)
        template_id = self._get_effective_template_id(prompt_context)

        # Check cache
        cache_key = f"{mask_pattern_hash}_final"
        cached = self.cache.get(video_id, cache_key, template_id)
        if cached and len(cached) >= 1:
            response = cached[0]
            p_yes = get_logit_confidence(response)
            ok = not is_skip_flagged(response) and p_yes > 0.5
            return response, p_yes, ok

        # Single query
        response = self.query_single(
            frames, action_label, video_id, removal_note, prompt_context
        )
        self._call_count += 1

        # Cache response
        self.cache.put(video_id, cache_key, template_id, [response])

        # Two-stage: check SKIP first, then binary P(YES)
        p_yes = get_logit_confidence(response)
        ok = not is_skip_flagged(response) and p_yes > 0.5
        return response, p_yes, ok

    def ok_search(self, response: OracleResponse) -> bool:
        """Check if response passes search-time screening.

        Args:
            response: Oracle response

        Returns:
            True if decision is YES
        """
        return response.decision == OracleDecision.YES

    @property
    def call_count(self) -> int:
        """Get total oracle call count."""
        return self._call_count

    @property
    def stats(self) -> dict:
        """Get oracle statistics."""
        return {
            "call_count": self._call_count,
            "cache_stats": self.cache.stats,
            "config": self.config.to_dict(),
        }
