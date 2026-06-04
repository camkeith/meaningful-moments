"""
Gemini API oracle implementation for MSS verification.

Implements OracleInterface using Google's google-genai SDK. Sends sampled
video frames as inline JPEG image parts and extracts decision-token logprobs
(when available) for two-stage YES/NO/SKIP scoring.

No local GPU required. Concurrency + rate limiting follow the same pattern
as DartmouthOracle (thread pool + sliding-window limiter).
"""

import base64
import io
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ._util_ratelimit import RateLimiter
from .oracle import (
    OracleCache,
    OracleConfig,
    OracleDecision,
    OracleInterface,
    OracleResponse,
    get_verification_prompt,
    logits_to_confidence,
    parse_oracle_response,
)

logger = logging.getLogger(__name__)


def _frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode an RGB frame as JPEG bytes for Gemini inline image parts."""
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _extract_decision_logprobs_gemini(
    chosen_candidates: List[Any],
    top_candidates: List[Any],
) -> Optional[Dict[str, float]]:
    """Extract YES/NO/SKIP logprobs from Gemini's logprobs_result.

    Gemini's logprobs_result has two parallel arrays:
      - chosen_candidates: per-position [{token, log_probability}, ...]
        (the actually-generated tokens)
      - top_candidates: per-position [{candidates: [{token, log_probability}, ...]}, ...]
        (top-k alternates at each position)

    We scan chosen_candidates for the first YES/NO/SKIP token, then read
    sibling logprobs for the other decision tokens from top_candidates at
    that same position.

    Args:
        chosen_candidates: List of {token, log_probability} dicts
        top_candidates: List of {candidates: [...]} dicts (parallel to chosen)

    Returns:
        Dict {"YES": logprob, "NO": logprob, "SKIP": logprob_or_missing} or
        None if a decision token wasn't found.
    """
    decision_tokens = {"YES", "NO", "SKIP"}

    for pos, chosen in enumerate(chosen_candidates):
        token = _get_token(chosen).strip().upper()
        if token not in decision_tokens:
            continue

        # Found the decision position. Read top_candidates at the same index.
        logprob_map: Dict[str, float] = {}

        if pos < len(top_candidates):
            top_at_pos = top_candidates[pos]
            cand_list = _get_candidates_list(top_at_pos)
            for cand in cand_list:
                t = _get_token(cand).strip().upper()
                if t in decision_tokens:
                    lp = _get_logprob(cand)
                    if lp is None:
                        continue
                    if t not in logprob_map or lp > logprob_map[t]:
                        logprob_map[t] = lp

        # Always include the chosen token itself
        chosen_lp = _get_logprob(chosen)
        if chosen_lp is not None and (token not in logprob_map or chosen_lp > logprob_map[token]):
            logprob_map[token] = chosen_lp

        if "YES" in logprob_map and "NO" in logprob_map:
            return logprob_map
        if logprob_map:
            return logprob_map

    return None


def _get_token(obj: Any) -> str:
    """Read a token field from either an SDK object or a plain dict."""
    if isinstance(obj, dict):
        return str(obj.get("token", "") or "")
    return str(getattr(obj, "token", "") or "")


def _get_logprob(obj: Any) -> Optional[float]:
    """Read a logprob field — Gemini SDK uses log_probability."""
    if isinstance(obj, dict):
        v = obj.get("log_probability", obj.get("logprob"))
    else:
        v = getattr(obj, "log_probability", None)
        if v is None:
            v = getattr(obj, "logprob", None)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _get_candidates_list(obj: Any) -> List[Any]:
    """Read the candidates list from a top_candidates entry."""
    if isinstance(obj, dict):
        return obj.get("candidates", []) or []
    return getattr(obj, "candidates", []) or []


class GeminiOracle(OracleInterface):
    """Google Gemini-based oracle for MSS verification.

    Sends frames as inline JPEG image parts via google-genai SDK. Extracts
    decision-token logprobs when the model returns them; falls back to
    text-based confidence otherwise.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        config: Optional[OracleConfig] = None,
        cache: Optional[OracleCache] = None,
        max_tokens: int = 4096,
        max_concurrent: int = 4,
        requests_per_minute: int = 30,
        prompt_template: Optional[str] = None,
        jpeg_quality: int = 85,
        logprobs_top_k: int = 5,
        **kwargs,
    ):
        """Initialize Gemini oracle.

        Args:
            model_name: Model identifier (e.g., "gemini-3-pro", "gemini-2.5-pro")
            api_key: Google AI API key
            config: Oracle configuration
            cache: Optional cache
            max_tokens: Maximum output tokens
            max_concurrent: Maximum concurrent API requests
            requests_per_minute: Rate limit (requests per 60-second window)
            prompt_template: Custom prompt template (uses default if None)
            jpeg_quality: JPEG encoding quality for frames (0-100)
            logprobs_top_k: Number of top logprobs to request per position
        """
        super().__init__(config or OracleConfig(), cache)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.jpeg_quality = jpeg_quality
        self.logprobs_top_k = logprobs_top_k
        # Late-2025: Google disabled logprobs on most Gemini models (returns
        # 400 "Logprobs is not enabled for this model"). We probe optimistically
        # and flip to False on the first such error, then retry without.
        self._logprobs_supported = True
        self._rate_limiter = RateLimiter(requests_per_minute)

        if prompt_template is None:
            self.prompt_template = get_verification_prompt(self.config.prompt_template_id)
        else:
            self.prompt_template = prompt_template

        try:
            from google import genai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "google-genai SDK is required for GeminiOracle. "
                "Install with: pip install google-genai"
            ) from e

        from google import genai

        self._client = genai.Client(api_key=api_key)

        logger.info(
            f"GeminiOracle initialized: model={model_name}, "
            f"max_tokens={max_tokens}, max_concurrent={max_concurrent}, "
            f"rate_limit={requests_per_minute}/min, logprobs=top-{logprobs_top_k}"
        )

    def query_single(
        self,
        frames: List[np.ndarray],
        action_label: str,
        video_id: str,
        removal_note: Optional[str] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> OracleResponse:
        """Query Gemini with a single video.

        Returns:
            OracleResponse with logprob-derived confidence (or text fallback)
        """
        from google.genai import types

        system_prompt, user_text = self._build_prompts(
            action_label, removal_note, prompt_context
        )

        # Build user content: text + frames as inline JPEG parts
        parts: List[Any] = [types.Part.from_text(text=user_text)]
        for frame in frames:
            jpeg_bytes = _frame_to_jpeg_bytes(frame, quality=self.jpeg_quality)
            parts.append(
                types.Part.from_bytes(mime_type="image/jpeg", data=jpeg_bytes)
            )

        contents = [types.Content(role="user", parts=parts)]

        def _build_cfg(with_logprobs: bool) -> "types.GenerateContentConfig":
            kwargs = dict(
                system_instruction=system_prompt,
                max_output_tokens=self.max_tokens,
                temperature=0.0,
            )
            if with_logprobs:
                kwargs["response_logprobs"] = True
                kwargs["logprobs"] = self.logprobs_top_k
            return types.GenerateContentConfig(**kwargs)

        try:
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=_build_cfg(self._logprobs_supported),
                )
            except Exception as e:
                msg = str(e)
                # Retry without logprobs whenever the API rejects them.
                # Don't gate on _logprobs_supported — under concurrent
                # requests, another worker may have already flipped it
                # to False after firing this request with logprobs=True.
                if "Logprobs is not enabled" in msg:
                    if self._logprobs_supported:
                        logger.warning(
                            f"Gemini logprobs disabled by API ({self.model_name}); "
                            f"falling back to text-only confidence for this oracle."
                        )
                        self._logprobs_supported = False
                    response = self._client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=_build_cfg(False),
                    )
                else:
                    raise

            raw_output = self._extract_text(response)
            logger.debug(f"Gemini raw output for {video_id}: {raw_output[:500]}")

            oracle_response = parse_oracle_response(raw_output)

            # Apply logprobs if present (no-op when unsupported)
            if self._logprobs_supported:
                self._apply_logprobs(oracle_response, response, video_id)

            return oracle_response

        except Exception as e:
            logger.error(f"Gemini oracle query failed for {video_id}: {e}")
            return OracleResponse(
                decision=OracleDecision.SKIP,
                confidence=0.0,
                evidence=f"Query failed: {str(e)[:50]}",
                raw_output=str(e),
            )

    def query_batch(
        self,
        batch: List[Dict[str, Any]],
        batch_size: int = 4,
    ) -> List[OracleResponse]:
        """Concurrent rate-limited batch via ThreadPoolExecutor.

        batch_size is ignored — concurrency is controlled by max_concurrent.
        """
        if not batch:
            return []

        responses: List[Optional[OracleResponse]] = [None] * len(batch)
        uncached: List[tuple] = []

        for i, item in enumerate(batch):
            template_id = self._get_effective_template_id(item.get("prompt_context"))
            cached = self.cache.get(
                item["video_id"], item["mask_pattern_hash"], template_id
            )
            if cached:
                responses[i] = cached[0]
            else:
                uncached.append((i, item))

        if not uncached:
            return responses

        logger.debug(
            f"Gemini batch: {len(batch)} items, "
            f"{len(batch) - len(uncached)} cached, {len(uncached)} to query"
        )

        def _query_one(idx: int, item: Dict[str, Any]) -> tuple:
            self._rate_limiter.acquire()
            resp = self.query_single(
                frames=item["frames"],
                action_label=item["action_label"],
                video_id=item["video_id"],
                removal_note=item.get("removal_note"),
                prompt_context=item.get("prompt_context"),
            )
            return idx, resp

        workers = min(self.max_concurrent, len(uncached))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_query_one, idx, item): idx
                for idx, item in uncached
            }
            for future in as_completed(futures):
                idx, resp = future.result()
                responses[idx] = resp
                item = batch[idx]
                template_id = self._get_effective_template_id(
                    item.get("prompt_context")
                )
                self.cache.put(
                    item["video_id"], item["mask_pattern_hash"],
                    template_id, [resp]
                )
                self._call_count += 1

        return responses

    def _build_prompts(
        self,
        action_label: str,
        removal_note: Optional[str],
        prompt_context: Optional[Dict[str, Any]],
    ) -> tuple:
        """Build system prompt + user text from optional context dict.

        Mirrors the dispatch logic in QwenOracle / DartmouthOracle so the
        same prompt templates work across all backends.
        """
        from .prompts import get_prompt

        if prompt_context:
            template_id = prompt_context.get("prompt_template_id")

            if template_id in ("mss_verification_ssv2", "direct_scoring_ssv2"):
                template = get_prompt(template_id)
                system_prompt = template.format(
                    action_template=prompt_context.get("action_template", action_label),
                    placeholders=prompt_context.get("placeholders", "unknown"),
                    duration=prompt_context.get("duration", "?"),
                    n_segments=prompt_context.get("n_segments", "?"),
                    segment_duration=prompt_context.get("segment_duration", "?"),
                )
                user_text = (
                    f"Verify if this video shows the action pattern: "
                    f"{prompt_context.get('action_template', action_label)}"
                )

            elif template_id in ("mss_verification_k400", "direct_scoring", "direct_scoring_k400", "direct_scoring_diving48"):
                template = get_prompt(template_id)
                system_prompt = template.format(
                    action_label=action_label,
                    duration=prompt_context.get("duration", "?"),
                    n_segments=prompt_context.get("n_segments", "?"),
                    segment_duration=prompt_context.get("segment_duration", "?"),
                )
                user_text = f"Verify if this video shows the action: {action_label}"

            else:
                system_prompt = self.prompt_template.format(action_label=action_label)
                user_text = f"Verify if this video shows the action: {action_label}"
        else:
            system_prompt = self.prompt_template.format(action_label=action_label)
            user_text = f"Verify if this video shows the action: {action_label}"

        if removal_note:
            user_text = f"{removal_note}\n\n{user_text}"

        return system_prompt, user_text

    def _extract_text(self, response: Any) -> str:
        """Extract the text content from a Gemini response object."""
        text = getattr(response, "text", None)
        if text:
            return text
        # Fallback: walk candidates -> content -> parts
        candidates = getattr(response, "candidates", []) or []
        if not candidates:
            return ""
        content = getattr(candidates[0], "content", None)
        if content is None:
            return ""
        parts = getattr(content, "parts", []) or []
        chunks = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                chunks.append(t)
        return "".join(chunks)

    def _apply_logprobs(
        self,
        oracle_response: OracleResponse,
        api_response: Any,
        video_id: str,
    ) -> None:
        """Extract decision-token logprobs from Gemini response and apply them."""
        try:
            candidates = getattr(api_response, "candidates", []) or []
            if not candidates:
                self._apply_text_fallback(oracle_response)
                return

            lp_result = getattr(candidates[0], "logprobs_result", None)
            if lp_result is None:
                logger.debug(f"No logprobs_result in Gemini response for {video_id}")
                self._apply_text_fallback(oracle_response)
                return

            chosen = getattr(lp_result, "chosen_candidates", []) or []
            top = getattr(lp_result, "top_candidates", []) or []

            if not chosen:
                logger.debug(f"Empty chosen_candidates for {video_id}")
                self._apply_text_fallback(oracle_response)
                return

            decision_logprobs = _extract_decision_logprobs_gemini(chosen, top)
            if decision_logprobs is None:
                logger.debug(
                    f"Could not find YES/NO/SKIP in Gemini logprobs for {video_id}"
                )
                self._apply_text_fallback(oracle_response)
                return

            logprob_yes = decision_logprobs.get("YES")
            logprob_no = decision_logprobs.get("NO")
            logprob_skip = decision_logprobs.get("SKIP")

            if logprob_yes is None or logprob_no is None:
                self._apply_text_fallback(oracle_response)
                return

            # softmax is shift-invariant, so logprobs work as logits here
            p_yes, p_skip = logits_to_confidence(
                logprob_yes, logprob_no, logprob_skip
            )

            text_decision = oracle_response.decision
            anomaly = (
                (text_decision == OracleDecision.YES and p_yes < 0.01)
                or (text_decision == OracleDecision.NO and p_yes > 0.99)
            )
            if anomaly:
                logger.warning(
                    f"Logprob anomaly for {video_id}: text says "
                    f"{text_decision.value} but P(YES|¬SKIP)={p_yes:.4f}. "
                    f"Falling back to text-based confidence."
                )
                self._apply_text_fallback(oracle_response)
                return

            oracle_response.logit_yes = logprob_yes
            oracle_response.logit_no = logprob_no
            oracle_response.logit_skip = logprob_skip
            oracle_response.logit_confidence = p_yes
            oracle_response.logit_p_skip = p_skip

            skip_str = f", SKIP={logprob_skip:.2f}" if logprob_skip is not None else ""
            pskip_str = f", P(SKIP)={p_skip:.3f}" if p_skip is not None else ""
            logger.debug(
                f"Gemini logprobs for {video_id}: "
                f"YES={logprob_yes:.2f}, NO={logprob_no:.2f}{skip_str}, "
                f"P(YES|¬SKIP)={p_yes:.3f}{pskip_str}"
            )

        except Exception as e:
            logger.warning(f"Failed to extract Gemini logprobs for {video_id}: {e}")
            self._apply_text_fallback(oracle_response)

    def _apply_text_fallback(self, oracle_response: OracleResponse) -> None:
        """Set logit_confidence from the text decision when logprobs unavailable."""
        if oracle_response.decision == OracleDecision.YES:
            oracle_response.logit_confidence = oracle_response.confidence
        elif oracle_response.decision == OracleDecision.NO:
            oracle_response.logit_confidence = 0.0
        else:
            oracle_response.logit_confidence = 0.0
            oracle_response.logit_p_skip = 1.0


# Known Gemini model identifiers. Add new releases here.
# These are surfaced via mss_extract.py's MODEL_CONFIGS merge.
GEMINI_MODEL_CONFIGS = {
    "gemini-3.1-pro": {
        "api_name": "gemini-3.1-pro-preview",
        "provider": "gemini",
    },
    "gemini-3-pro": {
        "api_name": "gemini-3-pro-preview",
        "provider": "gemini",
    },
    "gemini-2.5-pro": {
        "api_name": "gemini-2.5-pro",
        "provider": "gemini",
    },
    "gemini-2.5-flash": {
        "api_name": "gemini-2.5-flash",
        "provider": "gemini",
    },
}
