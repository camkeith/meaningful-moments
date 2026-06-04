"""
Dartmouth ChatAPI oracle implementation for MSS verification.

Implements the OracleInterface using ChatDartmouth (langchain_dartmouth),
which provides an OpenAI-compatible API with logprobs support.
No local GPU or model weights required.
"""

import base64
import io
import logging
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


def _frame_to_base64_jpeg(frame: np.ndarray, quality: int = 85) -> str:
    """Convert a numpy RGB frame to a base64-encoded JPEG data URL.

    Args:
        frame: RGB numpy array (H, W, 3)
        quality: JPEG compression quality (0-100)

    Returns:
        Data URL string: "data:image/jpeg;base64,..."
    """
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _extract_decision_logprobs(
    logprobs_content: List[dict],
) -> Optional[Dict[str, float]]:
    """Extract YES/NO/SKIP logprobs from OpenAI-compatible logprobs response.

    Searches through token logprobs to find the decision token position,
    then extracts logprobs for YES, NO, and SKIP from top_logprobs.

    Handles Qwen3 thinking tokens: skips past any <think>...</think> block
    in logprobs before scanning for the decision token.

    IMPORTANT: top_logprobs may contain multiple token variants that normalize
    to the same decision (e.g., 'YES', ' YES', 'yes'). We keep only the
    highest logprob variant for each decision token, since alternatives like
    'yes' (lowercase) have much lower logprob and would corrupt the softmax.

    Args:
        logprobs_content: List of token logprob entries from response metadata.
            Each entry has: {"token": str, "logprob": float, "top_logprobs": [...]}

    Returns:
        Dict with keys "YES", "NO", "SKIP" mapped to logprob values,
        or None if decision tokens not found.
    """
    decision_tokens = {"YES", "NO", "SKIP"}

    # Skip past thinking tokens: find </think> and start scanning after it.
    start_idx = 0
    for i, entry in enumerate(logprobs_content):
        tok = entry.get("token", "")
        if "</think>" in tok or tok.strip() == "</think>":
            start_idx = i + 1
            logger.debug(
                f"Found </think> at logprobs position {i}, "
                f"scanning from position {start_idx}"
            )
            break

    if start_idx > 0 and start_idx >= len(logprobs_content):
        logger.warning("</think> found at end of logprobs, no tokens remain")
        return None

    for entry in logprobs_content[start_idx:]:
        token = entry.get("token", "").strip().upper()
        if token not in decision_tokens:
            continue

        # Found the decision token position — extract all three from top_logprobs
        top_logprobs = entry.get("top_logprobs", [])

        # Build a lookup from top_logprobs, keeping only the HIGHEST logprob
        # variant for each normalized decision token. top_logprobs is sorted
        # descending, so 'YES' at -0.0 comes before 'yes' at -18.0.
        logprob_map = {}
        for tlp in top_logprobs:
            t = tlp.get("token", "").strip().upper()
            if t in decision_tokens:
                if t not in logprob_map or tlp["logprob"] > logprob_map[t]:
                    logprob_map[t] = tlp["logprob"]

        # The generated token itself is always available
        if token not in logprob_map:
            logprob_map[token] = entry["logprob"]

        # We need at least YES and NO
        if "YES" in logprob_map and "NO" in logprob_map:
            return logprob_map

        # If only one is present, we can still return what we have
        if logprob_map:
            logger.debug(
                f"Partial logprobs at decision token '{token}': {logprob_map}"
            )
            return logprob_map

    return None


class DartmouthOracle(OracleInterface):
    """Dartmouth ChatAPI-based oracle for MSS verification.

    Uses ChatDartmouth (langchain_dartmouth) to query Qwen VL models hosted
    on Dartmouth's chat.dartmouth.edu API. Supports logprobs extraction for
    P(YES) computation via 3-way softmax.
    """

    def __init__(
        self,
        model_name: str,
        chat_api_key: str,
        config: Optional[OracleConfig] = None,
        cache: Optional[OracleCache] = None,
        max_tokens: int = 4096,
        max_concurrent: int = 4,
        requests_per_minute: int = 15,
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """Initialize Dartmouth oracle.

        Args:
            model_name: Model identifier on Dartmouth API
                (e.g., "qwen.qwen3-vl-32b-instruct-fp8")
            chat_api_key: Dartmouth Chat API key (DARTMOUTH_CHAT_API_KEY)
            config: Oracle configuration
            cache: Optional cache
            max_tokens: Maximum tokens to generate
            max_concurrent: Maximum concurrent API requests
            requests_per_minute: Rate limit (requests per 60-second window)
            prompt_template: Custom prompt template (uses default if None)
        """
        super().__init__(config or OracleConfig(), cache)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self._rate_limiter = RateLimiter(requests_per_minute)

        if prompt_template is None:
            self.prompt_template = get_verification_prompt(self.config.prompt_template_id)
        else:
            self.prompt_template = prompt_template

        # Initialize ChatDartmouth with logprobs enabled. Some upstream
        # models (e.g. vertex_ai.gemini-3.x) reject logprobs at request
        # time — we fall back to a no-logprobs client on the first such
        # error and reuse it for the rest of the run.
        from langchain_dartmouth.llms import ChatDartmouth
        self._ChatDartmouth = ChatDartmouth
        self._chat_api_key = chat_api_key
        self._logprobs_supported = True

        self.llm = ChatDartmouth(
            model_name=model_name,
            dartmouth_chat_api_key=chat_api_key,
            max_tokens=max_tokens,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
        )

        logger.info(
            f"DartmouthOracle initialized: model={model_name}, "
            f"max_tokens={max_tokens}, max_concurrent={max_concurrent}, "
            f"rate_limit={requests_per_minute}/min, logprobs=True"
        )

    def query_single(
        self,
        frames: List[np.ndarray],
        action_label: str,
        video_id: str,
        removal_note: Optional[str] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> OracleResponse:
        """Query oracle with a single video via Dartmouth API.

        Args:
            frames: List of video frames (RGB numpy arrays)
            action_label: Action label to verify
            video_id: Video identifier
            removal_note: Optional note about removed segments (for CUT operator)
            prompt_context: Optional dict with additional prompt variables

        Returns:
            Oracle response with logprobs-based confidence
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build system prompt (same logic as QwenOracle)
        system_prompt, user_text = self._build_prompts(
            action_label, removal_note, prompt_context
        )

        # Encode frames as base64 JPEG images
        image_content = []
        for frame in frames:
            data_url = _frame_to_base64_jpeg(frame)
            image_content.append(
                {"type": "image_url", "image_url": {"url": data_url}}
            )

        # Build LangChain messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_text},
                    *image_content,
                ]
            ),
        ]

        try:
            # Call the API. Auto-fallback to a no-logprobs client if the
            # upstream model rejects logprobs (Gemini 3.x via Vertex AI).
            try:
                response = self.llm.invoke(messages)
            except Exception as e:
                msg = str(e)
                if "Logprobs is not supported" in msg or "Logprobs is not enabled" in msg:
                    if self._logprobs_supported:
                        logger.warning(
                            f"Dartmouth/{self.model_name}: logprobs not supported "
                            f"by upstream — switching to text-only confidence."
                        )
                        self._logprobs_supported = False
                        self.llm = self._ChatDartmouth(
                            model_name=self.model_name,
                            dartmouth_chat_api_key=self._chat_api_key,
                            max_tokens=self.max_tokens,
                            temperature=0.0,
                        )
                    response = self.llm.invoke(messages)
                else:
                    raise
            raw_output = response.content

            logger.debug(f"Dartmouth raw output for {video_id}: {raw_output[:500]}")

            # Parse JSON response
            oracle_response = parse_oracle_response(raw_output)

            # Extract logprobs for YES/NO/SKIP (no-op if unsupported)
            if self._logprobs_supported:
                self._apply_logprobs(oracle_response, response, video_id)

            return oracle_response

        except Exception as e:
            logger.error(f"Dartmouth oracle query failed for {video_id}: {e}")
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
        """Query oracle with concurrent API requests and rate limiting.

        Fires up to max_concurrent requests in parallel using threads,
        while respecting the requests_per_minute rate limit.

        Args:
            batch: List of query dicts (same schema as base class)
            batch_size: Ignored (concurrency controlled by max_concurrent)

        Returns:
            List of oracle responses in the same order as batch
        """
        if not batch:
            return []

        # Separate cached from uncached
        responses: List[Optional[OracleResponse]] = [None] * len(batch)
        uncached: List[tuple] = []  # (original_index, item)

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
            f"Dartmouth batch: {len(batch)} items, "
            f"{len(batch) - len(uncached)} cached, {len(uncached)} to query"
        )

        def _query_one(idx: int, item: Dict[str, Any]) -> tuple:
            """Rate-limited single query executed in a thread."""
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
                # Cache result
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
        """Build system prompt and user text from context.

        Returns:
            Tuple of (system_prompt, user_text)
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

    def _apply_logprobs(
        self,
        oracle_response: OracleResponse,
        api_response,
        video_id: str,
    ) -> None:
        """Extract logprobs from API response and apply to OracleResponse.

        Args:
            oracle_response: OracleResponse to update in-place
            api_response: Raw LangChain response with response_metadata
            video_id: Video identifier for logging
        """
        try:
            metadata = getattr(api_response, "response_metadata", {})
            logprobs_data = metadata.get("logprobs", {})
            content_logprobs = logprobs_data.get("content", [])

            if not content_logprobs:
                logger.debug(f"No logprobs in response for {video_id}")
                self._apply_text_fallback(oracle_response)
                return

            # Log first few tokens to help diagnose issues
            first_tokens = [
                entry.get("token", "?") for entry in content_logprobs[:8]
            ]
            logger.debug(
                f"Logprobs for {video_id}: {len(content_logprobs)} tokens, "
                f"first 8: {first_tokens}"
            )

            decision_logprobs = _extract_decision_logprobs(content_logprobs)

            if decision_logprobs is None:
                logger.debug(
                    f"Could not find YES/NO/SKIP in logprobs for {video_id}, "
                    f"using text-based fallback"
                )
                self._apply_text_fallback(oracle_response)
                return

            logprob_yes = decision_logprobs.get("YES")
            logprob_no = decision_logprobs.get("NO")
            logprob_skip = decision_logprobs.get("SKIP")

            if logprob_yes is not None and logprob_no is not None:
                # logprobs are log-probabilities (base e). Since softmax is
                # shift-invariant, softmax(logprobs) == softmax(logits).
                # Two-stage: binary P(YES|not SKIP) + separate P(SKIP)
                p_yes, p_skip = logits_to_confidence(
                    logprob_yes, logprob_no, logprob_skip
                )

                # Sanity check: if the logprob-based confidence contradicts
                # the text decision, the logprobs may still be corrupted.
                # Fall back to text-based confidence as a safety net.
                text_decision = oracle_response.decision
                anomaly = (
                    (text_decision == OracleDecision.YES and p_yes < 0.01)
                    or (text_decision == OracleDecision.NO and p_yes > 0.99)
                )
                if anomaly:
                    logger.warning(
                        f"Logprob anomaly for {video_id}: text says "
                        f"{text_decision.value} but P(YES|¬SKIP)={p_yes:.4f} "
                        f"(YES={logprob_yes:.2f}, NO={logprob_no:.2f}). "
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
                    f"Logprobs for {video_id}: "
                    f"YES={logprob_yes:.2f}, NO={logprob_no:.2f}{skip_str}, "
                    f"P(YES|¬SKIP)={p_yes:.3f}{pskip_str}"
                )
            else:
                logger.debug(
                    f"Incomplete logprobs for {video_id}: {decision_logprobs}, "
                    f"using text-based fallback"
                )
                self._apply_text_fallback(oracle_response)

        except Exception as e:
            logger.warning(f"Failed to extract logprobs for {video_id}: {e}")
            self._apply_text_fallback(oracle_response)

    def _apply_text_fallback(self, oracle_response: OracleResponse) -> None:
        """Apply text-based confidence fallback when logprobs unavailable."""
        if oracle_response.decision == OracleDecision.YES:
            oracle_response.logit_confidence = oracle_response.confidence
        elif oracle_response.decision == OracleDecision.NO:
            oracle_response.logit_confidence = 0.0
        else:  # SKIP
            oracle_response.logit_confidence = 0.0
            oracle_response.logit_p_skip = 1.0  # Flag as corrupted
