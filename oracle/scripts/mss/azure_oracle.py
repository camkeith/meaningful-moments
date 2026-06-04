"""
Azure OpenAI oracle implementation for MSS verification.

Implements OracleInterface using Azure OpenAI Service / AI Foundry deployments
of GPT-5.x family (and GPT-4o family). Sends sampled video frames as base64
image_url content blocks via the standard OpenAI chat completions API.

No local GPU required. Concurrency + rate limiting follow the same pattern as
GeminiOracle / DartmouthOracle (thread pool + sliding-window limiter).

Notes:
- GPT-5+ uses ``max_completion_tokens`` (not ``max_tokens``).
- GPT-5+ rejects ``temperature`` other than the default — we omit it.
- Logprobs requires ``logprobs=True, top_logprobs=N``; not all deployments
  expose decision-token logprobs reliably, so we auto-fall back to text-only
  confidence on the first ``Logprobs`` rejection (matches Gemini behavior).
"""

import base64
import io
import logging
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
    parse_oracle_response,
)

logger = logging.getLogger(__name__)


def _frame_to_data_url(frame: np.ndarray, quality: int = 85) -> str:
    """Encode an RGB frame as a base64 JPEG data URL."""
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# Known Azure OpenAI deployment families. The CLI maps a friendly key
# (e.g. ``gpt-5.5``) to a default deployment name; users can override via
# ``--azure-deployment`` or env var.
AZURE_MODEL_CONFIGS = {
    "gpt-5.5": {"api_name": "gpt-5.5",  "provider": "azure"},
    "gpt-5.4": {"api_name": "gpt-5.4",  "provider": "azure"},
    "gpt-5":   {"api_name": "gpt-5",    "provider": "azure"},
    "gpt-5-mini": {"api_name": "gpt-5-mini", "provider": "azure"},
    "gpt-4o":  {"api_name": "gpt-4o",   "provider": "azure"},
}


class AzureOpenAIOracle(OracleInterface):
    """Azure OpenAI Service / AI Foundry-based oracle for MSS verification."""

    def __init__(
        self,
        deployment_name: str,
        endpoint: str,
        api_key: str,
        api_version: str = "2024-12-01-preview",
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
        super().__init__(config or OracleConfig(), cache)
        self.deployment_name = deployment_name
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.jpeg_quality = jpeg_quality
        self.logprobs_top_k = logprobs_top_k
        # Auto-fallback flag: flipped to False on first "Logprobs" rejection.
        self._logprobs_supported = True
        self._rate_limiter = RateLimiter(requests_per_minute)

        if prompt_template is None:
            self.prompt_template = get_verification_prompt(self.config.prompt_template_id)
        else:
            self.prompt_template = prompt_template

        try:
            from openai import AzureOpenAI  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "openai SDK is required for AzureOpenAIOracle. "
                "Install with: pip install openai"
            ) from e

        from openai import AzureOpenAI

        # max_retries gives the SDK exponential-backoff retry on 429 / 5xx
        # (default is 2; we set higher because Azure's burst limit is finicky
        # under our concurrent batches even when nominally under RPM).
        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            max_retries=8,
            timeout=120.0,
        )

        logger.info(
            f"AzureOpenAIOracle initialized: deployment={deployment_name}, "
            f"endpoint={endpoint}, api_version={api_version}, "
            f"max_completion_tokens={max_tokens}, max_concurrent={max_concurrent}, "
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
        """Query oracle with a single video via Azure OpenAI chat completions."""
        system_prompt, user_text = self._build_prompts(
            action_label, removal_note, prompt_context
        )

        # Build user-message content: text first, then base64 JPEG frames.
        image_content = []
        for frame in frames:
            image_content.append(
                {"type": "image_url",
                 "image_url": {"url": _frame_to_data_url(frame, self.jpeg_quality)}}
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                *image_content,
            ]},
        ]

        def _call(with_logprobs: bool):
            kwargs = dict(
                model=self.deployment_name,
                messages=messages,
                max_completion_tokens=self.max_tokens,
            )
            if with_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = self.logprobs_top_k
            return self._client.chat.completions.create(**kwargs)

        try:
            try:
                response = _call(self._logprobs_supported)
            except Exception as e:
                msg = str(e)
                if (
                    "logprobs" in msg.lower()
                    and ("not supported" in msg.lower()
                         or "not enabled" in msg.lower()
                         or "is not allowed" in msg.lower())
                ):
                    if self._logprobs_supported:
                        logger.warning(
                            f"Azure deployment {self.deployment_name}: logprobs "
                            f"not supported — falling back to text-only confidence."
                        )
                        self._logprobs_supported = False
                    response = _call(False)
                else:
                    raise

            choice = response.choices[0]
            raw_output = choice.message.content or ""
            logger.debug(f"Azure raw output for {video_id}: {raw_output[:500]}")

            oracle_response = parse_oracle_response(raw_output)

            # Logprobs path: extract YES/NO/SKIP token probabilities. Most
            # GPT-5 deployments don't expose decision-token logprobs in a way
            # we can rely on for video direct-scoring, so the text-confidence
            # path is the primary signal regardless. We try once and shrug.
            if self._logprobs_supported:
                self._maybe_apply_logprobs(oracle_response, choice, video_id)

            return oracle_response

        except Exception as e:
            logger.error(f"Azure oracle query failed for {video_id}: {e}")
            return OracleResponse(
                decision=OracleDecision.SKIP,
                confidence=0.0,
                evidence=f"Query failed: {str(e)[:50]}",
                raw_output=str(e),
            )

    def _maybe_apply_logprobs(
        self,
        oracle_response: OracleResponse,
        choice: Any,
        video_id: str,
    ) -> None:
        """Best-effort logprob extraction for YES/NO/SKIP tokens."""
        try:
            lp = getattr(choice, "logprobs", None)
            if not lp:
                return
            content = getattr(lp, "content", None) or []
            decision_tokens = ("YES", "NO", "SKIP", "Yes", "No", "yes", "no")
            for entry in content:
                tok = getattr(entry, "token", "")
                if tok and any(d in tok for d in decision_tokens):
                    top = getattr(entry, "top_logprobs", None) or []
                    yes_lp = no_lp = None
                    for t in top:
                        ttok = getattr(t, "token", "")
                        tlp = getattr(t, "logprob", None)
                        if tlp is None:
                            continue
                        if "YES" in ttok.upper() and yes_lp is None:
                            yes_lp = tlp
                        elif "NO" in ttok.upper() and no_lp is None:
                            no_lp = tlp
                    if yes_lp is not None and no_lp is not None:
                        import math
                        denom = math.exp(yes_lp) + math.exp(no_lp)
                        if denom > 0:
                            oracle_response.logit_confidence = math.exp(yes_lp) / denom
                            oracle_response.logit_p_skip = 0.0
                    return
        except Exception as e:
            logger.debug(f"logprobs extraction skipped for {video_id}: {e}")

    def query_batch(
        self,
        batch: List[Dict[str, Any]],
        batch_size: int = 4,
    ) -> List[OracleResponse]:
        """Concurrent rate-limited batch via ThreadPoolExecutor."""
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
            f"Azure batch: {len(batch)} items, "
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
                template_id = self._get_effective_template_id(item.get("prompt_context"))
                self.cache.put(
                    item["video_id"], item["mask_pattern_hash"],
                    template_id, [resp]
                )
                self._call_count += 1

        return responses

    def _get_effective_template_id(self, prompt_context: Optional[Dict[str, Any]]) -> str:
        if prompt_context and prompt_context.get("prompt_template_id"):
            return prompt_context["prompt_template_id"]
        return self.config.prompt_template_id

    def _build_prompts(
        self,
        action_label: str,
        removal_note: Optional[str],
        prompt_context: Optional[Dict[str, Any]],
    ) -> tuple:
        """Build system prompt + user text. Mirrors GeminiOracle / DartmouthOracle."""
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

            elif template_id in ("mss_verification_k400", "direct_scoring",
                                 "direct_scoring_k400", "direct_scoring_diving48"):
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
