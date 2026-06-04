"""
Qwen VL oracle implementation for MSS verification.

Implements the OracleInterface using Qwen VL models.
Supports parallel batch processing across multiple GPUs.
"""

import logging
import os
import re
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

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


def _dump_parse_failure(video_id: str, raw_output: str, exc: BaseException) -> None:
    """Write the full failing raw_output to a sidecar file for later inspection.

    Looks at MSS_PARSE_FAILURES_DIR; if unset, silently does nothing.
    Env-var-based so the path survives multiprocessing forks/spawns into
    the parallel worker pool without needing extra plumbing.
    """
    dump_dir = os.environ.get("MSS_PARSE_FAILURES_DIR")
    if not dump_dir:
        return
    try:
        from datetime import datetime

        out_dir = Path(dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        # Sanitize video_id for filesystem
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", video_id)[:120]
        out_path = out_dir / f"{safe_id}_{ts}.txt"
        header = (
            f"# video_id: {video_id}\n"
            f"# timestamp: {ts}\n"
            f"# exception: {type(exc).__name__}: {exc}\n"
            f"# raw_output_len: {len(raw_output)}\n"
            f"# ---\n"
        )
        out_path.write_text(header + raw_output, encoding="utf-8")
    except Exception as dump_exc:  # never let dump failures propagate
        logger.debug(f"Failed to dump parse failure for {video_id}: {dump_exc}")


# Model configurations
# Note: qwen3-vl-32b is a dense model (NOT MoE), qwen3-vl-30b (A3B) is MoE
MODEL_CONFIGS = {
    "qwen3-vl-32b": {
        "hf_name": "Qwen/Qwen3-VL-32B-Instruct",
        "class_name": "Qwen3VLForConditionalGeneration",  # Dense model, not MoE
        "module": "transformers",
    },
    "qwen3-vl-30b": {
        "hf_name": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "class_name": "Qwen3VLMoeForConditionalGeneration",  # MoE model (A3B = Active 3B)
        "module": "transformers",
    },
    "qwen2.5-vl-32b": {
        "hf_name": "Qwen/Qwen2.5-VL-32B-Instruct",
        "class_name": "Qwen2_5_VLForConditionalGeneration",
        "module": "transformers",
    },
    "qwen2.5-vl-7b": {
        "hf_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "class_name": "Qwen2_5_VLForConditionalGeneration",
        "module": "transformers",
    },
    # Dartmouth API-hosted models (no local GPU required)
    "dartmouth:qwen3-vl-32b": {
        "hf_name": "qwen.qwen3-vl-32b-instruct-fp8",
        "provider": "dartmouth",
    },
}

# Mapping from local model keys to Dartmouth API model names.
# Used when --provider dartmouth is combined with a local model key like "qwen3-vl-32b".
DARTMOUTH_MODEL_MAP = {
    "qwen3-vl-32b": "qwen.qwen3-vl-32b-instruct-fp8",
    "gemini-3.1-pro": "vertex_ai.gemini-3.1-pro-preview",
    "gemini-3-flash":  "vertex_ai.gemini-3-flash-preview",
    "gemini-2.5-pro":   "vertex_ai.gemini-2.5-pro",
    "gemini-2.5-flash": "vertex_ai.gemini-2.5-flash",
}


def _bytes_to_gib(n: int) -> str:
    """Convert bytes to GiB string for max_memory."""
    return f"{max(int(n / 1024**3) - 0.1, 1)}GiB"


def load_qwen_model(
    model_key: str,
    num_gpus: int = 4,
    cache_dir: str = "./models/pretrained_oracle_models/hf_cache",
) -> Tuple:
    """Load Qwen VL model and processor.

    Args:
        model_key: Model key from MODEL_CONFIGS
        num_gpus: Number of GPUs to use
        cache_dir: HuggingFace cache directory

    Returns:
        Tuple of (model, processor)
    """
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model '{model_key}'. Available: {available}")

    config = MODEL_CONFIGS[model_key]
    hf_name = config["hf_name"]
    class_name = config["class_name"]

    import transformers

    model_class = getattr(transformers, class_name)

    # Device setup
    gpu_count = torch.cuda.device_count()
    gpus_to_use = list(range(min(num_gpus, gpu_count)))
    visible = len(gpus_to_use)

    logger.info(f"Loading model: {hf_name}")
    logger.info(f"CUDA devices visible: {gpu_count}, requesting: {num_gpus}")
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    if visible == 0:
        raise RuntimeError("No CUDA devices available!")

    # Log GPU info
    for i in gpus_to_use:
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / 1024**3
        logger.info(f"  GPU {i}: {props.name}, {total_mem:.1f} GiB")

    max_memory = {
        i: _bytes_to_gib(torch.cuda.get_device_properties(i).total_memory)
        for i in gpus_to_use
    }
    device_map = "balanced"

    logger.info(f"Using GPUs: {gpus_to_use}, device_map={device_map}")
    logger.info(f"max_memory: {max_memory}")

    # For Qwen3 MoE models, use the correct config class (fix transformers 5.0.0 bug)
    model_config = None
    if "qwen3" in model_key and "Moe" in class_name:
        from transformers import Qwen3VLMoeConfig
        logger.info("Using Qwen3VLMoeConfig for MoE model")
        model_config = Qwen3VLMoeConfig.from_pretrained(hf_name, cache_dir=cache_dir)
        # Fix missing pad_token_id in text_config
        if hasattr(model_config, "text_config"):
            tc = model_config.text_config
            if not hasattr(tc, "pad_token_id") or tc.pad_token_id is None:
                tc.pad_token_id = getattr(tc, "eos_token_id", 151643)
                logger.info(f"Patched text_config.pad_token_id = {tc.pad_token_id}")

    model = model_class.from_pretrained(
        hf_name,
        config=model_config,
        dtype=torch.bfloat16,
        attn_implementation=os.environ.get("ATTN_IMPL", "flash_attention_2"),
        device_map=device_map,
        max_memory=max_memory,
        cache_dir=cache_dir,
    )

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(hf_name, trust_remote_code=True, cache_dir=cache_dir)

    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        # Clear defaults that trigger warnings when do_sample=False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    return model, processor


class QwenOracle(OracleInterface):
    """Qwen VL-based oracle for MSS verification.

    Uses Qwen VL models to verify whether video contains evidence for action.
    """

    def __init__(
        self,
        model,
        processor,
        config: Optional[OracleConfig] = None,
        cache: Optional[OracleCache] = None,
        max_new_tokens: int = 4096,
        prompt_template: Optional[str] = None,
    ):
        """Initialize Qwen oracle.

        Args:
            model: Loaded Qwen VL model
            processor: Qwen VL processor
            config: Oracle configuration
            cache: Optional cache
            max_new_tokens: Maximum tokens to generate
            prompt_template: Custom prompt template (uses default if None)
        """
        super().__init__(config or OracleConfig(), cache)
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        # Per-batch peak GPU memory snapshot (populated by query_batch).
        self.last_peak_mem_gb: Dict[int, float] = {}
        # Load prompt template from file or use provided custom template
        if prompt_template is None:
            self.prompt_template = get_verification_prompt(self.config.prompt_template_id)
        else:
            self.prompt_template = prompt_template

    def _frames_to_video_content(
        self, frames: List[np.ndarray], fps: float = 30.0
    ) -> List[Image.Image]:
        """Convert frames to Qwen video content format.

        Args:
            frames: List of RGB numpy arrays
            fps: Assumed FPS for the frames

        Returns:
            List of PIL Images for Qwen processor
        """
        # qwen_vl_utils expects PIL Images, not raw numpy arrays
        return [Image.fromarray(frame) for frame in frames]

    def query_single(
        self,
        frames: List[np.ndarray],
        action_label: str,
        video_id: str,
        removal_note: Optional[str] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
        fps: Optional[float] = None,
    ) -> OracleResponse:
        """Query oracle with a single video.

        Args:
            frames: List of video frames (RGB numpy arrays)
            action_label: Action label to verify
            video_id: Video identifier
            removal_note: Optional note about removed segments (for CUT operator)
            prompt_context: Optional dict with additional prompt variables
                           (e.g., action_template, placeholders for SSv2)
            fps: Video FPS for processor metadata

        Returns:
            Oracle response
        """
        # Determine which prompt to use and format it
        from .prompts import get_prompt

        if prompt_context:
            template_id = prompt_context.get("prompt_template_id")

            if template_id in ("mss_verification_ssv2", "direct_scoring_ssv2"):
                # SSv2: use action template + placeholders
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
                # K400 / generic: use action label
                template = get_prompt(template_id)
                system_prompt = template.format(
                    action_label=action_label,
                    duration=prompt_context.get("duration", "?"),
                    n_segments=prompt_context.get("n_segments", "?"),
                    segment_duration=prompt_context.get("segment_duration", "?"),
                )
                user_text = f"Verify if this video shows the action: {action_label}"

            else:
                # Unknown template_id or generic: use default
                system_prompt = self.prompt_template.format(action_label=action_label)
                user_text = f"Verify if this video shows the action: {action_label}"
        else:
            # No context: use default template
            system_prompt = self.prompt_template.format(action_label=action_label)
            user_text = f"Verify if this video shows the action: {action_label}"

        if removal_note:
            user_text = f"{removal_note}\n\n{user_text}"

        # Convert frames to PIL Images for qwen_vl_utils
        pil_frames = self._frames_to_video_content(frames)

        # Get actual video FPS from prompt_context if available
        video_fps = float(prompt_context["video_fps"]) if prompt_context and "video_fps" in prompt_context else 2.0

        video_msg: Dict[str, Any] = {"type": "video", "video": pil_frames, "fps": video_fps}
        if prompt_context and prompt_context.get("max_pixels") is not None:
            video_msg["max_pixels"] = int(prompt_context["max_pixels"])

        # Build message for Qwen
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    video_msg,
                ],
            },
        ]

        try:
            # Process with Qwen
            from qwen_vl_utils import process_vision_info
            from transformers.video_utils import VideoMetadata

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                video_metadata=[VideoMetadata(
                    total_num_frames=len(pil_frames),
                    fps=video_fps,
                    frames_indices=list(range(len(pil_frames))),
                )],
            )

            # Move to device
            first_device = next(self.model.parameters()).device
            if hasattr(inputs, "to"):
                inputs = inputs.to(first_device)
            else:
                inputs = {
                    k: v.to(first_device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

            # Generate with output_scores to get logits
            with torch.no_grad():
                gen_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Extract generated IDs
            gen_ids = gen_outputs.sequences

            # Decode
            trimmed = gen_ids[:, inputs["input_ids"].shape[-1]:]
            raw_output = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Debug: log raw output
            logger.debug(f"Oracle raw output for {video_id}: {raw_output[:500]}")

            # Parse response
            response = parse_oracle_response(raw_output)

            # Extract logits for YES/NO at the position where the decision token is generated
            # For JSON output like {"decision": "YES", ...}, we need to find where YES/NO appears
            if gen_outputs.scores and len(gen_outputs.scores) > 0:
                try:
                    tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

                    # Get token IDs for YES and NO
                    yes_ids = tok.encode("YES", add_special_tokens=False)
                    no_ids = tok.encode("NO", add_special_tokens=False)
                    # Also check for SKIP
                    skip_ids = tok.encode("SKIP", add_special_tokens=False)

                    if yes_ids and no_ids:
                        yes_token_id = yes_ids[0]
                        no_token_id = no_ids[0]
                        skip_token_id = skip_ids[0] if skip_ids else None

                        # Find position where YES, NO, or SKIP was generated
                        decision_pos = None
                        for pos, token_id in enumerate(trimmed[0]):
                            tid = token_id.item()
                            if tid == yes_token_id or tid == no_token_id:
                                decision_pos = pos
                                break
                            if skip_token_id and tid == skip_token_id:
                                decision_pos = pos
                                break

                        if decision_pos is not None and decision_pos < len(gen_outputs.scores):
                            # Extract logits at the decision position
                            decision_logits = gen_outputs.scores[decision_pos][0]
                            logit_yes = decision_logits[yes_token_id].item()
                            logit_no = decision_logits[no_token_id].item()
                            logit_skip = None
                            if skip_token_id is not None:
                                logit_skip = decision_logits[skip_token_id].item()

                            # Two-stage: binary P(YES|not SKIP) + separate P(SKIP)
                            p_yes, p_skip = logits_to_confidence(logit_yes, logit_no, logit_skip)

                            response.logit_yes = logit_yes
                            response.logit_no = logit_no
                            response.logit_skip = logit_skip
                            response.logit_confidence = p_yes
                            response.logit_p_skip = p_skip

                            # Debug: show tokens around decision position
                            actual_token_id = trimmed[0][decision_pos].item()
                            actual_token = tok.decode([actual_token_id])
                            first_tokens = [tok.decode([t.item()]) for t in trimmed[0][:min(10, len(trimmed[0]))]]
                            skip_str = f", SKIP={logit_skip:.2f}" if logit_skip is not None else ""
                            pskip_str = f", P(SKIP)={p_skip:.3f}" if p_skip is not None else ""
                            logger.debug(
                                f"Logits for {video_id} (pos={decision_pos}, token='{actual_token}'): "
                                f"YES={logit_yes:.2f}, NO={logit_no:.2f}{skip_str}, "
                                f"P(YES|¬SKIP)={p_yes:.3f}{pskip_str}"
                            )
                            logger.debug(f"First 10 tokens: {first_tokens}")
                        else:
                            # Fallback: couldn't find decision position, use text-based
                            logger.debug(
                                f"Could not find decision token position for {video_id}, "
                                f"using text-based decision: {response.decision.value}"
                            )
                            # Use text confidence if YES, else 0
                            if response.decision == OracleDecision.YES:
                                response.logit_confidence = response.confidence
                            elif response.decision == OracleDecision.NO:
                                response.logit_confidence = 0.0
                            else:  # SKIP
                                response.logit_confidence = 0.0
                                response.logit_p_skip = 1.0  # Flag as corrupted

                except Exception as e:
                    logger.warning(f"Failed to extract logits for {video_id}: {e}")

            return response

        except Exception as e:
            logger.error(f"Oracle query failed for {video_id}: {e}")
            # If raw_output exists in scope, dump it for offline diagnosis.
            try:
                _dump_parse_failure(video_id, raw_output, e)
            except NameError:
                pass
            # Return SKIP on error
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
        """Query oracle with a batch of videos using TRUE batched inference.

        Processes multiple videos in a single forward pass through the sharded model.
        This is more efficient than sequential processing.

        Args:
            batch: List of dicts with keys:
                - frames: List of video frames
                - action_label: Action label to verify
                - video_id: Video identifier
                - mask_pattern_hash: Hash for caching
                - removal_note: Optional removal note
                - prompt_context: Optional prompt context
            batch_size: Number of videos to process in each forward pass

        Returns:
            List of oracle responses (same order as batch)
        """
        from qwen_vl_utils import process_vision_info

        # Separate cached from uncached
        responses = [None] * len(batch)
        uncached_indices = []
        uncached_items = []

        for i, item in enumerate(batch):
            template_id = self._get_effective_template_id(item.get("prompt_context"))
            cached = self.cache.get(
                item["video_id"],
                item["mask_pattern_hash"],
                template_id
            )
            if cached:
                responses[i] = cached[0]
            else:
                uncached_indices.append(i)
                uncached_items.append(item)

        if not uncached_items:
            return responses

        # Reset per-batch peak-memory snapshot before any forward passes.
        self.last_peak_mem_gb = {}
        if torch.cuda.is_available():
            for _g in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(_g)
                except RuntimeError:
                    pass

        # Process uncached items in batches
        for batch_start in range(0, len(uncached_items), batch_size):
            batch_end = min(batch_start + batch_size, len(uncached_items))
            current_batch = uncached_items[batch_start:batch_end]
            current_indices = uncached_indices[batch_start:batch_end]

            try:
                batch_responses = self._process_batch(current_batch)

                for idx, resp in zip(current_indices, batch_responses):
                    responses[idx] = resp
                    # Cache result
                    item = batch[idx]
                    template_id = self._get_effective_template_id(item.get("prompt_context"))
                    self.cache.put(
                        item["video_id"],
                        item["mask_pattern_hash"],
                        template_id,
                        [resp]
                    )
                    self._call_count += 1

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()

                # Halving fallback: keep halving the failing chunk until it fits
                # or reaches size 1. Items that still OOM at the current size are
                # carried into the next (smaller) round. The next outer batch
                # resets to the configured batch_size.
                pending = list(zip(current_indices, current_batch))
                prev_size = len(current_batch)
                fb_size = max(1, prev_size // 2)

                while pending:
                    logger.warning(
                        f"OOM at batch_size={prev_size}, "
                        f"retrying {len(pending)} item(s) at batch_size={fb_size}"
                    )
                    next_pending = []
                    for sub_start in range(0, len(pending), fb_size):
                        sub = pending[sub_start:sub_start + fb_size]
                        sub_items = [item for _, item in sub]
                        sub_indices = [idx for idx, _ in sub]
                        try:
                            sub_responses = self._process_batch(sub_items)
                            for idx, resp in zip(sub_indices, sub_responses):
                                responses[idx] = resp
                                item = batch[idx]
                                template_id = self._get_effective_template_id(item.get("prompt_context"))
                                self.cache.put(item["video_id"], item["mask_pattern_hash"], template_id, [resp])
                                self._call_count += 1
                            torch.cuda.empty_cache()
                        except RuntimeError as e2:
                            if "out of memory" not in str(e2).lower():
                                raise
                            torch.cuda.empty_cache()
                            next_pending.extend(sub)

                    if not next_pending:
                        break
                    if fb_size == 1:
                        # Already at the smallest unit; nothing left to halve.
                        failed_ids = [batch[idx].get("video_id", "?") for idx, _ in next_pending]
                        raise RuntimeError(
                            f"OOM persists at batch_size=1 for {len(next_pending)} item(s): "
                            f"{failed_ids[:5]}{'...' if len(failed_ids) > 5 else ''}"
                        )
                    prev_size = fb_size
                    fb_size = max(1, fb_size // 2)
                    pending = next_pending

            # Clear CUDA cache between batches
            torch.cuda.empty_cache()

        # Snapshot peak memory across all visible CUDA devices for this call.
        if torch.cuda.is_available():
            for _g in range(torch.cuda.device_count()):
                try:
                    self.last_peak_mem_gb[_g] = (
                        torch.cuda.max_memory_allocated(_g) / 1e9
                    )
                except RuntimeError:
                    pass

        return responses

    def _process_batch(self, items: List[Dict[str, Any]]) -> List[OracleResponse]:
        """Process a batch of items in a single forward pass.

        Args:
            items: List of query items (all will be processed together)

        Returns:
            List of OracleResponse objects
        """
        from qwen_vl_utils import process_vision_info
        from transformers.video_utils import VideoMetadata

        # Prepare all messages
        all_texts = []
        all_video_inputs = []
        all_image_inputs = []
        all_video_metadata = []

        for item in items:
            frames = item["frames"]
            action_label = item["action_label"]
            removal_note = item.get("removal_note")
            prompt_context = item.get("prompt_context")

            # Build prompt (same logic as query_single)
            if prompt_context:
                from .prompts import get_prompt
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

            pil_frames = self._frames_to_video_content(frames)

            video_fps = float(prompt_context["video_fps"]) if prompt_context and "video_fps" in prompt_context else 2.0

            video_msg: Dict[str, Any] = {"type": "video", "video": pil_frames, "fps": video_fps}
            if prompt_context and prompt_context.get("max_pixels") is not None:
                video_msg["max_pixels"] = int(prompt_context["max_pixels"])

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        video_msg,
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            all_texts.append(text)
            all_image_inputs.append(image_inputs)
            all_video_inputs.append(video_inputs)
            all_video_metadata.append(VideoMetadata(
                total_num_frames=len(pil_frames),
                fps=video_fps,
                frames_indices=list(range(len(pil_frames))),
            ))

        # Flatten video inputs (each item may have multiple videos/images)
        # For our use case, each item has exactly one video
        flat_images = []
        flat_videos = []
        for img_list, vid_list in zip(all_image_inputs, all_video_inputs):
            if img_list:
                flat_images.extend(img_list)
            if vid_list:
                flat_videos.extend(vid_list)

        # Process all inputs together
        inputs = self.processor(
            text=all_texts,
            images=flat_images if flat_images else None,
            videos=flat_videos if flat_videos else None,
            padding=True,
            return_tensors="pt",
            video_metadata=all_video_metadata,
        )

        # Move to device
        first_device = next(self.model.parameters()).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(first_device)
        else:
            inputs = {
                k: v.to(first_device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        # Batched generation
        with torch.no_grad():
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode all outputs
        gen_ids = gen_outputs.sequences
        input_len = inputs["input_ids"].shape[-1]
        trimmed = gen_ids[:, input_len:]

        raw_outputs = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Parse responses
        responses = []
        tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        yes_ids = tok.encode("YES", add_special_tokens=False)
        no_ids = tok.encode("NO", add_special_tokens=False)
        skip_ids = tok.encode("SKIP", add_special_tokens=False)
        yes_token_id = yes_ids[0] if yes_ids else None
        no_token_id = no_ids[0] if no_ids else None
        skip_token_id = skip_ids[0] if skip_ids else None

        for batch_idx, raw_output in enumerate(raw_outputs):
            video_id = items[batch_idx].get("video_id", f"batch_{batch_idx}")

            try:
                response = parse_oracle_response(raw_output)

                # Extract logits for this item
                if gen_outputs.scores and yes_token_id and no_token_id:
                    try:
                        # Find decision position in this item's output
                        item_tokens = trimmed[batch_idx]
                        decision_pos = None
                        for pos, token_id in enumerate(item_tokens):
                            tid = token_id.item()
                            if tid == yes_token_id or tid == no_token_id:
                                decision_pos = pos
                                break
                            if skip_token_id and tid == skip_token_id:
                                decision_pos = pos
                                break

                        if decision_pos is not None and decision_pos < len(gen_outputs.scores):
                            decision_logits = gen_outputs.scores[decision_pos][batch_idx]
                            logit_yes = decision_logits[yes_token_id].item()
                            logit_no = decision_logits[no_token_id].item()
                            logit_skip = None
                            if skip_token_id is not None:
                                logit_skip = decision_logits[skip_token_id].item()

                            # Two-stage: binary P(YES|not SKIP) + separate P(SKIP)
                            p_yes, p_skip = logits_to_confidence(logit_yes, logit_no, logit_skip)

                            response.logit_yes = logit_yes
                            response.logit_no = logit_no
                            response.logit_skip = logit_skip
                            response.logit_confidence = p_yes
                            response.logit_p_skip = p_skip
                    except Exception as e:
                        logger.debug(f"Failed to extract logits for batch item {batch_idx}: {e}")

                responses.append(response)

            except Exception as e:
                logger.error(
                    f"Failed to parse response for {video_id}: {e} "
                    f"(raw_output len={len(raw_output)})"
                )
                _dump_parse_failure(video_id, raw_output, e)
                responses.append(OracleResponse(
                    decision=OracleDecision.SKIP,
                    confidence=0.0,
                    evidence=f"Parse failed: {str(e)[:50]}",
                    raw_output=raw_output,
                ))

        return responses


# Global worker state for parallel processing
_worker_oracle: Optional["QwenOracle"] = None
_worker_gpu_id: Optional[int] = None
_worker_gpu_ids: List[int] = []


def _init_worker_on_gpu(model_key: str, gpu_ids: Union[int, List[int]], cache_dir: str, config_dict: dict, max_new_tokens: int = 256):
    """Initialize worker process with its own model on specified GPU(s)."""
    global _worker_oracle, _worker_gpu_id, _worker_gpu_ids

    import os

    # Normalize to list
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]

    # Set CUDA to only see this worker's GPU(s)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    _worker_gpu_id = gpu_ids[0]
    _worker_gpu_ids = list(gpu_ids)

    # Force CUDA to reinitialize with new visible devices
    import torch
    if torch.cuda.is_available():
        torch.cuda.init()

    num_gpus = len(gpu_ids)
    logger.info(f"Worker initializing on GPU(s) {gpu_ids} (CUDA sees {num_gpus} device(s))")

    # Load model across this worker's GPU(s)
    model, processor = load_qwen_model(model_key, num_gpus=num_gpus, cache_dir=cache_dir)

    config = OracleConfig(**config_dict) if config_dict else OracleConfig()
    _worker_oracle = QwenOracle(model, processor, config, max_new_tokens=max_new_tokens)

    logger.info(f"Worker on GPU(s) {gpu_ids} ready")


def _worker_query(item: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function to process a single query."""
    global _worker_oracle, _worker_gpu_id

    if _worker_oracle is None:
        raise RuntimeError("Worker not initialized")

    response = _worker_oracle.query_single(
        frames=item["frames"],
        action_label=item["action_label"],
        video_id=item["video_id"],
        removal_note=item.get("removal_note"),
        prompt_context=item.get("prompt_context"),
    )

    return {
        "index": item["_batch_index"],
        "response": response.to_dict(),
        "mask_pattern_hash": item["mask_pattern_hash"],
        "gpu_id": _worker_gpu_id,
    }


def _worker_query_batch(items: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
    """Worker function to process a batch of queries using batched inference."""
    global _worker_oracle, _worker_gpu_id, _worker_gpu_ids

    if _worker_oracle is None:
        raise RuntimeError("Worker not initialized")

    # Reset peak memory stats so we measure THIS batch's peak.
    import torch
    peak_mem_gb: Dict[int, float] = {}
    if torch.cuda.is_available():
        for _g in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(_g)
            except RuntimeError:
                pass

    responses = _worker_oracle.query_batch(items, batch_size=batch_size)

    if torch.cuda.is_available():
        # Map worker-local CUDA index → physical GPU ID for clarity in parent logs.
        for _local_idx in range(torch.cuda.device_count()):
            try:
                physical = (
                    _worker_gpu_ids[_local_idx]
                    if _local_idx < len(_worker_gpu_ids)
                    else _local_idx
                )
                peak_mem_gb[physical] = torch.cuda.max_memory_allocated(_local_idx) / 1e9
            except RuntimeError:
                pass

    results = []
    for i, (item, response) in enumerate(zip(items, responses)):
        result = {
            "index": item["_batch_index"],
            "response": response.to_dict() if response else None,
            "mask_pattern_hash": item["mask_pattern_hash"],
            "gpu_id": _worker_gpu_id,
        }
        # Stash peak-memory only on the first result to avoid duplication.
        if i == 0:
            result["peak_mem_gb"] = peak_mem_gb
            result["worker_batch_count"] = len(items)
        results.append(result)
    return results


def _create_worker_pool(model_key: str, gpu_groups: List[Union[int, List[int]]], cache_dir: str, config_dict: dict, max_new_tokens: int = 256, worker_batch_size: int = 1):
    """Create a pool of worker processes, each on its own GPU(s).

    Args:
        gpu_groups: List of GPU assignments per worker. Each element is either
            a single GPU ID (int) or a list of GPU IDs for multi-GPU workers.
        worker_batch_size: Number of clips each worker processes per forward pass.
            When > 1, the worker collects multiple items from its queue and
            processes them in a single batched forward pass.

    Returns a list of (process, input_queue, output_queue) tuples.
    """
    import queue
    from multiprocessing import Process, Queue

    workers = []

    for gpu_assignment in gpu_groups:
        input_q = Queue()
        output_q = Queue()

        def worker_loop(gpu_ids, model_key, cache_dir, config_dict, max_new_tokens, input_q, output_q, batch_size):
            """Worker main loop — collects up to batch_size items and processes them together."""
            _init_worker_on_gpu(model_key, gpu_ids, cache_dir, config_dict, max_new_tokens)

            items = []
            while True:
                try:
                    # Wait for at least one item
                    first = input_q.get(timeout=1.0)
                    if first is None:  # Shutdown signal
                        break

                    # Collect up to batch_size items (non-blocking after the first)
                    items = [first]
                    while len(items) < batch_size:
                        try:
                            item = input_q.get_nowait()
                            if item is None:  # Shutdown signal — put it back and finish this batch
                                input_q.put(None)
                                break
                            items.append(item)
                        except queue.Empty:
                            break

                    # Process as a batch
                    if len(items) == 1:
                        result = _worker_query(items[0])
                        output_q.put(result)
                    else:
                        results = _worker_query_batch(items, batch_size=len(items))
                        for result in results:
                            output_q.put(result)

                    items = []

                except queue.Empty:
                    continue
                except Exception as e:
                    # Report error for all items in the current batch
                    for it in items:
                        output_q.put({"error": str(e), "index": it.get("_batch_index", -1)})
                    items = []

        p = Process(
            target=worker_loop,
            args=(gpu_assignment, model_key, cache_dir, config_dict, max_new_tokens, input_q, output_q, worker_batch_size),
            daemon=True,
        )
        p.start()
        workers.append((p, input_q, output_q))
        logger.info(f"Started worker process for GPU(s) {gpu_assignment}")

    return workers


class ParallelQwenOracle(OracleInterface):
    """Parallel Qwen VL oracle using multiple GPUs.

    Spawns worker processes, each with its own model instance. Each worker
    can span one or more GPUs (controlled by gpus_per_instance).

    Examples:
        - 4 GPUs, gpus_per_instance=1: 4 workers (best for 7B models)
        - 4 GPUs, gpus_per_instance=2: 2 workers, each spanning 2 GPUs (best for 32B models)
    """

    def __init__(
        self,
        model_key: str = "qwen2.5-vl-7b",
        gpu_ids: List[int] = None,
        gpus_per_instance: int = 1,
        cache_dir: str = "./models/pretrained_oracle_models/hf_cache",
        config: Optional[OracleConfig] = None,
        cache: Optional[OracleCache] = None,
        max_new_tokens: int = 256,
        worker_batch_size: int = 4,
    ):
        """Initialize parallel oracle.

        Args:
            model_key: Model key from MODEL_CONFIGS
            gpu_ids: List of GPU IDs to use (default: all available)
            gpus_per_instance: Number of GPUs per worker instance (default: 1).
                Use 2+ for larger models (e.g., 32B) that don't fit on a single GPU.
            cache_dir: HuggingFace cache directory
            config: Oracle configuration
            cache: Optional cache
            max_new_tokens: Maximum tokens to generate per query
            worker_batch_size: Number of clips each worker processes per forward pass
        """
        super().__init__(config or OracleConfig(), cache)
        self.model_key = model_key
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.worker_batch_size = worker_batch_size
        # Per-batch peak GPU memory snapshot (populated as workers report).
        self.last_peak_mem_gb: Dict[int, float] = {}

        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        self.gpu_ids = gpu_ids
        self.gpus_per_instance = gpus_per_instance

        if len(gpu_ids) % gpus_per_instance != 0:
            raise ValueError(
                f"Number of GPUs ({len(gpu_ids)}) must be divisible by "
                f"gpus_per_instance ({gpus_per_instance})"
            )

        # Group GPUs into worker assignments
        self.gpu_groups = [
            gpu_ids[i:i + gpus_per_instance]
            for i in range(0, len(gpu_ids), gpus_per_instance)
        ]
        self.num_workers = len(self.gpu_groups)

        logger.info(f"ParallelQwenOracle configured: {self.num_workers} instance(s), "
                     f"{gpus_per_instance} GPU(s) each: {self.gpu_groups}")

        # Workers are initialized lazily on first batch query
        self._workers = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazily initialize worker processes."""
        if self._initialized:
            return

        logger.info(f"Spawning {self.num_workers} worker processes (batch_size={self.worker_batch_size})...")
        config_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else {}
        self._workers = _create_worker_pool(
            self.model_key,
            self.gpu_groups,
            self.cache_dir,
            config_dict,
            max_new_tokens=self.max_new_tokens,
            worker_batch_size=self.worker_batch_size,
        )
        self._initialized = True
        logger.info(f"Initialized {self.num_workers} parallel workers: {self.gpu_groups} (batch_size={self.worker_batch_size})")

    def query_single(
        self,
        frames: List[np.ndarray],
        action_label: str,
        video_id: str,
        removal_note: Optional[str] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> OracleResponse:
        """Query with single video (uses first worker)."""
        batch = [{
            "frames": frames,
            "action_label": action_label,
            "video_id": video_id,
            "mask_pattern_hash": "single",
            "removal_note": removal_note,
            "prompt_context": prompt_context,
        }]
        return self.query_batch(batch)[0]

    def query_batch(
        self,
        batch: List[Dict[str, Any]],
        batch_size: int = 1,
    ) -> List[OracleResponse]:
        """Query oracle with a batch of videos in parallel across GPUs.

        Each worker collects up to worker_batch_size items and processes them
        in a single batched forward pass for better throughput.

        Args:
            batch: List of query dicts
            batch_size: Accepted for API compatibility (worker_batch_size from
                __init__ controls per-worker batching).

        Returns:
            List of oracle responses (same order as batch)
        """
        if not batch:
            return []

        # Check cache first
        cached_responses = {}
        items_to_process = []

        for i, item in enumerate(batch):
            template_id = self._get_effective_template_id(item.get("prompt_context"))
            cached = self.cache.get(
                item["video_id"],
                item["mask_pattern_hash"],
                template_id
            )
            if cached:
                cached_responses[i] = cached[0]
            else:
                item["_batch_index"] = i
                items_to_process.append(item)

        if not items_to_process:
            # All cached
            return [cached_responses[i] for i in range(len(batch))]

        # Process uncached items in parallel
        self._ensure_initialized()

        results = {}
        pending_count = 0
        # Reset peak-memory snapshot for this batch (populated as workers report).
        self.last_peak_mem_gb = {}

        # Distribute items across workers (round-robin)
        for i, item in enumerate(items_to_process):
            worker_idx = i % self.num_workers
            _, input_q, _ = self._workers[worker_idx]
            input_q.put(item)
            pending_count += 1

        # Collect results from all workers
        import queue
        collected = 0
        while collected < pending_count:
            for _, _, output_q in self._workers:
                try:
                    result = output_q.get(timeout=0.1)
                    if "error" in result:
                        logger.error(f"Worker error: {result['error']}")
                        idx = result.get("index", -1)
                        if idx >= 0:
                            results[idx] = OracleResponse(
                                decision=OracleDecision.SKIP,
                                confidence=0.0,
                                evidence=f"Worker error: {result['error'][:50]}",
                            )
                    else:
                        idx = result["index"]
                        response = OracleResponse.from_dict(result["response"])
                        results[idx] = response

                        # Cache the result
                        item = batch[idx]
                        template_id = self._get_effective_template_id(item.get("prompt_context"))
                        self.cache.put(
                            item["video_id"],
                            item["mask_pattern_hash"],
                            template_id,
                            [response]
                        )
                        logger.debug(f"Got result for item {idx} from GPU {result.get('gpu_id')}")

                        # Worker stashes peak memory on the first result of each
                        # forward pass; merge into the per-batch dict (max across
                        # forward passes if a worker ran more than one).
                        worker_peak = result.get("peak_mem_gb")
                        if worker_peak:
                            for gid, gb in worker_peak.items():
                                prev = self.last_peak_mem_gb.get(gid, 0.0)
                                if gb > prev:
                                    self.last_peak_mem_gb[gid] = gb
                    collected += 1
                except queue.Empty:
                    continue

        # Merge cached and computed responses in original order
        final_responses = []
        for i in range(len(batch)):
            if i in cached_responses:
                final_responses.append(cached_responses[i])
            elif i in results:
                final_responses.append(results[i])
            else:
                # Should not happen, but fallback
                final_responses.append(OracleResponse(
                    decision=OracleDecision.SKIP,
                    confidence=0.0,
                    evidence="Response missing",
                ))

        return final_responses

    def shutdown(self):
        """Shutdown worker processes."""
        if self._workers:
            for proc, input_q, _ in self._workers:
                input_q.put(None)  # Shutdown signal
            for proc, _, _ in self._workers:
                proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.terminate()
            self._workers = None
            self._initialized = False
            logger.info("Parallel workers shut down")

    def __del__(self):
        """Cleanup worker processes."""
        self.shutdown()


def create_qwen_oracle(
    model_key: str = "qwen2.5-vl-7b",
    num_gpus: int = 4,
    cache_dir: str = "./models/pretrained_oracle_models/hf_cache",
    oracle_config: Optional[OracleConfig] = None,
    cache: Optional[OracleCache] = None,
    max_new_tokens: int = 4096,
) -> QwenOracle:
    """Create a QwenOracle instance.

    Args:
        model_key: Model key from MODEL_CONFIGS
        num_gpus: Number of GPUs to use
        cache_dir: HuggingFace cache directory
        oracle_config: Oracle configuration
        cache: Optional cache
        max_new_tokens: Maximum tokens to generate

    Returns:
        Configured QwenOracle instance
    """
    model, processor = load_qwen_model(model_key, num_gpus, cache_dir)
    return QwenOracle(model, processor, oracle_config, cache, max_new_tokens=max_new_tokens)
