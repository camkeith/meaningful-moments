"""
InternVL3 oracle implementation for MSS verification.

Implements OracleInterface using OpenGVLab/InternVL3 video-capable models.
Loads via HF transformers with trust_remote_code=True and shards across
multiple GPUs via device_map="balanced".

Inference path uses model.chat() (the canonical InternVL conversation API)
and applies text-based confidence to logit_confidence — InternVL3's chat
method does not surface generation scores, so YES/NO/SKIP token logits are
unavailable. For direct-scoring runs this is fine: per-segment importance
comes from the JSON output, and the precheck gate uses the model's
self-reported "confidence" field.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

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
    parse_oracle_response,
)

logger = logging.getLogger(__name__)


# Model registry. CLI/--model values map to HF repo IDs.
INTERNVL3_MODEL_CONFIGS = {
    "internvl3-38b": {
        "hf_name": "OpenGVLab/InternVL3-38B",
        "provider": "internvl3",
    },
    "internvl3-78b": {
        "hf_name": "OpenGVLab/InternVL3-78B",
        "provider": "internvl3",
    },
    "internvl3-14b": {
        "hf_name": "OpenGVLab/InternVL3-14B",
        "provider": "internvl3",
    },
    "internvl3-8b": {
        "hf_name": "OpenGVLab/InternVL3-8B",
        "provider": "internvl3",
    },
}


# ImageNet normalization stats used by the InternVL preprocessor.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448):
    """Build the InternVL image transform (resize + normalize)."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Pick the ratio from target_ratios that best matches the source image."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 1,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    """Tile-based preprocessing matching InternVL's dynamic_preprocess.

    For video we cap max_num=1 (1 tile per frame at 448x448) to keep token
    counts manageable across many segments. Increase max_num for higher
    spatial detail at the cost of more vision tokens per frame.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _frames_to_pixel_values(
    frames: List[np.ndarray],
    image_size: int = 448,
    max_num: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, List[int]]:
    """Convert frames to (pixel_values, num_patches_list) for InternVL3.

    Args:
        frames: List of RGB numpy arrays
        image_size: Tile resolution (448 is the standard for InternVL3)
        max_num: Maximum tiles per frame; 1 keeps tokens low for video
        dtype: Tensor dtype (bf16 matches model default)

    Returns:
        pixel_values: Tensor of shape (sum_tiles, 3, image_size, image_size)
        num_patches_list: List of tile counts per frame (parallel to frames)
    """
    transform = _build_transform(input_size=image_size)
    all_tiles: List[torch.Tensor] = []
    num_patches_list: List[int] = []

    for frame in frames:
        pil = Image.fromarray(frame)
        tiles = _dynamic_preprocess(
            pil, min_num=1, max_num=max_num, image_size=image_size, use_thumbnail=False
        )
        for tile in tiles:
            all_tiles.append(transform(tile))
        num_patches_list.append(len(tiles))

    pixel_values = torch.stack(all_tiles).to(dtype)
    return pixel_values, num_patches_list


def load_internvl3_model(
    model_key: str,
    num_gpus: int = 4,
    cache_dir: str = "./models/pretrained_oracle_models/hf_cache",
):
    """Load InternVL3 model + tokenizer with multi-GPU sharding.

    Returns (model, tokenizer).
    """
    if model_key not in INTERNVL3_MODEL_CONFIGS:
        available = ", ".join(INTERNVL3_MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown InternVL3 model '{model_key}'. Available: {available}")

    hf_name = INTERNVL3_MODEL_CONFIGS[model_key]["hf_name"]

    from transformers import AutoModel, AutoTokenizer

    gpu_count = torch.cuda.device_count()
    gpus_to_use = list(range(min(num_gpus, gpu_count)))
    visible = len(gpus_to_use)

    logger.info(f"Loading InternVL3 model: {hf_name}")
    logger.info(
        f"CUDA devices visible: {gpu_count}, requesting: {num_gpus}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
    )
    if visible == 0:
        raise RuntimeError("No CUDA devices available for InternVL3")

    for i in gpus_to_use:
        props = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GiB")

    max_memory = {
        i: f"{max(int(torch.cuda.get_device_properties(i).total_memory / 1024**3) - 0.1, 1)}GiB"
        for i in gpus_to_use
    }
    logger.info(f"Using GPUs: {gpus_to_use}, device_map=balanced, max_memory={max_memory}")

    model = AutoModel.from_pretrained(
        hf_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="balanced",
        max_memory=max_memory,
        cache_dir=cache_dir,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        hf_name, trust_remote_code=True, use_fast=False, cache_dir=cache_dir
    )

    return model, tokenizer


class InternVL3Oracle(OracleInterface):
    """InternVL3 video-capable oracle.

    Uses model.chat() for generation. Confidence is text-based — the chat
    method does not surface decision-token logits, so logit_yes/no/skip
    fields stay None and logit_confidence falls back to the model's
    self-reported "confidence" JSON field (mapped through the YES decision).
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[OracleConfig] = None,
        cache: Optional[OracleCache] = None,
        max_new_tokens: int = 4096,
        prompt_template: Optional[str] = None,
        image_size: int = 448,
        max_tiles_per_frame: int = 1,
    ):
        super().__init__(config or OracleConfig(), cache)
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size
        self.max_tiles_per_frame = max_tiles_per_frame

        if prompt_template is None:
            self.prompt_template = get_verification_prompt(self.config.prompt_template_id)
        else:
            self.prompt_template = prompt_template

    def query_single(
        self,
        frames: List[np.ndarray],
        action_label: str,
        video_id: str,
        removal_note: Optional[str] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> OracleResponse:
        """Query InternVL3 with a single video.

        Returns:
            OracleResponse with text-based logit_confidence (no token logits)
        """
        system_prompt, user_text = self._build_prompts(
            action_label, removal_note, prompt_context
        )

        try:
            pixel_values, num_patches_list = _frames_to_pixel_values(
                frames,
                image_size=self.image_size,
                max_num=self.max_tiles_per_frame,
                dtype=torch.bfloat16,
            )

            # Move pixel_values to the model's first parameter device
            first_device = next(self.model.parameters()).device
            pixel_values = pixel_values.to(first_device)

            # InternVL conversation: system + per-frame Image-N tag + user text
            # Each frame gets its own <image> placeholder so the model knows
            # frame ordering. This matches the InternVL video chat convention.
            frame_tags = "".join(
                f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))
            )
            question = f"{frame_tags}{user_text}"

            generation_config = dict(
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

            # InternVL's chat() exposes a system_message attribute on the model.
            # Set it before each call so we honor the per-call system prompt.
            old_system = getattr(self.model, "system_message", None)
            try:
                self.model.system_message = system_prompt
                raw_output = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=generation_config,
                    num_patches_list=num_patches_list,
                )
            finally:
                if old_system is not None:
                    self.model.system_message = old_system

            logger.debug(
                f"InternVL3 raw output for {video_id} ({len(num_patches_list)} frames): "
                f"{raw_output[:500]}"
            )

            response = parse_oracle_response(raw_output)
            self._apply_text_confidence(response)
            return response

        except Exception as e:
            logger.error(f"InternVL3 oracle query failed for {video_id}: {e}")
            return OracleResponse(
                decision=OracleDecision.SKIP,
                confidence=0.0,
                evidence=f"Query failed: {str(e)[:50]}",
                raw_output=str(e),
            )

    def query_batch(
        self,
        batch: List[Dict[str, Any]],
        batch_size: int = 1,
    ) -> List[OracleResponse]:
        """Sequential batch — InternVL3 chat doesn't expose a true batched path
        without re-implementing input_embeds construction. The OracleInterface
        default would cache twice; we override here just to handle the cache
        consistently with the other backends."""
        if not batch:
            return []

        responses: List[Optional[OracleResponse]] = [None] * len(batch)

        for i, item in enumerate(batch):
            template_id = self._get_effective_template_id(item.get("prompt_context"))
            cached = self.cache.get(
                item["video_id"], item["mask_pattern_hash"], template_id
            )
            if cached:
                responses[i] = cached[0]
                continue

            resp = self.query_single(
                frames=item["frames"],
                action_label=item["action_label"],
                video_id=item["video_id"],
                removal_note=item.get("removal_note"),
                prompt_context=item.get("prompt_context"),
            )
            responses[i] = resp
            self.cache.put(
                item["video_id"], item["mask_pattern_hash"], template_id, [resp]
            )
            self._call_count += 1

        return responses

    def _build_prompts(
        self,
        action_label: str,
        removal_note: Optional[str],
        prompt_context: Optional[Dict[str, Any]],
    ) -> tuple:
        """Build system + user text. Identical dispatch to QwenOracle."""
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

    def _apply_text_confidence(self, response: OracleResponse) -> None:
        """Map text decision + confidence into logit_confidence.

        InternVL3's chat path doesn't surface per-token logits, so the
        downstream P(YES) gate has to read from the model's JSON output
        instead. This is acceptable because direct-scoring runs only use
        the YES/NO decision as a gating step; the per-segment importance
        scores come from the segments[] array.
        """
        if response.decision == OracleDecision.YES:
            response.logit_confidence = response.confidence
        elif response.decision == OracleDecision.NO:
            response.logit_confidence = 0.0
        else:  # SKIP
            response.logit_confidence = 0.0
            response.logit_p_skip = 1.0


def create_internvl3_oracle(
    model_key: str,
    num_gpus: int = 4,
    cache_dir: str = "./models/pretrained_oracle_models/hf_cache",
    oracle_config: Optional[OracleConfig] = None,
    cache: Optional[OracleCache] = None,
    max_new_tokens: int = 4096,
    max_tiles_per_frame: int = 1,
) -> InternVL3Oracle:
    """Construct an InternVL3Oracle with a freshly loaded model."""
    model, tokenizer = load_internvl3_model(model_key, num_gpus, cache_dir)
    return InternVL3Oracle(
        model=model,
        tokenizer=tokenizer,
        config=oracle_config,
        cache=cache,
        max_new_tokens=max_new_tokens,
        max_tiles_per_frame=max_tiles_per_frame,
    )
