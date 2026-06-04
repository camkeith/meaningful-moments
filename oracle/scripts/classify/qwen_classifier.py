"""
QwenClassifier: Qwen3-VL closed-set top-5 SSv2 classifier.

Loads the same models as QwenOracle (via load_qwen_model) but with a different
prompt and a different output parser. Supports true batched inference with
an OOM-halving fallback identical in spirit to QwenOracle.query_batch.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # oracle/scripts/
from mss.qwen_oracle import load_qwen_model  # noqa: E402

from .parser import parse_top5, parse_top5_diagnostic
from .prompt import build_classification_messages

logger = logging.getLogger("classify.qwen")


class QwenClassifier:
    """Closed-set top-5 SSv2 classifier on top of a loaded Qwen3-VL model."""

    def __init__(self, model, processor, max_new_tokens: int = 128, max_retries: int = 1,
                 retry_temperature: float = 0.0):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        # Greedy only (do_sample=False). max_retries=1 means one attempt — no sampled retry.
        # For closed-set classification, sampling degrades reliability (Qwen3-VL model card
        # lists temperature controls, but the advice for exact-label ranking is to stay greedy).
        self.max_retries = max_retries
        self.retry_temperature = retry_temperature
        self.last_peak_mem_gb: Dict[int, float] = {}

    @staticmethod
    def _frames_to_pil(frames: List[np.ndarray]) -> List[Image.Image]:
        return [Image.fromarray(f) for f in frames]

    def _generate(self, items: List[Dict[str, Any]], sample: bool = False) -> List[str]:
        """Run one batched forward pass. Returns raw decoded strings.

        sample=False → greedy (deterministic). sample=True → temperature-sampled (for retries).
        """
        from qwen_vl_utils import process_vision_info
        from transformers.video_utils import VideoMetadata

        all_texts = []
        flat_videos: list = []
        flat_images: list = []
        all_metadata = []

        for item in items:
            pil_frames = self._frames_to_pil(item["frames"])
            fps = float(item.get("fps", 2.0))
            messages = build_classification_messages(pil_frames, video_fps=fps)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            all_texts.append(text)
            if image_inputs:
                flat_images.extend(image_inputs)
            if video_inputs:
                flat_videos.extend(video_inputs)
            all_metadata.append(VideoMetadata(
                total_num_frames=len(pil_frames),
                fps=fps,
                frames_indices=list(range(len(pil_frames))),
            ))

        inputs = self.processor(
            text=all_texts,
            images=flat_images if flat_images else None,
            videos=flat_videos if flat_videos else None,
            padding=True,
            return_tensors="pt",
            video_metadata=all_metadata,
        )

        first_device = next(self.model.parameters()).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(first_device)
        else:
            inputs = {
                k: (v.to(first_device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }

        # Explicit greedy decoding (do_sample=False + num_beams=1). HF generate() otherwise
        # inherits temperature/top_k/top_p from the model's generation_config, which can
        # silently introduce sampling behavior on closed-set classification.
        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "num_beams": 1}
        if sample:
            gen_kwargs.update({"do_sample": True, "temperature": self.retry_temperature, "top_p": 0.9})
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        trimmed = gen_ids[:, input_len:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def classify_batch(
        self,
        batch: List[Dict[str, Any]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Classify a list of items with up to batch_size per forward pass.

        Each item: {video_id, frames, fps}.
        Returns one result dict per item with keys:
          video_id, top5, parsed, normalization_applied, raw_output, elapsed_s
        """
        results: List[Optional[Dict[str, Any]]] = [None] * len(batch)

        # Reset peak-memory for this call.
        self.last_peak_mem_gb = {}
        if torch.cuda.is_available():
            for g in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(g)
                except RuntimeError:
                    pass

        def _run_chunk(indices: List[int], items: List[Dict[str, Any]]):
            t0 = time.perf_counter()
            raw_outputs = self._generate(items)
            dt_per = (time.perf_counter() - t0) / max(len(items), 1)

            # Retry loop: any item whose output doesn't parse is retried
            # individually up to max_retries times.
            for local_i, (idx, item, raw) in enumerate(zip(indices, items, raw_outputs)):
                top5, norm, reason = parse_top5_diagnostic(raw)
                attempt = 1
                while not top5 and attempt < self.max_retries:
                    logger.warning(
                        f"Parse failed for {item['video_id']} attempt {attempt} "
                        f"(reason={reason}); retrying with sampling (raw[:160]={raw[:160]!r})"
                    )
                    retry_outputs = self._generate([item], sample=True)
                    raw = retry_outputs[0]
                    top5, norm, reason = parse_top5_diagnostic(raw)
                    attempt += 1
                if not top5:
                    logger.warning(
                        f"Giving up on {item['video_id']} after {attempt} attempts "
                        f"(final reason={reason}); writing parsed=false"
                    )

                results[idx] = {
                    "video_id": item["video_id"],
                    "top5": top5,
                    "parsed": bool(top5),
                    "normalization_applied": norm,
                    "raw_output": raw,
                    "elapsed_s": round(dt_per, 3),
                    "retries": attempt - 1,
                }

        pending = [(i, it) for i, it in enumerate(batch)]
        cur_bs = batch_size
        while pending:
            # Process in chunks of cur_bs
            progress_made = False
            next_pending: List = []
            i = 0
            while i < len(pending):
                chunk = pending[i:i + cur_bs]
                indices = [p[0] for p in chunk]
                items = [p[1] for p in chunk]
                try:
                    _run_chunk(indices, items)
                    progress_made = True
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    logger.warning(
                        f"OOM at batch_size={cur_bs} for chunk of {len(chunk)}; "
                        f"deferring to smaller batch."
                    )
                    next_pending.extend(chunk)
                i += cur_bs

            if not next_pending:
                break
            if cur_bs == 1:
                failed_ids = [items[0]["video_id"] for _, items in next_pending[:5]]
                raise RuntimeError(
                    f"OOM persists at batch_size=1 for {len(next_pending)} items: {failed_ids}"
                )
            cur_bs = max(1, cur_bs // 2)
            pending = next_pending
            if not progress_made:
                # Everything OOMed at this size — halve again on next iteration.
                continue

        if torch.cuda.is_available():
            for g in range(torch.cuda.device_count()):
                try:
                    self.last_peak_mem_gb[g] = torch.cuda.max_memory_allocated(g) / 1e9
                except RuntimeError:
                    pass

        return [r for r in results if r is not None]


def create_classifier(
    model_key: str,
    num_gpus: int,
    cache_dir: str,
    max_new_tokens: int = 1024,
) -> QwenClassifier:
    model, processor = load_qwen_model(model_key, num_gpus=num_gpus, cache_dir=cache_dir)
    return QwenClassifier(model, processor, max_new_tokens=max_new_tokens)
