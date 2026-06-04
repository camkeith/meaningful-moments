"""VideoMAE V1 inference wrapper, mirroring ``oracle.scripts.vjepa2.inference.VJepa2Wrapper``.

Same interface, different recognizer:
    - forward kwarg is ``pixel_values`` (not ``pixel_values_videos``)
    - image_size is 224 (not 256)
    - num_labels is 400 (K400 classes)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from . import REGISTRY, VideoMaeProbeConfig, lookup_by_hf_id

LOG = logging.getLogger(__name__)


@dataclass
class Prediction:
    top_k_label_ids: list[int]
    top_k_label_strings: list[str]
    top_k_probs: list[float]
    full_logits: torch.Tensor


class VideoMaeWrapper:
    """Recognizer-agnostic interface (matches ``VJepa2Wrapper``):
        - ``frames_per_clip``, ``image_size``, ``processor``, ``model``
        - ``forward_batch(pixel_values)`` → logits
        - ``predict(video_path, kept_segments, protocol, k)``
    """

    def __init__(self, model, processor, probe: VideoMaeProbeConfig, device: str):
        self.model = model
        self.processor = processor
        self.probe = probe
        self.device = device
        self.frames_per_clip = probe.frames_per_clip
        self.image_size = probe.image_size

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "cuda:0",
        head_checkpoint: str | Path | None = None,
    ) -> "VideoMaeWrapper":
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        probe = lookup_by_hf_id(model_id)
        if probe is None:
            raise ValueError(
                f"Model id {model_id!r} is not registered in oracle.scripts.videomae.REGISTRY. "
                f"Known: {sorted(c.hf_model_id for c in REGISTRY.values() if c.hf_model_id)}"
            )
        if probe.status == "unavailable":
            raise NotImplementedError(f"VideoMAE probe {probe.key} is unavailable.")
        if probe.status == "placeholder":
            raise NotImplementedError(f"VideoMAE probe {probe.key} is a placeholder; not enabled in this change.")

        LOG.info("Loading VideoMAE weights: %s on %s (BF16)…", model_id, device)
        model = VideoMAEForVideoClassification.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        model.eval()
        model.to(device)
        processor = VideoMAEImageProcessor.from_pretrained(model_id)
        if head_checkpoint is not None:
            from .head import load_head
            LOG.info("Overlaying finetuned head from %s", head_checkpoint)
            head = load_head(head_checkpoint)
            head.copy_into_hf_model(model)
        return cls(model=model, processor=processor, probe=probe, device=device)

    def forward_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    @torch.inference_mode()
    def predict(
        self,
        video_path: str | Path,
        kept_segments=None,
        protocol: str = "repack",
        k: int = 5,
    ) -> Prediction:
        from oracle.scripts.vjepa2.adapter import build_input

        pixel_values = build_input(
            video_path=video_path,
            kept_segments=kept_segments,
            protocol=protocol,
            frames_per_clip=self.frames_per_clip,
            image_size=self.image_size,
            video_processor=self.processor,
        )
        pixel_values = pixel_values.to(self.device, dtype=torch.bfloat16)
        logits = self.forward_batch(pixel_values)[0].float()
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=k)
        id2label = self.model.config.id2label
        top_ids_list = [int(i) for i in top_ids.tolist()]
        return Prediction(
            top_k_label_ids=top_ids_list,
            top_k_label_strings=[id2label[i] for i in top_ids_list],
            top_k_probs=[float(p) for p in top_probs.tolist()],
            full_logits=logits.cpu(),
        )


__all__ = ["VideoMaeWrapper", "Prediction"]
