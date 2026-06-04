"""V-JEPA 2 ViT-L wrapper.

See the development design notes Decisions 6 (K400 gap),
8 (BF16, batch sizing, dual-GPU strategy), and 9 (Diving-48 placeholder).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from . import REGISTRY, ProbeConfig, lookup_by_hf_id
from .adapter import Protocol, Segment, build_input

LOG = logging.getLogger(__name__)


@dataclass
class Prediction:
    top_k_label_ids: list[int]
    top_k_label_strings: list[str]
    top_k_probs: list[float]
    full_logits: torch.Tensor


class VJepa2Wrapper:
    def __init__(self, model, processor, probe: ProbeConfig, device: str):
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
    ) -> "VJepa2Wrapper":
        from transformers import VJEPA2ForVideoClassification, VJEPA2VideoProcessor

        probe = lookup_by_hf_id(model_id)
        if probe is None:
            raise ValueError(
                f"Model id {model_id!r} is not registered in oracle.scripts.vjepa2.REGISTRY. "
                f"Known: {sorted(c.hf_model_id for c in REGISTRY.values() if c.hf_model_id)}"
            )

        if probe.status == "unavailable":
            raise NotImplementedError(
                f"V-JEPA 2 probe for {probe.key} is not released by Meta on Hugging Face. "
                "See the development design notes Decision 6 "
                "for remediation options (drop K400, train probe, route via different recognizer)."
            )

        if probe.status == "placeholder":
            raise NotImplementedError(
                f"V-JEPA 2 probe for {probe.key} is registered as a placeholder; "
                "Diving-48 inference is deferred to a follow-up change. See "
                "the development design notes Decision 9."
            )

        LOG.info("Loading V-JEPA 2 weights: %s on %s (BF16)…", model_id, device)
        model = VJEPA2ForVideoClassification.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        model.eval()
        model.to(device)
        processor = VJEPA2VideoProcessor.from_pretrained(model_id)
        if head_checkpoint is not None:
            from .head import load_head
            LOG.info("Overlaying finetuned head from %s", head_checkpoint)
            head = load_head(head_checkpoint)
            head.copy_into_hf_model(model)
        return cls(model=model, processor=processor, probe=probe, device=device)

    def forward_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values_videos=pixel_values)
        return outputs.logits

    @torch.inference_mode()
    def predict(
        self,
        video_path: str | Path,
        kept_segments: list[Segment] | None = None,
        protocol: Protocol = "repack",
        k: int = 5,
    ) -> Prediction:
        pixel_values = build_input(
            video_path=video_path,
            kept_segments=kept_segments,
            protocol=protocol,
            frames_per_clip=self.frames_per_clip,
            image_size=self.image_size,
            video_processor=self.processor,
        )
        pixel_values = pixel_values.to(self.device, dtype=torch.bfloat16)
        outputs = self.model(pixel_values_videos=pixel_values)
        logits = outputs.logits[0].float()
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


__all__ = ["VJepa2Wrapper", "Prediction"]
