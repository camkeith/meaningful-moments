"""VideoMAE V1 inference adapter — secondary classifier for K400.

Used as the cross-architecture / different-SSL-regime check for SAGE Experiment 5.
See the development design notes Decision 6 for why
this carries the K400 path: Meta did not release a V-JEPA 2 K400 probe.

The OpenGVLab VideoMAE V2 backbones are pretrained MAE encoders only (no
classifier head). The MCG-NJU V1 fine-tunes are the only K400 video classifiers
on Hugging Face. Methodologically the cross-arch argument is preserved — both
V1 and V2 are masked-autoencoding pretraining, contrasted against V-JEPA's
latent-prediction regime.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoMaeProbeConfig:
    key: str
    hf_model_id: str
    frames_per_clip: int
    image_size: int
    status: str  # "released" | "placeholder" | "unavailable"


REGISTRY: dict[str, VideoMaeProbeConfig] = {
    "k400-videomae-v1-large": VideoMaeProbeConfig(
        key="k400-videomae-v1-large",
        hf_model_id="MCG-NJU/videomae-large-finetuned-kinetics",
        frames_per_clip=16,
        image_size=224,
        status="released",
    ),
    "k400-videomae-v1-huge": VideoMaeProbeConfig(
        key="k400-videomae-v1-huge",
        hf_model_id="MCG-NJU/videomae-huge-finetuned-kinetics",
        frames_per_clip=16,
        image_size=224,
        status="released",
    ),
    "ssv2-videomae-v1-base": VideoMaeProbeConfig(
        key="ssv2-videomae-v1-base",
        hf_model_id="MCG-NJU/videomae-base-finetuned-ssv2",
        frames_per_clip=16,
        image_size=224,
        status="placeholder",  # not used in this change; available for follow-up cross-arch on SSv2
    ),
}


def lookup_by_hf_id(model_id: str) -> VideoMaeProbeConfig | None:
    for cfg in REGISTRY.values():
        if cfg.hf_model_id == model_id:
            return cfg
    return None


__all__ = ["VideoMaeProbeConfig", "REGISTRY", "lookup_by_hf_id"]
