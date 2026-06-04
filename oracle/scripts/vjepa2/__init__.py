"""V-JEPA 2 ViT-L inference adapter for SAGE Experiment 5.

Per the development design notes:
    - ``released``    : Meta probe is on Hugging Face and inference is supported here.
    - ``placeholder`` : probe is on Hugging Face but the path is deferred to a follow-up change.
    - ``unavailable`` : Meta did not release a probe at this size; see Decision 6.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeConfig:
    key: str
    hf_model_id: str
    frames_per_clip: int
    image_size: int
    status: str  # "released" | "placeholder" | "unavailable"


REGISTRY: dict[str, ProbeConfig] = {
    "ssv2-vitl-fpc16-256": ProbeConfig(
        key="ssv2-vitl-fpc16-256",
        hf_model_id="facebook/vjepa2-vitl-fpc16-256-ssv2",
        frames_per_clip=16,
        image_size=256,
        status="released",
    ),
    "diving48-vitl-fpc32-256": ProbeConfig(
        key="diving48-vitl-fpc32-256",
        hf_model_id="facebook/vjepa2-vitl-fpc32-256-diving48",
        frames_per_clip=32,
        image_size=256,
        status="released",
    ),
    "k400-vitl-fpc16-256": ProbeConfig(
        key="k400-vitl-fpc16-256",
        hf_model_id="",
        frames_per_clip=16,
        image_size=256,
        status="unavailable",
    ),
}


def lookup_by_hf_id(model_id: str) -> ProbeConfig | None:
    for cfg in REGISTRY.values():
        if cfg.hf_model_id == model_id:
            return cfg
    return None


__all__ = ["ProbeConfig", "REGISTRY", "lookup_by_hf_id"]
