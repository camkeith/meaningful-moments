"""Trainable head module for V-JEPA 2 linear-probe finetuning.

Mirrors the HF ``VJEPA2ForVideoClassification`` final classifier layout so a
trained state_dict can be copied straight back into the HF model at inference:

    pixel_values_videos                  ← frozen encoder input
        → model.vjepa2(...)              ← encoder (frozen)
        → model.pooler(last_hidden)      ← attentive pooler (frozen — paper-trained)
        → pooled (B, hidden=1024)        ← cached in Phase A
        → classifier (Linear)            ← trainable; what this module wraps
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)

    @classmethod
    def from_hf_model(cls, hf_model) -> "LinearHead":
        hidden_size = hf_model.classifier.in_features
        num_labels = hf_model.classifier.out_features
        head = cls(hidden_size, num_labels)
        head.classifier.load_state_dict(hf_model.classifier.state_dict())
        return head

    def copy_into_hf_model(self, hf_model) -> None:
        target_dtype = next(hf_model.classifier.parameters()).dtype
        target_device = next(hf_model.classifier.parameters()).device
        cls_sd = {k: v.to(target_device, dtype=target_dtype) for k, v in self.classifier.state_dict().items()}
        hf_model.classifier.load_state_dict(cls_sd)


def save_head(head: LinearHead, path) -> None:
    payload = {
        "classifier": head.classifier.state_dict(),
        "hidden_size": head.classifier.in_features,
        "num_labels": head.classifier.out_features,
    }
    torch.save(payload, str(path))


def load_head(path) -> LinearHead:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    head = LinearHead(payload["hidden_size"], payload["num_labels"])
    head.classifier.load_state_dict(payload["classifier"])
    return head


__all__ = ["LinearHead", "save_head", "load_head"]
