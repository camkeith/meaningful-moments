"""Trainable head module for VideoMAE-K400 frozen-encoder finetuning.

Mirrors the HF ``VideoMAEForVideoClassification`` head layout exactly so a trained
state_dict can be copied straight back into the HF model at inference time:

    last_hidden_state            (B, T*N, D)   ← encoder output (frozen)
        .mean(dim=1)             (B, D)        ← pooled feature (cached in Phase A)
        → fc_norm (LayerNorm)    (B, D)
        → classifier (Linear)    (B, num_labels)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrozenHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.fc_norm(pooled))

    @classmethod
    def from_hf_model(cls, hf_model) -> "FrozenHead":
        """Build a FrozenHead initialised from an HF VideoMAEForVideoClassification's weights."""
        hidden_size = hf_model.classifier.in_features
        num_labels = hf_model.classifier.out_features
        head = cls(hidden_size, num_labels)
        head.fc_norm.load_state_dict(hf_model.fc_norm.state_dict())
        head.classifier.load_state_dict(hf_model.classifier.state_dict())
        return head

    def copy_into_hf_model(self, hf_model) -> None:
        """Overwrite the HF model's fc_norm + classifier weights in place."""
        target_dtype = next(hf_model.classifier.parameters()).dtype
        target_device = next(hf_model.classifier.parameters()).device
        fc_norm_sd = {k: v.to(target_device, dtype=target_dtype) for k, v in self.fc_norm.state_dict().items()}
        cls_sd = {k: v.to(target_device, dtype=target_dtype) for k, v in self.classifier.state_dict().items()}
        hf_model.fc_norm.load_state_dict(fc_norm_sd)
        hf_model.classifier.load_state_dict(cls_sd)


def save_head(head: FrozenHead, path) -> None:
    payload = {
        "fc_norm": head.fc_norm.state_dict(),
        "classifier": head.classifier.state_dict(),
        "hidden_size": head.fc_norm.normalized_shape[0],
        "num_labels": head.classifier.out_features,
    }
    torch.save(payload, str(path))


def load_head(path) -> FrozenHead:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    head = FrozenHead(payload["hidden_size"], payload["num_labels"])
    head.fc_norm.load_state_dict(payload["fc_norm"])
    head.classifier.load_state_dict(payload["classifier"])
    return head


__all__ = ["FrozenHead", "save_head", "load_head"]
