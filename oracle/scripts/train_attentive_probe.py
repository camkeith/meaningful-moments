#!/usr/bin/env python
"""Phase B — finetune V-JEPA 2 paper-ckpt 4-block AttentiveClassifier on MSS-cut tokens.

Loads per-video token caches produced by ``oracle.scripts.vjepa2_official.cache_token_features``,
builds a fresh ``AttentiveClassifier`` (4-block × 16-head cross-attention pooler + Linear),
warm-starts it from the paper checkpoint's ``probe[0]`` weights, and trains it on the MSS-cut
input distribution. Saves only the best-val checkpoint (criterion = val_loss by default).

Selection rationale — we choose probe[0] from the paper's LR-sweep ensemble (the 4 heads
ckpt['classifiers']) since it's the highest-LR variant per Meta's config and tends to be the
strongest single head. At inference, the trained classifier replaces the entire ensemble.

Output:
    <output-dir>/attentive_probe_best.pt   — torch.save({"classifier": state_dict, "config": ...})
    <output-dir>/train_log.csv             — per-epoch train/val loss + top-1/top-5
    <output-dir>/val_metrics.json          — best + final metrics
    <output-dir>/config.json               — args + recognizer config snapshot
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

LOG = logging.getLogger("train_attentive_probe")


def _ddp_setup(force_enable: bool) -> tuple[bool, int, int, int]:
    """Return (is_ddp, local_rank, rank, world_size).

    Detects torchrun env (LOCAL_RANK / WORLD_SIZE / RANK). If --ddp is requested
    but env vars aren't set, error out — the user should launch via torchrun.
    """
    has_env = "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ
    if force_enable and not has_env:
        raise RuntimeError(
            "--ddp set but torchrun env vars (LOCAL_RANK / WORLD_SIZE) missing. "
            "Launch via `torchrun --nproc_per_node=N python -m oracle.scripts.train_attentive_probe --ddp ...`"
        )
    if not has_env:
        return False, 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ["WORLD_SIZE"])
    import torch.distributed as dist
    if not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return True, local_rank, rank, world_size


def _reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor across DDP ranks (sum)."""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recognizer", required=True,
                   choices=["vjepa2-official-ssv2", "vjepa2-official-diving48"])
    p.add_argument("--train-cache-dir", required=True, type=str,
                   help="Directory of per-video .pt token caches for train split.")
    p.add_argument("--val-cache-dir", required=True, type=str,
                   help="Directory of per-video .pt token caches for val split.")
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--epochs", type=int, default=100,
                   help="Default 100 — matches V-JEPA 2 paper's attentive-probe training length.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3,
                   help="Base LR. Default 5e-3 matches paper's highest LR variant. "
                        "With --keep-pretrained, multiplied by 0.1 for continued-training.")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="Default 0.01 matches paper config (final_weight_decay also 0.01).")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Default 0 matches paper (no warmup).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--keep-pretrained", action=argparse.BooleanOptionalAction, default=False,
        help="Warm-start the AttentiveClassifier from the paper-ckpt probe[0] weights. "
             "Default False = paper-faithful random init.",
    )
    p.add_argument(
        "--warmstart-lr-scale", type=float, default=1.0,
        help="LR multiplier applied when --keep-pretrained is on. "
             "1.0 = paper-faithful (paper LR full strength on top of warm-started weights). "
             "0.1 = conservative continued-training regime. Ignored when --no-keep-pretrained.",
    )
    p.add_argument(
        "--selection-metric", default="val_loss",
        choices=["val_loss", "val_top1"],
        help="Which val metric drives best-checkpoint selection.",
    )
    p.add_argument(
        "--patience", type=int, default=0,
        help="Early-stopping patience in epochs. If >0, stop after N consecutive epochs "
             "without improvement in the selection metric. 0 = disabled (run to args.epochs).",
    )
    p.add_argument("--device", default="cuda:0",
                   help="Ignored under DDP — torchrun assigns cuda:LOCAL_RANK per process.")
    p.add_argument(
        "--ddp", action=argparse.BooleanOptionalAction, default=False,
        help="Distributed Data Parallel mode. Launch via torchrun --nproc_per_node=N. "
             "Each process pins to cuda:LOCAL_RANK, the model is wrapped in DDP, "
             "and the train DataLoader uses DistributedSampler. Checkpoints and logs "
             "are written only on rank 0; metrics are all-reduced across ranks.",
    )
    p.add_argument(
        "--amp-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
        help="Autocast dtype for forward+backward. Paper uses bfloat16 (use_bfloat16: true). "
             "float32 = full precision (slow, no autocast).",
    )
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers — these load .pt files from disk in parallel.")
    # wandb (optional)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


# -------------------- Dataset --------------------

class TokenCacheDataset(Dataset):
    """One item per video, returning ALL clips stacked as (num_clips, N, D).

    Used by both train and val. The train loop computes per-clip CE losses
    against the same label, sums across clips, then averages across the
    video batch — matches the paper's `[criterion(o, labels) for o in
    coutputs]` followed by per-segment ``.backward()`` (sum-of-segment
    gradients per video). The val loop instead softmax-averages across
    the clips per video and computes top-1 of the averaged distribution.

    Cache files contain tokens of shape (num_clips, N, D) — typically
    num_clips = num_segments × num_views (paper-faithful train uses
    num_views=1, val uses num_views=3).
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(p for p in self.cache_dir.glob("*.pt") if not p.name.startswith("cache_metadata"))
        if not self.files:
            raise FileNotFoundError(f"No *.pt token files under {cache_dir}")
        probe = torch.load(str(self.files[0]), map_location="cpu", weights_only=False)
        toks = probe["tokens"]
        if toks.dim() == 3:
            self.num_clips_per_video = toks.shape[0]
        elif toks.dim() == 2:
            self.num_clips_per_video = 1
        else:
            raise RuntimeError(f"Unexpected token tensor dim {toks.dim()}")
        self.n_videos = len(self.files)

    def __len__(self) -> int:
        return self.n_videos

    def __getitem__(self, i: int):
        blob = torch.load(str(self.files[i]), map_location="cpu", weights_only=False)
        toks = blob["tokens"]
        if toks.dim() == 2:
            toks = toks.unsqueeze(0)  # legacy single-clip → (1, N, D)
        return toks, int(blob["label"])


def _collate(batch):
    tokens = torch.stack([b[0] for b in batch], dim=0)   # (B, num_clips, N, D)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return tokens, labels


# -------------------- Model construction --------------------

def _build_classifier_from_paper(recognizer: str, device: str, *, keep_pretrained: bool):
    """Build an AttentiveClassifier matching the paper config, optionally warm-started."""
    import sys as _sys
    from pathlib import Path as _P
    _MM_ROOT = _P(os.environ.get("MM_ROOT", _P(__file__).resolve().parents[2]))
    VJEPA2_REPO = _P(os.environ.get("VJEPA2_REPO", _MM_ROOT / "external" / "vjepa2"))
    if str(VJEPA2_REPO) not in _sys.path:
        _sys.path.insert(0, str(VJEPA2_REPO))
    from src.models.attentive_pooler import AttentiveClassifier
    from oracle.scripts.vjepa2_official.wrapper import PROBE_CONFIGS

    cfg = PROBE_CONFIGS[recognizer]
    classifier = AttentiveClassifier(
        embed_dim=1024,
        num_heads=cfg["num_heads"],
        depth=cfg["num_probe_blocks"],
        num_classes=cfg["num_classes"],
        use_activation_checkpointing=False,
    ).to(device)

    if keep_pretrained:
        ckpt = torch.load(str(cfg["probe_ckpt"]), map_location="cpu", weights_only=False)
        probe0_sd = {k.replace("module.", ""): v for k, v in ckpt["classifiers"][0].items()}
        msg = classifier.load_state_dict(probe0_sd, strict=True)
        LOG.info("Warm-started AttentiveClassifier from %s (probe[0]): %s", cfg["probe_ckpt"], msg)
    else:
        LOG.info("AttentiveClassifier initialized from scratch (no warm-start).")

    return classifier, cfg


# -------------------- Training utilities --------------------

def _topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int) -> int:
    _, pred = logits.topk(k, dim=-1)
    return int(pred.eq(target.unsqueeze(-1)).any(dim=-1).sum().item())


def _cosine_with_warmup(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


# -------------------- Main --------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    # DDP init — must come before any cuda allocation. Sets each rank's device.
    is_ddp, local_rank, rank, world_size = _ddp_setup(args.ddp)
    is_rank0 = (rank == 0)
    if is_ddp:
        args.device = f"cuda:{local_rank}"
        if is_rank0:
            LOG.info("DDP enabled: world_size=%d, local_rank=%d, rank=%d, device=%s",
                     world_size, local_rank, rank, args.device)
    # Suppress repetitive logging from non-rank-0 ranks.
    if is_ddp and not is_rank0:
        logging.getLogger().setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    if is_rank0:
        output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TokenCacheDataset(Path(args.train_cache_dir))
    val_ds = TokenCacheDataset(Path(args.val_cache_dir))
    if is_rank0:
        LOG.info(
            "train cache: %d videos × %d clips/video = %d clip-forwards/epoch;  val cache: %d videos × %d clips/video",
            train_ds.n_videos, train_ds.num_clips_per_video,
            train_ds.n_videos * train_ds.num_clips_per_video,
            val_ds.n_videos, val_ds.num_clips_per_video,
        )

    # Note: --batch-size is in VIDEOS per RANK, not clips. Per-step clip-forwards across ranks =
    # world_size × batch_size × num_clips.
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=_collate, drop_last=False, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, args.batch_size),
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=_collate, persistent_workers=args.num_workers > 0,
    )

    classifier, cfg = _build_classifier_from_paper(
        args.recognizer, args.device, keep_pretrained=args.keep_pretrained,
    )
    classifier.train()
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        classifier = DDP(classifier, device_ids=[local_rank], find_unused_parameters=False)

    base_lr = args.lr * (args.warmstart_lr_scale if args.keep_pretrained else 1.0)
    LOG.info(
        "Base LR = %.5g (args.lr=%g  warmstart=%s  warmstart_lr_scale=%g)",
        base_lr, args.lr, args.keep_pretrained, args.warmstart_lr_scale,
    )
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    wandb_run = None
    if args.wandb_project and is_rank0:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or output_dir.name,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                config={
                    "recognizer": args.recognizer,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "effective_base_lr": base_lr,
                    "weight_decay": args.weight_decay,
                    "warmup_epochs": args.warmup_epochs,
                    "selection_metric": args.selection_metric,
                    "keep_pretrained": bool(args.keep_pretrained),
                    "num_classes": cfg["num_classes"],
                    "num_heads": cfg["num_heads"],
                    "num_probe_blocks": cfg["num_probe_blocks"],
                    "n_train": len(train_ds),
                    "n_val": len(val_ds),
                    "train_cache_dir": str(args.train_cache_dir),
                    "val_cache_dir": str(args.val_cache_dir),
                },
            )
            LOG.info("wandb initialized (project=%s, run=%s)", args.wandb_project, wandb_run.name)
        except Exception as e:
            LOG.warning("Failed to init wandb (%s); continuing without it.", e)
            wandb_run = None

    log_csv = output_dir / "train_log.csv"
    if is_rank0:
        with log_csv.open("w") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_top1", "val_loss", "val_top1", "val_top5", "lr", "elapsed_s"]
            )

    def _better(new_val_loss, new_val_top1, best_val_loss, best_val_top1):
        if args.selection_metric == "val_loss":
            return new_val_loss < best_val_loss
        return new_val_top1 > best_val_top1

    best_val_loss = float("inf")
    best_val_top1 = -1.0
    best_val_top5 = -1.0
    best_epoch = -1
    epochs_since_improvement = 0
    early_stopped = False
    t0 = time.time()
    global_step = 0
    val_loss = float("nan")
    val_top1 = 0.0
    val_top5 = 0.0
    lr = base_lr

    # Autocast setup — matches paper's `torch.cuda.amp.autocast(..., enabled=use_bfloat16)`.
    amp_enabled = args.amp_dtype != "float32"
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.amp_dtype]
    # bf16 doesn't need GradScaler; fp16 does. Set up both paths.
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    LOG.info("AMP: enabled=%s  dtype=%s  scaler=%s", amp_enabled, args.amp_dtype, "GradScaler" if scaler else "none")

    for epoch in range(args.epochs):
        classifier.train()
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        n_train_correct_clip = 0
        n_train_total_clip = 0
        n_train_total_video = 0
        sum_video_loss = 0.0
        for tokens, y in train_loader:
            # tokens: (B, num_clips, N, D), y: (B,)
            tokens = tokens.to(args.device, non_blocking=True)
            # cast bf16 cache → autocast dtype (or fp32 if disabled)
            tokens = tokens.to(dtype=amp_dtype) if amp_enabled else tokens.float()
            y = y.to(args.device, non_blocking=True)
            B, num_clips = tokens.shape[0], tokens.shape[1]

            lr = _cosine_with_warmup(global_step, total_steps, warmup_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                # Forward all clips of all videos as a flat batch
                flat_tokens = tokens.view(B * num_clips, *tokens.shape[2:])
                flat_logits = classifier(flat_tokens)                         # (B*num_clips, C)

                # Per-clip CE against the video label. reduction='none' lets us
                # reshape and sum across clips per video, then mean across batch —
                # matching paper's "sum per-segment losses, mean over batch" gradient.
                y_expanded = y.unsqueeze(1).expand(-1, num_clips).reshape(-1)
                per_clip_loss = nn.functional.cross_entropy(
                    flat_logits.float(), y_expanded, reduction='none'
                )                                                              # (B*num_clips,)
                loss = per_clip_loss.view(B, num_clips).sum(dim=1).mean()      # scalar

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Bookkeeping. `loss` is the mean-over-batch of per-video summed losses;
            # convert to mean-per-clip for an interpretable train_loss log.
            sum_video_loss += float(loss.item()) * B
            n_train_total_video += B
            n_train_correct_clip += _topk_correct(flat_logits.detach(), y_expanded, 1)
            n_train_total_clip += B * num_clips
            global_step += 1
        # DDP: all-reduce raw counters before reducing to ratios.
        if is_ddp:
            counters = torch.tensor(
                [sum_video_loss, n_train_total_video, n_train_correct_clip, n_train_total_clip],
                device=args.device, dtype=torch.float64,
            )
            counters = _reduce_sum(counters)
            sum_video_loss, n_train_total_video, n_train_correct_clip, n_train_total_clip = counters.tolist()
        # train_loss reported = mean per-clip CE (paper-style summed loss / num_clips).
        train_loss = (sum_video_loss / max(1, n_train_total_video)) / max(1, num_clips)
        train_top1 = n_train_correct_clip / max(1, n_train_total_clip)

        classifier.eval()
        n_val_top1 = 0
        n_val_top5 = 0
        n_val_total = 0
        val_loss_sum = 0.0
        with torch.inference_mode():
            for tokens, y in val_loader:
                # tokens: (B, num_clips, N, D)
                tokens = tokens.to(args.device, non_blocking=True)
                tokens = tokens.to(dtype=amp_dtype) if amp_enabled else tokens.float()
                y = y.to(args.device, non_blocking=True)
                B, num_clips = tokens.shape[0], tokens.shape[1]
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    flat = tokens.view(B * num_clips, *tokens.shape[2:])      # (B*num_clips, N, D)
                    logits_flat = classifier(flat)                            # (B*num_clips, C)
                probs = torch.softmax(logits_flat.float(), dim=-1).view(B, num_clips, -1).mean(dim=1)   # (B, C)
                avg_logits = torch.log(probs.clamp_min(1e-12))            # back to logit-like for loss
                val_loss_sum += float(loss_fn(avg_logits, y).item()) * B
                n_val_top1 += _topk_correct(avg_logits, y, 1)
                n_val_top5 += _topk_correct(avg_logits, y, 5)
                n_val_total += B
        if is_ddp:
            vcounters = torch.tensor(
                [val_loss_sum, n_val_total, n_val_top1, n_val_top5],
                device=args.device, dtype=torch.float64,
            )
            vcounters = _reduce_sum(vcounters)
            val_loss_sum, n_val_total, n_val_top1, n_val_top5 = vcounters.tolist()
        val_loss = val_loss_sum / max(1, n_val_total)
        val_top1 = n_val_top1 / max(1, n_val_total)
        val_top5 = n_val_top5 / max(1, n_val_total)

        elapsed = time.time() - t0
        if is_rank0:
            with log_csv.open("a") as f:
                csv.writer(f).writerow(
                    [epoch, f"{train_loss:.4f}", f"{train_top1:.4f}",
                     f"{val_loss:.4f}", f"{val_top1:.4f}", f"{val_top5:.4f}",
                     f"{lr:.6f}", f"{elapsed:.1f}"]
                )
            LOG.info(
                "epoch %d/%d  train_loss=%.4f train_top1=%.4f  val_loss=%.4f val_top1=%.4f val_top5=%.4f  lr=%.5f  t=%.1fs",
                epoch, args.epochs - 1,
                train_loss, train_top1, val_loss, val_top1, val_top5, lr, elapsed,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": train_loss,
                        "train/top1": train_top1,
                        "val/loss": val_loss,
                        "val/top1": val_top1,
                        "val/top5": val_top5,
                        "lr": lr,
                        "elapsed_s": elapsed,
                    },
                    step=epoch,
                )

        improved = _better(val_loss, val_top1, best_val_loss, best_val_top1)
        if improved:
            best_val_loss = val_loss
            best_val_top1 = val_top1
            best_val_top5 = val_top5
            best_epoch = epoch
            epochs_since_improvement = 0
            if is_rank0:
                # If wrapped in DDP, save the underlying module's state dict (no `module.` prefix).
                state_to_save = (
                    classifier.module.state_dict() if hasattr(classifier, "module") else classifier.state_dict()
                )
                torch.save(
                    {
                        "classifier": state_to_save,
                        "config": {
                            "recognizer": args.recognizer,
                            "embed_dim": 1024,
                            "num_heads": cfg["num_heads"],
                            "num_probe_blocks": cfg["num_probe_blocks"],
                            "num_classes": cfg["num_classes"],
                        },
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_top1": val_top1,
                        "val_top5": val_top5,
                    },
                    str(output_dir / "attentive_probe_best.pt"),
                )
        else:
            epochs_since_improvement += 1

        if args.patience > 0 and epochs_since_improvement >= args.patience:
            LOG.info(
                "Early stopping at epoch %d: %d epochs since last improvement "
                "(patience=%d). Best %s @ epoch %d.",
                epoch, epochs_since_improvement, args.patience, args.selection_metric, best_epoch,
            )
            early_stopped = True
            break

    final_metrics = {
        "final_val_loss": val_loss,
        "final_val_top1": val_top1,
        "final_val_top5": val_top5,
        "best_val_loss": best_val_loss,
        "best_val_top1": best_val_top1,
        "best_val_top5": best_val_top5,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "num_classes": cfg["num_classes"],
        "epochs_run": epoch + 1,
        "epochs_budgeted": args.epochs,
        "early_stopped": early_stopped,
        "patience": args.patience,
    }
    if is_rank0:
        (output_dir / "val_metrics.json").write_text(json.dumps(final_metrics, indent=2, sort_keys=True))

    config = {
        "recognizer": args.recognizer,
        "train_cache_dir": str(args.train_cache_dir),
        "val_cache_dir": str(args.val_cache_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "effective_base_lr": base_lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "selection_metric": args.selection_metric,
        "keep_pretrained": bool(args.keep_pretrained),
        "seed": args.seed,
        "num_classes": cfg["num_classes"],
        "num_heads": cfg["num_heads"],
        "num_probe_blocks": cfg["num_probe_blocks"],
        "wandb_project": args.wandb_project,
        "wandb_run_id": wandb_run.id if wandb_run is not None else None,
    }
    if is_rank0:
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    if wandb_run is not None:
        wandb_run.summary.update(final_metrics)
        wandb_run.finish()

    if is_ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    if not is_rank0:
        return 0

    print()
    print("=" * 80)
    print(f"output_dir:        {output_dir}")
    print(f"  recognizer:      {args.recognizer}  (warm-start: {args.keep_pretrained})")
    print(f"  N train:         {len(train_ds)}  N val: {len(val_ds)}")
    print(f"  selection:       {args.selection_metric}  best epoch: {best_epoch}")
    print(f"  best val loss:   {best_val_loss:.4f}")
    print(f"  best val top1/5: {best_val_top1:.4f} / {best_val_top5:.4f}")
    print(f"  final val:       loss={val_loss:.4f} top1={val_top1:.4f} top5={val_top5:.4f}")
    print(f"  saved:           attentive_probe_best.pt")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
