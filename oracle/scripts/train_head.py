#!/usr/bin/env python
"""Phase B — generic linear/MLP head trainer over cached frozen-encoder features.

Reads ``shard_*.pt`` files from ``--train-cache-dir`` and ``--val-cache-dir``,
fits the head with AdamW + cosine schedule + linear warmup, and saves the best
checkpoint (by ``--selection-metric``) to disk. Supports both the VideoMAE
head (LayerNorm + Linear) and the V-JEPA 2 linear probe (Linear only) via
``--head-arch``.

Phase-A caches must come from the same ``--head-arch``'s ``cache_features.py``
so the cached feature semantics match the head's forward.

Output:
    <output-dir>/head_best.pt         — best-val checkpoint (only thing saved
                                        when --save-best-only)
    <output-dir>/head.pt              — final-epoch head (only if --no-save-best-only)
    <output-dir>/config.json          — args + cache shapes + hparams
    <output-dir>/train_log.csv        — per-epoch metrics
    <output-dir>/val_metrics.json     — final + best val top-1/top-5 + val loss
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

LOG = logging.getLogger("train_head")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--head-arch", required=True, choices=["videomae", "vjepa2", "vjepa2_official"],
        help=(
            "Head architecture: 'videomae' = LayerNorm + Linear; "
            "'vjepa2' = Linear only (HF port, post-AttentivePooler features); "
            "'vjepa2_official' = Linear only (paper-ckpt, post-AttentiveClassifier-pooler features). "
            "vjepa2 and vjepa2_official share the same LinearHead module — the difference is just "
            "which encoder/pooler the cached features came from."
        ),
    )
    p.add_argument("--train-cache-dir", required=True, type=str)
    p.add_argument(
        "--val-cache-dir", default=None, type=str,
        help="If omitted, falls back to a stratified 90/10 split of the train cache (legacy).",
    )
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--legacy-val-frac", type=float, default=0.1,
        help="Only used when --val-cache-dir is omitted.",
    )
    p.add_argument(
        "--model-id", default=None,
        help="Used only when --keep-pretrained-head is set (warm-start from HF weights).",
    )
    p.add_argument(
        "--keep-pretrained-head", action="store_true",
        help="Warm-start the head from the HF model's existing weights and use 1/10 LR.",
    )
    p.add_argument(
        "--selection-metric", default="val_loss",
        choices=["val_loss", "val_top1"],
        help="Which val metric drives best-checkpoint selection.",
    )
    p.add_argument(
        "--save-best-only", action=argparse.BooleanOptionalAction, default=True,
        help="If true (default), only head_best.pt is written; final-epoch head.pt is skipped.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers. Default 0 — dataset is in-memory tensors; workers add IPC overhead with no I/O win.",
    )
    # --- wandb (optional) ---
    p.add_argument("--wandb-project", default=None, help="Enable wandb logging under this project")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument(
        "--wandb-mode", default="online", choices=["online", "offline", "disabled"],
        help="Override wandb mode. Use 'offline' if the node has no internet.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


# -------------------- Head dispatch --------------------

def _load_head_module(head_arch: str):
    if head_arch == "videomae":
        from oracle.scripts.videomae.head import FrozenHead, save_head
        return FrozenHead, save_head
    if head_arch in ("vjepa2", "vjepa2_official"):
        from oracle.scripts.vjepa2.head import LinearHead, save_head
        return LinearHead, save_head
    raise ValueError(f"Unknown head-arch: {head_arch!r}")


def _build_head_fresh(head_arch: str, hidden_size: int, num_labels: int) -> nn.Module:
    HeadCls, _ = _load_head_module(head_arch)
    return HeadCls(hidden_size, num_labels)


def _build_head_warmstart(head_arch: str, model_id: str, hidden_size: int, num_labels: int) -> nn.Module:
    HeadCls, _ = _load_head_module(head_arch)
    if head_arch == "vjepa2_official":
        # Warm-start from the paper checkpoint's probe[0].linear (Meta's trained final layer).
        # `model_id` here is the recognizer key (e.g., "vjepa2-official-ssv2"), not an HF id.
        from oracle.scripts.vjepa2_official.wrapper import PROBE_CONFIGS
        if model_id not in PROBE_CONFIGS:
            raise ValueError(
                f"--model-id={model_id!r} is not a vjepa2_official recognizer key. "
                f"Known: {sorted(PROBE_CONFIGS)}"
            )
        cfg = PROBE_CONFIGS[model_id]
        LOG.info("Warm-starting V-JEPA 2 paper-ckpt head from %s (probe[0].linear) …", cfg["probe_ckpt"])
        ckpt = torch.load(str(cfg["probe_ckpt"]), map_location="cpu", weights_only=False)
        probe0_sd = {k.replace("module.", ""): v for k, v in ckpt["classifiers"][0].items()}
        lin_w = probe0_sd["linear.weight"]   # (num_classes, embed_dim)
        lin_b = probe0_sd["linear.bias"]     # (num_classes,)
        in_features = lin_w.shape[1]
        out_features = lin_w.shape[0]
        if in_features != hidden_size:
            raise ValueError(f"paper-ckpt hidden={in_features} ≠ cache hidden={hidden_size}")
        if out_features != num_labels:
            raise ValueError(f"paper-ckpt labels={out_features} ≠ cache labels={num_labels}")
        head = HeadCls(hidden_size, num_labels)
        with torch.no_grad():
            head.classifier.weight.copy_(lin_w.float())
            head.classifier.bias.copy_(lin_b.float())
        return head

    if head_arch == "videomae":
        from transformers import VideoMAEForVideoClassification
        LOG.info("Warm-starting VideoMAE head from %s …", model_id)
        m = VideoMAEForVideoClassification.from_pretrained(model_id, torch_dtype=torch.float32)
    else:
        from transformers import VJEPA2ForVideoClassification
        LOG.info("Warm-starting V-JEPA 2 head from %s …", model_id)
        m = VJEPA2ForVideoClassification.from_pretrained(model_id, torch_dtype=torch.float32)
    head = HeadCls.from_hf_model(m)
    if head.classifier.in_features != hidden_size:
        raise ValueError(f"HF hidden={head.classifier.in_features} ≠ cache hidden={hidden_size}")
    if head.classifier.out_features != num_labels:
        raise ValueError(f"HF labels={head.classifier.out_features} ≠ cache labels={num_labels}")
    del m
    return head


# -------------------- Cache loading --------------------

def _load_cache(cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor, list[str], int | None]:
    """Returns (features, labels, video_ids, num_labels_from_metadata_or_None)."""
    shards = sorted(cache_dir.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.pt files under {cache_dir}")
    feats_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    vids_all: list[str] = []
    for s in shards:
        blob = torch.load(str(s), map_location="cpu", weights_only=False)
        feats_all.append(blob["features"])
        labels_all.append(blob["labels"])
        vids_all.extend(list(blob["video_ids"]))
    features = torch.cat(feats_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    # Trust the recognizer-reported num_labels recorded at extraction time.
    num_labels_meta: int | None = None
    for meta_path in sorted(cache_dir.glob("cache_metadata_shard_*.json")):
        try:
            meta = json.load(open(meta_path))
            if meta.get("num_labels"):
                num_labels_meta = int(meta["num_labels"])
                break
        except Exception:
            continue
    return features, labels, vids_all, num_labels_meta


def _stratified_split(labels: torch.Tensor, val_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    classes = torch.unique(labels)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for c in classes.tolist():
        idx = (labels == c).nonzero(as_tuple=False).squeeze(-1)
        perm = torch.randperm(idx.numel(), generator=g)
        idx = idx[perm]
        n_val = max(1, int(round(idx.numel() * val_frac))) if idx.numel() > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long)


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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cache_dir = Path(args.train_cache_dir)
    LOG.info("Loading TRAIN cache from %s …", train_cache_dir)
    train_features, train_labels, _, num_labels_meta = _load_cache(train_cache_dir)
    n_train_cache, hidden_size = train_features.shape
    LOG.info(
        "Train cache: N=%d hidden_size=%d unique_labels=%d num_labels_meta=%s",
        n_train_cache, hidden_size, int(train_labels.unique().numel()), num_labels_meta,
    )

    if args.val_cache_dir:
        val_cache_dir = Path(args.val_cache_dir)
        LOG.info("Loading VAL cache from %s …", val_cache_dir)
        val_features, val_labels, _, val_num_labels_meta = _load_cache(val_cache_dir)
        if val_features.shape[1] != hidden_size:
            raise ValueError(
                f"Val hidden={val_features.shape[1]} ≠ train hidden={hidden_size}; "
                "caches must come from the same recognizer."
            )
        if val_num_labels_meta and num_labels_meta and val_num_labels_meta != num_labels_meta:
            raise ValueError(
                f"Val num_labels={val_num_labels_meta} ≠ train num_labels={num_labels_meta}"
            )
        train_x, train_y = train_features, train_labels
        LOG.info("Using external val cache: N_val=%d", val_features.shape[0])
    else:
        LOG.info("No --val-cache-dir; using legacy stratified split (frac=%.3f)", args.legacy_val_frac)
        train_idx, val_idx = _stratified_split(train_labels, args.legacy_val_frac, args.seed)
        train_x = train_features[train_idx]
        train_y = train_labels[train_idx]
        val_features = train_features[val_idx]
        val_labels = train_labels[val_idx]

    # Final num_labels: trust cache_metadata first, then per-arch default, then max(labels)+1.
    arch_defaults = {"videomae": 400, "vjepa2": None, "vjepa2_official": None}
    derived_max = int(max(train_labels.max().item(), val_labels.max().item())) + 1
    num_labels = num_labels_meta or arch_defaults.get(args.head_arch) or derived_max
    if num_labels < derived_max:
        raise ValueError(
            f"num_labels={num_labels} from cache metadata is smaller than max observed label "
            f"id+1={derived_max}. Cache extraction was likely truncated."
        )
    LOG.info("Effective num_labels=%d (from cache_metadata=%s)", num_labels, num_labels_meta)

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_features, val_labels)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Head
    if args.keep_pretrained_head:
        if args.model_id is None:
            raise ValueError("--keep-pretrained-head requires --model-id")
        head = _build_head_warmstart(args.head_arch, args.model_id, hidden_size, num_labels)
    else:
        head = _build_head_fresh(args.head_arch, hidden_size, num_labels)
    head = head.to(args.device)
    _, save_head_fn = _load_head_module(args.head_arch)

    base_lr = args.lr * (0.1 if args.keep_pretrained_head else 1.0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    # ----- wandb (optional) -----
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or output_dir.name,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                config={
                    "head_arch": args.head_arch,
                    "model_id": args.model_id,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "effective_base_lr": base_lr,
                    "weight_decay": args.weight_decay,
                    "warmup_epochs": args.warmup_epochs,
                    "selection_metric": args.selection_metric,
                    "keep_pretrained_head": bool(args.keep_pretrained_head),
                    "hidden_size": hidden_size,
                    "num_labels": num_labels,
                    "n_train": int(train_x.shape[0]),
                    "n_val": int(val_features.shape[0]),
                    "train_cache_dir": str(train_cache_dir),
                    "val_cache_dir": args.val_cache_dir,
                },
            )
            LOG.info("wandb initialized (project=%s, run=%s)", args.wandb_project, wandb_run.name)
        except Exception as e:
            LOG.warning("Failed to init wandb (%s); continuing without it.", e)
            wandb_run = None

    log_csv = output_dir / "train_log.csv"
    with log_csv.open("w") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_top1", "val_loss", "val_top1", "val_top5", "lr", "elapsed_s"]
        )

    def _selection_better(new_val_loss: float, new_val_top1: float,
                          best_val_loss: float, best_val_top1: float) -> bool:
        if args.selection_metric == "val_loss":
            return new_val_loss < best_val_loss
        return new_val_top1 > best_val_top1

    best_val_loss = float("inf")
    best_val_top1 = -1.0
    best_val_top5 = -1.0
    best_epoch = -1
    t0 = time.time()
    global_step = 0
    val_loss = float("nan")
    val_top1 = 0.0
    val_top5 = 0.0
    lr = base_lr

    for epoch in range(args.epochs):
        # ----- Train -----
        head.train()
        n_train_correct = 0
        n_train_total = 0
        sum_loss = 0.0
        for x, y in train_loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            lr = _cosine_with_warmup(global_step, total_steps, warmup_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            logits = head(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.item()) * x.shape[0]
            n_train_correct += _topk_correct(logits.detach(), y, 1)
            n_train_total += x.shape[0]
            global_step += 1
        train_loss = sum_loss / max(1, n_train_total)
        train_top1 = n_train_correct / max(1, n_train_total)

        # ----- Val -----
        head.eval()
        n_val_top1 = 0
        n_val_top5 = 0
        n_val_total = 0
        val_loss_sum = 0.0
        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)
                logits = head(x)
                val_loss_sum += float(loss_fn(logits, y).item()) * x.shape[0]
                n_val_top1 += _topk_correct(logits, y, 1)
                n_val_top5 += _topk_correct(logits, y, 5)
                n_val_total += x.shape[0]
        val_loss = val_loss_sum / max(1, n_val_total)
        val_top1 = n_val_top1 / max(1, n_val_total)
        val_top5 = n_val_top5 / max(1, n_val_total)

        elapsed = time.time() - t0
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

        if _selection_better(val_loss, val_top1, best_val_loss, best_val_top1):
            best_val_loss = val_loss
            best_val_top1 = val_top1
            best_val_top5 = val_top5
            best_epoch = epoch
            save_head_fn(head, output_dir / "head_best.pt")

    # Save final head only if not in best-only mode
    if not args.save_best_only:
        save_head_fn(head, output_dir / "head.pt")

    final_metrics = {
        "final_val_loss": val_loss,
        "final_val_top1": val_top1,
        "final_val_top5": val_top5,
        "best_val_loss": best_val_loss,
        "best_val_top1": best_val_top1,
        "best_val_top5": best_val_top5,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "n_train": int(train_x.shape[0]),
        "n_val": int(val_features.shape[0]),
        "hidden_size": hidden_size,
        "num_labels": num_labels,
        "epochs_run": args.epochs,
    }
    (output_dir / "val_metrics.json").write_text(json.dumps(final_metrics, indent=2, sort_keys=True))

    config = {
        "head_arch": args.head_arch,
        "train_cache_dir": str(train_cache_dir),
        "val_cache_dir": args.val_cache_dir,
        "model_id": args.model_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "effective_base_lr": base_lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "selection_metric": args.selection_metric,
        "save_best_only": bool(args.save_best_only),
        "seed": args.seed,
        "keep_pretrained_head": bool(args.keep_pretrained_head),
        "hidden_size": hidden_size,
        "num_labels": num_labels,
        "n_train_cache_total": n_train_cache,
        "n_train_used": int(train_x.shape[0]),
        "n_val_used": int(val_features.shape[0]),
        "wandb_project": args.wandb_project,
        "wandb_run_id": wandb_run.id if wandb_run is not None else None,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    if wandb_run is not None:
        wandb_run.summary.update(final_metrics)
        wandb_run.finish()

    print()
    print("=" * 80)
    print(f"output_dir:        {output_dir}")
    print(f"  head_arch:       {args.head_arch}")
    print(f"  N train cache:   {n_train_cache}  N train used: {train_x.shape[0]}  N val: {val_features.shape[0]}")
    print(f"  selection:       {args.selection_metric}  best epoch: {best_epoch}")
    print(f"  best val loss:   {best_val_loss:.4f}")
    print(f"  best val top1/5: {best_val_top1:.4f} / {best_val_top5:.4f}")
    print(f"  final val:       loss={val_loss:.4f} top1={val_top1:.4f} top5={val_top5:.4f}")
    print(f"  saved:           head_best.pt" + ("" if args.save_best_only else " + head.pt"))
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
