#!/usr/bin/env python
"""
MSS (Minimal Sufficient Subset) pseudo-label extraction script.

Extracts segment importance labels from videos using oracle verification.
Uses logit-based P(YES) > 0.5 for all decisions (single query per verification).

Usage:
    # Single video
    python mss_extract.py --video data/video.mp4 --label "picking up" --output output.json

    # Dataset (CSV with video_path, label columns)
    python mss_extract.py --dataset data/labels.csv --output-dir pseudo_labels/mss/

    # With custom parameters
    python mss_extract.py --dataset data/labels.csv --model qwen2.5-vl-7b \
        --r-runs 10 --delta-t 0.5

    # List available models
    python mss_extract.py --list-models
"""

import argparse
import atexit
import errno
import fcntl
import json
import logging
import os
import shutil
import signal
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mss import (
    MSSExtractor,
    MSSResult,
    assign_labels,
    segment_video,
)
from mss.extraction import MSSConfig
from mss.labeling import VideoLabels, save_video_labels, summarize_labels
from mss.masking import MaskConfig, MaskOperator
from mss.oracle import OracleCache, OracleConfig
from mss.dartmouth_oracle import DartmouthOracle
from mss.qwen_oracle import (
    MODEL_CONFIGS as QWEN_MODEL_CONFIGS,
    DARTMOUTH_MODEL_MAP,
    create_qwen_oracle,
    ParallelQwenOracle,
)
from mss.internvl3_oracle import (
    INTERNVL3_MODEL_CONFIGS,
    create_internvl3_oracle,
)
from mss.gemini_oracle import GEMINI_MODEL_CONFIGS, GeminiOracle
from mss.azure_oracle import AZURE_MODEL_CONFIGS, AzureOpenAIOracle

# Unified model registry: merge per-backend configs so --model choices and
# --list-models work uniformly across Qwen / InternVL3 / Gemini / Dartmouth /
# Azure OpenAI.
MODEL_CONFIGS = {
    **QWEN_MODEL_CONFIGS,
    **INTERNVL3_MODEL_CONFIGS,
    **GEMINI_MODEL_CONFIGS,
    **AZURE_MODEL_CONFIGS,
}


def _resolve_provider(model_key: str, explicit_provider: str) -> str:
    """Map a --model key to its backend provider.

    Explicit --provider always wins. Otherwise infer from the model key:
      - internvl3-* → internvl3
      - gemini-*    → gemini
      - dartmouth:* or qwen.*  → dartmouth
      - everything else (qwen2.5-vl-*, qwen3-vl-*, etc.) → local
    """
    if explicit_provider and explicit_provider != "auto":
        return explicit_provider
    if model_key is None:
        return "local"
    cfg = MODEL_CONFIGS.get(model_key, {})
    p = cfg.get("provider")
    if p:
        return p
    if model_key.startswith("internvl3"):
        return "internvl3"
    if model_key.startswith("gemini"):
        return "gemini"
    if model_key.startswith("dartmouth:"):
        return "dartmouth"
    if model_key.startswith("gpt-"):
        return "azure"
    return "local"

# Setup basic logging (will be reconfigured with file handler later).
# %(worker_tag)s is injected by _WorkerTagFilter so every line shows which
# worker emitted it (useful when tail-following many worker logs at once).
class _WorkerTagFilter(logging.Filter):
    tag: str = "-"

    def filter(self, record):
        record.worker_tag = self.tag
        return True


_WORKER_TAG_FILTER = _WorkerTagFilter()
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s [%(worker_tag)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
# Filters attached to a logger only fire for records emitted *at* that logger,
# not for records propagating up from descendants. Attach the filter to every
# handler instead so %(worker_tag)s renders for all log lines (including the
# many messages emitted before setup_logging() runs).
for _h in logging.getLogger().handlers:
    _h.addFilter(_WORKER_TAG_FILTER)
logger = logging.getLogger("mss_extract")


def setup_logging(log_file: Optional[Path] = None, verbose: bool = False):
    """Configure logging with optional file output.

    Args:
        log_file: Path to log file (None for console only)
        verbose: Enable DEBUG level logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Get root logger and mss loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Update existing handlers
    for handler in root_logger.handlers:
        handler.setLevel(level)

    # Keep noisy HTTP/LLM loggers at INFO even in verbose mode —
    # at DEBUG they dump full request/response bodies with base64 images.
    for noisy in ("urllib3", "httpcore", "httpx", "langchain", "openai"):
        logging.getLogger(noisy).setLevel(max(level, logging.INFO))

    # Add the worker-tag filter to every handler so the format string's
    # %(worker_tag)s renders even on stream handlers that bypass the root
    # logger's filter chain.
    for handler in root_logger.handlers:
        if _WORKER_TAG_FILTER not in handler.filters:
            handler.addFilter(_WORKER_TAG_FILTER)

    # Add file handler if log_file specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        file_handler.addFilter(_WORKER_TAG_FILTER)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


# ============================================================================
# Shared work-queue primitives
#
# Multiple slurm jobs can write to the same out_dir and avoid duplicating each
# other's work by claiming videos via atomic mkdir(.claims/<video_id>). Crashed
# claims (mtime older than the stale threshold) are reclaimable. A single
# results.jsonl is shared via fcntl.flock so the data-viz-client backend can
# read it without falling back to slow filesystem-counting.
# ============================================================================


# Set of video_ids this worker currently holds claims for. Populated by
# try_claim, drained by release_claim and the shutdown handler. Used so a
# preempted worker can hand its work back to the queue within the slurm
# grace period instead of stranding it for the full stale window.
_OWNED_CLAIMS: set = set()
_CLAIM_DIR_FOR_SHUTDOWN: Optional[Path] = None


def _release_owned_claims() -> None:
    """Release every claim this worker still holds. Idempotent + crash-safe."""
    if _CLAIM_DIR_FOR_SHUTDOWN is None:
        return
    held = list(_OWNED_CLAIMS)
    if not held:
        return
    for vid in held:
        try:
            shutil.rmtree(_CLAIM_DIR_FOR_SHUTDOWN / vid, ignore_errors=True)
        except Exception:
            pass
    _OWNED_CLAIMS.clear()


def install_shutdown_handler(claim_dir: Path) -> None:
    """Release in-flight claims on SIGTERM/SIGUSR1 and at normal exit.

    Slurm preemption sends SIGTERM with a small grace period (~30-60s on
    gpu_preempt). Without this handler, claims would survive until the stale
    threshold (default 20 min) — wasteful in a thin fleet. With this handler,
    the next worker to come online picks them up immediately.
    """
    global _CLAIM_DIR_FOR_SHUTDOWN
    _CLAIM_DIR_FOR_SHUTDOWN = claim_dir
    atexit.register(_release_owned_claims)

    def _handler(signum, _frame):
        logger.warning(
            f"Received signal {signum}; releasing {len(_OWNED_CLAIMS)} claim(s) before exit"
        )
        _release_owned_claims()
        # Restore default and re-raise so the parent shell sees the right exit.
        try:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        except Exception:
            sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            # Not all signals are available in all contexts (subprocesses, etc.)
            pass


def _claim_meta_dict(extra: Optional[dict] = None) -> dict:
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        **(extra or {}),
    }


def _claim_path(claim_dir: Path, video_id: str) -> Path:
    return claim_dir / video_id


def is_claim_fresh(claim_dir: Path, video_id: str, stale_seconds: float) -> bool:
    """Return True if a claim exists and is younger than stale_seconds."""
    p = _claim_path(claim_dir, video_id)
    try:
        age = time.time() - p.stat().st_mtime
    except FileNotFoundError:
        return False
    return age < stale_seconds


def try_claim(
    claim_dir: Path,
    video_id: str,
    stale_seconds: float,
    extra_meta: Optional[dict] = None,
) -> bool:
    """Atomically claim a video for processing.

    Returns True on success (caller owns the work). Returns False if another
    worker holds a fresh claim. If a stale claim exists, it is removed and
    reclaimed.
    """
    p = _claim_path(claim_dir, video_id)

    # Stale-claim sweep for this specific video before trying to claim
    try:
        age = time.time() - p.stat().st_mtime
        if age >= stale_seconds:
            shutil.rmtree(p, ignore_errors=True)
    except FileNotFoundError:
        pass

    try:
        p.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return False
        raise

    # Best-effort meta.json — losing this doesn't invalidate the claim
    try:
        meta_path = p / "meta.json"
        meta_path.write_text(json.dumps(_claim_meta_dict(extra_meta), indent=2))
    except OSError as exc:
        logger.debug(f"could not write claim meta for {video_id}: {exc}")

    _OWNED_CLAIMS.add(video_id)
    return True


def release_claim(claim_dir: Path, video_id: str) -> None:
    """Remove the claim dir for a video. Idempotent."""
    p = _claim_path(claim_dir, video_id)
    shutil.rmtree(p, ignore_errors=True)
    _OWNED_CLAIMS.discard(video_id)


def sweep_stale_claims(claim_dir: Path, stale_seconds: float) -> int:
    """Remove any claim dirs older than stale_seconds. Returns count removed."""
    if not claim_dir.exists():
        return 0
    now = time.time()
    removed = 0
    for entry in claim_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            age = now - entry.stat().st_mtime
        except FileNotFoundError:
            continue
        if age >= stale_seconds:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed


def scan_done_and_claimed(
    output_dir: Path,
    claim_dir: Optional[Path],
    stale_seconds: float,
) -> tuple:
    """Bulk-list done video_ids and fresh-claim video_ids in one readdir each.

    Replaces per-row stat() calls in shared-queue startup. For SSv2 train
    (168k videos) this drops the pending-rows scan from ~2 min to a few
    seconds because NFS readdir of one directory is dramatically cheaper
    than 100k+ separate `Path.exists()` syscalls.

    Returns (done_set, claimed_fresh_set). Stale claims are excluded — the
    caller relies on this to avoid skipping reclaimable videos.
    """
    done: set = set()
    try:
        with os.scandir(output_dir) as it:
            for entry in it:
                name = entry.name
                if not name.endswith(".json"):
                    continue
                if name == "config.json" or name == "results.jsonl":
                    continue
                # Strip the trailing ".json"
                done.add(name[:-5])
    except FileNotFoundError:
        pass

    claimed: set = set()
    if claim_dir is not None:
        cutoff = time.time() - stale_seconds
        try:
            with os.scandir(claim_dir) as it:
                for entry in it:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if st.st_mtime > cutoff:
                        claimed.add(entry.name)
        except FileNotFoundError:
            pass

    return done, claimed


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON to `path` atomically via temp + os.replace.

    Concurrent extractor + annotation writes (data-viz-client) rely on
    os.replace's atomicity guarantee on POSIX/NFS. Temp file lives in the
    target directory so the rename stays on the same filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{int(time.time() * 1e6)}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def locked_append_jsonl(path: Path, line: str) -> None:
    """Append one JSON line under an exclusive flock.

    Lock is held for the duration of one write+flush. NFSv4 (Discovery's
    mount) supports POSIX advisory locks. Lines are independently parseable,
    so a worker that dies between flock and write leaves the file consistent.
    """
    if not line.endswith("\n"):
        line = line + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_APPEND so concurrent writers always extend the file even if another
    # process truncates between open and write (it shouldn't, but cheap insurance).
    with open(path, "a", encoding="utf-8") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


def parse_args():
    """Parse command line arguments.

    Returns parsed args with CLI args taking priority over config files.
    Priority order (highest to lowest):
    1. Command-line arguments (always override)
    2. User-specified --config file
    3. Default config file (mss/default_config.json)
    """
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent.parent

    default_output_dir = project_dir / "pseudo_labels" / "mss"

    parser = argparse.ArgumentParser(
        description="Extract MSS pseudo-labels from videos using oracle verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single video
    python mss_extract.py --video data/video.mp4 --label "picking up" --output labels.json

    # Dataset CSV (columns: video_path, label)
    python mss_extract.py --dataset data/labels.csv --output-dir pseudo_labels/mss/

    # With custom model and parameters (CLI args override config)
    python mss_extract.py --dataset data/labels.csv \\
        --model qwen2.5-vl-32b \\
        --r-runs 15 \\
        --delta-t 0.5 \\
        --visible-gpus 0,1,2,3

Config priority: CLI args > --config file > default_config.json
""",
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--video", "-v", type=Path, help="Single video file path"
    )
    input_group.add_argument(
        "--dataset", "-d", type=Path, help="Dataset CSV file (columns: video_path, label)"
    )

    # Single video options
    parser.add_argument(
        "--label", "-l", type=str, help="Action label for single video mode"
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output JSON path for single video mode"
    )

    # Dataset options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Output directory for dataset mode (default: {default_output_dir})",
    )
    parser.add_argument(
        "--video-col",
        type=str,
        default="video_path",
        help="Column name for video paths in dataset CSV (default: video_path)",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Column name for labels in dataset CSV (default: label)",
    )

    # Model options - default=None to detect CLI override
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use for oracle",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--visible-gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1,2,3'). Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=True,
        help="Disable oracle result caching (default: True - caching disabled)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable oracle result caching (overrides --no-cache)",
    )

    # MSS parameters - default=None to detect CLI override
    parser.add_argument(
        "--delta-t",
        type=float,
        default=None,
        help="Segment length in seconds",
    )
    parser.add_argument(
        "--r-runs",
        type=int,
        default=None,
        help="Number of MSS runs",
    )

    # Score-based strategy parameters
    parser.add_argument(
        "--removal-strategy",
        type=str,
        default=None,
        choices=["score", "random"],
        help="Removal strategy: 'score' (deterministic) or 'random' (legacy)",
    )
    parser.add_argument(
        "--tau-final",
        type=float,
        default=None,
        help="Min removal confidence for removability (default: 0.88, only if --use-thresholds)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Max confidence drop for removability (default: 0.20, only if --use-thresholds)",
    )
    parser.add_argument(
        "--tau-min",
        type=float,
        default=None,
        help="Min baseline confidence to continue extraction (default: 0.80, only if --use-thresholds)",
    )
    parser.add_argument(
        "--use-thresholds",
        action="store_true",
        default=None,
        help="Apply tau_final and delta thresholds (default: False, only P(YES) > 0.5 matters)",
    )
    parser.add_argument(
        "--direct-scoring",
        action="store_true",
        default=None,
        help="Use oracle's single-query importance scores instead of greedy removal loop",
    )

    # Masking options - default=None to detect CLI override
    parser.add_argument(
        "--mask-operator",
        type=str,
        default=None,
        choices=["cut", "blur", "freeze", "black", "mean"],
        help="Masking operator (cut=remove & concatenate, blur/black/freeze/mean=replace)",
    )

    # Prompt template options
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        choices=["auto", "generic", "ssv2", "k400", "diving48"],
        help=(
            "Prompt template for oracle verification. "
            "'auto' (default) detects from dataset columns. "
            "'ssv2' for Something-Something V2, 'k400' for Kinetics-400, "
            "'diving48' for Diving-48 (FINA-rule diving phases), "
            "'generic' for basic action verification."
        ),
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Load configuration from JSON file (CLI args still override)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from most recent run for this model (skip already processed videos)",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Resume from a specific output directory (e.g., pseudo_labels/mss/qwen3-vl-32b_20260203_135102/)",
    )
    parser.add_argument(
        "--shared-queue",
        action="store_true",
        help="Multiple slurm jobs share one out_dir and coordinate via atomic "
             "mkdir(.claims/<video_id>) so heterogeneous GPU types can pull "
             "from the same work queue without skew. Pair with --resume-dir "
             "(or rely on auto-resume creating a deterministic dir name).",
    )
    parser.add_argument(
        "--stale-claim-minutes",
        type=float,
        default=20.0,
        help="With --shared-queue: a claim older than this many minutes is "
             "treated as belonging to a dead worker and reclaimed. Default 20 "
             "(safe for A5500 4-way + qwen3-vl-32b worst-case OOM cascade).",
    )
    parser.add_argument(
        "--claim-dir",
        type=Path,
        default=None,
        help="With --shared-queue: directory holding claim markers. Defaults "
             "to <out_dir>/.claims so it sits inside the run dir (invisible "
             "to data-viz-client backend, which only treats subdirs with "
             "config.json as runs).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N videos from dataset",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without processing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "local", "dartmouth", "internvl3", "gemini", "azure"],
        help=(
            "Inference provider: 'auto' (default, infer from --model), 'local' "
            "(Qwen GPU), 'internvl3' (OpenGVLab GPU), 'gemini' (Google API), "
            "or 'dartmouth' (Dartmouth API)."
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel GPU processing (spawns multiple model instances across GPUs)",
    )
    parser.add_argument(
        "--gpus-per-instance",
        type=int,
        default=1,
        help="GPUs per model instance in parallel mode (default: 1 for 7B; use 2 for 32B models)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=60,
        help="Number of masked videos to process per forward pass (default: 60). "
             "Ignored for direct-scoring runs when --token-budget is set.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Direct-scoring only: max total input tokens per batch. When set, "
             "videos are pre-probed for duration + resolution and bin-packed via "
             "First-Fit Decreasing into batches whose summed estimated tokens stay "
             "under this budget. Overrides --batch-size for direct scoring.",
    )
    parser.add_argument(
        "--max-batch-videos",
        type=int,
        default=16,
        help="Hard cap on videos per bin during token-budget packing (default: 16). "
             "Prevents the long tail of small clips from being packed into a single "
             "bin that OOMs on per-video activation overhead despite fitting the "
             "token budget. Only affects --token-budget mode.",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=None,
        help="Subsample video frames so the rate sent to the oracle is <= this "
             "value (default: native fps, no subsampling). Recommended for K400 "
             "where native 30 fps gives ~15 frames per 0.5s segment — the model "
             "doesn't need that many frames to score per-segment importance. "
             "Setting --max-fps 12 matches SSv2's effective rate and gives ~6 "
             "frames/segment at ~2.5x throughput.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Cap per-frame resolution (Qwen smart_resize max_pixels). Default "
             "is Qwen's native 12,845,056 (HD-friendly). For K400 where actions "
             "are macro-scale, --max-pixels 102400 (~320x320, matches SSv2) "
             "gives ~3-5x throughput at minor quality cost.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=15,
        help="Max API requests per minute for dartmouth provider (default: 15)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Max concurrent API requests for dartmouth/gemini providers (default: 4)",
    )
    parser.add_argument(
        "--max-tiles-per-frame",
        type=int,
        default=1,
        help=(
            "InternVL3 only: max image tiles per video frame (default: 1). "
            "Increase to 2-4 for higher spatial detail at the cost of more "
            "vision tokens per frame."
        ),
    )
    parser.add_argument(
        "--azure-deployment",
        type=str,
        default=None,
        help=(
            "Azure OpenAI deployment name (overrides AZURE_OPENAI_DEPLOYMENT "
            "env var and MODEL_CONFIGS lookup). e.g. 'gpt-5.5'."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )

    return parser.parse_args()


def get_default_config_path() -> Path:
    """Get path to the default config file.

    Looks for YAML first, then JSON.
    """
    mss_dir = Path(__file__).parent / "mss"
    yaml_path = mss_dir / "default_config.yaml"
    json_path = mss_dir / "default_config.json"

    if yaml_path.exists():
        return yaml_path
    return json_path


def load_config_file(config_path: Path) -> dict:
    """Load configuration from JSON or YAML file.

    Args:
        config_path: Path to config file (.json, .yaml, or .yml)

    Returns:
        Configuration dictionary
    """
    suffix = config_path.suffix.lower()

    with open(config_path, "r") as f:
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError(
                    "PyYAML is required to load YAML config files. "
                    "Install with: pip install pyyaml"
                )
        else:
            return json.load(f)


def apply_config_to_args(args, config: dict, only_if_none: bool = False):
    """Apply config file settings to args.

    Args:
        args: Parsed arguments
        config: Configuration dictionary
        only_if_none: If True, only set values that are currently None in args
                      (used to let CLI args override config)
    """
    def set_if_allowed(attr: str, value):
        """Set attribute if allowed by only_if_none setting."""
        if only_if_none and getattr(args, attr, None) is not None:
            return  # CLI arg was provided, don't override
        if value is not None:
            setattr(args, attr, value)

    # GPU settings
    if "visible_gpus" in config:
        set_if_allowed("visible_gpus", config["visible_gpus"])
    if "num_gpus" in config:
        set_if_allowed("num_gpus", config["num_gpus"])

    # Model settings
    if "model" in config:
        set_if_allowed("model", config["model"])
    if "cache_dir" in config and config["cache_dir"]:
        if not only_if_none or args.cache_dir is None:
            args.cache_dir = Path(config["cache_dir"])

    # MSS config
    if "mss_config" in config:
        mss = config["mss_config"]
        if "delta_t" in mss:
            set_if_allowed("delta_t", mss["delta_t"])
        if "r_runs" in mss:
            set_if_allowed("r_runs", mss["r_runs"])
        if "seed" in mss:
            set_if_allowed("seed", mss["seed"])
        if "show_progress" in mss:
            # Handle show_progress -> no_progress inversion
            if not only_if_none or not args.no_progress:
                args.show_progress = mss["show_progress"]
        if "removal_strategy" in mss:
            set_if_allowed("removal_strategy", mss["removal_strategy"])
        if "tau_final" in mss:
            set_if_allowed("tau_final", mss["tau_final"])
        if "delta" in mss:
            set_if_allowed("delta", mss["delta"])
        if "tau_min" in mss:
            set_if_allowed("tau_min", mss["tau_min"])
        if "use_thresholds" in mss:
            set_if_allowed("use_thresholds", mss["use_thresholds"])
        if "direct_scoring" in mss:
            set_if_allowed("direct_scoring", mss["direct_scoring"])

    # Mask config
    if "mask_config" in config:
        mask = config["mask_config"]
        if "operator" in mask:
            set_if_allowed("mask_operator", mask["operator"])



def detect_dataset_type(df: pd.DataFrame) -> str:
    """Auto-detect dataset type from DataFrame columns.

    Args:
        df: Dataset DataFrame

    Returns:
        Dataset type: 'ssv2', 'diving48', 'k400', or 'generic'
    """
    columns = set(df.columns)

    # SSv2 has 'template' and 'placeholders' columns
    if "template" in columns:
        return "ssv2"

    # Diving-48 has 'class_id' + 'raw_class_name' columns (built by
    # scripts/oracle_agreement/build_diving48_csv.py)
    if "raw_class_name" in columns and "class_id" in columns:
        return "diving48"

    # K400 typically has just video_path and label (or class)
    # Could add more heuristics here if needed
    return "generic"


_SSV2_PLACEHOLDER_LOOKUP: Optional[Dict[str, Dict[str, str]]] = None


def _load_ssv2_placeholder_lookup() -> Dict[str, Dict[str, str]]:
    """Build a video_id → {label, template, placeholders} lookup from the SSv2
    CSVs that DO carry placeholder info (val_full, train_full, val_pilot50,
    val_sample). Used to backfill the SSv2 prompt's `OBJECT(S)` slot when the
    input CSV lacks a `placeholders` column (e.g., the official test split or
    pilots sampled from it).

    Returns an empty dict if no source CSVs are available; callers fall back
    to "unknown" placeholders in that case.
    """
    global _SSV2_PLACEHOLDER_LOOKUP
    if _SSV2_PLACEHOLDER_LOOKUP is not None:
        return _SSV2_PLACEHOLDER_LOOKUP

    lookup: Dict[str, Dict[str, str]] = {}
    candidates = [
        Path("data/csvs/ssv2/val_full.csv"),
        Path("data/csvs/ssv2/train_full.csv"),
        Path("data/csvs/ssv2/val_pilot50.csv"),
        Path("data/csvs/ssv2/val_sample.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not {"id", "label", "placeholders"}.issubset(df.columns):
            continue
        for _, row in df.iterrows():
            vid = str(row["id"])
            if vid in lookup:
                continue  # earlier source wins (val_full preferred over train_full)
            lookup[vid] = {
                "label": str(row["label"]),
                "template": str(row.get("template", "")),
                "placeholders": str(row["placeholders"]),
            }
    _SSV2_PLACEHOLDER_LOOKUP = lookup
    return lookup


def build_prompt_context(
    template_type: str,
    action_label: str,
    row_metadata: Optional[dict] = None,
    direct_scoring: bool = False,
) -> Optional[dict]:
    """Build prompt context based on template type and row data.

    Args:
        template_type: 'ssv2', 'k400', or 'generic'
        action_label: The action label (for SSv2, this already has objects substituted)
        row_metadata: Additional metadata from dataset row
        direct_scoring: If True, use direct scoring prompts instead of MSS verification

    Returns:
        Prompt context dict or None for generic template
    """
    # Per-row override: if the row metadata carries a `dataset` hint (used
    # by the oracle-agreement mixed sample CSV), use it to pick the right
    # template even when the CLI was run with --prompt-template auto on a
    # heterogeneous dataset.
    if row_metadata and row_metadata.get("dataset"):
        ds = str(row_metadata["dataset"]).lower().strip()
        if ds == "ssv2":
            template_type = "ssv2"
        elif ds == "k400":
            template_type = "k400"
        elif ds == "diving48":
            template_type = "diving48"
        elif ds == "charades":
            # No bespoke prompt for charades; fall back to generic. Direct-scoring
            # mode picks up the standard direct_scoring.md template.
            template_type = "generic"

    if template_type == "ssv2":
        # For SSv2, the action_label already has objects substituted
        # e.g., "spinning cube that quickly stops spinning"
        # Use it directly as the action_template
        placeholders_str = "unknown"
        action_template = action_label

        placeholders_raw = None
        if row_metadata and "placeholders" in row_metadata and row_metadata["placeholders"]:
            placeholders_raw = row_metadata["placeholders"]

        # Backfill from val_full/train_full when the input CSV lacks placeholders
        # (e.g., test split, or a pilot resampled from test). The lookup also
        # provides the substituted `label` so we keep ACTION and OBJECT(S) in
        # sync — otherwise we'd render "Pushing something..." next to a real
        # object name.
        if not placeholders_raw and row_metadata:
            vid = row_metadata.get("video_id") or row_metadata.get("id")
            if vid is not None:
                hit = _load_ssv2_placeholder_lookup().get(str(vid))
                if hit:
                    placeholders_raw = hit["placeholders"]
                    action_template = hit["label"]

        if placeholders_raw:
            placeholders = placeholders_raw
            if isinstance(placeholders, str):
                try:
                    placeholders = json.loads(placeholders)
                except json.JSONDecodeError:
                    pass
            if isinstance(placeholders, list):
                placeholders_str = ", ".join(str(p) for p in placeholders)
            else:
                placeholders_str = str(placeholders)

        template_id = "direct_scoring_ssv2" if direct_scoring else "mss_verification_ssv2"
        return {
            "action_template": action_template,
            "placeholders": placeholders_str,
            "prompt_template_id": template_id,
        }

    elif template_type == "k400":
        template_id = "direct_scoring_k400" if direct_scoring else "mss_verification_k400"
        return {
            "action_label": action_label,
            "prompt_template_id": template_id,
        }

    elif template_type == "diving48":
        # Diving-48 only has a direct-scoring variant; greedy-removal mode falls
        # back to the generic mss_verification template.
        if direct_scoring:
            return {
                "action_label": action_label,
                "prompt_template_id": "direct_scoring_diving48",
            }
        return None

    # Generic
    if direct_scoring:
        return {
            "action_label": action_label,
            "prompt_template_id": "direct_scoring",
        }
    return None


def _build_direct_scoring_batches(
    pending_rows: list,
    args,
    logger,
) -> tuple[list[list], Optional[list[int]]]:
    """Group pending_rows into batches for direct-scoring forward passes.

    If --token-budget is set, probe each video for (duration, w, h), estimate
    input tokens via Qwen smart-resize + patch math, and First-Fit-Decreasing
    pack into bins ≤ budget. Otherwise fall back to flat --batch-size N.

    Returns (batches, per_batch_costs). per_batch_costs is None for flat mode.
    """
    if args.token_budget is None:
        bs = args.batch_size
        return [pending_rows[i:i + bs] for i in range(0, len(pending_rows), bs)], None

    from mss.cost_estimation import estimate_input_tokens, probe_video_dimensions
    from mss.bin_packer import pack_first_fit_decreasing

    logger.info(
        f"Token-budget batching: probing {len(pending_rows)} videos for "
        f"duration + resolution..."
    )
    costs: list[int] = []
    probe_failures = 0
    # Apply caps to the cost estimate so the budget number reflects what Qwen
    # will actually see, not the native video.
    max_fps = getattr(args, "max_fps", None)
    max_pixels = getattr(args, "max_pixels", None)
    for r in pending_rows:
        try:
            duration, w, h, fps = probe_video_dimensions(Path(r["video_path"]))
            effective_fps = min(fps, max_fps) if max_fps is not None else fps
            kwargs = {}
            if max_pixels is not None:
                kwargs["max_pixels"] = max_pixels
            costs.append(estimate_input_tokens(duration, w, h, effective_fps, **kwargs))
        except Exception as e:
            # Fall back to a budget-sized cost so it gets its own bin and the
            # runtime OOM-halving fallback handles whatever's actually wrong.
            logger.warning(f"Probe failed for {r['video_path']}: {e}")
            costs.append(args.token_budget)
            probe_failures += 1

    bins = pack_first_fit_decreasing(
        costs, args.token_budget, max_items_per_bin=args.max_batch_videos
    )
    if probe_failures:
        logger.warning(f"{probe_failures} probe failures fell through to per-video bins")

    sizes = [len(b.indices) for b in bins]
    logger.info(
        f"Bin-packed into {len(bins)} batches "
        f"(min={min(b.total_cost for b in bins):,} tok, "
        f"max={max(b.total_cost for b in bins):,} tok, "
        f"max_size={max(sizes)} videos, avg_size={sum(sizes) / len(bins):.1f} videos, "
        f"cap={args.max_batch_videos})"
    )

    batches = [[pending_rows[i] for i in b.indices] for b in bins]
    return batches, [b.total_cost for b in bins]


def process_single_video(
    video_path: Path,
    action_label: str,
    extractor: MSSExtractor,
    video_index: Optional[int] = None,
    total_videos: Optional[int] = None,
    metadata: Optional[dict] = None,
    prompt_context: Optional[dict] = None,
) -> VideoLabels:
    """Process a single video and return labels.

    Args:
        video_path: Path to video file
        action_label: Ground-truth action label
        extractor: MSS extractor instance
        video_index: Current video index (for progress display)
        total_videos: Total number of videos (for progress display)
        metadata: Additional dataset-specific metadata to save with labels
        prompt_context: Optional dict with additional prompt variables
                       (e.g., action_template, placeholders for SSv2)

    Returns:
        VideoLabels with extraction results
    """
    # Extract MSS
    mss_result = extractor.extract(
        video_path,
        action_label,
        video_index=video_index,
        total_videos=total_videos,
        prompt_context=prompt_context,
    )

    # Generate labels (binary: kept in MSS = Important, removed = Unimportant)
    labels = VideoLabels.from_mss_result(
        mss_result, video_path, action_label, metadata=metadata,
    )

    return labels


def main():
    args = parse_args()

    # Handle list-models
    if args.list_models:
        print("Available models:")
        for key, config in MODEL_CONFIGS.items():
            ident = config.get("hf_name") or config.get("api_name") or "(remote)"
            provider = config.get("provider", "local")
            print(f"  {key}: {ident}  [{provider}]")
        return

    # Setup verbose logging early so config loading messages appear
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configs in priority order (lowest to highest):
    # 1. Default config (mss/default_config.json)
    # 2. User-specified --config file
    # 3. CLI args (already in args, only_if_none=True preserves them)

    default_config_path = get_default_config_path()
    if default_config_path.exists():
        logger.debug(f"Loading default config from {default_config_path}")
        default_config = load_config_file(default_config_path)
        apply_config_to_args(args, default_config, only_if_none=True)
    else:
        logger.warning(f"Default config not found: {default_config_path}")

    # Load user config file if provided (still respects CLI args)
    if args.config:
        if not args.config.exists():
            raise ValueError(f"Config file not found: {args.config}")
        logger.info(f"Loading user config from {args.config}")
        user_config = load_config_file(args.config)
        apply_config_to_args(args, user_config, only_if_none=True)

    # Handle show_progress: --no-progress flag overrides config
    if args.no_progress:
        args.show_progress = False
    elif not hasattr(args, "show_progress"):
        args.show_progress = True

    # Resolve provider before any setup that depends on it. After this point,
    # args.provider is one of: "local" | "dartmouth" | "internvl3" | "gemini".
    args.provider = _resolve_provider(args.model, args.provider)
    logger.info(f"Resolved provider: {args.provider}")

    # Set CUDA_VISIBLE_DEVICES (skip for API providers)
    if args.provider in ("dartmouth", "gemini", "azure"):
        logger.info(f"{args.provider} provider: skipping GPU setup (API-based inference)")
    elif args.visible_gpus and not args.parallel:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.visible_gpus}")
    elif args.parallel and args.visible_gpus:
        logger.info(f"Parallel mode: workers will use GPUs {args.visible_gpus}")

    # Set provider-specific defaults
    if args.provider == "dartmouth" and args.model is None:
        args.model = "dartmouth:qwen3-vl-32b"
        logger.info(f"Dartmouth provider: defaulting model to {args.model}")

    # Validate input
    if args.video and not args.label:
        raise ValueError("--label is required when using --video")
    if not args.video and not args.dataset:
        raise ValueError("Either --video or --dataset is required")

    # Build configurations from final args
    oracle_config = OracleConfig()

    # Build MSSConfig, using defaults for any None values
    mss_kwargs = {
        "delta_t": args.delta_t,
        "r_runs": args.r_runs,
        "seed": args.seed,
        "show_progress": args.show_progress,
    }
    # Only add score-strategy params if they're set (otherwise use dataclass defaults)
    if args.removal_strategy is not None:
        mss_kwargs["removal_strategy"] = args.removal_strategy
    if hasattr(args, "tau_final") and args.tau_final is not None:
        mss_kwargs["tau_final"] = args.tau_final
    if args.delta is not None:
        mss_kwargs["delta"] = args.delta
    if args.tau_min is not None:
        mss_kwargs["tau_min"] = args.tau_min
    if args.use_thresholds is not None:
        mss_kwargs["use_thresholds"] = args.use_thresholds
    if hasattr(args, "direct_scoring") and args.direct_scoring is not None:
        mss_kwargs["direct_scoring"] = args.direct_scoring
    if args.max_fps is not None:
        mss_kwargs["max_fps"] = args.max_fps
    if args.max_pixels is not None:
        mss_kwargs["max_pixels"] = args.max_pixels
    mss_config = MSSConfig(**mss_kwargs)

    mask_config = MaskConfig(
        operator=MaskOperator(args.mask_operator),
    )

    # Print configuration
    logger.info("=" * 60)
    logger.info("MSS Extraction Configuration")
    logger.info("=" * 60)
    logger.info(f"  Provider: {args.provider}")
    logger.info(f"  Model: {args.model}")
    if args.provider in ("local", "internvl3"):
        logger.info(f"  Visible GPUs: {args.visible_gpus or 'all'}")
        logger.info(f"  Num GPUs: {args.num_gpus}")
        logger.info(f"  Cache dir: {args.cache_dir}")
    if args.provider == "internvl3":
        logger.info(f"  Max tiles per frame: {args.max_tiles_per_frame}")
    logger.info(f"  Delta-t: {mss_config.delta_t}s")
    logger.info(f"  R runs: {mss_config.r_runs}")
    logger.info(f"  Removal strategy: {mss_config.removal_strategy}")
    logger.info(f"  Direct scoring: {mss_config.direct_scoring}")
    logger.info(f"  Use thresholds: {mss_config.use_thresholds}")
    if mss_config.use_thresholds:
        logger.info(f"    tau_final (min removal conf): {mss_config.tau_final}")
        logger.info(f"    delta (max drop): {mss_config.delta}")
        logger.info(f"    tau_min (min baseline): {mss_config.tau_min}")
    logger.info(f"  Decision threshold: P(YES) > 0.5 (logit-based)")
    logger.info(f"  Labeling: Binary (Important/Unimportant based on MSS membership)")
    logger.info(f"  Mask operator: {mask_config.operator.value}")
    logger.info(f"  Show progress: {args.show_progress}")

    if args.dry_run:
        logger.info("Dry run - exiting without processing")
        return

    # Setup cache (disabled by default, use --use-cache to enable)
    use_cache = args.use_cache and not args.no_cache
    if use_cache and args.dataset:
        cache_dir = args.output_dir / "cache"
        cache = OracleCache(cache_dir)
        logger.info(f"  Oracle cache: enabled ({cache_dir})")
    else:
        cache = OracleCache()  # In-memory only, no persistence
        logger.info("  Oracle cache: disabled (use --use-cache to enable)")

    # Determine max_new_tokens: direct scoring prompts produce much longer output
    # (full segments array vs. short YES/NO JSON)
    max_new_tokens = 4096
    if mss_config.direct_scoring:
        max_new_tokens = 4096
        logger.info(f"  Direct scoring: max_new_tokens={max_new_tokens} (extended for segment annotations)")

    # Load model and create oracle
    logger.info(f"Loading model {args.model}...")

    if args.provider == "dartmouth":
        # Dartmouth API mode: no local GPU required
        from dotenv import load_dotenv
        load_dotenv()

        chat_api_key = os.environ.get("DARTMOUTH_CHAT_API_KEY")
        if not chat_api_key:
            raise ValueError(
                "DARTMOUTH_CHAT_API_KEY must be set in "
                "environment or .env file for dartmouth provider"
            )

        # Resolve model name for Dartmouth API:
        # 1. "dartmouth:qwen3-vl-32b" -> lookup in MODEL_CONFIGS
        # 2. "qwen3-vl-32b" (local key) -> lookup in DARTMOUTH_MODEL_MAP
        # 3. "qwen.qwen3-vl-32b-instruct-fp8" (raw API name) -> use as-is
        model_config = MODEL_CONFIGS.get(args.model, {})
        if model_config.get("provider") == "dartmouth":
            api_model_name = model_config["hf_name"]
        elif args.model in DARTMOUTH_MODEL_MAP:
            api_model_name = DARTMOUTH_MODEL_MAP[args.model]
        else:
            api_model_name = args.model

        logger.info(
            f"Using Dartmouth API: model={api_model_name}, "
            f"max_concurrent={args.max_concurrent}, rate_limit={args.rate_limit}/min"
        )
        oracle = DartmouthOracle(
            model_name=api_model_name,
            chat_api_key=chat_api_key,
            config=oracle_config,
            cache=cache,
            max_concurrent=args.max_concurrent,
            requests_per_minute=args.rate_limit,
            max_tokens=max_new_tokens,
        )
    elif args.provider == "gemini":
        # Gemini API mode: no local GPU required
        from dotenv import load_dotenv
        load_dotenv()

        gemini_api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) must be set in "
                "environment or .env file for gemini provider"
            )

        # Resolve model name for Gemini API:
        #   "gemini-3-pro" → MODEL_CONFIGS lookup yields api_name
        #   raw API id (e.g. "gemini-2.5-pro-latest") → use as-is
        model_config = MODEL_CONFIGS.get(args.model, {})
        api_model_name = model_config.get("api_name") or args.model

        logger.info(
            f"Using Gemini API: model={api_model_name}, "
            f"max_concurrent={args.max_concurrent}, rate_limit={args.rate_limit}/min"
        )
        oracle = GeminiOracle(
            model_name=api_model_name,
            api_key=gemini_api_key,
            config=oracle_config,
            cache=cache,
            max_concurrent=args.max_concurrent,
            requests_per_minute=args.rate_limit,
            max_tokens=max_new_tokens,
        )
    elif args.provider == "azure":
        # Azure OpenAI / AI Foundry mode: no local GPU required.
        from dotenv import load_dotenv
        load_dotenv()

        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_api_version = os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
        )
        if not azure_endpoint or not azure_key:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set "
                "in environment or .env file for azure provider"
            )

        # Deployment name resolution:
        #   1. Explicit --azure-deployment / AZURE_OPENAI_DEPLOYMENT env var
        #   2. MODEL_CONFIGS lookup → api_name
        #   3. raw model key (e.g. "gpt-5.5")
        deployment_name = (
            getattr(args, "azure_deployment", None)
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or MODEL_CONFIGS.get(args.model, {}).get("api_name")
            or args.model
        )

        logger.info(
            f"Using Azure OpenAI: endpoint={azure_endpoint}, "
            f"deployment={deployment_name}, api_version={azure_api_version}, "
            f"max_concurrent={args.max_concurrent}, rate_limit={args.rate_limit}/min"
        )
        oracle = AzureOpenAIOracle(
            deployment_name=deployment_name,
            endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=azure_api_version,
            config=oracle_config,
            cache=cache,
            max_concurrent=args.max_concurrent,
            requests_per_minute=args.rate_limit,
            max_tokens=max_new_tokens,
        )
    elif args.provider == "internvl3":
        # InternVL3 local mode: tensor-parallel single instance.
        # Parallel-pool mode is not supported for InternVL3 — model sizes
        # (38B/78B) require multi-GPU sharding for a single instance, so
        # spawning N copies isn't memory-feasible on the available hardware.
        if args.parallel:
            logger.warning(
                "--parallel is not supported for InternVL3; falling back to "
                "single tensor-parallel instance with all visible GPUs"
            )
        # Honor --visible-gpus for tensor-parallel sharding
        if args.visible_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
            logger.info(f"InternVL3: set CUDA_VISIBLE_DEVICES={args.visible_gpus}")
        if args.visible_gpus:
            n_gpus = len(args.visible_gpus.split(","))
        elif args.num_gpus is not None:
            n_gpus = args.num_gpus
        else:
            import torch
            n_gpus = torch.cuda.device_count()
        oracle = create_internvl3_oracle(
            model_key=args.model,
            num_gpus=n_gpus,
            cache_dir=str(args.cache_dir),
            oracle_config=oracle_config,
            cache=cache,
            max_new_tokens=max_new_tokens,
            max_tiles_per_frame=args.max_tiles_per_frame,
        )
    elif args.parallel:
        # Parallel mode: spawn model instances across GPUs
        gpu_ids = None
        if args.visible_gpus:
            gpu_ids = [int(g) for g in args.visible_gpus.split(",")]
        gpus_per = getattr(args, "gpus_per_instance", 1)
        n_gpus = len(gpu_ids) if gpu_ids else args.num_gpus
        n_instances = n_gpus // gpus_per
        logger.info(f"Using PARALLEL mode: {n_instances} instance(s), {gpus_per} GPU(s) each")
        oracle = ParallelQwenOracle(
            model_key=args.model,
            gpu_ids=gpu_ids,
            gpus_per_instance=gpus_per,
            cache_dir=str(args.cache_dir),
            config=oracle_config,
            cache=cache,
            max_new_tokens=max_new_tokens,
            worker_batch_size=args.batch_size,
        )
    else:
        # Standard mode: single model with tensor parallelism
        oracle = create_qwen_oracle(
            model_key=args.model,
            num_gpus=args.num_gpus,
            cache_dir=str(args.cache_dir),
            oracle_config=oracle_config,
            cache=cache,
            max_new_tokens=max_new_tokens,
        )

    # Create extractor
    extractor = MSSExtractor(oracle, mss_config, mask_config, batch_size=args.batch_size)

    # Process single video
    if args.video:
        start_time = time.time()

        # Setup log file next to output
        output_path = args.output or args.video.with_suffix(".mss.json")
        log_file = output_path.with_suffix(".log")
        setup_logging(log_file, args.verbose)

        logger.info("=" * 60)
        logger.info("Single Video MSS Extraction")
        logger.info("=" * 60)
        logger.info(f"  Video: {args.video}")
        logger.info(f"  Label: {args.label}")

        labels = process_single_video(
            args.video, args.label, extractor,
            video_index=0, total_videos=1,
        )

        elapsed = time.time() - start_time

        # Save output (output_path already set above for log file)
        save_video_labels(labels, output_path)

        # Print detailed summary
        summary = summarize_labels(labels.segments)
        logger.info("=" * 60)
        logger.info("Extraction Results")
        logger.info("=" * 60)
        logger.info(f"  Precheck passed: {labels.mss_result.precheck_passed}")
        logger.info(f"  Precheck P(YES): {labels.mss_result.precheck_vote_yes:.3f}")
        logger.info(f"  Total segments: {summary.get('total_segments', 0)}")
        logger.info(f"  Important (kept in MSS): {summary.get('important_count', 0)}")
        logger.info(f"  Unimportant (removed): {summary.get('unimportant_count', 0)}")
        logger.info(f"  MSS runs: {len(labels.mss_result.mss_runs)}")
        logger.info(f"  Total oracle calls: {labels.mss_result.total_oracle_calls}")
        logger.info(f"  Processing time: {elapsed:.1f}s")
        logger.info(f"  Output saved to: {output_path}")
        logger.info(f"  Log saved to: {log_file}")

        return

    # Process dataset
    logger.info(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)

    if args.video_col not in df.columns:
        raise ValueError(f"Column '{args.video_col}' not found in dataset")
    if args.label_col not in df.columns:
        raise ValueError(f"Column '{args.label_col}' not found in dataset")

    # Apply limit if specified
    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited to first {args.limit} videos")

    # Determine prompt template type
    if args.prompt_template and args.prompt_template != "auto":
        template_type = args.prompt_template
        logger.info(f"Using prompt template: {template_type} (specified)")
    else:
        template_type = detect_dataset_type(df)
        logger.info(f"Using prompt template: {template_type} (auto-detected)")

    # Set the correct prompt_template_id on oracle_config so it's saved accurately
    direct_scoring = mss_config.direct_scoring
    if template_type == "ssv2":
        oracle_config.prompt_template_id = "direct_scoring_ssv2" if direct_scoring else "mss_verification_ssv2"
    elif template_type == "k400":
        oracle_config.prompt_template_id = "direct_scoring_k400" if direct_scoring else "mss_verification_k400"
    elif template_type == "diving48" and direct_scoring:
        oracle_config.prompt_template_id = "direct_scoring_diving48"
    elif direct_scoring:
        oracle_config.prompt_template_id = "direct_scoring"
    # else: keep default "mss_verification"

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.shared_queue and not args.resume_dir:
        logger.error(
            "--shared-queue requires --resume-dir so all workers write into "
            "the same run directory. Pass e.g. "
            "--resume-dir pseudo_labels/mss/<model>_<RUN_TS>/"
        )
        sys.exit(2)

    if args.resume_dir:
        # Resume from specific directory
        output_dir = args.resume_dir
        if not output_dir.exists():
            if args.shared_queue:
                # First worker into a shared-queue run creates the dir; the
                # later mkdir(exist_ok=True) is a no-op for everyone else.
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Shared queue: created run directory {output_dir}")
            else:
                logger.error(f"Resume directory does not exist: {output_dir}")
                sys.exit(1)
        # Extract original timestamp from directory name if possible
        dir_name = output_dir.name
        if "_" in dir_name:
            timestamp = dir_name.split("_", 1)[-1]  # e.g., "20260203_135102"
        logger.info(f"Resuming from: {output_dir}")
    elif args.resume:
        # Find most recent run for this model
        existing_dirs = sorted(
            args.output_dir.glob(f"{args.model}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if existing_dirs:
            output_dir = existing_dirs[0]
            # Extract original timestamp from directory name
            dir_name = output_dir.name
            if "_" in dir_name:
                timestamp = dir_name.split("_", 1)[-1]
            logger.info(f"Resuming from most recent run: {output_dir}")
        else:
            # No existing run, create new
            output_dir = args.output_dir / f"{args.model}_{timestamp}"
            logger.info(f"No existing run found, creating new: {output_dir}")
    else:
        # Create new timestamped directory
        output_dir = args.output_dir / f"{args.model}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Worker identity used for log prefixes and results.jsonl provenance.
    # Compute once so save_result writes the same dict for every video.
    worker_info = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    # Compact log prefix: "<short_host>/a<task>" for slurm, "<short_host>/<pid>" otherwise.
    short_host = worker_info["host"].split(".")[0] if worker_info["host"] else "?"
    if worker_info["slurm_array_task"] is not None:
        _WORKER_TAG_FILTER.tag = f"{short_host}/a{worker_info['slurm_array_task']}"
    else:
        _WORKER_TAG_FILTER.tag = f"{short_host}/{worker_info['pid']}"

    # Shared work-queue setup. With --shared-queue, multiple slurm jobs point
    # at the same output_dir and coordinate via atomic mkdir(.claims/<vid>).
    # Each worker uses a per-worker log file so concurrent writers don't
    # clobber each other; the shared extraction.log is reserved for the
    # legacy single-worker case.
    stale_seconds = max(1.0, float(args.stale_claim_minutes) * 60.0)
    claim_dir: Optional[Path] = None
    if args.shared_queue:
        claim_dir = (args.claim_dir if args.claim_dir is not None else output_dir / ".claims")
        claim_dir.mkdir(parents=True, exist_ok=True)
        removed = sweep_stale_claims(claim_dir, stale_seconds)
        if removed:
            logger.info(f"Swept {removed} stale claim(s) older than {args.stale_claim_minutes} min")
        # Release in-flight claims on preemption (SIGTERM) so the next worker
        # picks them up immediately instead of waiting for the stale window.
        install_shutdown_handler(claim_dir)

    # Expose parse-failure dump directory to the oracle (read by qwen_oracle
    # via env var so it survives multiprocessing fork into worker pool).
    parse_failures_dir = output_dir / "parse_failures"
    os.environ["MSS_PARSE_FAILURES_DIR"] = str(parse_failures_dir)

    # Setup log file in output directory. Per-worker file in shared mode so
    # concurrent jobs don't clobber each other's logs (mode="w" on the shared
    # extraction.log would wipe history every time a worker restarts).
    if args.shared_queue:
        log_file = output_dir / f"extraction.{socket.gethostname()}.{os.getpid()}.log"
    else:
        log_file = output_dir / "extraction.log"
    setup_logging(log_file, args.verbose)
    logger.info(f"Logging to: {log_file}")

    # Save configuration. In shared-queue mode multiple workers race here;
    # only the first one to land writes config.json (it's identical across
    # workers up to per-worker GPU IDs, which the website doesn't read).
    config_path = output_dir / "config.json"
    if args.shared_queue and config_path.exists():
        logger.info("Shared queue: config.json already present, leaving as-is")
    else:
        config_payload = {
            "provider": args.provider,
            "model": args.model,
            "visible_gpus": args.visible_gpus,
            "num_gpus": args.num_gpus,
            "cache_dir": str(args.cache_dir),
            "mss_config": mss_config.to_dict(),
            "oracle_config": oracle_config.to_dict(),
            "mask_config": mask_config.to_dict(),
            "dataset": str(args.dataset),
            "prompt_template": template_type,
            "batch_size": args.batch_size,
            "parallel": args.parallel,
            "gpus_per_instance": getattr(args, "gpus_per_instance", 1),
            "limit": args.limit,
            "timestamp": timestamp,
            "shared_queue": bool(args.shared_queue),
        }
        atomic_write_json(config_path, config_payload)

    # Track progress
    results_file = output_dir / "results.jsonl"
    processed = set()

    # Resume from checkpoint if requested. Shared-queue mode always behaves
    # as a resume: another worker may have populated results.jsonl already.
    if (args.resume or args.resume_dir or args.shared_queue) and results_file.exists():
        with open(results_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data["video_id"])
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
        logger.info(f"Resuming: {len(processed)} videos already processed")

    # Process videos
    total_videos = len(df)
    videos_to_process = total_videos - len(processed)

    logger.info(f"Processing {videos_to_process} videos ({len(processed)} already done)")

    # Outer progress bar for videos
    pbar_videos = tqdm(
        total=total_videos,
        initial=len(processed),
        desc="Videos",
        unit="video",
        position=0,
        ncols=120,
    )

    # Stats tracking
    already_processed = len(processed)
    stats = {
        "processed": len(processed),
        "failed": 0,
        "precheck_failed": 0,
        "total_oracle_calls": 0,
        "total_time": 0,
        "session_processed": 0,  # Videos processed in this session
    }

    # In shared-queue mode, pre-scan done JSONs + fresh claims via bulk
    # readdir. For 168k-video datasets this drops startup from ~2 min of NFS
    # stat calls to a few seconds. Stale claims are excluded so try_claim()
    # at process time can still reclaim them.
    done_set: set = set()
    claimed_set: set = set()
    if args.shared_queue:
        scan_start = time.time()
        done_set, claimed_set = scan_done_and_claimed(
            output_dir, claim_dir, stale_seconds
        )
        logger.info(
            f"Shared-queue scan: {len(done_set)} done, "
            f"{len(claimed_set)} in-flight ({time.time() - scan_start:.1f}s)"
        )

    # Collect rows to process (skipping already-processed). In shared-queue
    # mode we ALSO skip videos whose <video_id>.json already exists (another
    # worker finished them before our resume-load saw their results.jsonl
    # entry) and videos with a fresh claim. Stale claims are NOT skipped here
    # — try_claim() at process time will reclaim them, ensuring crashed work
    # gets retried by whichever worker grabs it first.
    pending_rows = []
    skipped_claimed = 0
    skipped_completed_by_other = 0
    for idx, row in df.iterrows():
        video_path = Path(row[args.video_col])
        video_id = video_path.stem
        if video_id in processed:
            continue
        if args.shared_queue:
            if video_id in done_set:
                processed.add(video_id)
                skipped_completed_by_other += 1
                continue
            if video_id in claimed_set:
                skipped_claimed += 1
                continue

        action_label = str(row[args.label_col])

        # Collect extra columns as metadata
        metadata = {}
        for col in df.columns:
            if col not in (args.video_col, args.label_col):
                val = row[col]
                if hasattr(val, "item"):
                    val = val.item()
                elif pd.isna(val):
                    val = None
                metadata[col] = val

        prompt_context = build_prompt_context(
            template_type, action_label, metadata,
            direct_scoring=mss_config.direct_scoring,
        )

        pending_rows.append({
            "idx": idx,
            "video_path": video_path,
            "action_label": action_label,
            "video_id": video_id,
            "metadata": metadata,
            "prompt_context": prompt_context,
        })

    if args.shared_queue:
        # Advance the bar past videos other workers already finished or claimed
        # so the displayed total reflects the global state, not just our slice.
        if skipped_completed_by_other:
            pbar_videos.update(skipped_completed_by_other)
        if skipped_claimed:
            pbar_videos.update(skipped_claimed)
        logger.info(
            f"Shared queue: {len(pending_rows)} videos to attempt, "
            f"{skipped_completed_by_other} already done by others, "
            f"{skipped_claimed} currently claimed by others"
        )

    # Helper to save one result (shared by both paths)
    def save_result(results_fp, video_info, labels, elapsed):
        video_id = video_info["video_id"]
        video_path = video_info["video_path"]
        action_label = video_info["action_label"]
        metadata = video_info["metadata"]
        row_idx = video_info["idx"]

        video_output = output_dir / f"{video_id}.json"
        # Inject elapsed time into the label dict before saving
        label_dict = labels.to_dict()
        label_dict["elapsed_s"] = round(elapsed, 2)
        # Atomic write so concurrent annotation writes (data-viz-client uses
        # the same temp+rename pattern) can't tear a partial JSON.
        atomic_write_json(video_output, label_dict)

        summary = summarize_labels(labels.segments)
        result_line = {
            "video_id": video_id,
            "video_path": str(video_path),
            "action_label": action_label,
            "timestamp": label_dict.get("timestamp"),
            "precheck_passed": labels.mss_result.precheck_passed,
            "precheck_vote_yes": labels.mss_result.precheck_vote_yes,
            "summary": summary,
            "segment_weights": [
                seg.get("frequency", seg.get("weight", 1.0 if seg.get("label") == "important" else 0.0))
                for seg in label_dict.get("segments", [])
            ],
            "oracle_calls": labels.mss_result.total_oracle_calls,
            "elapsed_s": round(elapsed, 2),
            # Worker provenance — preserves the audit trail after the
            # claim's meta.json is rmtree'd on success.
            "worker": worker_info,
        }
        if args.shared_queue:
            # Multi-writer path: append under flock to a single shared file
            # so the data-viz-client backend's fast-path summary reader keeps
            # working. results_fp is unused in this branch.
            locked_append_jsonl(results_file, json.dumps(result_line))
        else:
            results_fp.write(json.dumps(result_line) + "\n")
            results_fp.flush()

        if args.shared_queue and claim_dir is not None:
            release_claim(claim_dir, video_id)

        processed.add(video_id)
        stats["processed"] += 1
        stats["session_processed"] += 1
        stats["total_time"] += elapsed
        stats["total_oracle_calls"] += labels.mss_result.total_oracle_calls

        if not labels.mss_result.precheck_passed:
            stats["precheck_failed"] += 1
            logger.warning(
                f"[{row_idx + 1}/{total_videos}] Video '{video_id}' precheck failed - "
                f"oracle cannot reliably classify full video"
            )
        else:
            logger.info(
                f"[{row_idx + 1}/{total_videos}] Completed '{video_id}': "
                f"important={summary.get('important_count', 0)}, "
                f"unimportant={summary.get('unimportant_count', 0)}, "
                f"oracle_calls={labels.mss_result.total_oracle_calls}, "
                f"time={elapsed:.1f}s"
            )

    with open(results_file, "a") as results_fp:
        if mss_config.direct_scoring:
            # Build the list of batches up-front so the loop body is the same
            # whether we're using flat batch_size or token-budget bin-packing.
            batches, batch_costs = _build_direct_scoring_batches(
                pending_rows, args, logger
            )
            # Make batches mutable so we can backfill from upcoming batches
            # when claim contention shrinks the current one (flat mode only).
            batches = [list(b) for b in batches]
            target_batch_size = args.batch_size
            # Backfill is only safe in flat-batch mode. Token-budget batches
            # are sized to a token cap, so pulling extra rows could OOM.
            backfill_enabled = (
                args.shared_queue
                and claim_dir is not None
                and args.token_budget is None
            )
            logger.info(
                f"Direct scoring batch mode: processing {len(pending_rows)} videos "
                f"in {len(batches)} batches "
                f"({'token-budget' if args.token_budget else f'flat batch_size={args.batch_size}'}"
                f"{' + backfill' if backfill_enabled else ''})"
            )

            def _try_claim_row(r, batch_idx_for_meta) -> Optional[dict]:
                """Try to claim r; return r on success, None if contended/done.

                Handles the post-claim re-check for the narrow race where
                another worker finishes the same video between our pending-
                rows build and our try_claim.
                """
                if not try_claim(
                    claim_dir, r["video_id"], stale_seconds,
                    extra_meta={"batch_idx": batch_idx_for_meta},
                ):
                    return None
                if (output_dir / f"{r['video_id']}.json").exists():
                    release_claim(claim_dir, r["video_id"])
                    return None
                return r

            for batch_idx, batch_rows in enumerate(batches, start=1):
                if not batch_rows:
                    # Drained by an earlier batch's backfill — nothing to do.
                    continue

                # Shared-queue: claim each video right before processing.
                # Items losing the race are dropped — another worker has them.
                # If the result is short of the target size and we're in flat
                # mode, pull rows from upcoming batches so each forward pass
                # stays at full batch size.
                if args.shared_queue and claim_dir is not None:
                    original_size = len(batch_rows)
                    claimed_rows = []
                    skipped = 0
                    for r in batch_rows:
                        if (got := _try_claim_row(r, batch_idx)) is not None:
                            claimed_rows.append(got)
                        else:
                            skipped += 1

                    backfill_count = 0
                    if backfill_enabled and len(claimed_rows) < target_batch_size:
                        # batches[batch_idx] is the next batch (batch_idx is
                        # 1-indexed; list is 0-indexed → batches[batch_idx] is
                        # the NEXT entry).
                        next_idx = batch_idx
                        while (
                            len(claimed_rows) < target_batch_size
                            and next_idx < len(batches)
                        ):
                            future_batch = batches[next_idx]
                            remaining_in_future = []
                            for r in future_batch:
                                if len(claimed_rows) >= target_batch_size:
                                    remaining_in_future.append(r)
                                    continue
                                if (got := _try_claim_row(r, batch_idx)) is not None:
                                    claimed_rows.append(got)
                                    backfill_count += 1
                                else:
                                    skipped += 1
                            batches[next_idx] = remaining_in_future
                            next_idx += 1

                    if skipped:
                        # Advance the bar for items we attempted but lost.
                        pbar_videos.update(skipped)
                    if skipped or backfill_count:
                        from_current = len(claimed_rows) - backfill_count
                        logger.info(
                            f"Batch {batch_idx}: kept {len(claimed_rows)}/{target_batch_size} "
                            f"({from_current} from this batch's {original_size}, "
                            f"{backfill_count} backfilled, "
                            f"{skipped} lost to other workers)"
                        )
                    batch_rows = claimed_rows
                    if not batch_rows:
                        continue

                batch_videos = [
                    {
                        "video_path": r["video_path"],
                        "action_label": r["action_label"],
                        "video_id": r["video_id"],
                        "prompt_context": r["prompt_context"],
                        "video_index": r["idx"],
                        "total_videos": total_videos,
                    }
                    for r in batch_rows
                ]

                pbar_videos.set_description(f"Videos [batch {batch_idx}]")

                try:
                    batch_time = time.time()
                    mss_results = extractor.extract_batch_direct(batch_videos)
                    batch_elapsed = time.time() - batch_time
                    per_video_time = batch_elapsed / max(len(batch_rows), 1)

                    # Peak VRAM is measured by the oracle (worker-side in
                    # parallel mode, locally otherwise) and exposed via
                    # last_peak_mem_gb: {physical_gpu_id: peak_gb}.
                    _peak_map = getattr(oracle, "last_peak_mem_gb", None) or {}
                    if _peak_map:
                        _peak_str = ", ".join(
                            f"gpu{_g}={_peak_map[_g]:.1f}GB"
                            for _g in sorted(_peak_map.keys())
                        )
                        _max_peak = max(_peak_map.values())
                        cost_str = (
                            f", est_tokens={batch_costs[batch_idx - 1]:,}"
                            if batch_costs is not None else ""
                        )
                        logger.info(
                            f"Batch {batch_idx} peak VRAM: "
                            f"{_peak_str} (max={_max_peak:.1f}GB, "
                            f"batch_size={len(batch_rows)}{cost_str})"
                        )

                    for row_info, mss_result in zip(batch_rows, mss_results):
                        labels = VideoLabels.from_mss_result(
                            mss_result,
                            row_info["video_path"],
                            row_info["action_label"],
                            metadata=row_info["metadata"] if row_info["metadata"] else None,
                        )
                        save_result(results_fp, row_info, labels, per_video_time)
                        pbar_videos.update(1)

                    pbar_videos.set_postfix({
                        "done": stats["processed"],
                        "fail": stats["failed"],
                        "batch_t": f"{batch_elapsed:.1f}s",
                    })

                except Exception as e:
                    # On batch failure, fall back to sequential for this batch.
                    # Claims acquired above are still held — save_result releases
                    # on success, the except branch below releases on failure so
                    # another worker can retry.
                    logger.error(
                        f"Batch failed: {e}, falling back to sequential",
                        exc_info=args.verbose,
                    )
                    for row_info in batch_rows:
                        try:
                            video_start = time.time()
                            labels = process_single_video(
                                row_info["video_path"],
                                row_info["action_label"],
                                extractor,
                                video_index=row_info["idx"],
                                total_videos=total_videos,
                                metadata=row_info["metadata"] if row_info["metadata"] else None,
                                prompt_context=row_info["prompt_context"],
                            )
                            save_result(results_fp, row_info, labels, time.time() - video_start)
                        except Exception as e2:
                            stats["failed"] += 1
                            logger.error(
                                f"Failed '{row_info['video_id']}': {e2}",
                                exc_info=args.verbose,
                            )
                            if args.shared_queue and claim_dir is not None:
                                release_claim(claim_dir, row_info["video_id"])
                        finally:
                            pbar_videos.update(1)

        else:
            # Standard sequential processing
            for row_info in pending_rows:
                video_id = row_info["video_id"]
                action_label = row_info["action_label"]

                # Shared-queue: claim before processing. Lose the race → skip.
                # Re-check JSON existence after claiming to handle the narrow
                # race where another worker finished between pending-rows
                # build and our claim attempt.
                if args.shared_queue and claim_dir is not None:
                    if not try_claim(claim_dir, video_id, stale_seconds):
                        pbar_videos.update(1)
                        continue
                    if (output_dir / f"{video_id}.json").exists():
                        release_claim(claim_dir, video_id)
                        pbar_videos.update(1)
                        continue

                pbar_videos.set_description(
                    f"Videos [{video_id[:20]}...]" if len(video_id) > 20 else f"Videos [{video_id}]"
                )
                pbar_videos.set_postfix({
                    "label": action_label[:15] + "..." if len(action_label) > 15 else action_label,
                    "done": stats["processed"],
                    "fail": stats["failed"],
                })

                try:
                    video_start = time.time()
                    labels = process_single_video(
                        row_info["video_path"],
                        action_label,
                        extractor,
                        video_index=row_info["idx"],
                        total_videos=total_videos,
                        metadata=row_info["metadata"] if row_info["metadata"] else None,
                        prompt_context=row_info["prompt_context"],
                    )
                    save_result(results_fp, row_info, labels, time.time() - video_start)

                except Exception as e:
                    stats["failed"] += 1
                    logger.error(
                        f"[{row_info['idx'] + 1}/{total_videos}] Failed to process '{video_id}': {e}",
                        exc_info=args.verbose,
                    )
                    if args.shared_queue and claim_dir is not None:
                        release_claim(claim_dir, video_id)

                finally:
                    pbar_videos.update(1)

    pbar_videos.close()

    # Final summary
    session_processed = stats["session_processed"]
    avg_time = stats["total_time"] / max(session_processed, 1)
    logger.info("=" * 60)
    logger.info("MSS Extraction Complete")
    logger.info("=" * 60)
    logger.info(f"  Videos processed (this session): {session_processed}")
    logger.info(f"  Videos processed (total): {stats['processed']}/{total_videos}")
    if already_processed > 0:
        logger.info(f"  Videos skipped (already done): {already_processed}")
    logger.info(f"  Precheck failures: {stats['precheck_failed']}")
    logger.info(f"  Processing errors: {stats['failed']}")
    logger.info(f"  Total oracle calls: {stats['total_oracle_calls']}")
    logger.info(f"  Total time: {stats['total_time']:.1f}s")
    logger.info(f"  Avg time per video: {avg_time:.1f}s")
    logger.info(f"  Results saved to: {output_dir}")
    logger.info(f"  Log saved to: {log_file}")
    logger.info(f"  Oracle cache stats: {oracle.stats}")


if __name__ == "__main__":
    main()
