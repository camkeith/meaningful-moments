"""
Parallel classifier worker pool.

Mirrors ParallelQwenOracle's topology (N workers, each spanning `gpus_per_instance`
GPUs, communicating via input/output multiprocessing queues), but initializes
QwenClassifier instead of QwenOracle. Kept separate to avoid touching the
existing MSS infrastructure.

For the production config (4 GPUs, gpus_per_instance=4), this reduces to a single
worker that owns all 4 GPUs with tensor-parallel sharding — same shape as current
MSS direct-scoring runs.
"""

import logging
import os
import queue
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional

logger = logging.getLogger("classify.parallel")


_worker_classifier = None
_worker_gpu_ids: List[int] = []


def _init_worker(model_key: str, gpu_ids: List[int], cache_dir: str, max_new_tokens: int):
    global _worker_classifier, _worker_gpu_ids

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    _worker_gpu_ids = list(gpu_ids)

    import torch
    if torch.cuda.is_available():
        torch.cuda.init()

    from .qwen_classifier import create_classifier
    _worker_classifier = create_classifier(
        model_key=model_key,
        num_gpus=len(gpu_ids),
        cache_dir=cache_dir,
        max_new_tokens=max_new_tokens,
    )
    logger.info(f"Classifier worker ready on GPU(s) {gpu_ids}")


def _worker_loop(
    gpu_ids: List[int],
    model_key: str,
    cache_dir: str,
    max_new_tokens: int,
    input_q,
    output_q,
    batch_size: int,
):
    _init_worker(model_key, gpu_ids, cache_dir, max_new_tokens)

    while True:
        try:
            first = input_q.get(timeout=1.0)
        except queue.Empty:
            continue

        if first is None:
            break

        items = [first]
        while len(items) < batch_size:
            try:
                it = input_q.get_nowait()
                if it is None:
                    input_q.put(None)  # propagate shutdown to any other worker
                    break
                items.append(it)
            except queue.Empty:
                break

        try:
            results = _worker_classifier.classify_batch(items, batch_size=len(items))
            peak = dict(_worker_classifier.last_peak_mem_gb)
            # Map local CUDA indices to physical GPU ids for clarity.
            physical_peak = {}
            for local_idx, gb in peak.items():
                physical = _worker_gpu_ids[local_idx] if local_idx < len(_worker_gpu_ids) else local_idx
                physical_peak[physical] = gb

            for i, r in enumerate(results):
                r = dict(r)
                if i == 0:
                    r["_peak_mem_gb"] = physical_peak
                    r["_batch_n"] = len(items)
                output_q.put(r)
        except Exception as e:
            logger.exception(f"Worker error processing batch of {len(items)}: {e}")
            for it in items:
                output_q.put({"video_id": it["video_id"], "error": str(e)})


class ParallelClassifier:
    """Wraps N classifier workers behind a single classify(items) call."""

    def __init__(
        self,
        model_key: str,
        gpu_ids: List[int],
        gpus_per_instance: int,
        cache_dir: str,
        max_new_tokens: int = 128,
        worker_batch_size: int = 35,
    ):
        if len(gpu_ids) % gpus_per_instance != 0:
            raise ValueError(
                f"len(gpu_ids)={len(gpu_ids)} must be divisible by gpus_per_instance={gpus_per_instance}"
            )
        self.model_key = model_key
        self.gpu_ids = gpu_ids
        self.gpus_per_instance = gpus_per_instance
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.worker_batch_size = worker_batch_size

        self.gpu_groups = [
            gpu_ids[i:i + gpus_per_instance]
            for i in range(0, len(gpu_ids), gpus_per_instance)
        ]
        self.num_workers = len(self.gpu_groups)
        self._workers: Optional[list] = None
        self.last_peak_mem_gb: Dict[int, float] = {}

        logger.info(
            f"ParallelClassifier: {self.num_workers} worker(s), "
            f"gpus_per_instance={gpus_per_instance}, worker_batch_size={worker_batch_size}, "
            f"groups={self.gpu_groups}"
        )

    def _ensure_initialized(self):
        if self._workers is not None:
            return
        self._workers = []
        for group in self.gpu_groups:
            iq = Queue()
            oq = Queue()
            p = Process(
                target=_worker_loop,
                args=(group, self.model_key, self.cache_dir, self.max_new_tokens, iq, oq, self.worker_batch_size),
                daemon=True,
            )
            p.start()
            self._workers.append((p, iq, oq))
            logger.info(f"Started classifier worker for GPU(s) {group}")

    def classify(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Distribute items across workers; return results in the same order."""
        if not items:
            return []
        self._ensure_initialized()

        id_to_pos = {it["video_id"]: i for i, it in enumerate(items)}
        results: List[Optional[Dict[str, Any]]] = [None] * len(items)

        # Round-robin dispatch.
        for i, it in enumerate(items):
            _, iq, _ = self._workers[i % self.num_workers]
            iq.put(it)

        self.last_peak_mem_gb = {}
        collected = 0
        while collected < len(items):
            for _, _, oq in self._workers:
                try:
                    r = oq.get(timeout=0.1)
                except queue.Empty:
                    continue
                peak = r.pop("_peak_mem_gb", None)
                r.pop("_batch_n", None)
                if peak:
                    for gid, gb in peak.items():
                        prev = self.last_peak_mem_gb.get(gid, 0.0)
                        if gb > prev:
                            self.last_peak_mem_gb[gid] = gb

                pos = id_to_pos.get(r["video_id"])
                if pos is None:
                    logger.warning(f"Received result for unknown video_id={r.get('video_id')}")
                else:
                    results[pos] = r
                collected += 1

        return [r for r in results if r is not None]

    def shutdown(self):
        if self._workers is None:
            return
        for _, iq, _ in self._workers:
            iq.put(None)
        for p, _, _ in self._workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        self._workers = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
