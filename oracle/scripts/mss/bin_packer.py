"""First-Fit Decreasing bin-packer for token-budget batching.

FFD is a 11/9-OPT approximation for bin packing — for a token-budget batching
problem with hundreds of items, the gap to optimal is negligible. Items larger
than the budget go in their own bin and rely on the runtime OOM-halving fallback
to recover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class Bin:
    """A packed batch."""
    indices: List[int]   # original positions in the input list
    total_cost: int


def pack_first_fit_decreasing(
    costs: Sequence[int],
    budget: int,
    max_items_per_bin: int | None = None,
) -> List[Bin]:
    """Pack items into bins of capacity `budget` using First-Fit Decreasing.

    Args:
        costs: per-item cost (token count); positions are preserved as indices.
        budget: max total cost per bin.
        max_items_per_bin: optional hard cap on bin size. Useful when the cost
            metric (tokens) under-counts per-item overhead — e.g. a long tail
            of small videos can pack to budget but still OOM because each video
            has activation/KV cost that doesn't scale with its content size.

    Returns:
        List of Bin in pack order. Items exceeding `budget` are placed alone.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if max_items_per_bin is not None and max_items_per_bin < 1:
        raise ValueError(f"max_items_per_bin must be >= 1, got {max_items_per_bin}")

    indexed = sorted(enumerate(costs), key=lambda x: -x[1])

    bins: List[Bin] = []
    for orig_idx, cost in indexed:
        placed = False
        for b in bins:
            if b.total_cost + cost > budget:
                continue
            if max_items_per_bin is not None and len(b.indices) >= max_items_per_bin:
                continue
            b.indices.append(orig_idx)
            b.total_cost += cost
            placed = True
            break
        if not placed:
            bins.append(Bin(indices=[orig_idx], total_cost=cost))

    return bins
