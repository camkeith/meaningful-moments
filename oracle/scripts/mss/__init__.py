"""
MSS (Minimal Sufficient Subset) Pseudo-Labeling Framework.

This module implements counterfactual importance scoring for video segments
using oracle verification with logit-based P(YES) > 0.5 decision threshold.

Key components:
- segments: Video segmentation utilities
- masking: Frame masking operators (cut, blur, freeze, black, mean)
- oracle: Oracle interface with single logit-based query
- extraction: Greedy MSS extraction algorithm
- labeling: Binary importance labels (Important/Unimportant based on MSS membership)
"""

from .segments import Segment, segment_video
from .masking import MaskOperator, MaskingInfo, mask_video
from .oracle import OracleDecision, OracleResponse, OracleInterface
from .extraction import MSSExtractor, MSSResult
from .labeling import ImportanceLabel, assign_labels

__all__ = [
    "Segment",
    "segment_video",
    "MaskOperator",
    "MaskingInfo",
    "mask_video",
    "OracleDecision",
    "OracleResponse",
    "OracleInterface",
    "MSSExtractor",
    "MSSResult",
    "ImportanceLabel",
    "assign_labels",
]
