"""
Segmentation Module

Handles time series segmentation and windowing for activity recognition.
"""

from .window_segmenter import WindowSegmenter
from .adaptive_segmenter import AdaptiveSegmenter

__all__ = ["WindowSegmenter", "AdaptiveSegmenter"] 