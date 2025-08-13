"""
Preprocessing module for sequence preparation

Handles chunking, sequence generation, and dataset loading
for neural network training. Provides comprehensive functionality
for converting raw poetry datasets into training-ready sequences
with advanced chunking strategies that preserve 95%+ of data.
"""

from .sequence_generator import SequenceGenerator
from .dataset_loader import PoetryDatasetLoader

__all__ = [
    'SequenceGenerator',
    'PoetryDatasetLoader'
]