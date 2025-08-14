"""
RNN Autoencoder Models for Poetry Dimensionality Reduction

This module provides the core neural network architectures for the poetry
autoencoder project, implementing a theoretically-grounded approach to
dimensionality reduction with RNNs.

Architecture Overview:
    - VanillaRNNCell: Educational RNN cell implementation with clear mathematics
    - RNNEncoder: Sequences → Compressed representation (bottleneck)
    - RNNDecoder: Bottleneck → Reconstructed sequences
    - RNNAutoencoder: Complete encoder-decoder architecture

Theory Foundation:
    Based on theoretical analysis showing optimal compression from 300D → 15-20D
    for poetry embeddings, achieving O(ε^-600) → O(ε^-35) complexity reduction.
"""

from .rnn_cell import VanillaRNNCell
from .encoder import RNNEncoder
from .decoder import RNNDecoder
from .autoencoder import RNNAutoencoder

__all__ = [
    'VanillaRNNCell',
    'RNNEncoder', 
    'RNNDecoder',
    'RNNAutoencoder'
]
