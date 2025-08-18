"""
Core API modules for Poetry RNN Autoencoder.

This package contains the main autoencoder class, training functions,
and generation utilities - extracted from the original enhanced_interface
for better maintainability and modularity.
"""

from .autoencoder import PoetryAutoencoder
from .training import train_hybrid_loss  
from .generation import PoetryGenerationConfig
from .denoising import (
    DenoisingConfig, 
    NoiseType, 
    DenoisingPoetryAutoencoder,
    create_denoising_autoencoder
)

__all__ = [
    'PoetryAutoencoder',
    'train_hybrid_loss',
    'PoetryGenerationConfig',
    'DenoisingConfig',
    'NoiseType', 
    'DenoisingPoetryAutoencoder',
    'create_denoising_autoencoder'
]