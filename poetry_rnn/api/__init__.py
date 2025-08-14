"""
High-Level API for Poetry RNN Autoencoder

This module provides a simple, intuitive interface for training RNN autoencoders
on poetry text data. It abstracts away the complexity of the underlying system
while preserving full control for advanced users.

Main Interface:
    poetry_autoencoder() - One-line function to train an autoencoder
    RNN class - High-level model interface with lazy initialization

Configuration:
    ArchitectureConfig - Model architecture settings
    TrainingConfig - Training parameters and curriculum learning  
    DataConfig - Data processing and embedding settings

Factory Functions:
    design_autoencoder() - Create architecture configurations
    curriculum_learning() - Create training configurations
    fetch_data() - Create data configurations

Example Usage:
    >>> from poetry_rnn.api import poetry_autoencoder
    >>> 
    >>> # Simple one-line usage
    >>> model = poetry_autoencoder("poems.json")
    >>> 
    >>> # Advanced usage with custom configuration
    >>> model = poetry_autoencoder(
    ...     data_path="poems.json",
    ...     design={"hidden_size": 512, "bottleneck_size": 64, "rnn_type": "lstm"},
    ...     training={"epochs": 30, "curriculum_phases": 4},
    ...     output_dir="./results"
    ... )
    >>> 
    >>> # Generate poetry from trained model
    >>> poem = model.generate(seed_text="In the beginning", length=100)
"""

from .config import ArchitectureConfig, TrainingConfig, DataConfig
from .factories import design_autoencoder, curriculum_learning, fetch_data, preset_architecture, quick_training, production_training
from .main import RNN, poetry_autoencoder
from .utils import find_glove_embeddings, detect_data_format, auto_detect_device

__all__ = [
    # Main API
    'poetry_autoencoder',
    'RNN',
    
    # Configuration classes
    'ArchitectureConfig',
    'TrainingConfig', 
    'DataConfig',
    
    # Factory functions
    'design_autoencoder',
    'curriculum_learning',
    'fetch_data',
    'preset_architecture',
    'quick_training',
    'production_training',
    
    # Utilities
    'find_glove_embeddings',
    'detect_data_format',
    'auto_detect_device'
]