"""
Poetry RNN Autoencoder Package

A modular, production-grade implementation for preprocessing poetry text
and training RNN autoencoders for dimensionality reduction.

This package provides both high-level and low-level APIs:

High-Level API (Recommended):
- poetry_autoencoder(): One-line function for complete training
- RNN: High-level model interface with lazy initialization
- Configuration dataclasses with validation and presets

Low-Level API (Advanced Users):
- PoetryPreprocessor: Detailed preprocessing pipeline control
- AutoencoderDataset: PyTorch-compatible dataset with efficient loading
- Individual model components and trainers

Example Usage (High-Level):
    >>> from poetry_rnn.api import poetry_autoencoder
    >>> 
    >>> # One-line training
    >>> model = poetry_autoencoder("poems.json")
    >>> 
    >>> # Custom configuration
    >>> model = poetry_autoencoder(
    ...     data_path="poems.json",
    ...     design={"hidden_size": 512, "bottleneck_size": 64, "rnn_type": "lstm"},
    ...     training={"epochs": 30, "curriculum_phases": 4},
    ...     output_dir="./results"
    ... )
    >>> 
    >>> # Generate poetry
    >>> poem = model.generate("In the beginning", length=100)

Example Usage (Low-Level):
    >>> from poetry_rnn import PoetryPreprocessor, AutoencoderDataset
    >>> 
    >>> # Detailed preprocessing control
    >>> preprocessor = PoetryPreprocessor()
    >>> results = preprocessor.process_poems("poems.json")
    >>> 
    >>> # Create PyTorch dataset
    >>> dataset = AutoencoderDataset(
    ...     sequences=results['sequences'],
    ...     embedding_sequences=results['embedding_sequences'],
    ...     attention_masks=results['attention_masks'],
    ...     metadata=results['metadata'],
    ...     vocabulary=results['vocabulary']
    ... )

Author: Created through collaborative learning with Claude
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Poetry RNN Collaborative Project"

# High-level API imports
from .pipeline import PoetryPreprocessor
from .dataset import AutoencoderDataset, create_poetry_datasets, create_poetry_dataloaders
from .config import Config

# New High-Level API
from .api import poetry_autoencoder, RNN

# Core component imports
from .tokenization.poetry_tokenizer import PoetryTokenizer
from .embeddings.glove_manager import GLoVeEmbeddingManager
from .preprocessing.sequence_generator import SequenceGenerator
from .preprocessing.dataset_loader import PoetryDatasetLoader

# Configuration classes
from .config import (
    TokenizationConfig, 
    EmbeddingConfig, 
    ChunkingConfig, 
    CooccurrenceConfig
)

# Utility imports
from .utils.io import ArtifactManager

# Export all public classes and functions
__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # High-level API (New - Recommended)
    "poetry_autoencoder",
    "RNN",
    
    # High-level API (Legacy - Still Supported)
    "PoetryPreprocessor",
    "AutoencoderDataset", 
    "create_poetry_datasets",
    "create_poetry_dataloaders",
    
    # Core components
    "PoetryTokenizer",
    "GLoVeEmbeddingManager", 
    "SequenceGenerator",
    "PoetryDatasetLoader",
    
    # Configuration
    "Config",
    "TokenizationConfig",
    "EmbeddingConfig", 
    "ChunkingConfig",
    "CooccurrenceConfig",
    
    # Utilities
    "ArtifactManager"
]

# Package-level convenience functions
def quick_preprocess(poems_path, config=None, **kwargs):
    """
    Quick preprocessing function for simple use cases.
    
    Args:
        poems_path: Path to poetry JSON file
        config: Optional configuration (uses default if None)
        **kwargs: Additional arguments passed to process_poems()
        
    Returns:
        Preprocessing results dictionary
        
    Example:
        >>> results = quick_preprocess("poems.json", save_artifacts=True)
        >>> print(f"Generated {len(results['sequences'])} sequences")
    """
    preprocessor = PoetryPreprocessor(config=config)
    return preprocessor.process_poems(poems_path, **kwargs)


def load_dataset(artifacts_path, split="full", **kwargs):
    """
    Quick dataset loading function for simple use cases.
    
    Args:
        artifacts_path: Path to preprocessing artifacts
        split: Dataset split ("full", "train", "val", "test")
        **kwargs: Additional arguments passed to AutoencoderDataset
        
    Returns:
        AutoencoderDataset instance
        
    Example:
        >>> dataset = load_dataset("artifacts/", split="train")
        >>> print(f"Loaded {len(dataset)} training samples")
    """
    return AutoencoderDataset(artifacts_path=artifacts_path, split=split, **kwargs)


# Add convenience functions to __all__
__all__.extend(["quick_preprocess", "load_dataset"])