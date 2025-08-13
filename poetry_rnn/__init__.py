"""
Poetry RNN Autoencoder Package

A modular, production-grade implementation for preprocessing poetry text
and training RNN autoencoders for dimensionality reduction.

This package provides a complete pipeline from raw poetry JSON files to 
training-ready PyTorch datasets, with support for:

High-Level API:
- PoetryPreprocessor: One-line setup for complete preprocessing pipeline
- AutoencoderDataset: PyTorch-compatible dataset with efficient loading
- Configuration system: Centralized settings management

Core Components:
- Poetry-specific tokenization preserving semantic elements
- GLoVe embedding integration with vocabulary alignment  
- Sliding window chunking for maximum data preservation
- Co-occurrence matrix analysis for effective dimensionality
- Comprehensive testing and validation

Example Usage:
    >>> from poetry_rnn import PoetryPreprocessor, AutoencoderDataset
    >>> 
    >>> # Simple preprocessing
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
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Poetry RNN Collaborative Project"

# High-level API imports
from .pipeline import PoetryPreprocessor
from .dataset import AutoencoderDataset, create_poetry_datasets, create_poetry_dataloaders
from .config import Config

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
    
    # High-level API
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