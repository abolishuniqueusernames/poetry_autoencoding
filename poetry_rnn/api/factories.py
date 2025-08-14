"""
Factory functions for creating API configurations.

These functions provide convenient ways to create configuration objects
with sensible defaults, validation, and preset options. They implement
the "configuration over code" philosophy while making common use cases
simple and discoverable.

Design Philosophy:
- Sensible defaults based on theoretical analysis
- Progressive disclosure (simple → advanced parameters)
- Immediate validation and helpful error messages
- Preset configurations for common architectures
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import math

from .config import ArchitectureConfig, TrainingConfig, DataConfig
from .utils import find_glove_embeddings, detect_data_format, auto_detect_device


def design_autoencoder(
    hidden_size: int = 512,
    bottleneck_size: int = 64,
    input_size: int = 300,
    rnn_type: str = 'lstm',
    num_layers: int = 1,
    dropout: float = 0.1,
    **kwargs
) -> ArchitectureConfig:
    """
    Create an architecture configuration for RNN autoencoder.
    
    This factory function provides sensible defaults based on theoretical
    analysis of optimal dimensionality reduction for poetry embeddings.
    
    Args:
        hidden_size: RNN hidden state dimension (default: 512)
        bottleneck_size: Compressed representation dimension (default: 64)
        input_size: Input embedding dimension (default: 300 for GLoVe)
        rnn_type: Type of RNN cell ('vanilla', 'lstm', 'gru')
        num_layers: Number of RNN layers (default: 1)
        dropout: Dropout probability for regularization (default: 0.1)
        **kwargs: Additional parameters for ArchitectureConfig
    
    Returns:
        ArchitectureConfig: Validated architecture configuration
        
    Raises:
        ValueError: If configuration parameters are invalid
        
    Example:
        >>> config = design_autoencoder(hidden_size=256, bottleneck_size=32)
        >>> print(f"Compression ratio: {config.get_compression_ratio():.1f}x")
        >>> print(f"Estimated parameters: {config.estimate_parameters():,}")
    """
    config = ArchitectureConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        bottleneck_size=bottleneck_size,
        rnn_type=rnn_type.lower(),
        num_layers=num_layers,
        dropout=dropout,
        **kwargs
    )
    
    # Validate configuration
    config.validate()
    
    # Print helpful information
    compression_ratio = config.get_compression_ratio()
    estimated_params = config.estimate_parameters()
    
    print(f"Architecture configured:")
    print(f"  Compression: {input_size}D → {bottleneck_size}D ({compression_ratio:.1f}x reduction)")
    print(f"  RNN type: {rnn_type.upper()}")
    print(f"  Estimated parameters: {estimated_params:,}")
    
    if compression_ratio > 20:
        print(f"  Note: High compression ratio may cause information loss")
    elif compression_ratio < 5:
        print(f"  Note: Low compression ratio may not provide significant benefits")
    
    return config


def preset_architecture(preset: str, **overrides) -> ArchitectureConfig:
    """
    Create architecture configuration from preset templates.
    
    Presets are based on empirical testing and theoretical analysis,
    providing good starting points for different use cases and hardware
    constraints.
    
    Args:
        preset: Preset name ('tiny', 'small', 'medium', 'large', 'xlarge', 'research')
        **overrides: Parameters to override in the preset
    
    Returns:
        ArchitectureConfig: Configured architecture
        
    Available Presets:
        - tiny: Minimal model for testing (128/16D)
        - small: Small model for limited hardware (256/32D) 
        - medium: Balanced model for most use cases (512/64D)
        - large: Large model for high-quality results (768/96D)
        - xlarge: Very large model for research (1024/128D)
        - research: Experimental configuration for investigation
        
    Example:
        >>> config = preset_architecture('medium', dropout=0.2)
        >>> config = preset_architecture('large', rnn_type='gru')
    """
    presets = {
        'tiny': {
            'hidden_size': 128,
            'bottleneck_size': 16,
            'dropout': 0.05,
            'num_layers': 1,
            'rnn_type': 'lstm'
        },
        'small': {
            'hidden_size': 256,
            'bottleneck_size': 32,
            'dropout': 0.1,
            'num_layers': 1,
            'rnn_type': 'lstm'
        },
        'medium': {
            'hidden_size': 512,
            'bottleneck_size': 64,
            'dropout': 0.1,
            'num_layers': 1,
            'rnn_type': 'lstm'
        },
        'large': {
            'hidden_size': 768,
            'bottleneck_size': 96,
            'dropout': 0.15,
            'num_layers': 2,
            'rnn_type': 'lstm',
            'use_batch_norm': True
        },
        'xlarge': {
            'hidden_size': 1024,
            'bottleneck_size': 128,
            'dropout': 0.2,
            'num_layers': 2,
            'rnn_type': 'lstm',
            'use_batch_norm': True,
            'residual': True
        },
        'research': {
            'hidden_size': 512,
            'bottleneck_size': 20,  # Based on theory optimal ~15-20D
            'dropout': 0.1,
            'num_layers': 1,
            'rnn_type': 'lstm',
            'attention': True,
            'bidirectional': True
        }
    }
    
    if preset not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available presets: {available}")
    
    # Get base configuration and apply overrides
    preset_config = presets[preset].copy()
    preset_config.update(overrides)
    
    print(f"Using '{preset}' architecture preset")
    return design_autoencoder(**preset_config)


def curriculum_learning(
    phases: int = 4,
    epochs: int = 30,
    decay_type: str = 'linear',
    initial_teacher_forcing: float = 0.9,
    final_teacher_forcing: float = 0.1,
    learning_rate: float = 2.5e-4,
    batch_size: int = 16,
    **kwargs
) -> TrainingConfig:
    """
    Create curriculum learning training configuration.
    
    Implements progressive training strategy where model learns shorter
    sequences first, then gradually increases complexity. Based on
    theoretical insights about RNN training stability.
    
    Args:
        phases: Number of curriculum phases (default: 4)
        epochs: Total training epochs (default: 30)
        decay_type: Teacher forcing decay ('linear', 'exponential', 'cosine')
        initial_teacher_forcing: Starting teacher forcing ratio (default: 0.9)
        final_teacher_forcing: Final teacher forcing ratio (default: 0.1)
        learning_rate: Initial learning rate (default: 2.5e-4)
        batch_size: Training batch size (default: 16)
        **kwargs: Additional TrainingConfig parameters
        
    Returns:
        TrainingConfig: Configured training setup
        
    Example:
        >>> config = curriculum_learning(phases=6, epochs=60, decay_type='exponential')
        >>> print("Training phases:", config.phase_epochs)
        >>> print("Teacher forcing schedule:", config.teacher_forcing_schedule)
    """
    # Distribute epochs across phases
    if epochs < phases:
        raise ValueError(f"Total epochs ({epochs}) must be >= phases ({phases})")
    
    phase_epochs = _distribute_epochs(epochs, phases)
    
    # Generate teacher forcing schedule
    teacher_forcing_schedule = _generate_teacher_forcing_schedule(
        phases, decay_type, initial_teacher_forcing, final_teacher_forcing
    )
    
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        curriculum_phases=phases,
        phase_epochs=phase_epochs,
        teacher_forcing_schedule=teacher_forcing_schedule,
        **kwargs
    )
    
    # Validate configuration
    config.validate()
    
    print(f"Curriculum learning configured:")
    print(f"  Phases: {phases}")
    print(f"  Epochs per phase: {phase_epochs}")
    print(f"  Teacher forcing: {initial_teacher_forcing:.2f} → {final_teacher_forcing:.2f} ({decay_type})")
    print(f"  Schedule: {[f'{tf:.2f}' for tf in teacher_forcing_schedule]}")
    
    return config


def fetch_data(
    data_path: str,
    embedding_type: str = 'glove',
    max_length: int = 50,
    chunking_method: str = 'sliding',
    split_ratios: tuple = (0.8, 0.15, 0.05),
    cache: bool = True,
    **kwargs
) -> DataConfig:
    """
    Create data configuration with auto-detection and validation.
    
    Automatically detects data format, locates embeddings, and configures
    preprocessing pipeline for optimal performance with poetry data.
    
    Args:
        data_path: Path to poetry dataset file
        embedding_type: Type of embeddings ('glove', 'word2vec', 'custom')
        max_length: Maximum sequence length for processing
        chunking_method: Chunking strategy ('sliding', 'truncate', 'pad')
        split_ratios: Train/validation/test split ratios
        cache: Whether to cache preprocessed data
        **kwargs: Additional DataConfig parameters
        
    Returns:
        DataConfig: Configured data processing setup
        
    Example:
        >>> config = fetch_data("poems.json", max_length=100)
        >>> print(f"Expected vocab size: {config.get_expected_vocab_size()}")
    """
    # Auto-detect data format
    data_format = detect_data_format(data_path)
    print(f"Detected data format: {data_format}")
    
    # Auto-locate embeddings if not specified
    embedding_path = kwargs.get('embedding_path')
    if embedding_type == 'glove' and not embedding_path:
        embedding_path = find_glove_embeddings()
        if embedding_path:
            print(f"Found GLoVe embeddings: {embedding_path}")
        else:
            print("GLoVe embeddings not found - will create mock embeddings")
        kwargs['embedding_path'] = embedding_path
    
    config = DataConfig(
        data_path=data_path,
        embedding_type=embedding_type,
        max_sequence_length=max_length,
        chunking_method=chunking_method,
        split_ratios=split_ratios,
        cache_preprocessed=cache,
        **kwargs
    )
    
    # Validate configuration
    config.validate()
    
    # Print helpful information
    expected_vocab = config.get_expected_vocab_size()
    preprocessing_time = config.estimate_preprocessing_time()
    
    print(f"Data configuration:")
    print(f"  Dataset: {Path(data_path).name}")
    print(f"  Expected vocabulary size: ~{expected_vocab:,}")
    print(f"  Sequence length: {max_length}")
    print(f"  Chunking: {chunking_method}")
    print(f"  Split ratios: {split_ratios}")
    print(f"  Estimated preprocessing time: {preprocessing_time['estimated_minutes']:.1f} minutes")
    
    return config


def quick_training(
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    **kwargs
) -> TrainingConfig:
    """
    Create a simplified training configuration for quick experiments.
    
    Provides faster training with reduced curriculum complexity,
    suitable for testing and rapid iteration.
    
    Args:
        epochs: Number of training epochs (default: 15)
        batch_size: Batch size (default: 32, larger for speed)
        learning_rate: Learning rate (default: 1e-3, higher for speed)
        **kwargs: Additional training parameters
        
    Returns:
        TrainingConfig: Quick training configuration
    """
    return TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        curriculum_phases=2,  # Simplified curriculum
        phase_epochs=[epochs//2, epochs - epochs//2],
        teacher_forcing_schedule=[0.8, 0.2],
        save_every=max(1, epochs//5),  # Save less frequently
        **kwargs
    )


def production_training(
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    **kwargs
) -> TrainingConfig:
    """
    Create production-quality training configuration.
    
    Optimized for best model quality with extensive curriculum learning,
    careful learning rate scheduling, and comprehensive monitoring.
    
    Args:
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size (default: 8, smaller for stability)
        learning_rate: Learning rate (default: 1e-4, conservative)
        **kwargs: Additional training parameters
        
    Returns:
        TrainingConfig: Production training configuration
    """
    phases = min(6, epochs // 10)  # More phases for long training
    return curriculum_learning(
        phases=phases,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        decay_type='exponential',
        scheduler='plateau',
        early_stopping_patience=20,
        gradient_clip=0.5,  # Conservative clipping
        **kwargs
    )


# Helper functions

def _distribute_epochs(total_epochs: int, phases: int) -> List[int]:
    """Distribute epochs across curriculum phases."""
    if phases == 1:
        return [total_epochs]
    
    # Use slightly increasing epoch counts per phase
    base_epochs = total_epochs // phases
    remaining = total_epochs % phases
    
    phase_epochs = []
    for i in range(phases):
        epochs = base_epochs
        if i >= phases - remaining:  # Distribute remainder to later phases
            epochs += 1
        phase_epochs.append(epochs)
    
    return phase_epochs


def _generate_teacher_forcing_schedule(
    phases: int, 
    decay_type: str, 
    initial: float, 
    final: float
) -> List[float]:
    """Generate teacher forcing schedule based on decay type."""
    if phases == 1:
        return [initial]
    
    if decay_type.lower() == 'linear':
        # Linear decay
        step = (initial - final) / (phases - 1)
        return [initial - i * step for i in range(phases)]
    
    elif decay_type.lower() == 'exponential':
        # Exponential decay
        ratio = (final / initial) ** (1 / (phases - 1))
        return [initial * (ratio ** i) for i in range(phases)]
    
    elif decay_type.lower() == 'cosine':
        # Cosine annealing
        schedule = []
        for i in range(phases):
            progress = i / (phases - 1)
            value = final + (initial - final) * (1 + math.cos(math.pi * progress)) / 2
            schedule.append(value)
        return schedule
    
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")


# Validation helpers

def validate_configuration_compatibility(
    architecture: ArchitectureConfig,
    training: TrainingConfig,
    data: DataConfig
) -> None:
    """
    Validate that configurations are compatible with each other.
    
    Checks for common configuration mismatches that could cause
    training issues or poor performance.
    """
    warnings = []
    errors = []
    
    # Check dimension compatibility
    if architecture.input_size != data.embedding_dim:
        errors.append(f"Architecture input size ({architecture.input_size}) "
                     f"doesn't match embedding dimension ({data.embedding_dim})")
    
    # Check sequence length compatibility
    if data.max_sequence_length > 100 and training.batch_size > 16:
        warnings.append(f"Large sequences ({data.max_sequence_length}) with large batch size "
                       f"({training.batch_size}) may cause memory issues")
    
    # Check training complexity vs model size
    param_count = architecture.estimate_parameters()
    if param_count > 1000000 and training.learning_rate > 1e-3:
        warnings.append(f"Large model ({param_count:,} params) with high learning rate "
                       f"({training.learning_rate}) may be unstable")
    
    if param_count < 50000 and training.epochs > 50:
        warnings.append(f"Small model ({param_count:,} params) with many epochs "
                       f"({training.epochs}) may overfit")
    
    # Print warnings and errors
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    if errors:
        error_msg = "Configuration compatibility errors:\n" + "\n".join(f"  ❌ {e}" for e in errors)
        raise ValueError(error_msg)
    
    if not warnings and not errors:
        print("✅ Configuration compatibility check passed")