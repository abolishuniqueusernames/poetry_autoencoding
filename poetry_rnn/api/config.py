"""
Configuration dataclasses for Poetry RNN Autoencoder API

Provides type-safe, validated configuration objects for all aspects
of autoencoder training. Each configuration class includes:
- Sensible defaults for immediate use
- Comprehensive validation
- Immutable design (using frozen dataclasses where appropriate)
- Clear documentation of all parameters

Architecture follows the theory-driven approach established in the project,
with defaults based on mathematical analysis of optimal dimensions.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import torch


@dataclass
class ArchitectureConfig:
    """
    Configuration for RNN Autoencoder architecture.
    
    Based on theoretical analysis showing optimal compression from 300D → 15-20D
    for poetry embeddings, achieving O(ε^-600) → O(ε^-35) complexity reduction.
    
    Attributes:
        input_size: Embedding dimension (typically 300 for GLoVe)
        hidden_size: RNN hidden state dimension
        bottleneck_size: Compressed representation dimension
        rnn_type: Type of RNN cell ('vanilla', 'lstm', 'gru')
        num_layers: Number of RNN layers in encoder/decoder
        dropout: Dropout probability for regularization
        use_batch_norm: Whether to use batch normalization
        bidirectional: Whether to use bidirectional RNN in encoder
        attention: Whether to add attention mechanism
        residual: Whether to use residual connections
    """
    
    # Core dimensions
    input_size: int = 300
    hidden_size: int = 512
    bottleneck_size: int = 64
    
    # Model architecture
    rnn_type: str = 'lstm'
    num_layers: int = 1
    
    # Regularization
    dropout: float = 0.1
    use_batch_norm: bool = True
    
    # Advanced features
    bidirectional: bool = False
    attention: bool = False
    residual: bool = False
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []
        
        # Dimension checks
        if self.bottleneck_size >= self.hidden_size:
            errors.append(f"Bottleneck size ({self.bottleneck_size}) must be smaller than hidden size ({self.hidden_size})")
        
        if self.input_size <= 0:
            errors.append(f"Input size must be positive, got {self.input_size}")
            
        if self.hidden_size <= 0:
            errors.append(f"Hidden size must be positive, got {self.hidden_size}")
            
        if self.bottleneck_size <= 0:
            errors.append(f"Bottleneck size must be positive, got {self.bottleneck_size}")
        
        # RNN type validation
        valid_rnn_types = {'vanilla', 'lstm', 'gru'}
        if self.rnn_type.lower() not in valid_rnn_types:
            errors.append(f"RNN type must be one of {valid_rnn_types}, got '{self.rnn_type}'")
        
        # Layer count validation
        if self.num_layers < 1:
            errors.append(f"Number of layers must be >= 1, got {self.num_layers}")
            
        if self.num_layers > 4:
            # Warn about potential training difficulties
            print(f"Warning: {self.num_layers} layers may be difficult to train. Consider using fewer layers.")
        
        # Dropout validation
        if not (0.0 <= self.dropout <= 1.0):
            errors.append(f"Dropout must be in [0, 1], got {self.dropout}")
        
        # Theoretical dimension checks based on project analysis
        if self.bottleneck_size > 128:
            print(f"Warning: Large bottleneck size ({self.bottleneck_size}). "
                  f"Theory suggests optimal compression is 15-20D for poetry.")
        
        if self.bottleneck_size < 8:
            print(f"Warning: Very small bottleneck size ({self.bottleneck_size}). "
                  f"May lose too much information.")
        
        if errors:
            raise ValueError("Architecture configuration validation failed:\n" + "\n".join(errors))
    
    def get_compression_ratio(self) -> float:
        """Calculate the compression ratio achieved by the bottleneck."""
        return self.input_size / self.bottleneck_size
    
    def estimate_parameters(self) -> int:
        """Estimate the total number of model parameters."""
        # Simplified parameter counting based on RNN type
        if self.rnn_type.lower() == 'lstm':
            # LSTM has 4 gates, each with input and hidden weights plus bias
            encoder_params = 4 * (self.input_size * self.hidden_size + self.hidden_size**2 + self.hidden_size)
            decoder_params = 4 * (self.bottleneck_size * self.hidden_size + self.hidden_size**2 + self.hidden_size)
        else:
            # Simplified for vanilla RNN and GRU
            encoder_params = self.input_size * self.hidden_size + self.hidden_size**2 + self.hidden_size
            decoder_params = self.bottleneck_size * self.hidden_size + self.hidden_size**2 + self.hidden_size
        
        # Bottleneck linear layers
        bottleneck_params = self.hidden_size * self.bottleneck_size + self.bottleneck_size
        output_params = self.hidden_size * self.input_size + self.input_size
        
        total = encoder_params + decoder_params + bottleneck_params + output_params
        return int(total * self.num_layers)


@dataclass
class TrainingConfig:
    """
    Configuration for RNN Autoencoder training.
    
    Implements curriculum learning strategy based on theoretical insights
    showing RNNs benefit from gradual complexity increase and adaptive
    learning rates for stability.
    
    Attributes:
        epochs: Total number of training epochs
        batch_size: Mini-batch size for training
        learning_rate: Initial learning rate
        curriculum_phases: Number of curriculum learning phases
        phase_epochs: Epochs per curriculum phase
        teacher_forcing_schedule: Teacher forcing ratios per phase
        optimizer: Optimizer type ('adam', 'sgd', 'adamw')
        weight_decay: L2 regularization strength
        gradient_clip: Gradient clipping threshold
        scheduler: Learning rate scheduler ('plateau', 'cosine', 'exponential', 'none')
        save_every: Save checkpoint every N epochs
        log_every: Log metrics every N batches
        early_stopping_patience: Early stopping patience
        device: Training device ('auto', 'cpu', 'cuda', 'cuda:0')
        num_workers: DataLoader worker processes
        pin_memory: Whether to pin memory for GPU transfer
    """
    
    # Basic training parameters
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 2.5e-4
    
    # Curriculum learning
    curriculum_phases: int = 4
    phase_epochs: List[int] = field(default_factory=lambda: [8, 10, 12, 15])
    teacher_forcing_schedule: List[float] = field(default_factory=lambda: [0.9, 0.7, 0.3, 0.1])
    
    # Optimization
    optimizer: str = 'adam'
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    scheduler: str = 'plateau'
    
    # Monitoring and checkpointing
    save_every: int = 5
    log_every: int = 20
    early_stopping_patience: int = 10
    
    # Hardware configuration
    device: str = 'auto'
    num_workers: int = 4
    pin_memory: bool = True
    
    def validate(self) -> None:
        """Validate training configuration parameters."""
        errors = []
        
        # Basic parameter validation
        if self.epochs < 1:
            errors.append(f"Epochs must be >= 1, got {self.epochs}")
        
        if self.batch_size < 1:
            errors.append(f"Batch size must be >= 1, got {self.batch_size}")
        
        if self.learning_rate <= 0:
            errors.append(f"Learning rate must be positive, got {self.learning_rate}")
        
        # Curriculum learning validation
        if self.curriculum_phases < 1:
            errors.append(f"Curriculum phases must be >= 1, got {self.curriculum_phases}")
        
        if len(self.phase_epochs) != self.curriculum_phases:
            errors.append(f"Phase epochs list length ({len(self.phase_epochs)}) "
                         f"must match curriculum phases ({self.curriculum_phases})")
        
        if len(self.teacher_forcing_schedule) != self.curriculum_phases:
            errors.append(f"Teacher forcing schedule length ({len(self.teacher_forcing_schedule)}) "
                         f"must match curriculum phases ({self.curriculum_phases})")
        
        if sum(self.phase_epochs) != self.epochs:
            errors.append(f"Sum of phase epochs ({sum(self.phase_epochs)}) "
                         f"must equal total epochs ({self.epochs})")
        
        # Validate teacher forcing ratios
        for i, ratio in enumerate(self.teacher_forcing_schedule):
            if not (0.0 <= ratio <= 1.0):
                errors.append(f"Teacher forcing ratio at phase {i} must be in [0, 1], got {ratio}")
        
        # Check that teacher forcing generally decreases (curriculum learning principle)
        if len(self.teacher_forcing_schedule) > 1:
            for i in range(len(self.teacher_forcing_schedule) - 1):
                if self.teacher_forcing_schedule[i] < self.teacher_forcing_schedule[i + 1]:
                    print(f"Warning: Teacher forcing increases from phase {i} to {i+1}. "
                          f"Curriculum learning typically uses decreasing ratios.")
        
        # Optimizer validation
        valid_optimizers = {'adam', 'sgd', 'adamw'}
        if self.optimizer.lower() not in valid_optimizers:
            errors.append(f"Optimizer must be one of {valid_optimizers}, got '{self.optimizer}'")
        
        # Scheduler validation
        valid_schedulers = {'plateau', 'cosine', 'exponential', 'none'}
        if self.scheduler.lower() not in valid_schedulers:
            errors.append(f"Scheduler must be one of {valid_schedulers}, got '{self.scheduler}'")
        
        # Hardware validation
        if self.num_workers < 0:
            errors.append(f"Number of workers must be >= 0, got {self.num_workers}")
        
        # Device validation
        if self.device != 'auto':
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                errors.append(f"CUDA device requested but not available: {self.device}")
        
        if errors:
            raise ValueError("Training configuration validation failed:\n" + "\n".join(errors))
    
    def get_effective_learning_rates(self) -> List[float]:
        """Calculate effective learning rates per phase if using scheduling."""
        # Simplified calculation - actual implementation would depend on scheduler
        if self.scheduler.lower() == 'exponential':
            decay_factor = 0.9
            return [self.learning_rate * (decay_factor ** i) for i in range(self.curriculum_phases)]
        elif self.scheduler.lower() == 'cosine':
            # Rough approximation
            import math
            return [self.learning_rate * (1 + math.cos(math.pi * i / self.curriculum_phases)) / 2 
                   for i in range(self.curriculum_phases)]
        else:
            return [self.learning_rate] * self.curriculum_phases
    
    def estimate_training_time(self, samples_per_epoch: int) -> Dict[str, float]:
        """Estimate training time based on configuration."""
        # Very rough estimates - actual time depends on hardware
        batches_per_epoch = max(1, samples_per_epoch // self.batch_size)
        total_batches = batches_per_epoch * self.epochs
        
        # Rough estimates: simple model ~0.1s/batch, complex model ~0.5s/batch
        time_per_batch = 0.3  # seconds, rough average
        
        estimated_seconds = total_batches * time_per_batch
        estimated_minutes = estimated_seconds / 60
        estimated_hours = estimated_minutes / 60
        
        return {
            'total_batches': total_batches,
            'estimated_seconds': estimated_seconds,
            'estimated_minutes': estimated_minutes,
            'estimated_hours': estimated_hours
        }


@dataclass
class DataConfig:
    """
    Configuration for data processing and loading.
    
    Handles poetry-specific data processing including tokenization,
    embedding alignment, and sequence chunking strategies optimized
    for the alt-lit poetry aesthetic.
    
    Attributes:
        data_path: Path to poetry dataset file
        split_ratios: Train/validation/test split ratios
        max_sequence_length: Maximum sequence length for chunking
        embedding_type: Type of embeddings to use
        embedding_dim: Embedding dimensionality
        embedding_path: Path to pre-trained embedding file
        chunking_method: Sequence chunking strategy
        window_size: Sliding window size for chunking
        overlap: Overlap between chunks
        cache_preprocessed: Whether to cache preprocessed data
        cache_dir: Directory for cached artifacts
    """
    
    # Dataset configuration
    data_path: str
    split_ratios: Tuple[float, float, float] = (0.8, 0.15, 0.05)  # train/val/test
    
    # Sequence processing
    max_sequence_length: int = 50
    
    # Embedding configuration
    embedding_type: str = 'glove'
    embedding_dim: int = 300
    embedding_path: Optional[str] = None
    
    # Chunking strategy
    chunking_method: str = 'sliding'  # 'sliding', 'truncate', 'pad'
    window_size: int = 50
    overlap: int = 10
    
    # Caching
    cache_preprocessed: bool = True
    cache_dir: str = 'preprocessed_artifacts'
    
    def validate(self) -> None:
        """Validate data configuration parameters."""
        errors = []
        
        # Path validation
        if not self.data_path:
            errors.append("Data path cannot be empty")
        
        data_file = Path(self.data_path)
        if not data_file.exists():
            errors.append(f"Data file does not exist: {self.data_path}")
        
        # Split ratios validation
        if len(self.split_ratios) != 3:
            errors.append(f"Split ratios must have 3 values (train/val/test), got {len(self.split_ratios)}")
        
        if abs(sum(self.split_ratios) - 1.0) > 1e-6:
            errors.append(f"Split ratios must sum to 1.0, got {sum(self.split_ratios)}")
        
        for i, ratio in enumerate(self.split_ratios):
            if not (0.0 <= ratio <= 1.0):
                errors.append(f"Split ratio {i} must be in [0, 1], got {ratio}")
        
        # Sequence length validation
        if self.max_sequence_length < 1:
            errors.append(f"Max sequence length must be >= 1, got {self.max_sequence_length}")
        
        if self.max_sequence_length > 500:
            print(f"Warning: Very long sequences ({self.max_sequence_length}). "
                  f"May cause memory issues or slow training.")
        
        # Embedding validation
        valid_embedding_types = {'glove', 'word2vec', 'custom'}
        if self.embedding_type.lower() not in valid_embedding_types:
            errors.append(f"Embedding type must be one of {valid_embedding_types}, got '{self.embedding_type}'")
        
        if self.embedding_dim < 1:
            errors.append(f"Embedding dimension must be >= 1, got {self.embedding_dim}")
        
        if self.embedding_path and not Path(self.embedding_path).exists():
            print(f"Warning: Embedding file not found: {self.embedding_path}. "
                  f"Will attempt auto-detection or create mock embeddings.")
        
        # Chunking validation
        valid_chunking_methods = {'sliding', 'truncate', 'pad'}
        if self.chunking_method.lower() not in valid_chunking_methods:
            errors.append(f"Chunking method must be one of {valid_chunking_methods}, got '{self.chunking_method}'")
        
        if self.chunking_method.lower() == 'sliding':
            if self.window_size <= self.overlap:
                errors.append(f"Window size ({self.window_size}) must be larger than overlap ({self.overlap})")
            
            if self.overlap < 0:
                errors.append(f"Overlap must be >= 0, got {self.overlap}")
        
        # Cache validation
        if self.cache_preprocessed:
            cache_path = Path(self.cache_dir)
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                errors.append(f"Cannot create cache directory: {self.cache_dir}")
        
        if errors:
            raise ValueError("Data configuration validation failed:\n" + "\n".join(errors))
    
    def get_expected_vocab_size(self) -> int:
        """Estimate vocabulary size based on data file."""
        try:
            import json
            with open(self.data_path) as f:
                data = json.load(f)
            
            # Rough estimate: ~500-2000 unique words per poem
            if isinstance(data, list):
                num_poems = len(data)
            else:
                num_poems = len(data.get('poems', []))
            
            # Conservative estimate: average 1000 unique words per poem with overlap
            estimated_vocab = min(50000, num_poems * 200)  # Cap at reasonable size
            return estimated_vocab
            
        except Exception:
            # Fallback estimate for unknown data format
            return 10000
    
    def estimate_preprocessing_time(self) -> Dict[str, float]:
        """Estimate preprocessing time based on data size."""
        try:
            data_file = Path(self.data_path)
            file_size_mb = data_file.stat().st_size / (1024 * 1024)
            
            # Rough estimates based on file size
            time_per_mb = 5.0  # seconds per MB of JSON data
            estimated_seconds = file_size_mb * time_per_mb
            
            return {
                'file_size_mb': file_size_mb,
                'estimated_seconds': estimated_seconds,
                'estimated_minutes': estimated_seconds / 60
            }
        except Exception:
            return {'estimated_seconds': 60, 'estimated_minutes': 1}  # Fallback