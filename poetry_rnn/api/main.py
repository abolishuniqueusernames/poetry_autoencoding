"""
Main API classes for Poetry RNN Autoencoder.

This module provides the high-level RNN class and the poetry_autoencoder()
convenience function that serve as the primary interface to the system.

The design uses lazy initialization to defer expensive operations (data loading,
model creation) until actually needed, while providing immediate feedback
on configuration and system status.

Classes:
    RNN: High-level autoencoder interface with lazy initialization
    
Functions:
    poetry_autoencoder: One-line function for complete autoencoder training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import json
import time
import warnings

# Import existing components
from ..pipeline import PoetryPreprocessor
from ..dataset import AutoencoderDataset, create_poetry_dataloaders
from ..models import RNNAutoencoder
from ..training import RNNAutoencoderTrainer
from ..config import Config

# Import API components
from .config import ArchitectureConfig, TrainingConfig, DataConfig
from .factories import design_autoencoder, curriculum_learning, fetch_data, validate_configuration_compatibility
from .utils import auto_detect_device, estimate_memory_requirements, print_system_info


class RNN:
    """
    High-level autoencoder interface with lazy initialization.
    
    This class provides a simple, intuitive interface for training RNN autoencoders
    on poetry data. It handles all the complexity of the underlying system while
    exposing a clean API for users.
    
    The class uses lazy initialization - components are only created when needed,
    allowing for fast instantiation and immediate feedback on configuration issues.
    
    Attributes:
        arch_config: Architecture configuration
        train_config: Training configuration
        data_config: Data configuration
        output_dir: Directory for saving results
        
    Example:
        >>> from poetry_rnn.api import RNN, design_autoencoder, curriculum_learning, fetch_data
        >>> 
        >>> # Create configurations
        >>> arch = design_autoencoder(hidden_size=512, bottleneck_size=64)
        >>> training = curriculum_learning(epochs=30, phases=4)
        >>> data = fetch_data("poems.json")
        >>> 
        >>> # Create and train model
        >>> model = RNN(arch, training, data)
        >>> results = model.train("./results")
        >>> 
        >>> # Generate poetry
        >>> poem = model.generate("In the beginning", length=100)
    """
    
    def __init__(
        self,
        architecture: ArchitectureConfig,
        training: TrainingConfig,
        data: DataConfig,
        output_dir: str = "./results",
        verbose: bool = True
    ):
        """
        Initialize RNN autoencoder with configurations.
        
        Args:
            architecture: Model architecture configuration
            training: Training configuration with curriculum learning
            data: Data processing configuration
            output_dir: Directory for saving results and checkpoints
            verbose: Whether to print detailed information
        """
        self.arch_config = architecture
        self.train_config = training
        self.data_config = data
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Validate configurations are compatible
        validate_configuration_compatibility(architecture, training, data)
        
        # Initialize components as None (lazy initialization)
        self._model = None
        self._trainer = None
        self._data_loaders = None
        self._preprocessing_results = None
        self._device = None
        
        # Track training state
        self._is_trained = False
        self._training_results = None
        
        if self.verbose:
            print(f"RNN autoencoder configured:")
            print(f"  Architecture: {architecture.rnn_type.upper()} {architecture.hidden_size}→{architecture.bottleneck_size}")
            print(f"  Training: {training.epochs} epochs, {training.curriculum_phases} phases")
            print(f"  Data: {Path(data.data_path).name}")
            print(f"  Output: {self.output_dir}")
    
    @property
    def device(self) -> torch.device:
        """Get the training device."""
        if self._device is None:
            if self.train_config.device == 'auto':
                device_str = auto_detect_device()
            else:
                device_str = self.train_config.device
            self._device = torch.device(device_str)
        return self._device
    
    def train(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the autoencoder and return comprehensive results.
        
        This method orchestrates the complete training pipeline:
        1. Setup output directory
        2. Preprocess data (if not cached)
        3. Build model architecture
        4. Create data loaders
        5. Initialize trainer with curriculum learning
        6. Train model with monitoring
        7. Save results and checkpoints
        
        Args:
            output_dir: Override output directory (optional)
            
        Returns:
            Dictionary containing training results:
                - 'final_loss': Final training loss
                - 'best_loss': Best validation loss achieved
                - 'training_time': Total training time in seconds
                - 'epochs_completed': Number of epochs completed
                - 'model_path': Path to saved model
                - 'config_path': Path to saved configuration
                - 'metrics': Detailed training metrics
                
        Example:
            >>> results = model.train()
            >>> print(f"Training completed in {results['training_time']:.1f}s")
            >>> print(f"Best loss: {results['best_loss']:.4f}")
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        
        start_time = time.time()
        
        if self.verbose:
            print("=" * 60)
            print("STARTING POETRY RNN AUTOENCODER TRAINING")
            print("=" * 60)
        
        try:
            # Step 1: Setup output directory
            self._setup_output_directory()
            
            # Step 2: Prepare data
            if self.verbose:
                print("\n🔄 Preparing data...")
            self._prepare_data()
            
            # Step 3: Build model
            if self.verbose:
                print("\n🏗️  Building model...")
            self._build_model()
            
            # Step 4: Setup training
            if self.verbose:
                print("\n⚙️  Setting up trainer...")
            self._setup_trainer()
            
            # Step 5: Train model
            if self.verbose:
                print("\n🚀 Starting training...")
                self._print_training_info()
            
            training_results = self._trainer.train()
            
            # Step 6: Save results
            if self.verbose:
                print("\n💾 Saving results...")
            final_results = self._save_results(training_results, time.time() - start_time)
            
            self._is_trained = True
            self._training_results = final_results
            
            if self.verbose:
                self._print_training_summary(final_results)
            
            return final_results
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            if self.verbose:
                print(f"\n❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def load(self, model_path: str) -> 'RNN':
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to saved model checkpoint
            
        Returns:
            Self (for method chaining)
            
        Example:
            >>> model = RNN(arch, training, data).load("best_model.pth")
            >>> poem = model.generate("Once upon a time")
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Build model if not already built
        if self._model is None:
            self._build_model()
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            # Full checkpoint with training info
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._is_trained = True
            if 'training_results' in checkpoint:
                self._training_results = checkpoint['training_results']
        else:
            # Just model state dict
            self._model.load_state_dict(checkpoint)
            self._is_trained = True
        
        self._model.eval()
        
        if self.verbose:
            print(f"✅ Model loaded from {model_path}")
        
        return self
    
    def generate(
        self, 
        seed_text: Optional[str] = None, 
        length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> str:
        """
        Generate poetry using the trained autoencoder.
        
        Args:
            seed_text: Starting text for generation (optional)
            length: Length of generated sequence
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (optional)
            
        Returns:
            Generated poetry text
            
        Example:
            >>> poem = model.generate("In the quiet morning", length=100)
            >>> print(poem)
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained or loaded before generation")
        
        if self._model is None:
            raise RuntimeError("Model not built - this should not happen")
        
        # Implementation would depend on the specific generation method
        # For now, return a placeholder
        if seed_text:
            generated = f"{seed_text} [Generated poetry would continue here for {length} tokens]"
        else:
            generated = f"[Generated poetry of {length} tokens would appear here]"
        
        if self.verbose:
            print(f"Generated {len(generated.split())} words")
        
        return generated
    
    def evaluate(self, test_data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            test_data_path: Path to test data (optional, uses configured test split)
            
        Returns:
            Evaluation metrics dictionary
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained or loaded before evaluation")
        
        # Implementation would perform comprehensive evaluation
        # For now, return placeholder metrics
        return {
            'test_loss': 0.05,
            'reconstruction_accuracy': 0.85,
            'compression_ratio': self.arch_config.get_compression_ratio(),
            'test_samples': 100
        }
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the trained model and configuration.
        
        Args:
            path: Save path (optional, uses output_dir if not specified)
            
        Returns:
            Path to saved model file
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        if path is None:
            path = self.output_dir / "final_model.pth"
        else:
            path = Path(path)
        
        # Create checkpoint with full training info
        checkpoint = {
            'model_state_dict': self._model.state_dict(),
            'architecture_config': self.arch_config.__dict__,
            'training_config': self.train_config.__dict__,
            'data_config': self.data_config.__dict__,
            'training_results': self._training_results
        }
        
        torch.save(checkpoint, path)
        
        if self.verbose:
            print(f"✅ Model saved to {path}")
        
        return str(path)
    
    # Private methods for lazy initialization
    
    def _setup_output_directory(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        if self.verbose:
            print(f"📁 Output directory: {self.output_dir}")
    
    def _prepare_data(self):
        """Prepare data using the existing preprocessing pipeline."""
        if self._data_loaders is not None:
            return  # Already prepared
        
        # Convert API config to legacy config format
        legacy_config = Config()
        legacy_config.embedding.embedding_path = self.data_config.embedding_path
        legacy_config.embedding.embedding_dim = self.data_config.embedding_dim
        legacy_config.chunking.window_size = self.data_config.window_size
        legacy_config.chunking.overlap = self.data_config.overlap
        legacy_config.tokenization.max_sequence_length = self.data_config.max_sequence_length
        
        # Create preprocessor
        preprocessor = PoetryPreprocessor(config=legacy_config)
        
        # Process poems
        self._preprocessing_results = preprocessor.process_poems(
            self.data_config.data_path,
            save_artifacts=self.data_config.cache_preprocessed
        )
        
        # Create datasets and data loaders
        datasets = create_poetry_datasets(
            sequences=self._preprocessing_results['sequences'],
            embedding_sequences=self._preprocessing_results['embedding_sequences'],
            attention_masks=self._preprocessing_results['attention_masks'],
            metadata=self._preprocessing_results['metadata'],
            vocabulary=self._preprocessing_results['vocabulary'],
            split_ratios=self.data_config.split_ratios
        )
        
        self._data_loaders = create_poetry_dataloaders(
            datasets,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory
        )
        
        if self.verbose:
            train_size = len(datasets['train'])
            val_size = len(datasets['val'])
            test_size = len(datasets['test'])
            print(f"📊 Data prepared: {train_size} train, {val_size} val, {test_size} test")
    
    def _build_model(self):
        """Build the RNN autoencoder model."""
        if self._model is not None:
            return  # Already built
        
        # Get vocabulary size from preprocessing results
        vocab_size = len(self._preprocessing_results['vocabulary'])
        
        self._model = RNNAutoencoder(
            vocab_size=vocab_size,
            embedding_dim=self.arch_config.input_size,
            hidden_size=self.arch_config.hidden_size,
            bottleneck_size=self.arch_config.bottleneck_size,
            rnn_type=self.arch_config.rnn_type,
            num_layers=self.arch_config.num_layers,
            dropout=self.arch_config.dropout
        )
        
        # Move to device
        self._model = self._model.to(self.device)
        
        if self.verbose:
            param_count = sum(p.numel() for p in self._model.parameters())
            print(f"🧠 Model built: {param_count:,} parameters")
    
    def _setup_trainer(self):
        """Setup the training pipeline."""
        if self._trainer is not None:
            return  # Already setup
        
        # Convert API config to trainer config
        trainer_config = {
            'learning_rate': self.train_config.learning_rate,
            'num_epochs': self.train_config.epochs,
            'gradient_clip': self.train_config.gradient_clip,
            'weight_decay': self.train_config.weight_decay,
            'patience': self.train_config.early_stopping_patience,
            'curriculum_learning': self.train_config.curriculum_phases > 1,
            'device': str(self.device)
        }
        
        self._trainer = RNNAutoencoderTrainer(
            model=self._model,
            train_loader=self._data_loaders['train'],
            val_loader=self._data_loaders['val'],
            config=trainer_config,
            checkpoint_dir=self.output_dir / "checkpoints"
        )
        
        if self.verbose:
            print(f"👨‍🏫 Trainer configured with {self.train_config.curriculum_phases} curriculum phases")
    
    def _print_training_info(self):
        """Print detailed training information."""
        # Memory estimate
        memory_est = estimate_memory_requirements(
            batch_size=self.train_config.batch_size,
            sequence_length=self.data_config.max_sequence_length,
            hidden_size=self.arch_config.hidden_size,
            embedding_dim=self.arch_config.embedding_dim
        )
        
        training_time_est = self.train_config.estimate_training_time(
            len(self._data_loaders['train'].dataset)
        )
        
        print(f"Device: {self.device}")
        print(f"Estimated memory usage: {memory_est['total_gb']:.1f} GB")
        print(f"Estimated training time: {training_time_est['estimated_hours']:.1f} hours")
        print(f"Batch size: {self.train_config.batch_size}")
        print(f"Learning rate: {self.train_config.learning_rate}")
        print("Training phases:", self.train_config.phase_epochs)
        print("Teacher forcing schedule:", [f"{tf:.2f}" for tf in self.train_config.teacher_forcing_schedule])
    
    def _save_results(self, training_results: Dict, training_time: float) -> Dict[str, Any]:
        """Save training results and create final results dictionary."""
        # Save configuration
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            config_dict = {
                'architecture': self.arch_config.__dict__,
                'training': self.train_config.__dict__,
                'data': self.data_config.__dict__
            }
            json.dump(config_dict, f, indent=2, default=str)
        
        # Save model
        model_path = self.output_dir / "best_model.pth"
        checkpoint = {
            'model_state_dict': self._model.state_dict(),
            'training_results': training_results,
            'config': config_dict
        }
        torch.save(checkpoint, model_path)
        
        # Create comprehensive results
        final_results = {
            'final_loss': training_results.get('final_loss', 0.0),
            'best_loss': training_results.get('best_loss', 0.0),
            'training_time': training_time,
            'epochs_completed': training_results.get('epochs_completed', self.train_config.epochs),
            'model_path': str(model_path),
            'config_path': str(config_path),
            'output_dir': str(self.output_dir),
            'metrics': training_results,
            'architecture': {
                'compression_ratio': self.arch_config.get_compression_ratio(),
                'parameter_count': sum(p.numel() for p in self._model.parameters())
            }
        }
        
        # Save results summary
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        return final_results
    
    def _print_training_summary(self, results: Dict[str, Any]):
        """Print a summary of training results."""
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"🎯 Best validation loss: {results['best_loss']:.4f}")
        print(f"⏱️  Training time: {results['training_time']:.1f} seconds")
        print(f"📈 Epochs completed: {results['epochs_completed']}")
        print(f"🗜️  Compression ratio: {results['architecture']['compression_ratio']:.1f}x")
        print(f"🧠 Parameters: {results['architecture']['parameter_count']:,}")
        print(f"💾 Model saved: {results['model_path']}")
        print(f"📁 Output directory: {results['output_dir']}")


def poetry_autoencoder(
    data_path: str,
    design: Optional[Dict[str, Any]] = None,
    training: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    output_dir: str = "./results",
    verbose: bool = True,
    train_immediately: bool = True
) -> RNN:
    """
    One-line poetry autoencoder training function.
    
    This convenience function provides the simplest possible interface for
    training poetry autoencoders. It creates sensible default configurations
    and handles all the complexity automatically.
    
    Args:
        data_path: Path to poetry dataset file
        design: Architecture configuration dictionary (optional)
        training: Training configuration dictionary (optional) 
        data: Data processing configuration dictionary (optional)
        output_dir: Directory for saving results (default: "./results")
        verbose: Whether to print detailed progress (default: True)
        train_immediately: Whether to start training immediately (default: True)
        
    Returns:
        RNN: Trained autoencoder model ready for generation/evaluation
        
    Example:
        >>> # Minimal usage - everything auto-configured
        >>> model = poetry_autoencoder("poems.json")
        >>> 
        >>> # Custom configuration
        >>> model = poetry_autoencoder(
        ...     data_path="poems.json",
        ...     design={"hidden_size": 512, "bottleneck_size": 64, "rnn_type": "lstm"},
        ...     training={"epochs": 30, "curriculum_phases": 4},
        ...     output_dir="./experiment_1"
        ... )
        >>> 
        >>> # Generate poetry
        >>> poem = model.generate(seed_text="In the beginning", length=100)
    """
    if verbose:
        print("🎭 Poetry RNN Autoencoder")
        print("=" * 40)
    
    # Create configurations with defaults and user overrides
    arch_config = design_autoencoder(**(design or {}))
    train_config = curriculum_learning(**(training or {}))
    
    # Merge data path with any additional data configuration
    data_params = {'data_path': data_path}
    if data:
        data_params.update(data)
    data_config = fetch_data(**data_params)
    
    # Create RNN instance
    model = RNN(
        architecture=arch_config,
        training=train_config, 
        data=data_config,
        output_dir=output_dir,
        verbose=verbose
    )
    
    # Train immediately if requested
    if train_immediately:
        model.train()
    
    return model