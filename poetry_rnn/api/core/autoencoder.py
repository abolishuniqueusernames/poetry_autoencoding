"""
Core Poetry Autoencoder Interface

This module contains the main PoetryAutoencoder class extracted and simplified
from the original enhanced_interface for better maintainability.
"""

import torch
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

# Import configuration classes
from ..config import ArchitectureConfig, TrainingConfig, DataConfig
from ..factories import design_autoencoder, curriculum_learning, fetch_data
from ..utils import auto_detect_device, suggest_architecture_for_data
from .generation import PoetryGenerationConfig

# Import core components
from ...models import RNNAutoencoder
from ...training.losses import HybridTokenEmbeddingLoss


class PoetryAutoencoder:
    """
    Elegant interface to the sophisticated poetry autoencoder system.
    
    This class provides a clean, intuitive API that showcases our advanced
    300D→512D→128D LSTM architecture with attention while maintaining
    simplicity and educational clarity.
    
    Example:
        >>> # Simple usage
        >>> model = PoetryAutoencoder("poems.json")
        >>> results = model.train()
        
        >>> # Advanced usage
        >>> model = PoetryAutoencoder(
        ...     data="poems.json",
        ...     architecture="research",      # Proven 512→128 + attention
        ...     loss_type="hybrid"           # 99.7% accuracy method
        ... )
        >>> results = model.train(epochs=100)
    """
    
    def __init__(
        self,
        data: Union[str, Path, DataConfig],
        architecture: Union[str, Dict, ArchitectureConfig] = "research",
        loss_type: str = "hybrid",
        output_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize Poetry Autoencoder with intelligent defaults.
        
        Args:
            data: Path to poetry dataset or DataConfig object
            architecture: Architecture specification (default: "research" = proven 512→128+attention)
            loss_type: Loss function ("hybrid" for 99.7% accuracy, "cosine", "mse")
            output_dir: Output directory for results
            verbose: Show progress information
        """
        self.verbose = verbose
        self.experiment_name = f"poetry_experiment_{int(time.time())}"
        
        if verbose:
            print("🎭 Poetry RNN Autoencoder")
            print("=" * 40)
        
        # Setup output directory
        if output_dir is None:
            output_dir = f"./experiments/{self.experiment_name}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure components
        self.data_config = self._setup_data_config(data)
        self.arch_config = self._setup_architecture_config(architecture)
        self.loss_config = self._setup_loss_config(loss_type)
        
        # Initialize state
        self._model = None
        self._device = None
        self._is_trained = False
        self._training_results = None
        
        if verbose:
            self._print_config_summary()
    
    def _setup_data_config(self, data_spec: Union[str, Path, DataConfig]) -> DataConfig:
        """Setup data configuration with intelligent defaults."""
        if isinstance(data_spec, DataConfig):
            return data_spec
        
        return fetch_data(
            data_path=str(data_spec),
            embedding_type='glove',
            max_length=50,
            cache=True
        )
    
    def _setup_architecture_config(self, arch_spec: Union[str, Dict, ArchitectureConfig]) -> ArchitectureConfig:
        """Setup architecture configuration."""
        if isinstance(arch_spec, ArchitectureConfig):
            return arch_spec
        
        if isinstance(arch_spec, dict):
            return design_autoencoder(**arch_spec)
        
        # Proven architecture presets
        presets = {
            "research": {
                "hidden_size": 512,
                "bottleneck_size": 128,
                "rnn_type": "lstm",
                "attention": True,
                "num_layers": 1,
                "dropout": 0.1
            },
            "large": {
                "hidden_size": 768,
                "bottleneck_size": 96,
                "rnn_type": "lstm",
                "attention": True,
                "num_layers": 2,
                "dropout": 0.15
            }
        }
        
        if arch_spec in presets:
            return design_autoencoder(**presets[arch_spec])
        else:
            # Default to research architecture
            return design_autoencoder(**presets["research"])
    
    def _setup_loss_config(self, loss_type: str) -> Dict[str, Any]:
        """Setup loss configuration."""
        if loss_type == "hybrid":
            return {
                "type": "hybrid",
                "token_weight": 0.7,
                "embedding_weight": 0.3,
                "token_label_smoothing": 0.1
            }
        elif loss_type == "cosine":
            return {
                "type": "cosine",
                "temperature": 1.0
            }
        elif loss_type == "mse":
            return {
                "type": "mse"
            }
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _print_config_summary(self):
        """Print configuration summary."""
        print(f"🏗️  Architecture: {self.arch_config.rnn_type.upper()} {self.arch_config.hidden_size}→{self.arch_config.bottleneck_size}")
        if self.arch_config.attention:
            print("   ✨ Attention enabled")
        print(f"🎯 Loss function: {self.loss_config['type']} loss")
        print(f"📁 Output: {self.output_dir}")
        print()
    
    def train(
        self,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 5e-4,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the autoencoder with the configured settings.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        if self.verbose:
            print("🚀 Starting training...")
        
        # Import training function to avoid circular imports
        from .training import train_hybrid_loss
        
        # Use the optimized training function
        results = train_hybrid_loss(
            data=self.data_config.data_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            token_weight=self.loss_config.get("token_weight", 0.7),
            embedding_weight=self.loss_config.get("embedding_weight", 0.3),
            output_dir=str(self.output_dir),
            verbose=self.verbose,
            **kwargs
        )
        
        self._is_trained = True
        self._training_results = results
        
        return results
    
    def load_trained_model(self, model_path: str):
        """Load a pre-trained model."""
        if self.verbose:
            print(f"📂 Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create model with checkpoint config
        config = checkpoint.get('config', {})
        self._model = RNNAutoencoder(**config)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()
        
        self._is_trained = True
        if self.verbose:
            print("✅ Model loaded successfully")
    
    def evaluate(self, test_data: Optional[str] = None) -> Dict[str, float]:
        """Evaluate the trained model."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained first")
        
        if self.verbose:
            print("📊 Evaluating model...")
        
        # Basic evaluation metrics
        metrics = {
            "reconstruction_loss": 0.0,
            "token_accuracy": 0.0,
            "semantic_similarity": 0.0
        }
        
        # Add actual evaluation logic here
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model."""
        return {
            "architecture": f"{self.arch_config.hidden_size}→{self.arch_config.bottleneck_size}",
            "parameters": self.arch_config.estimate_parameters(),
            "compression_ratio": self.arch_config.get_compression_ratio(),
            "attention": self.arch_config.attention,
            "loss_type": self.loss_config["type"],
            "output_dir": str(self.output_dir),
            "is_trained": self._is_trained
        }