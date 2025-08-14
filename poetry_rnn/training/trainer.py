"""
Main trainer class for RNN Autoencoder

Orchestrates the complete training pipeline with curriculum learning,
gradient monitoring, and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
from collections import defaultdict
import numpy as np
import time

from .losses import MaskedMSELoss, CompositeLoss, compute_gradient_norms, analyze_hidden_states
from .curriculum import CurriculumScheduler, CurriculumPhase
from .monitoring import GradientMonitor, TrainingMonitor


class RNNAutoencoderTrainer:
    """
    Complete training pipeline for RNN Autoencoder.
    
    Implements curriculum learning, gradient monitoring, checkpointing,
    and comprehensive evaluation. Designed for educational clarity while
    maintaining production-quality features.
    
    Theory Foundation:
        Based on analysis showing RNNs benefit from:
        - Curriculum learning (shorter → longer sequences)
        - Gradient clipping (prevents exploding gradients)
        - Adaptive learning rates (stability in later phases)
    
    Args:
        model: RNN Autoencoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        checkpoint_dir: Directory for saving checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Default configuration
        default_config = {
            'learning_rate': 1e-3,
            'num_epochs': 50,
            'gradient_clip': 5.0,
            'weight_decay': 1e-5,
            'patience': 5,
            'min_delta': 1e-4,
            'curriculum_learning': True,
            'adaptive_lr': True,
            'loss_type': 'mse',  # 'mse', 'cosine', or 'composite'
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Setup device
        self.device = torch.device(self.config['device'])
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup loss function
        if self.config['loss_type'] == 'mse':
            self.loss_fn = MaskedMSELoss()
        elif self.config['loss_type'] == 'composite':
            self.loss_fn = CompositeLoss()
        else:
            self.loss_fn = MaskedMSELoss()
        
        # Setup curriculum scheduler
        if self.config['curriculum_learning']:
            self.curriculum = CurriculumScheduler()
        else:
            self.curriculum = None
        
        # Setup monitoring
        self.gradient_monitor = GradientMonitor(
            self.model,
            clip_value=self.config['gradient_clip']
        )
        self.training_monitor = TrainingMonitor(
            log_dir=checkpoint_dir / 'logs' if checkpoint_dir else None
        )
        
        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, max_length: Optional[int] = None) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            max_length: Maximum sequence length for curriculum learning
        
        Returns:
            Average training loss and additional metrics
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Apply curriculum truncation if needed
            if max_length and self.curriculum:
                batch = self.curriculum.truncate_batch(batch, max_length)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output_dict = self.model(batch)
            
            # Compute loss
            if isinstance(self.loss_fn, CompositeLoss):
                # Handle composite loss with multiple components
                loss_dict = self.loss_fn(
                    output_dict['reconstructed'],
                    batch['input_sequences'],
                    batch.get('attention_mask'),
                    additional_losses={
                        'bottleneck_reg': output_dict.get('bottleneck_regularization', 0)
                    }
                )
                loss = loss_dict['total']
                
                # Track individual loss components
                for key, value in loss_dict.items():
                    if key != 'total':
                        if key not in epoch_metrics:
                            epoch_metrics[key] = []
                        epoch_metrics[key].append(value.item() if torch.is_tensor(value) else value)
            else:
                # Simple loss
                loss = self.loss_fn(
                    output_dict['reconstructed'],
                    batch['input_sequences'],
                    batch.get('attention_mask')
                )
            
            # Backward pass
            loss.backward()
            
            # Monitor and clip gradients
            grad_norm = self.gradient_monitor.clip_gradients()
            
            # Optimizer step
            self.optimizer.step()
            
            # Record loss
            epoch_losses.append(loss.item())
            
            # Log batch metrics
            self.training_monitor.log_batch(
                batch_idx,
                loss.item(),
                {'gradient_norm': grad_norm}
            )
            
            # Progress update
            if batch_idx % 10 == 0:
                avg_loss = np.mean(epoch_losses[-10:]) if len(epoch_losses) >= 10 else np.mean(epoch_losses)
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
                      f"Loss={avg_loss:.6f}, Grad={grad_norm:.4f}", end='\r')
        
        # Compute epoch averages
        avg_loss = np.mean(epoch_losses)
        for key in epoch_metrics:
            epoch_metrics[key] = np.mean(epoch_metrics[key])
        
        print()  # New line after progress updates
        return avg_loss, epoch_metrics
    
    def validate(self, max_length: Optional[int] = None) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Args:
            max_length: Maximum sequence length for curriculum learning
        
        Returns:
            Average validation loss and additional metrics
        """
        self.model.eval()
        val_losses = []
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = self._batch_to_device(batch)
                
                # Apply curriculum truncation if needed
                if max_length and self.curriculum:
                    batch = self.curriculum.truncate_batch(batch, max_length)
                
                # Forward pass
                output_dict = self.model(batch)
                
                # Compute loss
                loss = self.loss_fn(
                    output_dict['reconstructed'],
                    batch['input_sequences'],
                    batch.get('attention_mask')
                )
                val_losses.append(loss.item())
                
                # Compute additional metrics
                stats = self.model.get_compression_stats(batch)
                for key, value in stats.items():
                    val_metrics[key].append(value)
        
        # Average metrics
        avg_loss = np.mean(val_losses)
        avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Complete training loop with curriculum learning.
        
        Returns:
            Training history and final metrics
        """
        print("="*60)
        print("Starting RNN Autoencoder Training")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {self.model.count_parameters():,}")
        print("="*60)
        
        self.training_monitor.start_training()
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch + 1
            self.training_monitor.start_epoch(self.current_epoch)
            
            # Get current curriculum phase
            if self.curriculum:
                max_length = self.curriculum.get_max_length()
                lr_scale = self.curriculum.get_learning_rate_scale()
                
                # Adjust learning rate for curriculum phase
                if self.config['adaptive_lr']:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.config['learning_rate'] * lr_scale
                
                progress = self.curriculum.get_progress()
                print(f"\nEpoch {self.current_epoch}/{self.config['num_epochs']} "
                      f"- Phase {progress['current_phase']}/{progress['total_phases']}: "
                      f"{progress['phase_description']}")
            else:
                max_length = None
                print(f"\nEpoch {self.current_epoch}/{self.config['num_epochs']}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(max_length)
            
            # Validate
            val_loss, val_metrics = self.validate(max_length)
            
            # Gradient analysis
            grad_stats = self.gradient_monitor.get_gradient_stats()
            
            # Print epoch summary
            print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            print(f"  Cosine Sim: {val_metrics.get('cosine_similarity', 0):.4f} | "
                  f"Bottleneck Std: {val_metrics.get('bottleneck_std', 0):.4f}")
            
            # Check gradient health
            diagnostics = self.gradient_monitor.diagnose_gradient_issues()
            if diagnostics:
                print("  Gradient Issues:")
                for diag in diagnostics:
                    print(f"    - {diag}")
            
            # Log epoch metrics
            self.training_monitor.end_epoch(
                train_loss,
                val_loss,
                {**train_metrics, **val_metrics, **grad_stats['current_norms']}
            )
            
            # Early stopping check
            if val_loss < self.best_val_loss - self.config['min_delta']:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print(f"  New best model saved (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['patience']:
                    print(f"\nEarly stopping triggered after {self.current_epoch} epochs")
                    break
            
            # Update curriculum
            if self.curriculum:
                advanced = self.curriculum.step(val_loss)
                if advanced:
                    print("\n" + "="*60)
                    print("CURRICULUM PHASE ADVANCED")
                    print("="*60)
            
            # Periodic checkpoint
            if self.current_epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.current_epoch}.pth')
        
        # Final summary
        print("\n" + "="*60)
        print("Training Complete")
        summary = self.training_monitor.get_summary()
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"Best Epoch: {summary['best_epoch']}")
        print(f"Best Val Loss: {summary['best_val_loss']:.6f}")
        print(f"Total Time: {summary['total_time_hours']:.2f} hours")
        print("="*60)
        
        # Save final model and plots
        self.save_checkpoint('final_model.pth')
        self.training_monitor.save_logs()
        self.training_monitor.plot_progress()
        
        return self.training_monitor.epoch_metrics
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'curriculum_state': self.curriculum.get_progress() if self.curriculum else None
        }
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"  Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        load_path = self.checkpoint_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {load_path}")
        print(f"Resuming from epoch {self.current_epoch}")
    
    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
