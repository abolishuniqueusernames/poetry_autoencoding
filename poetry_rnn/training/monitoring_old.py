"""
Training monitoring utilities for RNN Autoencoder

Provides comprehensive monitoring of training progress, gradient flow,
and model performance metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import time
import json
from pathlib import Path


class GradientMonitor:
    """
    Monitors gradient flow through the network during training.
    
    Tracks gradient norms, detects vanishing/exploding gradients, and
    provides diagnostics for training stability issues.
    
    Theory Note:
        RNNs are prone to vanishing/exploding gradients due to repeated
        matrix multiplication. Monitoring helps detect and address these issues.
    """
    
    def __init__(
        self,
        model: nn.Module,
        clip_value: float = 5.0,
        vanishing_threshold: float = 1e-6,
        exploding_threshold: float = 10.0
    ):
        self.model = model
        self.clip_value = clip_value
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        
        # History tracking
        self.gradient_history = defaultdict(lambda: deque(maxlen=100))
        self.clip_counts = 0
        self.vanishing_counts = 0
        self.exploding_counts = 0
    
    def compute_gradient_norms(self) -> Dict[str, float]:
        """
        Compute gradient norms for all model parameters.
        
        Returns:
            Dictionary mapping parameter names to gradient norms
        """
        grad_norms = {}
        total_norm = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                grad_norms[name] = param_norm
                total_norm += param_norm ** 2
                
                # Track history
                self.gradient_history[name].append(param_norm)
        
        total_norm = total_norm ** 0.5
        grad_norms['total'] = total_norm
        self.gradient_history['total'].append(total_norm)
        
        # Check for issues
        if total_norm < self.vanishing_threshold:
            self.vanishing_counts += 1
        elif total_norm > self.exploding_threshold:
            self.exploding_counts += 1
        
        return grad_norms
    
    def clip_gradients(self) -> float:
        """
        Clip gradients to prevent exploding gradients.
        
        Returns:
            Total gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.clip_value
        )
        
        if total_norm > self.clip_value:
            self.clip_counts += 1
        
        return total_norm.item()
    
    def get_gradient_stats(self) -> Dict[str, Any]:
        """
        Get statistics about gradient behavior.
        
        Returns:
            Dictionary with gradient statistics
        """
        stats = {
            'clip_counts': self.clip_counts,
            'vanishing_counts': self.vanishing_counts,
            'exploding_counts': self.exploding_counts,
            'current_norms': {}
        }
        
        # Compute average norms for key components
        for key in ['encoder', 'decoder', 'total']:
            history = []
            for name, values in self.gradient_history.items():
                if key in name or key == 'total':
                    if values:
                        history.extend(values)
            
            if history:
                stats['current_norms'][key] = {
                    'mean': sum(history) / len(history),
                    'max': max(history),
                    'min': min(history)
                }
        
        return stats
    
    def diagnose_gradient_issues(self) -> List[str]:
        """
        Diagnose potential gradient flow issues.
        
        Returns:
            List of diagnostic messages
        """
        diagnostics = []
        
        total_updates = len(self.gradient_history['total'])
        
        if total_updates > 0:
            vanishing_rate = self.vanishing_counts / total_updates
            exploding_rate = self.exploding_counts / total_updates
            clip_rate = self.clip_counts / total_updates
            
            if vanishing_rate > 0.1:
                diagnostics.append(
                    f"High vanishing gradient rate: {vanishing_rate:.1%}. "
                    "Consider: reducing sequence length, using LSTM/GRU, "
                    "or adjusting learning rate."
                )
            
            if exploding_rate > 0.1:
                diagnostics.append(
                    f"High exploding gradient rate: {exploding_rate:.1%}. "
                    "Consider: reducing learning rate, increasing gradient clipping, "
                    "or using gradient normalization."
                )
            
            if clip_rate > 0.5:
                diagnostics.append(
                    f"Frequent gradient clipping: {clip_rate:.1%}. "
                    "Model may be unstable. Consider reducing learning rate."
                )
        
        return diagnostics


class TrainingMonitor:
    """
    Comprehensive training progress monitor.
    
    Tracks losses, metrics, timing, and provides visualization-ready data
    for training analysis.
    """
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        save_frequency: int = 10
    ):
        self.log_dir = Path(log_dir) if log_dir else Path('training_logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        
        # Metrics tracking
        self.epoch_metrics = defaultdict(list)
        self.batch_metrics = defaultdict(list)
        
        # Timing
        self.epoch_times = []
        self.start_time = None
        self.epoch_start_time = None
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Current state
        self.current_epoch = 0
        self.total_batches = 0
    
    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def start_epoch(self, epoch: int):
        """Mark the start of an epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
    
    def end_epoch(
        self,
        train_loss: float,
        val_loss: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Mark the end of an epoch and record metrics.
        
        Args:
            train_loss: Average training loss for the epoch
            val_loss: Average validation loss
            additional_metrics: Any additional metrics to track
        """
        # Record timing
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # Record metrics
        self.epoch_metrics['epoch'].append(self.current_epoch)
        self.epoch_metrics['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.epoch_metrics['val_loss'].append(val_loss)
            
            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = self.current_epoch
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.epoch_metrics[key].append(value)
        
        # Save logs periodically
        if self.current_epoch % self.save_frequency == 0:
            self.save_logs()
    
    def log_batch(
        self,
        batch_idx: int,
        loss: float,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log metrics for a single batch.
        
        Args:
            batch_idx: Batch index within epoch
            loss: Batch loss
            additional_metrics: Any additional batch metrics
        """
        self.total_batches += 1
        
        self.batch_metrics['epoch'].append(self.current_epoch)
        self.batch_metrics['batch_idx'].append(batch_idx)
        self.batch_metrics['loss'].append(loss)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.batch_metrics[key].append(value)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get training summary statistics.
        
        Returns:
            Dictionary with training summary
        """
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
        
        summary = {
            'total_epochs': self.current_epoch,
            'total_batches': self.total_batches,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'total_time_hours': total_time / 3600,
            'avg_epoch_time_minutes': avg_epoch_time / 60,
            'current_train_loss': self.epoch_metrics['train_loss'][-1] if self.epoch_metrics['train_loss'] else None,
            'current_val_loss': self.epoch_metrics['val_loss'][-1] if self.epoch_metrics['val_loss'] else None
        }
        
        return summary
    
    def save_logs(self):
        """Save training logs to disk."""
        # Save epoch metrics
        epoch_log_path = self.log_dir / 'epoch_metrics.json'
        with open(epoch_log_path, 'w') as f:
            json.dump(dict(self.epoch_metrics), f, indent=2)
        
        # Save summary
        summary_path = self.log_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
    
    def plot_progress(self, save_path: Optional[Path] = None):
        """
        Generate training progress plots.
        
        Args:
            save_path: Where to save the plot (uses log_dir if None)
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.epoch_metrics['epoch']:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Loss curves
            ax = axes[0, 0]
            ax.plot(self.epoch_metrics['epoch'], self.epoch_metrics['train_loss'], 
                   label='Train Loss', marker='o')
            if 'val_loss' in self.epoch_metrics:
                ax.plot(self.epoch_metrics['epoch'], self.epoch_metrics['val_loss'],
                       label='Val Loss', marker='s')
                ax.axvline(x=self.best_epoch, color='r', linestyle='--', 
                          label=f'Best Epoch ({self.best_epoch})')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Learning rate if tracked
            if 'learning_rate' in self.epoch_metrics:
                ax = axes[0, 1]
                ax.plot(self.epoch_metrics['epoch'], self.epoch_metrics['learning_rate'])
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Learning Rate')
                ax.set_title('Learning Rate Schedule')
                ax.grid(True, alpha=0.3)
            
            # Gradient norms if tracked
            if 'gradient_norm' in self.epoch_metrics:
                ax = axes[1, 0]
                ax.plot(self.epoch_metrics['epoch'], self.epoch_metrics['gradient_norm'])
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Gradient Norm')
                ax.set_title('Gradient Magnitude')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
            
            # Timing
            if self.epoch_times:
                ax = axes[1, 1]
                ax.plot(range(1, len(self.epoch_times) + 1), self.epoch_times)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Time (seconds)')
                ax.set_title('Epoch Duration')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.log_dir / 'training_progress.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"Training progress plot saved to {save_path}")
            
        except ImportError:
            print("Matplotlib not available, skipping plot generation")