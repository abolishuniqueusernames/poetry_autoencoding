"""
Optimized RNN Autoencoder Trainer with Performance Enhancements

This module implements an optimized training loop with:
- Efficient batch processing with minimal overhead
- Real-time performance monitoring and profiling
- Gradient accumulation support
- Mixed precision training support (when GPU available)
- Bottleneck regularization
- Teacher forcing scheduling
- Async checkpointing integration
- Memory-efficient operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Tuple, List
import time
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler

# Import new loss functions
from .losses import get_loss_function, EnhancedCosineLoss, MaskedMSELoss

logger = logging.getLogger(__name__)


class OptimizedRNNTrainer:
    """
    High-performance trainer for RNN Autoencoder with extensive optimizations.
    
    Features:
    - Multi-threaded data loading with prefetching
    - Gradient accumulation for larger effective batch sizes
    - Mixed precision training support
    - Async checkpointing to eliminate I/O blocking
    - Real-time performance profiling
    - Adaptive gradient clipping
    - Bottleneck regularization
    - Teacher forcing ratio scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        async_checkpointer: Optional[Any] = None,
        artifact_manager: Optional[Any] = None,
        performance_monitor: Optional[Any] = None
    ):
        """
        Initialize optimized trainer.
        
        Args:
            model: RNN autoencoder model
            train_loader: Training data loader (should be optimized)
            val_loader: Validation data loader
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            config: Training configuration
            device: Device for training
            async_checkpointer: Async checkpoint manager
            artifact_manager: Async artifact manager
            performance_monitor: Performance monitoring instance
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.device = device or torch.device('cpu')
        self.async_checkpointer = async_checkpointer
        self.artifact_manager = artifact_manager
        self.performance_monitor = performance_monitor
        
        # Training configuration
        self.num_epochs = self.config.get('training', {}).get('num_epochs', 100)
        self.gradient_clip_norm = self.config.get('training', {}).get('gradient_clip_norm', 1.0)
        self.bottleneck_reg_weight = self.config.get('training', {}).get('bottleneck_regularization', 0.001)
        self.early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 20)
        self.min_delta = self.config.get('training', {}).get('min_delta', 1e-5)
        
        # Optimization configuration
        self.gradient_accumulation_steps = self.config.get('optimization', {}).get('gradient_accumulation_steps', 1)
        self.use_mixed_precision = self.config.get('optimization', {}).get('mixed_precision', False) and torch.cuda.is_available()
        
        # Monitoring configuration
        self.log_interval = self.config.get('monitoring', {}).get('log_interval', 10)
        self.checkpoint_interval = self.config.get('monitoring', {}).get('checkpoint_interval', 5)
        
        # Setup mixed precision if enabled
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Initialize loss function based on configuration
        self._setup_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_cosine_similarity = 0.0
        self.patience_counter = 0
        
        # Performance tracking
        self.batch_times = []
        self.throughput_history = []
        self.gradient_norms = []
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True):
        """
        Load a checkpoint and optionally resume training from that point.
        
        Args:
            checkpoint_path: Path to checkpoint file
            resume_training: If True, restore training state (epoch, optimizer, scheduler)
                           If False, only load model weights
        
        Returns:
            Dictionary with checkpoint info and loaded epoch
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✅ Model weights loaded successfully")
        else:
            # Fallback for older checkpoint format
            self.model.load_state_dict(checkpoint)
            logger.info("✅ Model weights loaded (legacy format)")
        
        if resume_training:
            # Restore training state
            if 'epoch' in checkpoint:
                self.current_epoch = checkpoint['epoch']
                logger.info(f"📍 Resuming from epoch {self.current_epoch}")
            
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                # Ensure initial_lr is set for scheduler compatibility
                for param_group in self.optimizer.param_groups:
                    if 'initial_lr' not in param_group:
                        param_group['initial_lr'] = param_group['lr']
                
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("✅ Optimizer state restored")
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("✅ Scheduler state restored")
            
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
                logger.info(f"📊 Best validation loss: {self.best_val_loss:.6f}")
            
            if 'best_cosine_similarity' in checkpoint:
                self.best_cosine_similarity = checkpoint.get('best_cosine_similarity', 0.0)
                logger.info(f"📊 Best cosine similarity: {self.best_cosine_similarity:.4f}")
            
            # Reset patience counter for early stopping
            self.patience_counter = 0
            logger.info("🔄 Early stopping patience counter reset")
        
        # Extract checkpoint metadata
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'val_loss': checkpoint.get('best_val_loss', float('inf')),
            'cosine_similarity': checkpoint.get('best_cosine_similarity', 0.0),
            'config': checkpoint.get('config', {}),
            'val_metrics': checkpoint.get('val_metrics', {})
        }
        
        logger.info("🎯 Checkpoint loaded successfully!")
        return checkpoint_info
        
    def _setup_loss_function(self):
        """Initialize loss function based on configuration."""
        # Get loss configuration
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'mse')
        
        if loss_type == 'cosine':
            # Cosine similarity loss configuration
            self.loss_fn = EnhancedCosineLoss(
                mse_weight=loss_config.get('mse_weight', 0.1),  # 10% MSE for stability
                temperature=loss_config.get('temperature', 1.0),
                sequence_level=loss_config.get('sequence_level', False),
                normalize_targets=loss_config.get('normalize_targets', True)
            )
            logger.info(f"Initialized EnhancedCosineLoss with mse_weight={loss_config.get('mse_weight', 0.1)}")
            
        elif loss_type == 'composite':
            # Composite loss with bottleneck regularization built-in
            self.loss_fn = get_loss_function(
                'composite',
                reconstruction_loss='cosine',
                bottleneck_weight=loss_config.get('bottleneck_weight', 0.001),
                **loss_config.get('cosine_kwargs', {})
            )
            logger.info("Initialized CompositeLoss with cosine reconstruction")
            
        else:
            # Default MSE loss
            self.loss_fn = MaskedMSELoss()
            logger.info("Initialized MaskedMSELoss (default)")
            
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch with optimizations.
        
        Returns:
            Average loss and metrics dictionary
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = defaultdict(list)
        
        # Timing
        epoch_start = time.perf_counter()
        data_time = 0
        compute_time = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.perf_counter()
            
            # Move batch to device (data loading time)
            data_start = time.perf_counter()
            batch = self._move_batch_to_device(batch)
            data_time += time.perf_counter() - data_start
            
            # Compute forward pass (compute time)
            compute_start = time.perf_counter()
            
            # Mixed precision context
            if self.use_mixed_precision:
                with autocast():
                    loss, metrics = self._compute_loss(batch)
                    loss = loss / self.gradient_accumulation_steps
            else:
                loss, metrics = self._compute_loss(batch)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm
                )
                self.gradient_norms.append(grad_norm.item())
                
                # Optimizer step
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            compute_time += time.perf_counter() - compute_start
            
            # Record metrics
            epoch_losses.append(loss.item() * self.gradient_accumulation_steps)
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            
            # Batch timing
            batch_time = time.perf_counter() - batch_start
            self.batch_times.append(batch_time)
            
            # Calculate throughput
            batch_size = batch['input_sequences'].size(0)
            sequences_per_sec = batch_size / batch_time
            self.throughput_history.append(sequences_per_sec)
            
            # Logging
            if batch_idx % self.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-self.log_interval:]) if len(epoch_losses) >= self.log_interval else np.mean(epoch_losses)
                avg_throughput = np.mean(self.throughput_history[-self.log_interval:]) if len(self.throughput_history) >= self.log_interval else np.mean(self.throughput_history)
                
                logger.info(
                    f"  Batch [{batch_idx}/{len(self.train_loader)}] | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Throughput: {avg_throughput:.1f} seq/s | "
                    f"Grad Norm: {grad_norm.item() if 'grad_norm' in locals() else 0:.3f}"
                )
        
        # Epoch statistics
        epoch_time = time.perf_counter() - epoch_start
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        # Performance metrics
        avg_metrics['epoch_time'] = epoch_time
        avg_metrics['data_time'] = data_time
        avg_metrics['compute_time'] = compute_time
        avg_metrics['avg_batch_time'] = np.mean(self.batch_times)
        avg_metrics['sequences_per_sec'] = np.mean(self.throughput_history)
        avg_metrics['avg_gradient_norm'] = np.mean(self.gradient_norms) if self.gradient_norms else 0
        
        return avg_loss, avg_metrics
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model with efficient batch processing.
        
        Returns:
            Average validation loss and metrics
        """
        self.model.eval()
        val_losses = []
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                output = self.model(batch)
                
                # Compute loss
                loss = self._compute_validation_loss(batch, output)
                val_losses.append(loss.item())
                
                # Compute metrics
                metrics = self._compute_metrics(batch, output)
                for key, value in metrics.items():
                    val_metrics[key].append(value)
        
        avg_loss = np.mean(val_losses)
        avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss with regularization.
        
        Args:
            batch: Input batch
            
        Returns:
            Loss tensor and metrics dictionary
        """
        # Forward pass
        output = self.model(batch)
        
        # Use the configured loss function
        mask = batch.get('attention_mask', None)
        
        # Check if using composite loss (handles bottleneck regularization internally)
        if hasattr(self.loss_fn, 'bottleneck_weight'):
            # Composite loss handles everything
            total_loss = self.loss_fn(
                predictions=output['reconstructed'],
                targets=batch['input_sequences'],
                bottleneck=output.get('bottleneck', None),
                mask=mask
            )
            reconstruction_loss = total_loss  # For metrics
            bottleneck_reg = 0
        else:
            # Standard loss function + manual bottleneck regularization
            reconstruction_loss = self.loss_fn(
                predictions=output['reconstructed'],
                targets=batch['input_sequences'],
                mask=mask
            )
            
            # Bottleneck regularization
            bottleneck_reg = 0
            if self.bottleneck_reg_weight > 0 and 'bottleneck' in output:
                # L2 regularization on bottleneck representations
                bottleneck_reg = self.bottleneck_reg_weight * output['bottleneck'].pow(2).mean()
            
            # Total loss
            total_loss = reconstruction_loss + bottleneck_reg
        
        # Metrics
        metrics = {
            'reconstruction_loss': reconstruction_loss.item(),
            'bottleneck_reg': bottleneck_reg.item() if torch.is_tensor(bottleneck_reg) else bottleneck_reg,
            'total_loss': total_loss.item()
        }
        
        # Add bottleneck statistics
        if 'bottleneck' in output:
            with torch.no_grad():
                metrics['bottleneck_mean'] = output['bottleneck'].mean().item()
                metrics['bottleneck_std'] = output['bottleneck'].std().item()
                metrics['bottleneck_sparsity'] = (output['bottleneck'].abs() < 0.01).float().mean().item()
        
        return total_loss, metrics
    
    def _compute_validation_loss(self, batch: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute validation loss without regularization.
        
        Args:
            batch: Input batch
            output: Model output
            
        Returns:
            Validation loss
        """
        # Use the same loss function as training, but without bottleneck regularization
        mask = batch.get('attention_mask', None)
        
        if hasattr(self.loss_fn, 'bottleneck_weight'):
            # For composite loss, just use reconstruction component
            reconstruction_loss = self.loss_fn.reconstruction_loss(
                predictions=output['reconstructed'],
                targets=batch['input_sequences'],
                mask=mask
            )
        else:
            # Use the configured loss function directly
            reconstruction_loss = self.loss_fn(
                predictions=output['reconstructed'],
                targets=batch['input_sequences'],
                mask=mask
            )
        
        return reconstruction_loss
    
    def _compute_metrics(self, batch: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            batch: Input batch
            output: Model output
            
        Returns:
            Metrics dictionary
        """
        metrics = {}
        
        # Cosine similarity
        with torch.no_grad():
            # Flatten sequences for comparison
            input_flat = batch['input_sequences'].reshape(-1, batch['input_sequences'].size(-1))
            output_flat = output['reconstructed'].reshape(-1, output['reconstructed'].size(-1))
            
            # Compute cosine similarity
            cosine_sim = F.cosine_similarity(input_flat, output_flat, dim=1)
            
            # Apply mask if available
            if 'attention_mask' in batch:
                mask_flat = batch['attention_mask'].reshape(-1)
                cosine_sim = cosine_sim * mask_flat
                metrics['cosine_similarity'] = (cosine_sim.sum() / mask_flat.sum()).item()
            else:
                metrics['cosine_similarity'] = cosine_sim.mean().item()
            
            # Bottleneck statistics
            if 'bottleneck' in output:
                metrics['bottleneck_mean'] = output['bottleneck'].mean().item()
                metrics['bottleneck_std'] = output['bottleneck'].std().item()
                metrics['bottleneck_l2'] = output['bottleneck'].norm(p=2, dim=-1).mean().item()
                
                # Information retention (variance preserved)
                input_var = batch['input_sequences'].var(dim=1).mean()
                bottleneck_var = output['bottleneck'].var(dim=-1).mean()
                metrics['variance_ratio'] = (bottleneck_var / input_var).item()
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Efficiently move batch to device with pinned memory support.
        
        Args:
            batch: Input batch
            
        Returns:
            Batch on target device
        """
        device_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                # Non-blocking transfer if memory is pinned
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch
    
    def train(self) -> Dict[str, Any]:
        """
        Complete optimized training loop.
        
        Returns:
            Dictionary with best metrics and training statistics
        """
        logger.info("Starting optimized training...")
        training_start = time.perf_counter()
        
        # Initialize best metrics (may be overridden by resumed checkpoint)
        best_metrics = {
            'best_val_loss': self.best_val_loss,
            'best_cosine_similarity': self.best_cosine_similarity,
            'best_epoch': self.current_epoch,
            'total_time': 0,
            'avg_throughput': 0
        }
        
        all_throughputs = []
        
        # Support resumed training - start from current_epoch + 1
        start_epoch = self.current_epoch + 1 if self.current_epoch > 0 else 1
        
        for epoch in range(start_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.perf_counter()
            
            logger.info(f"\nEpoch {epoch}/{self.num_epochs}")
            logger.info("-" * 40)
            
            # Training
            train_loss, train_metrics = self.train_epoch()
            
            # Validation
            val_loss, val_metrics = self.validate()
            
            # Update learning rate (handle different scheduler types)
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    # ReduceLROnPlateau requires validation metric
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Epoch summary
            epoch_time = time.perf_counter() - epoch_start
            all_throughputs.append(train_metrics['sequences_per_sec'])
            
            logger.info(
                f"Epoch {epoch} Summary:\n"
                f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}\n"
                f"  Cosine Similarity: {val_metrics['cosine_similarity']:.4f}\n"
                f"  Learning Rate: {current_lr:.2e}\n"
                f"  Throughput: {train_metrics['sequences_per_sec']:.1f} seq/s\n"
                f"  Epoch Time: {epoch_time:.2f}s"
            )
            
            # Performance monitoring
            if self.performance_monitor:
                # Compute attention statistics for monitoring
                attention_stats = None
                if hasattr(self.model, 'use_attention') and self.model.use_attention:
                    # Use a sample batch from validation for attention analysis
                    sample_batch = next(iter(self.val_loader))
                    for key in sample_batch:
                        if isinstance(sample_batch[key], torch.Tensor):
                            sample_batch[key] = sample_batch[key].to(self.device)
                    attention_stats = self._compute_attention_statistics(sample_batch)
                
                self.performance_monitor.log_epoch(
                    epoch,
                    {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'cosine_similarity': val_metrics['cosine_similarity'],
                        'learning_rate': current_lr,
                        'sequences_per_sec': train_metrics['sequences_per_sec'],
                        'gradient_norm': train_metrics['avg_gradient_norm'],
                        'bottleneck_std': val_metrics.get('bottleneck_std', 0),
                        'variance_ratio': val_metrics.get('variance_ratio', 0),
                        'throughput': train_metrics['sequences_per_sec']  # Alias for compatibility
                    },
                    attention_stats  # Pass attention statistics
                )
            
            # Check for improvement
            improved = False
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                best_metrics['best_val_loss'] = val_loss
                best_metrics['best_epoch'] = epoch
                self.patience_counter = 0
                improved = True
            else:
                self.patience_counter += 1
            
            if val_metrics['cosine_similarity'] > best_metrics['best_cosine_similarity']:
                best_metrics['best_cosine_similarity'] = val_metrics['cosine_similarity']
                improved = True
            
            # Checkpointing
            if improved or epoch % self.checkpoint_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_loss': self.best_val_loss,
                    'val_metrics': val_metrics,
                    'config': self.config
                }
                
                if self.async_checkpointer:
                    # Async non-blocking save
                    checkpoint_name = f"checkpoint_epoch_{epoch}.pth" if not improved else "best_model.pth"
                    self.async_checkpointer.save_checkpoint_async(checkpoint, checkpoint_name)
                    if improved:
                        logger.info(f"  ✓ New best model (async saved)")
                else:
                    # Fallback to synchronous save
                    checkpoint_path = Path(self.config['paths']['checkpoint_dir']) / f"checkpoint_epoch_{epoch}.pth"
                    torch.save(checkpoint, checkpoint_path)
            
            # Enhanced early stopping with attention monitoring
            early_stop = False
            
            # Standard early stopping based on validation loss
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"\n⏹️  Early stopping triggered after {epoch} epochs (no improvement)")
                early_stop = True
            
            # Attention degradation monitoring (if enabled and attention is available)
            if (hasattr(self.model, 'use_attention') and self.model.use_attention and 
                attention_stats and self.config.get('monitoring', {}).get('attention_monitoring', False)):
                
                attention_entropy = attention_stats.get('avg_attention_entropy', 0)
                attention_diversity = attention_stats.get('attention_diversity', 0)
                
                # Check for attention degradation
                entropy_threshold = self.config.get('monitoring', {}).get('attention_entropy_threshold', 0.5)
                diversity_min = self.config.get('monitoring', {}).get('attention_diversity_min', 0.001)
                
                if attention_entropy > entropy_threshold:
                    logger.warning(f"⚠️  Attention entropy high: {attention_entropy:.3f} > {entropy_threshold}")
                    logger.warning("   Attention patterns becoming too diffuse")
                
                if attention_diversity < diversity_min:
                    logger.warning(f"⚠️  Attention diversity low: {attention_diversity:.4f} < {diversity_min}")
                    logger.warning("   Attention heads becoming too similar")
                
                # Stop if attention severely degraded
                if attention_entropy > entropy_threshold * 1.5 and attention_diversity < diversity_min * 0.5:
                    logger.info(f"\n⚠️  Early stopping due to attention degradation at epoch {epoch}")
                    logger.info(f"   Entropy: {attention_entropy:.3f}, Diversity: {attention_diversity:.4f}")
                    early_stop = True
            
            if early_stop:
                break
            
            # Check if we've reached target performance
            if val_metrics['cosine_similarity'] >= 0.95:
                logger.info(f"\n✓ Target performance reached! Cosine similarity: {val_metrics['cosine_similarity']:.4f}")
                break
        
        # Training complete
        total_time = time.perf_counter() - training_start
        best_metrics['total_time'] = total_time
        best_metrics['avg_throughput'] = np.mean(all_throughputs)
        
        return best_metrics
    
    def _compute_attention_statistics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute attention statistics for monitoring.
        
        Returns attention entropy, diversity, and other metrics if the model
        has attention mechanisms, otherwise returns zeros.
        """
        # Check if model has attention
        if not hasattr(self.model, 'use_attention') or not self.model.use_attention:
            return {
                'avg_attention_entropy': 0.0,
                'attention_diversity': 0.0,
                'attention_sharpness': 0.0
            }
        
        try:
            self.model.eval()
            with torch.no_grad():
                # Forward pass to get attention weights
                output = self.model(batch)
                
                # Check if the decoder provides attention weights
                decoder = self.model.decoder
                if hasattr(decoder, 'last_attention_weights') and decoder.last_attention_weights is not None:
                    attention_weights = decoder.last_attention_weights
                    
                    # Compute entropy (attention sharpness)
                    # Higher entropy = more diffuse attention, lower entropy = more focused
                    log_attention = torch.log(attention_weights + 1e-8)
                    entropy = -(attention_weights * log_attention).sum(dim=-1)  # [batch, heads, seq_len]
                    avg_entropy = entropy.mean().item()
                    
                    # Compute diversity (how different are the attention heads)
                    # Higher diversity = heads attend to different things
                    num_heads = attention_weights.size(1)
                    if num_heads > 1:
                        # Pairwise cosine similarity between attention heads
                        heads_flat = attention_weights.view(attention_weights.size(0), num_heads, -1)  # [batch, heads, seq*seq]
                        heads_norm = F.normalize(heads_flat, p=2, dim=-1)
                        
                        # Compute pairwise similarities
                        similarities = torch.bmm(heads_norm, heads_norm.transpose(1, 2))  # [batch, heads, heads]
                        
                        # Get upper triangular (excluding diagonal) and compute diversity
                        mask = torch.triu(torch.ones(num_heads, num_heads), diagonal=1).bool()
                        avg_similarity = similarities[:, mask].mean().item()
                        diversity = 1.0 - avg_similarity  # Higher diversity = lower similarity
                    else:
                        diversity = 0.0
                    
                    # Compute sharpness (how focused is the attention)
                    max_attention = attention_weights.max(dim=-1)[0]  # [batch, heads, seq_len]
                    avg_sharpness = max_attention.mean().item()
                    
                    return {
                        'avg_attention_entropy': avg_entropy,
                        'attention_diversity': diversity,
                        'attention_sharpness': avg_sharpness
                    }
                else:
                    # Attention decoder exists but no weights available
                    return {
                        'avg_attention_entropy': 0.0,
                        'attention_diversity': 0.0, 
                        'attention_sharpness': 0.0
                    }
                    
        except Exception as e:
            logger.warning(f"Failed to compute attention statistics: {e}")
            return {
                'avg_attention_entropy': 0.0,
                'attention_diversity': 0.0,
                'attention_sharpness': 0.0
            }
        finally:
            self.model.train()