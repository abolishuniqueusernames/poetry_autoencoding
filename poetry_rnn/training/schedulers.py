"""
Advanced Learning Rate Schedulers for Optimized Training

This module provides sophisticated learning rate scheduling strategies
including cosine annealing with warm restarts and exponential decay.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Optional, List
import numpy as np


class CosineAnnealingWarmRestartsWithDecay(_LRScheduler):
    """
    Cosine Annealing with Warm Restarts and Decay Factor.
    
    This scheduler implements SGDR (Stochastic Gradient Descent with Warm Restarts)
    with an additional decay factor applied after each restart, leading to
    progressively smaller learning rates over time.
    
    The learning rate follows a cosine annealing schedule within each cycle,
    and the maximum learning rate is multiplied by decay_factor after each restart.
    
    Args:
        optimizer: Wrapped optimizer
        T_0: Number of epochs for the first restart
        T_mult: Factor to increase T_i after each restart (default: 2)
        eta_min: Minimum learning rate (default: 1e-6)
        decay_factor: Factor to decay max LR after each restart (default: 0.95)
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 2,
        eta_min: float = 1e-6,
        decay_factor: float = 0.95,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay_factor = decay_factor
        
        # Track restart information
        self.T_cur = 0  # Current position within the current cycle
        self.T_i = T_0  # Current cycle length
        self.cycle = 0  # Current cycle number
        self.decay_multiplier = 1.0  # Current decay multiplier
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch."""
        if self.last_epoch == 0:
            return self.base_lrs
        
        # Check if we need to restart
        if self.T_cur >= self.T_i:
            # Restart: reset counter and update cycle parameters
            self.cycle += 1
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult
            self.decay_multiplier *= self.decay_factor
        
        # Calculate cosine annealing within current cycle
        lrs = []
        for base_lr in self.base_lrs:
            # Maximum learning rate for this cycle (with decay)
            max_lr = base_lr * self.decay_multiplier
            
            # Cosine annealing
            lr = self.eta_min + (max_lr - self.eta_min) * \
                 (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            
            lrs.append(lr)
        
        self.T_cur += 1
        return lrs
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Cosine Annealing with Linear Warmup.
    
    Learning rate starts from a small value and linearly increases to the
    initial learning rate over warmup_epochs, then follows cosine annealing.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of epochs for warmup
        max_epochs: Total number of training epochs
        warmup_start_lr: Starting learning rate for warmup (default: 1e-8)
        eta_min: Minimum learning rate after annealing (default: 1e-6)
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class PolynomialLR(_LRScheduler):
    """
    Polynomial Learning Rate Decay.
    
    The learning rate is decayed using a polynomial function:
    lr = base_lr * (1 - epoch/max_epochs) ^ power
    
    Args:
        optimizer: Wrapped optimizer
        max_epochs: Maximum number of epochs
        power: Power of the polynomial (default: 1.0 for linear decay)
        eta_min: Minimum learning rate (default: 1e-6)
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        power: float = 1.0,
        eta_min: float = 1e-6,
        last_epoch: int = -1
    ):
        self.max_epochs = max_epochs
        self.power = power
        self.eta_min = eta_min
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch."""
        if self.last_epoch >= self.max_epochs:
            return [self.eta_min for _ in self.base_lrs]
        
        factor = (1 - self.last_epoch / self.max_epochs) ** self.power
        return [
            self.eta_min + (base_lr - self.eta_min) * factor
            for base_lr in self.base_lrs
        ]


class CyclicalLR(_LRScheduler):
    """
    Cyclical Learning Rate with Triangular Schedule.
    
    The learning rate cyclically varies between base_lr and max_lr following
    a triangular wave pattern. This can help escape local minima and find
    better solutions.
    
    Args:
        optimizer: Wrapped optimizer
        base_lr: Lower boundary of learning rate
        max_lr: Upper boundary of learning rate
        step_size: Number of training iterations in half cycle
        mode: One of {'triangular', 'triangular2', 'exp_range'}
        gamma: Constant for 'exp_range' mode (default: 0.99)
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        max_lr: float,
        step_size: int,
        mode: str = 'triangular',
        gamma: float = 0.99,
        last_epoch: int = -1
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        if mode not in ['triangular', 'triangular2', 'exp_range']:
            raise ValueError(f"Invalid mode: {mode}")
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch."""
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale = 1.0
        elif self.mode == 'triangular2':
            scale = 1 / (2 ** (cycle - 1))
        else:  # exp_range
            scale = self.gamma ** self.last_epoch
        
        lrs = []
        for base_lr in self.base_lrs:
            base = self.base_lr * scale
            amplitude = (self.max_lr - self.base_lr) * scale
            lr = base + amplitude * max(0, 1 - x)
            lrs.append(lr)
        
        return lrs


class AdaptiveLRScheduler:
    """
    Adaptive Learning Rate Scheduler based on validation metrics.
    
    This scheduler monitors validation loss and adaptively adjusts the learning
    rate based on performance plateaus. It combines ideas from ReduceLROnPlateau
    with smooth transitions.
    
    Args:
        optimizer: Wrapped optimizer
        mode: One of 'min' or 'max'
        factor: Factor by which to reduce learning rate
        patience: Number of epochs with no improvement to wait
        threshold: Threshold for measuring improvement
        cooldown: Number of epochs to wait before resuming normal operation
        min_lr: Minimum learning rate
        smoothing: Exponential moving average smoothing factor
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 1e-7,
        smoothing: float = 0.9
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.smoothing = smoothing
        
        # State tracking
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_reduction_epoch = 0
        self.smoothed_metric = None
        
        # Comparison function
        self.is_better = (lambda a, b: a < b - threshold) if mode == 'min' else \
                        (lambda a, b: a > b + threshold)
    
    def step(self, metric: float, epoch: Optional[int] = None):
        """
        Update learning rate based on validation metric.
        
        Args:
            metric: Current validation metric
            epoch: Current epoch number
        """
        # Apply exponential moving average smoothing
        if self.smoothed_metric is None:
            self.smoothed_metric = metric
        else:
            self.smoothed_metric = self.smoothing * self.smoothed_metric + (1 - self.smoothing) * metric
        
        current_metric = self.smoothed_metric
        
        # Check if in cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        
        # Check for improvement
        if self.is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        # Reduce learning rate if patience exceeded
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
    
    def _reduce_lr(self, epoch: Optional[int] = None):
        """Reduce learning rate by factor."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if epoch is not None:
                print(f"Epoch {epoch}: Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    config: dict
) -> Optional[_LRScheduler]:
    """
    Factory function to create learning rate schedulers.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler to create
        config: Scheduler configuration
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_type == 'cosine_annealing_warm_restarts':
        return CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2),
            eta_min=config.get('eta_min', 1e-6),
            decay_factor=config.get('decay_factor', 0.95)
        )
    
    elif scheduler_type == 'warmup_cosine':
        return WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=config.get('warmup_epochs', 5),
            max_epochs=config.get('max_epochs', 100),
            warmup_start_lr=config.get('warmup_start_lr', 1e-8),
            eta_min=config.get('eta_min', 1e-6)
        )
    
    elif scheduler_type == 'polynomial':
        return PolynomialLR(
            optimizer,
            max_epochs=config.get('max_epochs', 100),
            power=config.get('power', 1.0),
            eta_min=config.get('eta_min', 1e-6)
        )
    
    elif scheduler_type == 'cyclical':
        return CyclicalLR(
            optimizer,
            base_lr=config.get('base_lr', 1e-4),
            max_lr=config.get('max_lr', 1e-3),
            step_size=config.get('step_size', 100),
            mode=config.get('mode', 'triangular'),
            gamma=config.get('gamma', 0.99)
        )
    
    else:
        return None