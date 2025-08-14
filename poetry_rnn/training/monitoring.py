"""
Adaptive Gradient Monitoring System for RNN Training

Provides comprehensive gradient flow analysis with adaptive clipping,
automatic problem detection, and training stability recommendations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json


class GradientMonitor:
    """
    Adaptive gradient monitoring system for RNN training.
    
    Provides comprehensive gradient flow analysis with adaptive clipping,
    automatic problem detection, and training stability recommendations.
    
    Key Features:
    - Adaptive gradient clipping based on gradient statistics
    - Real-time gradient flow diagnostics
    - Vanishing/exploding gradient detection
    - Training stability warnings and recommendations
    - History tracking with configurable window sizes
    
    Usage:
        monitor = GradientMonitor(model, clip_value=1.0, adaptive=True)
        
        # In training loop after loss.backward():
        grad_norm = monitor.clip_and_monitor()
        diagnostics = monitor.diagnose_gradient_issues()
        for msg in diagnostics:
            print(msg)
        optimizer.step()
    
    Args:
        model: PyTorch model to monitor
        clip_value: Initial gradient clipping value (will adapt if adaptive=True)
        history_size: Number of gradient norms to keep in history
        vanishing_threshold: Threshold below which gradients are considered vanishing
        exploding_threshold: Threshold above which gradients are considered exploding
        adaptive: Whether to adaptively adjust clipping value
    """
    
    def __init__(
        self,
        model: nn.Module,
        clip_value: float = 5.0,
        history_size: int = 100,
        vanishing_threshold: float = 1e-6,
        exploding_threshold: float = 10.0,
        adaptive: bool = True
    ):
        self.model = model
        self.initial_clip_value = clip_value
        self.clip_value = clip_value
        self.history_size = history_size
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.adaptive = adaptive
        
        # Gradient history with efficient deque storage
        self.gradient_history = {
            'pre_clip': deque(maxlen=history_size),
            'post_clip': deque(maxlen=history_size),
            'clip_ratio': deque(maxlen=history_size),
            'clip_value_history': deque(maxlen=history_size)
        }
        
        # Adaptive clipping parameters
        self.adaptation_window = 20  # Window size for adaptation decisions
        self.clip_adjustment_factor = 0.15  # How aggressively to adjust
        self.min_clip_value = 0.1
        self.max_clip_value = 50.0
        
        # Diagnostic tracking
        self.step_count = 0
        self.last_adaptation_step = 0
        self.adaptation_frequency = 10  # Adapt every N steps
        
    def compute_gradient_norms(self) -> Dict[str, float]:
        """
        Compute gradient norms for the model.
        
        Returns:
            Dictionary with gradient norms:
            - 'total': L2 norm of all gradients
            - Individual parameter norms (optional for debugging)
        """
        total_norm_squared = 0.0
        grad_norms = {}
        
        # Compute total gradient norm efficiently
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm_squared = param.grad.data.norm(2).item() ** 2
                total_norm_squared += param_norm_squared
        
        grad_norms['total'] = total_norm_squared ** 0.5
        return grad_norms
    
    def clip_and_monitor(self) -> float:
        """
        Perform gradient clipping with monitoring and adaptive adjustment.
        
        This is the main method called during training. It:
        1. Records pre-clipping gradient norms
        2. Adaptively adjusts clipping value if needed
        3. Performs gradient clipping
        4. Records post-clipping statistics
        
        Returns:
            Pre-clipping gradient norm for logging
        """
        self.step_count += 1
        
        # Compute pre-clipping gradient norms
        pre_clip_norms = self.compute_gradient_norms()
        pre_clip_total = pre_clip_norms['total']
        
        # Store pre-clipping statistics
        self.gradient_history['pre_clip'].append(pre_clip_total)
        
        # Adaptive clipping adjustment (every N steps)
        if (self.adaptive and 
            self.step_count - self.last_adaptation_step >= self.adaptation_frequency and
            len(self.gradient_history['pre_clip']) >= self.adaptation_window):
            
            old_clip_value = self.clip_value
            self._adapt_clip_value()
            
            # Track clip value changes
            if abs(self.clip_value - old_clip_value) > 0.01:
                self.last_adaptation_step = self.step_count
        
        # Record current clip value
        self.gradient_history['clip_value_history'].append(self.clip_value)
        
        # Perform gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
        
        # Compute post-clipping gradient norms
        post_clip_norms = self.compute_gradient_norms()
        post_clip_total = post_clip_norms['total']
        
        # Store post-clipping statistics
        self.gradient_history['post_clip'].append(post_clip_total)
        
        # Compute clipping ratio (how much was clipped)
        clip_ratio = post_clip_total / max(pre_clip_total, 1e-8)
        self.gradient_history['clip_ratio'].append(clip_ratio)
        
        return pre_clip_total
    
    def _adapt_clip_value(self):
        """
        Adaptively adjust gradient clipping value based on recent statistics.
        
        Strategy:
        - If gradients consistently exceed clip value → increase clip value
        - If gradients consistently well below clip value → decrease clip value
        - If high variance in gradients → slightly increase clip value for stability
        - Always respect min/max bounds
        """
        # Get recent gradient statistics
        recent_pre_clip = list(self.gradient_history['pre_clip'])[-self.adaptation_window:]
        recent_clip_ratios = list(self.gradient_history['clip_ratio'])[-self.adaptation_window:]
        
        if len(recent_pre_clip) < self.adaptation_window:
            return
        
        avg_norm = np.mean(recent_pre_clip)
        std_norm = np.std(recent_pre_clip)
        avg_clip_ratio = np.mean(recent_clip_ratios)
        
        # Decision logic for adaptation
        adjustment_factor = 0.0
        
        # Case 1: Gradients consistently exceed clip value (aggressive clipping)
        if avg_clip_ratio < 0.5:  # More than 50% clipped on average
            adjustment_factor = self.clip_adjustment_factor
            reason = "aggressive_clipping"
        
        # Case 2: Gradients consistently well below clip value (minimal clipping)
        elif avg_clip_ratio > 0.95 and avg_norm < self.clip_value * 0.3:
            adjustment_factor = -self.clip_adjustment_factor * 0.7  # More conservative reduction
            reason = "minimal_clipping"
        
        # Case 3: High variance suggests instability
        elif std_norm > avg_norm * 1.5:
            adjustment_factor = self.clip_adjustment_factor * 0.5  # Smaller increase for stability
            reason = "high_variance"
        
        # Apply adjustment
        if abs(adjustment_factor) > 0.01:
            new_clip_value = self.clip_value * (1 + adjustment_factor)
            new_clip_value = max(self.min_clip_value, min(new_clip_value, self.max_clip_value))
            
            self.clip_value = new_clip_value
    
    def diagnose_gradient_issues(self) -> List[str]:
        """
        Analyze recent gradients and identify potential training issues.
        
        Provides actionable warnings and recommendations based on gradient
        flow patterns. This is the main diagnostic interface.
        
        Returns:
            List of diagnostic messages with warnings and recommendations
        """
        diagnostics = []
        
        if len(self.gradient_history['pre_clip']) < 5:
            return ["📊 Insufficient gradient history for diagnosis (need ≥5 steps)"]
        
        # Get recent statistics
        recent_pre = list(self.gradient_history['pre_clip'])[-10:]
        recent_post = list(self.gradient_history['post_clip'])[-10:]
        recent_ratios = list(self.gradient_history['clip_ratio'])[-10:]
        
        avg_norm = np.mean(recent_pre)
        std_norm = np.std(recent_pre)
        avg_ratio = np.mean(recent_ratios)
        
        # 1. Check for vanishing gradients
        if avg_norm < self.vanishing_threshold:
            diagnostics.append(f"🚨 VANISHING GRADIENTS: Average norm {avg_norm:.2e} < {self.vanishing_threshold:.2e}")
            diagnostics.append("   💡 Recommendations:")
            diagnostics.append("      • Reduce learning rate by 2-5x")
            diagnostics.append("      • Check weight initialization (use Xavier/He)")
            diagnostics.append("      • Consider skip connections or residual architecture")
            diagnostics.append("      • Verify loss function is not saturated")
        
        # 2. Check for exploding gradients
        elif avg_norm > self.exploding_threshold:
            diagnostics.append(f"🚨 EXPLODING GRADIENTS: Average norm {avg_norm:.2f} > {self.exploding_threshold}")
            diagnostics.append(f"   📎 Current clip value: {self.clip_value:.2f} (adaptive: {self.adaptive})")
            diagnostics.append("   💡 Recommendations:")
            diagnostics.append("      • Reduce learning rate by 5-10x")
            diagnostics.append("      • Enable adaptive clipping if not already on")
            diagnostics.append("      • Check for NaN values in loss/data")
            diagnostics.append("      • Consider gradient accumulation to reduce batch size")
        
        # 3. Check for gradient instability (high variance)
        elif len(recent_pre) > 1 and std_norm > avg_norm * 2:
            diagnostics.append(f"⚠️ UNSTABLE GRADIENTS: High variance (std={std_norm:.3f}, mean={avg_norm:.3f})")
            diagnostics.append("   💡 Recommendations:")
            diagnostics.append("      • Consider learning rate warmup/scheduling")
            diagnostics.append("      • Increase batch size if possible")
            diagnostics.append("      • Check data shuffling and augmentation")
            diagnostics.append(f"      • Current adaptive clipping: {'ON' if self.adaptive else 'OFF'}")
        
        # 4. Analyze clipping effectiveness
        if len(recent_ratios) >= 5:
            if avg_ratio < 0.2:
                diagnostics.append(f"📎 AGGRESSIVE CLIPPING: Average ratio {avg_ratio:.3f} - gradients heavily clipped")
                diagnostics.append("   💡 Consider: Reducing learning rate instead of relying on clipping")
            
            elif avg_ratio > 0.98:
                diagnostics.append(f"📎 MINIMAL CLIPPING: Average ratio {avg_ratio:.3f} - clipping rarely active")
                diagnostics.append("   💡 Consider: Current clipping value may be too high")
        
        # 5. Adaptive clipping status
        if self.adaptive and len(self.gradient_history['clip_value_history']) > 1:
            clip_values = list(self.gradient_history['clip_value_history'])
            if len(clip_values) >= 20:
                recent_clip_change = abs(clip_values[-1] - clip_values[-20])
                if recent_clip_change > self.initial_clip_value * 0.5:
                    diagnostics.append(f"🔧 ADAPTIVE CLIPPING ACTIVE: Value changed from {clip_values[-20]:.2f} to {clip_values[-1]:.2f}")
        
        # 6. Overall health summary
        if not diagnostics:
            diagnostics.append(f"✅ HEALTHY GRADIENTS: norm={avg_norm:.4f}±{std_norm:.4f}, clip_ratio={avg_ratio:.3f}")
            diagnostics.append(f"   📊 Current clip value: {self.clip_value:.2f} (adaptive: {'ON' if self.adaptive else 'OFF'})")
        
        return diagnostics
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive gradient monitoring statistics.
        
        Returns detailed statistics suitable for logging, visualization,
        or integration with training monitoring systems.
        
        Returns:
            Dictionary with gradient statistics and health indicators
        """
        if len(self.gradient_history['pre_clip']) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 gradient steps for statistics",
                "step_count": self.step_count
            }
        
        # Get recent data
        recent_pre = list(self.gradient_history['pre_clip'])[-20:]
        recent_post = list(self.gradient_history['post_clip'])[-20:]
        recent_ratios = list(self.gradient_history['clip_ratio'])[-20:]
        
        stats = {
            'step_count': self.step_count,
            'monitoring_config': {
                'adaptive_clipping': self.adaptive,
                'current_clip_value': self.clip_value,
                'initial_clip_value': self.initial_clip_value,
                'vanishing_threshold': self.vanishing_threshold,
                'exploding_threshold': self.exploding_threshold
            },
            'gradient_norms': {
                'pre_clip_mean': float(np.mean(recent_pre)),
                'pre_clip_std': float(np.std(recent_pre)),
                'pre_clip_latest': float(recent_pre[-1]),
                'post_clip_mean': float(np.mean(recent_post)),
                'post_clip_std': float(np.std(recent_post)),
                'post_clip_latest': float(recent_post[-1])
            },
            'clipping_analysis': {
                'average_clip_ratio': float(np.mean(recent_ratios)),
                'clipping_frequency': float(sum(1 for r in recent_ratios if r < 0.99) / len(recent_ratios)),
                'aggressive_clipping_freq': float(sum(1 for r in recent_ratios if r < 0.5) / len(recent_ratios)),
                'minimal_clipping_freq': float(sum(1 for r in recent_ratios if r > 0.98) / len(recent_ratios))
            },
            'health_indicators': {
                'vanishing_risk': float(np.mean(recent_pre) < self.vanishing_threshold),
                'exploding_risk': float(np.mean(recent_pre) > self.exploding_threshold),
                'instability_risk': float(np.std(recent_pre) > np.mean(recent_pre) * 2) if len(recent_pre) > 1 else 0.0,
                'overall_health_score': self._compute_health_score(recent_pre, recent_ratios)
            },
            'adaptation_info': {
                'adaptations_made': self.step_count // self.adaptation_frequency if self.adaptive else 0,
                'last_adaptation_step': self.last_adaptation_step,
                'clip_value_range': {
                    'min': float(np.min(list(self.gradient_history['clip_value_history']))) if self.gradient_history['clip_value_history'] else self.clip_value,
                    'max': float(np.max(list(self.gradient_history['clip_value_history']))) if self.gradient_history['clip_value_history'] else self.clip_value
                }
            }
        }
        
        return stats
    
    def _compute_health_score(self, recent_norms: List[float], recent_ratios: List[float]) -> float:
        """
        Compute overall gradient health score (0=bad, 1=excellent).
        
        Considers multiple factors:
        - Gradient magnitude (not too small/large)
        - Gradient stability (low variance)
        - Clipping effectiveness (moderate clipping)
        """
        if len(recent_norms) < 2:
            return 0.5
        
        avg_norm = np.mean(recent_norms)
        std_norm = np.std(recent_norms)
        avg_ratio = np.mean(recent_ratios)
        
        # Score components (each 0-1)
        magnitude_score = 1.0
        if avg_norm < self.vanishing_threshold:
            magnitude_score = 0.1  # Very bad
        elif avg_norm > self.exploding_threshold:
            magnitude_score = 0.2  # Bad
        elif self.vanishing_threshold * 10 <= avg_norm <= self.exploding_threshold * 0.1:
            magnitude_score = 1.0  # Good range
        else:
            magnitude_score = 0.7  # Okay
        
        # Stability score (lower variance is better)
        stability_score = 1.0 / (1.0 + (std_norm / max(avg_norm, 1e-8)))
        
        # Clipping score (moderate clipping is optimal)
        if 0.7 <= avg_ratio <= 0.95:
            clipping_score = 1.0  # Optimal range
        elif 0.5 <= avg_ratio < 0.7:
            clipping_score = 0.7  # Moderate clipping
        elif avg_ratio < 0.5:
            clipping_score = 0.3  # Heavy clipping
        else:
            clipping_score = 0.8  # Minimal clipping (okay)
        
        # Weighted combination
        health_score = (0.4 * magnitude_score + 0.3 * stability_score + 0.3 * clipping_score)
        return float(np.clip(health_score, 0.0, 1.0))


def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Standalone function to compute gradient norms for any model.
    
    Utility function for quick gradient analysis without full monitoring.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with gradient norm statistics
    """
    total_norm_squared = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm_squared = param.grad.data.norm(2).item() ** 2
            total_norm_squared += param_norm_squared
            param_count += 1
    
    return {
        'total': total_norm_squared ** 0.5,
        'average': (total_norm_squared ** 0.5) / max(param_count, 1),
        'param_count': param_count
    }


def analyze_hidden_states(hidden_states: torch.Tensor, prefix: str = '') -> Dict[str, float]:
    """
    Analyze RNN hidden state statistics for debugging.
    
    Provides statistics about hidden state activations that can indicate
    training issues like saturation or dead neurons.
    
    Args:
        hidden_states: Tensor of hidden states [batch, seq_len, hidden_size]
        prefix: Prefix for returned keys (e.g., 'encoder_', 'decoder_')
        
    Returns:
        Dictionary with hidden state statistics
    """
    if hidden_states.numel() == 0:
        return {f'{prefix}error': 'Empty hidden states'}
    
    # Flatten for analysis
    flat_hidden = hidden_states.view(-1)
    
    # Basic statistics
    mean_activation = flat_hidden.mean().item()
    std_activation = flat_hidden.std().item()
    
    # Activation range
    min_activation = flat_hidden.min().item()
    max_activation = flat_hidden.max().item()
    
    # Saturation analysis (for tanh: values near ±1)
    saturation_threshold = 0.9
    saturated_count = torch.sum(torch.abs(flat_hidden) > saturation_threshold).item()
    saturation_rate = saturated_count / flat_hidden.numel()
    
    # Dead neuron analysis (consistently near zero)
    dead_threshold = 0.01
    dead_count = torch.sum(torch.abs(flat_hidden) < dead_threshold).item()
    dead_rate = dead_count / flat_hidden.numel()
    
    return {
        f'{prefix}mean_activation': mean_activation,
        f'{prefix}std_activation': std_activation,
        f'{prefix}min_activation': min_activation,
        f'{prefix}max_activation': max_activation,
        f'{prefix}saturation_rate': saturation_rate,
        f'{prefix}dead_rate': dead_rate,
        f'{prefix}activation_range': max_activation - min_activation
    }


class TrainingMonitor:
    """
    Simple training progress monitor for basic metrics tracking.
    
    Lightweight monitoring for epoch-level training metrics, loss curves,
    and basic performance tracking.
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else Path('logs')
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.epoch_metrics = defaultdict(list)
        self.batch_metrics = []
        self.start_time = None
        self.epoch_times = []
        
    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None, **kwargs):
        """Log epoch-level metrics."""
        self.epoch_metrics['epoch'].append(epoch)
        self.epoch_metrics['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.epoch_metrics['val_loss'].append(val_loss)
        
        # Log additional metrics
        for key, value in kwargs.items():
            self.epoch_metrics[key].append(value)
        
        # Track epoch time
        if self.start_time:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
            self.start_time = time.time()  # Reset for next epoch
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.epoch_metrics['train_loss']:
            return {"error": "No training data logged"}
        
        train_losses = self.epoch_metrics['train_loss']
        summary = {
            'epochs_completed': len(train_losses),
            'final_train_loss': train_losses[-1],
            'best_train_loss': min(train_losses),
            'loss_improvement': train_losses[0] - train_losses[-1] if len(train_losses) > 1 else 0,
        }
        
        if 'val_loss' in self.epoch_metrics:
            val_losses = self.epoch_metrics['val_loss']
            summary.update({
                'final_val_loss': val_losses[-1],
                'best_val_loss': min(val_losses),
                'val_loss_improvement': val_losses[0] - val_losses[-1] if len(val_losses) > 1 else 0
            })
        
        if self.epoch_times:
            summary['average_epoch_time'] = np.mean(self.epoch_times)
            summary['total_training_time'] = sum(self.epoch_times)
        
        return summary