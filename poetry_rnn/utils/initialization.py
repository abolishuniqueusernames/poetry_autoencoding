"""
RNN-Specific Weight Initialization Utilities

Implements optimal weight initialization strategies for different RNN architectures
following current best practices from neural network research.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional


def init_vanilla_rnn_weights(module: nn.Module, gain_ih: float = 1.0, gain_hh: float = 1.0):
    """
    Initialize vanilla RNN weights with proper orthogonal recurrent connections.
    
    Strategy:
    - weight_ih (input-to-hidden): Xavier uniform initialization
    - weight_hh (hidden-to-hidden): Orthogonal initialization  
    - biases: Zero initialization
    
    This follows Saxe et al. (2014) recommendations for RNN initialization.
    Orthogonal initialization prevents vanishing gradients in recurrent connections.
    
    Args:
        module: RNN module to initialize
        gain_ih: Gain factor for input-to-hidden weights
        gain_hh: Gain factor for hidden-to-hidden weights (typically 1.0)
    """
    for name, param in module.named_parameters():
        if 'weight_ih' in name:
            # Input-to-hidden: Xavier initialization
            nn.init.xavier_uniform_(param, gain=gain_ih)
        elif 'weight_hh' in name:
            # Hidden-to-hidden: Orthogonal initialization for gradient flow
            nn.init.orthogonal_(param, gain=gain_hh)
        elif 'bias' in name:
            # Biases: Zero initialization (standard for vanilla RNNs)
            nn.init.zeros_(param)


def init_lstm_weights(module: nn.Module, forget_bias: float = 1.0):
    """
    Initialize LSTM weights with proper gate-specific initialization.
    
    Strategy:
    - weight_ih: Xavier uniform initialization
    - weight_hh: Orthogonal initialization
    - bias_ih/bias_hh: Zeros, except forget gate bias = forget_bias
    
    The forget gate bias is initialized to a positive value (typically 1.0)
    to encourage the LSTM to remember information initially.
    
    Args:
        module: LSTM module to initialize
        forget_bias: Initial bias for forget gate (default: 1.0)
    """
    for name, param in module.named_parameters():
        if 'weight_ih' in name:
            # Input-to-hidden: Xavier initialization
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            # Hidden-to-hidden: Orthogonal initialization
            nn.init.orthogonal_(param)
        elif 'bias_ih' in name or 'bias_hh' in name:
            # Initialize all biases to zero first
            nn.init.zeros_(param)
            
            # Set forget gate bias to positive value
            # LSTM bias layout: [input, forget, cell, output]
            hidden_size = param.size(0) // 4
            forget_start = hidden_size
            forget_end = 2 * hidden_size
            param.data[forget_start:forget_end].fill_(forget_bias)


def init_gru_weights(module: nn.Module):
    """
    Initialize GRU weights following best practices.
    
    Strategy:
    - weight_ih: Xavier uniform initialization
    - weight_hh: Orthogonal initialization
    - biases: Zero initialization
    
    Args:
        module: GRU module to initialize
    """
    for name, param in module.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)


def init_linear_layer(module: nn.Linear, gain: float = 1.0, bias_fill: float = 0.0):
    """
    Initialize linear/projection layers with appropriate scaling.
    
    Args:
        module: Linear layer to initialize
        gain: Xavier gain factor (use < 1.0 for bottleneck stability)
        bias_fill: Value to fill bias with (typically 0.0)
    """
    nn.init.xavier_uniform_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias_fill)


def init_rnn_model(model: nn.Module, rnn_type: str = 'vanilla', 
                   bottleneck_gain: float = 0.1, projection_gain: float = 0.5):
    """
    Initialize an entire RNN-based model with appropriate schemes.
    
    Automatically detects and initializes different layer types:
    - RNN layers: Type-specific initialization
    - Linear layers: Xavier with configurable gain
    - Bottleneck layers: Small gain for compression stability
    
    Args:
        model: Model to initialize
        rnn_type: Type of RNN ('vanilla', 'lstm', 'gru')
        bottleneck_gain: Gain for bottleneck compression layers (small for stability)
        projection_gain: Gain for regular projection layers
    """
    for name, module in model.named_modules():
        # RNN layers
        if isinstance(module, nn.RNN):
            init_vanilla_rnn_weights(module)
        elif isinstance(module, nn.LSTM):
            init_lstm_weights(module)
        elif isinstance(module, nn.GRU):
            init_gru_weights(module)
        
        # Custom vanilla RNN cell (our implementation)
        elif hasattr(module, 'weight_ih') and hasattr(module, 'weight_hh'):
            init_vanilla_rnn_weights(module)
        
        # Linear layers with context-aware gain
        elif isinstance(module, nn.Linear):
            # Detect bottleneck layers by name or size
            is_bottleneck = (
                'bottleneck' in name.lower() or 
                'compression' in name.lower() or
                'projection' in name.lower() and module.out_features < 30
            )
            
            if is_bottleneck:
                init_linear_layer(module, gain=bottleneck_gain)
            else:
                init_linear_layer(module, gain=projection_gain)


def get_initialization_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Generate a summary of initialization applied to model layers.
    
    Useful for debugging and verifying initialization was applied correctly.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with initialization summary
    """
    summary = {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'layers': {},
        'initialization_schemes': set()
    }
    
    for name, module in model.named_modules():
        if len(list(module.parameters())) == 0:
            continue
            
        layer_info = {
            'type': type(module).__name__,
            'parameters': sum(p.numel() for p in module.parameters(recurse=False))
        }
        
        # Detect initialization scheme
        if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            layer_info['init_scheme'] = f"{module.__class__.__name__.lower()}_optimized"
            summary['initialization_schemes'].add(layer_info['init_scheme'])
        elif hasattr(module, 'weight_ih') and hasattr(module, 'weight_hh'):
            layer_info['init_scheme'] = "vanilla_rnn_optimized"
            summary['initialization_schemes'].add(layer_info['init_scheme'])
        elif isinstance(module, nn.Linear):
            if module.out_features < 30:
                layer_info['init_scheme'] = "bottleneck_linear"
            else:
                layer_info['init_scheme'] = "standard_linear"
            summary['initialization_schemes'].add(layer_info['init_scheme'])
        
        if layer_info['parameters'] > 0:
            summary['layers'][name] = layer_info
    
    summary['initialization_schemes'] = list(summary['initialization_schemes'])
    return summary


def verify_initialization_quality(model: nn.Module, verbose: bool = True) -> Dict[str, float]:
    """
    Verify that initialization produces reasonable weight distributions.
    
    Checks for common initialization problems:
    - Weights too large/small (saturation/vanishing)
    - Poor variance preservation
    - Asymmetric distributions
    
    Args:
        model: Model to verify
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary with quality metrics
    """
    import numpy as np
    
    metrics = {
        'total_params': 0,
        'weight_std_mean': 0.0,
        'weight_std_std': 0.0,
        'bias_nonzero_fraction': 0.0,
        'large_weight_fraction': 0.0,  # |w| > 2
        'small_weight_fraction': 0.0   # |w| < 0.01
    }
    
    weight_stds = []
    bias_nonzero_count = 0
    total_bias_count = 0
    large_weight_count = 0
    small_weight_count = 0
    total_weight_count = 0
    
    for name, param in model.named_parameters():
        param_data = param.data.cpu().numpy()
        metrics['total_params'] += param.numel()
        
        if 'weight' in name:
            std = np.std(param_data)
            weight_stds.append(std)
            
            large_weight_count += np.sum(np.abs(param_data) > 2.0)
            small_weight_count += np.sum(np.abs(param_data) < 0.01)
            total_weight_count += param.numel()
            
            if verbose:
                print(f"{name}: std={std:.4f}, mean={np.mean(param_data):.4f}, "
                      f"min={np.min(param_data):.4f}, max={np.max(param_data):.4f}")
        
        elif 'bias' in name:
            nonzero_count = np.sum(np.abs(param_data) > 1e-6)
            bias_nonzero_count += nonzero_count
            total_bias_count += param.numel()
            
            if verbose:
                print(f"{name}: nonzero_fraction={nonzero_count/param.numel():.3f}, "
                      f"mean={np.mean(param_data):.4f}")
    
    if weight_stds:
        metrics['weight_std_mean'] = np.mean(weight_stds)
        metrics['weight_std_std'] = np.std(weight_stds)
    
    if total_bias_count > 0:
        metrics['bias_nonzero_fraction'] = bias_nonzero_count / total_bias_count
    
    if total_weight_count > 0:
        metrics['large_weight_fraction'] = large_weight_count / total_weight_count
        metrics['small_weight_fraction'] = small_weight_count / total_weight_count
    
    if verbose:
        print(f"\nInitialization Quality Summary:")
        print(f"  Weight std mean: {metrics['weight_std_mean']:.4f} ± {metrics['weight_std_std']:.4f}")
        print(f"  Bias nonzero fraction: {metrics['bias_nonzero_fraction']:.3f}")
        print(f"  Large weights (>2): {metrics['large_weight_fraction']:.3f}")
        print(f"  Small weights (<0.01): {metrics['small_weight_fraction']:.3f}")
    
    return metrics