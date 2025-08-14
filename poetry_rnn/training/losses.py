"""
Loss functions for RNN Autoencoder training

Provides specialized loss functions that handle variable-length sequences
and masked positions in poetry data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class MaskedMSELoss(nn.Module):
    """
    Masked Mean Squared Error for variable-length sequences.
    
    Only computes loss on non-padded tokens, ensuring the model learns
    to reconstruct actual poetry content rather than padding tokens.
    
    Args:
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
        epsilon: Small value for numerical stability
    """
    
    def __init__(self, reduction: str = 'mean', epsilon: float = 1e-8):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute masked MSE loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, embedding_dim]
            targets: Target sequences [batch_size, seq_len, embedding_dim]
            mask: Boolean mask [batch_size, seq_len] where True = valid position
        
        Returns:
            Scalar loss value (if reduction='mean' or 'sum')
            or tensor of losses (if reduction='none')
        """
        # Compute element-wise squared error
        mse = (predictions - targets) ** 2  # [batch, seq_len, embedding_dim]
        
        if mask is not None:
            # Expand mask to match embedding dimension
            mask_expanded = mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            
            # Apply mask: zero out padded positions
            mse = mse * mask_expanded
            
            if self.reduction == 'mean':
                # Mean over valid positions only
                # Count total valid elements across batch
                valid_elements = mask_expanded.sum()
                # Each valid position has embedding_dim elements
                valid_elements = valid_elements * mse.shape[-1]
                return mse.sum() / (valid_elements + self.epsilon)
            elif self.reduction == 'sum':
                return mse.sum()
            else:
                return mse
        else:
            # No mask provided, use standard reduction
            if self.reduction == 'mean':
                return mse.mean()
            elif self.reduction == 'sum':
                return mse.sum()
            else:
                return mse


class CosineReconstructionLoss(nn.Module):
    """
    Cosine similarity-based reconstruction loss.
    
    Measures the angular difference between predicted and target embeddings,
    focusing on semantic direction rather than magnitude. This is particularly
    suitable for word embeddings where direction encodes meaning.
    
    Args:
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(CosineReconstructionLoss, self).__init__()
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, embedding_dim]
            targets: Target sequences [batch_size, seq_len, embedding_dim]
            mask: Boolean mask [batch_size, seq_len] where True = valid position
        
        Returns:
            Scalar loss value (1 - cosine_similarity)
        """
        # Compute cosine similarity for each token
        cos_sim = F.cosine_similarity(predictions, targets, dim=-1)  # [batch, seq_len]
        
        # Convert to loss (1 - similarity)
        loss = 1 - cos_sim
        
        if mask is not None:
            # Apply mask
            loss = loss * mask.float()
            
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple loss functions with weights.
    
    Allows flexible combination of different loss objectives, such as:
    - MSE for reconstruction accuracy
    - Cosine similarity for semantic preservation
    - Regularization terms for bottleneck sparsity
    
    Args:
        loss_weights: Dictionary mapping loss names to weights
        losses: Dictionary mapping loss names to loss modules
    """
    
    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        losses: Optional[Dict[str, nn.Module]] = None
    ):
        super(CompositeLoss, self).__init__()
        
        # Default loss configuration
        if losses is None:
            losses = {
                'mse': MaskedMSELoss(),
                'cosine': CosineReconstructionLoss()
            }
        
        if loss_weights is None:
            loss_weights = {
                'mse': 1.0,
                'cosine': 0.1
            }
        
        self.losses = nn.ModuleDict(losses)
        self.loss_weights = loss_weights
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        additional_losses: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, embedding_dim]
            targets: Target sequences [batch_size, seq_len, embedding_dim]
            mask: Boolean mask [batch_size, seq_len]
            additional_losses: Extra losses (e.g., regularization terms)
        
        Returns:
            Dictionary with:
                - 'total': Total weighted loss
                - Individual loss components
        """
        loss_dict = {}
        total_loss = 0
        
        # Compute each configured loss
        for name, loss_fn in self.losses.items():
            if name in self.loss_weights:
                loss_value = loss_fn(predictions, targets, mask)
                weighted_loss = self.loss_weights[name] * loss_value
                loss_dict[name] = loss_value
                total_loss = total_loss + weighted_loss
        
        # Add any additional losses (e.g., regularization)
        if additional_losses:
            for name, loss_value in additional_losses.items():
                if name in self.loss_weights:
                    weighted_loss = self.loss_weights[name] * loss_value
                    loss_dict[name] = loss_value
                    total_loss = total_loss + weighted_loss
        
        loss_dict['total'] = total_loss
        return loss_dict


def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient norms for each parameter group to monitor gradient flow.
    
    This helps detect vanishing/exploding gradients, a key challenge in RNN training
    as predicted by theoretical analysis.
    
    Args:
        model: PyTorch model with computed gradients
    
    Returns:
        Dictionary mapping parameter names to gradient norms
    """
    grad_norms = {}
    total_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            grad_norms[name] = param_norm
            total_norm += param_norm ** 2
    
    grad_norms['total'] = total_norm ** 0.5
    
    # Group by module for summary
    module_norms = {}
    for name, norm in grad_norms.items():
        if name != 'total':
            module = name.split('.')[0]
            if module not in module_norms:
                module_norms[module] = []
            module_norms[module].append(norm)
    
    # Add module averages
    for module, norms in module_norms.items():
        grad_norms[f'{module}_avg'] = sum(norms) / len(norms)
    
    return grad_norms


def analyze_hidden_states(
    hidden_states: torch.Tensor,
    name: str = ""
) -> Dict[str, float]:
    """
    Analyze RNN hidden states to understand information flow.
    
    Provides insights into:
    - Activation patterns (saturation, dead neurons)
    - Temporal dynamics (how states evolve over time)
    - Information propagation through the network
    
    Args:
        hidden_states: Hidden state tensor [batch_size, seq_len, hidden_dim]
        name: Identifier for the hidden states (e.g., 'encoder', 'decoder')
    
    Returns:
        Dictionary of statistics about the hidden states
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    
    # Compute statistics over time and batch dimensions
    mean_activation = hidden_states.mean(dim=(0, 1))  # [hidden_dim]
    std_activation = hidden_states.std(dim=(0, 1))    # [hidden_dim]
    
    # Compute temporal dynamics (how much states change over time)
    if seq_len > 1:
        temporal_diff = hidden_states[:, 1:] - hidden_states[:, :-1]
        temporal_variance = temporal_diff.var(dim=(0, 1))  # [hidden_dim]
    else:
        temporal_variance = torch.zeros_like(mean_activation)
    
    # Saturation analysis (neurons stuck at extremes)
    saturation_high = (hidden_states > 0.9).float().mean().item()
    saturation_low = (hidden_states < -0.9).float().mean().item()
    
    # Dead neuron analysis (neurons with very low activation)
    dead_neurons = (torch.abs(hidden_states) < 0.1).float().mean().item()
    
    prefix = f'{name}_' if name else ''
    
    stats = {
        f'{prefix}mean_activation': mean_activation.mean().item(),
        f'{prefix}std_activation': std_activation.mean().item(),
        f'{prefix}temporal_variance': temporal_variance.mean().item(),
        f'{prefix}saturation_high': saturation_high,
        f'{prefix}saturation_low': saturation_low,
        f'{prefix}dead_neurons': dead_neurons,
        f'{prefix}activation_range': (hidden_states.max() - hidden_states.min()).item()
    }
    
    return stats