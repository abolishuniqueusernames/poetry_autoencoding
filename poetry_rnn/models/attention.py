"""
Multi-Head Self-Attention Module

Implements self-attention mechanism based on rigorous mathematical theory from
SELF-ATTENTION-THEORY.md. All parameters are theory-optimized for poetry
sequence reconstruction.

Key theoretical foundations:
- Theorem 4.4: Optimal temperature τ = √d_k minimizes variance
- Theorem 6.1: Multi-head attention performs subspace decomposition  
- Theorem 8.3: Constant gradient path length O(1) vs RNN's O(n)
- Definition 4.1: Attention(Q,K,V) = softmax(QK^T/√d_k)V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Theory-Optimized Parameters
    
    Implementation based on Definition 4.1 and Theorem 6.1 from theory:
    - Performs implicit subspace decomposition across h heads
    - Each head attends to different aspects of sequence relationships
    - Provides O(1) gradient path length for all position pairs
    
    Architecture:
        Input: [batch_size, seq_len, d_model]
        Queries/Keys/Values: Linear projections to [batch_size, seq_len, d_k/d_v]
        Attention: softmax(QK^T/√d_k)V per head
        Output: Concatenated heads projected back to d_model
    
    Args:
        d_model: Model dimension (512 for poetry RNN hidden states)
        num_heads: Number of attention heads (8, theory-optimal for poetry)
        dropout: Dropout probability for attention weights
        temperature_scale: Temperature scaling factor (default: √d_k from Theorem 4.4)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature_scale: Optional[float] = None
    ):
        super(MultiHeadAttention, self).__init__()
        
        # Validate dimensions for theory-optimal configuration
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.d_v = self.d_k  # Same as d_k for standard attention
        
        # Theorem 4.4: Optimal temperature τ = √d_k minimizes variance
        self.temperature = temperature_scale if temperature_scale is not None else math.sqrt(self.d_k)
        
        # Linear projections for Q, K, V (Definition 4.1)
        # Use separate projections per head for maximum expressiveness
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)  
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection to combine heads
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout for attention weights (not applied to values)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize weights using theory-motivated scheme
        self._init_weights()
        
        logger.info(f"MultiHeadAttention initialized:")
        logger.info(f"  d_model={d_model}, num_heads={num_heads}, d_k={self.d_k}")
        logger.info(f"  Temperature: {self.temperature:.3f} (theory-optimal)")
        logger.info(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self):
        """
        Initialize weights using Xavier uniform initialization.
        
        Theory: Maintains gradient variance across layers for stable training.
        Each projection matrix is initialized to preserve signal magnitude.
        """
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Multi-head attention forward pass.
        
        Implements Definition 4.1: Attention(Q,K,V) = softmax(QK^T/√d_k)V
        extended to multiple heads with subspace decomposition.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model] 
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Attention mask [batch_size, seq_len_q, seq_len_k] (1=attend, 0=mask)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attention output [batch_size, seq_len_q, d_model]
            attention_weights: Optional attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)
        
        # Ensure key and value have same sequence length
        assert seq_len_k == seq_len_v, f"Key ({seq_len_k}) and value ({seq_len_v}) must have same sequence length"
        
        # 1. Linear projections for Q, K, V
        Q = self.W_q(query)  # [batch_size, seq_len_q, d_model]
        K = self.W_k(key)    # [batch_size, seq_len_k, d_model]
        V = self.W_v(value)  # [batch_size, seq_len_v, d_model]
        
        # 2. Reshape for multi-head attention
        # Split d_model into num_heads × d_k
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.num_heads, self.d_v).transpose(1, 2)
        # Shape: [batch_size, num_heads, seq_len, d_k/d_v]
        
        # 3. Compute attention for all heads in parallel
        attention_output, attention_weights = self._scaled_dot_product_attention(
            Q, K, V, mask, return_attention
        )
        # attention_output: [batch_size, num_heads, seq_len_q, d_v]
        # attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k] or None
        
        # 4. Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        # Shape: [batch_size, seq_len_q, d_model]
        
        # 5. Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights
    
    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Scaled dot-product attention (Definition 4.1).
        
        Computes attention weights as softmax(QK^T/τ) where τ is temperature.
        Applies optional masking and dropout for regularization.
        
        Args:
            Q: Queries [batch_size, num_heads, seq_len_q, d_k]
            K: Keys [batch_size, num_heads, seq_len_k, d_k]
            V: Values [batch_size, num_heads, seq_len_v, d_v]
            mask: Attention mask [batch_size, seq_len_q, seq_len_k] or None
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attention output [batch_size, num_heads, seq_len_q, d_v]
            attention_weights: Optional attention weights
        """
        # Compute attention scores: QK^T/√d_k (Definition 4.1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.temperature
        # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention dimensions
            if mask.dim() == 3:  # [batch_size, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            elif mask.dim() == 2:  # [seq_len_q, seq_len_k]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_q, seq_len_k]
            
            # Apply mask: set masked positions to large negative value
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights: softmax(scores)
        attention_weights = F.softmax(scores, dim=-1)
        # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # Apply dropout to attention weights (not values)
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values: AV
        output = torch.matmul(attention_weights, V)
        # Shape: [batch_size, num_heads, seq_len_q, d_v]
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None


class SelfAttention(MultiHeadAttention):
    """
    Self-Attention Layer (simplified interface for same Q, K, V).
    
    Convenience wrapper around MultiHeadAttention where query, key, and value
    are all the same input tensor. Common use case for encoder self-attention.
    """
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Self-attention forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len] (1=attend, 0=mask)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Self-attention output [batch_size, seq_len, d_model]
            attention_weights: Optional attention weights
        """
        return super().forward(x, x, x, mask, return_attention)


class CrossAttention(MultiHeadAttention):
    """
    Cross-Attention Layer (encoder-decoder attention).
    
    Specialized for encoder-decoder attention where queries come from decoder
    and keys/values come from encoder. This solves the exponential accuracy
    decay by providing direct access to encoder representations.
    """
    
    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Cross-attention forward pass.
        
        Args:
            decoder_state: Decoder queries [batch_size, seq_len_dec, d_model]
            encoder_states: Encoder keys/values [batch_size, seq_len_enc, d_model]
            mask: Attention mask [batch_size, seq_len_dec, seq_len_enc]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Cross-attention output [batch_size, seq_len_dec, d_model]
            attention_weights: Optional attention weights
        """
        return super().forward(
            query=decoder_state,
            key=encoder_states, 
            value=encoder_states,
            mask=mask,
            return_attention=return_attention
        )


# Utility functions for attention analysis and debugging

def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention weights to measure attention sharpness.
    
    From theory: Higher entropy = more distributed attention
    Lower entropy = more focused attention
    
    Args:
        attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        
    Returns:
        entropy: [batch_size, num_heads, seq_len_q] attention entropy per query
    """
    # Add small epsilon to prevent log(0)
    eps = 1e-8
    attention_weights_safe = attention_weights + eps
    
    # Compute entropy: -∑ p_i log p_i
    entropy = -(attention_weights_safe * torch.log(attention_weights_safe)).sum(dim=-1)
    
    return entropy


def visualize_attention_pattern(
    attention_weights: torch.Tensor,
    head_idx: int = 0,
    batch_idx: int = 0
) -> torch.Tensor:
    """
    Extract attention pattern for visualization.
    
    Args:
        attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        head_idx: Which attention head to visualize
        batch_idx: Which batch element to visualize
        
    Returns:
        pattern: [seq_len_q, seq_len_k] attention matrix for visualization
    """
    return attention_weights[batch_idx, head_idx].detach()


if __name__ == "__main__":
    # Quick test of attention module
    print("Testing MultiHeadAttention...")
    
    # Create attention layer with poetry-optimized parameters
    attention = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)
    
    # Test with sample data
    batch_size, seq_len = 2, 20
    x = torch.randn(batch_size, seq_len, 512)
    
    # Self-attention test
    output, weights = attention(x, x, x, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention entropy (mean): {compute_attention_entropy(weights).mean():.3f}")
    
    print("✅ MultiHeadAttention test passed!")