"""
Positional Encoding for Self-Attention

Implements positional encodings based on mathematical theory from SELF-ATTENTION-THEORY.md.
All designs are theory-motivated for optimal sequence position representation.

Key theoretical foundations:
- Theorem 7.1: Positional encoding necessary due to permutation invariance
- Theorem 7.2: Sinusoidal encoding enables relative position dependencies  
- Theorem 7.3: Uniqueness guarantee up to position n = 10000^(d/2)π
- Definition 7.1: PE(pos,2i) = sin(pos/10000^(2i/d))
"""

import torch
import torch.nn as nn
import math
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Definition 7.1)
    
    Provides position information to attention mechanism through sinusoidal
    functions with different frequencies. Based on theoretical analysis
    showing optimal properties for relative position modeling.
    
    Mathematical formulation:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Key properties (from theory):
    - Relative position property (Theorem 7.2): PE(pos+k) = T_k · PE(pos)
    - Uniqueness up to max_len = 10000^(d/2)π (Theorem 7.3)
    - Linear transformations preserve relative distances
    
    Args:
        d_model: Model dimension (must match attention d_model)
        max_len: Maximum sequence length (50 for poetry chunks)
        dropout: Dropout rate for regularization
    """
    
    def __init__(
        self,
        d_model: int = 512,
        max_len: int = 50,
        dropout: float = 0.1
    ):
        super(SinusoidalPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Verify theoretical uniqueness bound (Theorem 7.3)
        # Use log space to avoid overflow for large d_model
        log_uniqueness_bound = (d_model / 2) * math.log(10000) + math.log(math.pi)
        uniqueness_bound = math.exp(min(log_uniqueness_bound, 700))  # Cap to prevent overflow
        if max_len > uniqueness_bound:
            logger.warning(f"max_len ({max_len}) exceeds uniqueness bound (~{uniqueness_bound:.0e})")
        
        # Pre-compute positional encodings for efficiency
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term: 10000^(2i/d_model) for i = 0, 1, ..., d_model//2
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sinusoidal functions (Definition 7.1)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cos
        
        # Add batch dimension and register as buffer (not parameter)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
        logger.info(f"SinusoidalPositionalEncoding initialized:")
        logger.info(f"  d_model={d_model}, max_len={max_len}")
        logger.info(f"  Uniqueness bound: {uniqueness_bound:.0f} positions")
        logger.info(f"  Theoretical properties: relative position encoding")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            x + positional_encoding: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # Ensure sequence length doesn't exceed pre-computed range
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds max_len ({self.max_len})")
        
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        
        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x
    
    def get_position_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encoding for a specific sequence length.
        
        Args:
            seq_len: Desired sequence length
            
        Returns:
            pe: Positional encoding [1, seq_len, d_model]
        """
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds max_len ({self.max_len})")
        
        return self.pe[:, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding
    
    Alternative to sinusoidal encoding where position embeddings are learned
    parameters. May be more effective for fixed-length sequences like poetry
    chunks where positions have semantic meaning.
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout rate for regularization
    """
    
    def __init__(
        self,
        d_model: int = 512,
        max_len: int = 50,
        dropout: float = 0.1
    ):
        super(LearnedPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Learnable position embeddings
        self.pe = nn.Embedding(max_len, d_model)
        
        # Initialize with small random values
        nn.init.normal_(self.pe.weight, mean=0, std=0.1)
        
        logger.info(f"LearnedPositionalEncoding initialized:")
        logger.info(f"  d_model={d_model}, max_len={max_len}")
        logger.info(f"  Parameters: {self.pe.weight.numel():,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            x + positional_encoding: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds max_len ({self.max_len})")
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get position embeddings
        pos_emb = self.pe(positions)
        
        # Add to input
        x = x + pos_emb
        
        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding
    
    Encodes relative distances between positions rather than absolute positions.
    Based on the theoretical insight that attention should depend on relative
    rather than absolute positions (Theorem 7.2).
    
    Args:
        d_model: Model dimension
        max_relative_distance: Maximum relative distance to encode
        num_heads: Number of attention heads (for head-specific biases)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        max_relative_distance: int = 50,
        num_heads: int = 8
    ):
        super(RelativePositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_relative_distance = max_relative_distance
        self.num_heads = num_heads
        
        # Learnable relative position embeddings
        # Range: [-max_relative_distance, max_relative_distance]
        self.relative_positions = nn.Embedding(
            2 * max_relative_distance + 1,
            num_heads
        )
        
        # Initialize with small values
        nn.init.normal_(self.relative_positions.weight, mean=0, std=0.1)
        
        logger.info(f"RelativePositionalEncoding initialized:")
        logger.info(f"  d_model={d_model}, max_distance={max_relative_distance}")
        logger.info(f"  num_heads={num_heads}")
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate relative position bias for attention scores.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            bias: Relative position bias [num_heads, seq_len, seq_len]
        """
        # Create relative position matrix
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).expand(seq_len, -1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clamp to maximum distance and shift to positive indices
        distance_mat_clipped = torch.clamp(
            distance_mat,
            -self.max_relative_distance,
            self.max_relative_distance
        ) + self.max_relative_distance
        
        # Get relative position embeddings
        final_mat = self.relative_positions(distance_mat_clipped)
        # Shape: [seq_len, seq_len, num_heads]
        
        # Transpose to [num_heads, seq_len, seq_len] for attention
        final_mat = final_mat.permute(2, 0, 1)
        
        return final_mat


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal (lower triangular) mask for autoregressive attention.
    
    Prevents attention to future positions during autoregressive generation.
    Essential for decoder self-attention to maintain causality.
    
    Args:
        seq_len: Sequence length
        device: Device for tensor creation
        
    Returns:
        mask: Causal mask [seq_len, seq_len] where 1=attend, 0=mask
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def create_padding_mask(
    sequences: torch.Tensor,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Create padding mask from input sequences.
    
    Prevents attention to padded positions in variable-length sequences.
    
    Args:
        sequences: Input sequences [batch_size, seq_len]
        pad_token_id: Token ID used for padding
        
    Returns:
        mask: Padding mask [batch_size, seq_len] where 1=real, 0=padding
    """
    return (sequences != pad_token_id).long()


if __name__ == "__main__":
    # Test positional encoding modules
    print("Testing Positional Encoding modules...")
    
    # Test parameters
    batch_size, seq_len, d_model = 2, 20, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test sinusoidal encoding
    print("\n1. Testing SinusoidalPositionalEncoding...")
    sin_pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=50)
    x_sin = sin_pe(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {x_sin.shape}")
    print(f"  Position encoding shape: {sin_pe.get_position_encoding(seq_len).shape}")
    
    # Test learned encoding
    print("\n2. Testing LearnedPositionalEncoding...")
    learned_pe = LearnedPositionalEncoding(d_model=d_model, max_len=50)
    x_learned = learned_pe(x)
    print(f"  Output shape: {x_learned.shape}")
    print(f"  Learnable parameters: {learned_pe.pe.weight.numel():,}")
    
    # Test relative encoding
    print("\n3. Testing RelativePositionalEncoding...")
    rel_pe = RelativePositionalEncoding(d_model=d_model, num_heads=8)
    rel_bias = rel_pe(seq_len)
    print(f"  Relative bias shape: {rel_bias.shape}")
    print(f"  Range: [{rel_bias.min():.3f}, {rel_bias.max():.3f}]")
    
    # Test utility functions
    print("\n4. Testing utility functions...")
    causal_mask = create_causal_mask(seq_len)
    print(f"  Causal mask shape: {causal_mask.shape}")
    print(f"  Causal property verified: {torch.triu(causal_mask, diagonal=1).sum() == 0}")
    
    sequences = torch.randint(0, 1000, (batch_size, seq_len))
    sequences[:, -5:] = 0  # Add some padding
    padding_mask = create_padding_mask(sequences, pad_token_id=0)
    print(f"  Padding mask shape: {padding_mask.shape}")
    print(f"  Padding detected: {(padding_mask == 0).sum().item()} positions")
    
    print("\n✅ All positional encoding tests passed!")