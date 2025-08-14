"""
Complete RNN Autoencoder Architecture

Combines encoder and decoder into a unified model for poetry dimensionality
reduction and reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Union

from .encoder import RNNEncoder
from .decoder import RNNDecoder


class RNNAutoencoder(nn.Module):
    """
    Complete RNN Autoencoder: Encoder + Bottleneck + Decoder
    
    This class combines the encoder and decoder components into a unified
    architecture for dimensionality reduction and reconstruction of poetry
    sequences. The model learns a compressed representation that captures
    the semantic essence of poetry while discarding redundant information.
    
    Mathematical Framework:
    -----------------------
    Given input sequence X ∈ ℝ^{B×T×D} where:
        B = batch size
        T = sequence length (max 50)
        D = embedding dimension (300)
    
    The autoencoder computes:
        1. Encoding: z = Encoder(X) ∈ ℝ^{B×d}
        2. Decoding: X̂ = Decoder(z) ∈ ℝ^{B×T×D}
        3. Loss: L = MSE(X, X̂, mask) + λ·||z||₂ (with regularization)
    
    The bottleneck dimension d << D enforces compression and learns
    a low-dimensional manifold of poetry semantics.
    
    Theory Foundation:
    -----------------
    Based on analysis showing:
    - Poetry embeddings have ~15-20 effective dimensions
    - Compression from 300D → 18D reduces complexity O(ε^-600) → O(ε^-35)
    - Curriculum learning improves convergence for variable-length sequences
    
    Args:
        config: Configuration dict with model hyperparameters, or individual args:
        input_size: Dimension of input embeddings (300 for GLoVe)
        hidden_size: RNN hidden state dimension (64-128)
        bottleneck_dim: Compressed representation dimension (15-20)
        max_seq_len: Maximum sequence length (50)
        rnn_type: Type of RNN ('vanilla', 'lstm', 'gru')
        num_layers: Number of RNN layers (1-2)
        dropout: Dropout probability (0.1-0.3)
        teacher_forcing_ratio: Teacher forcing probability during training
        bottleneck_regularization: L2 regularization weight for bottleneck
    """
    
    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        input_size: int = 300,
        hidden_size: int = 128,
        bottleneck_dim: int = 18,
        max_seq_len: int = 50,
        rnn_type: str = 'vanilla',
        num_layers: int = 1,
        dropout: float = 0.1,
        teacher_forcing_ratio: float = 0.5,
        bottleneck_regularization: float = 0.01,
        **kwargs
    ):
        super(RNNAutoencoder, self).__init__()
        
        # Handle config object or dict
        if config is not None:
            if hasattr(config, 'get'):  # dict-like interface
                input_size = config.get('input_size', input_size)
                hidden_size = config.get('hidden_size', hidden_size)
                bottleneck_dim = config.get('bottleneck_dim', bottleneck_dim)
                max_seq_len = config.get('max_seq_len', max_seq_len)
                rnn_type = config.get('rnn_type', rnn_type)
                num_layers = config.get('num_layers', num_layers)
                dropout = config.get('dropout', dropout)
                teacher_forcing_ratio = config.get('teacher_forcing_ratio', teacher_forcing_ratio)
                bottleneck_regularization = config.get('bottleneck_regularization', bottleneck_regularization)
            else:  # config object interface
                # Use embedding dimension from config if available
                if hasattr(config, 'embedding') and hasattr(config.embedding, 'embedding_dim'):
                    input_size = config.embedding.embedding_dim
                if hasattr(config, 'chunking') and hasattr(config.chunking, 'window_size'):
                    max_seq_len = config.chunking.window_size
        
        # Store dimensions for reference
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.max_seq_len = max_seq_len
        self.bottleneck_regularization = bottleneck_regularization
        
        # Initialize encoder
        self.encoder = RNNEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            bottleneck_dim=bottleneck_dim,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=True
        )
        
        # Initialize decoder
        self.decoder = RNNDecoder(
            bottleneck_dim=bottleneck_dim,
            hidden_size=hidden_size,
            output_size=input_size,
            max_seq_len=max_seq_len,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
            use_start_token=True,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        # Compression ratio for logging
        self.compression_ratio = input_size / bottleneck_dim
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize model weights using Xavier/He initialization.
        
        Theory Note:
            Proper initialization is crucial for RNN training. We use:
            - Xavier for linear layers (maintains variance)
            - Orthogonal for RNN weights (helps with gradient flow)
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'rnn' in name:
                    # Orthogonal initialization for RNN weights
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    # Xavier initialization for linear layers
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Zero initialization for biases
                nn.init.zeros_(param)
    
    def forward(
        self,
        batch_dict: Dict[str, torch.Tensor],
        return_hidden: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete autoencoder.
        
        Args:
            batch_dict: Batch dictionary with keys:
                - 'input_sequences': [batch_size, seq_len, embedding_dim]
                - 'attention_mask': [batch_size, seq_len] (optional)
                - 'metadata': List of dictionaries with chunk information
            return_hidden: Whether to return hidden states for analysis
        
        Returns:
            Output dictionary containing:
                - 'reconstructed': Reconstructed sequences [batch_size, seq_len, embedding_dim]
                - 'bottleneck': Compressed representations [batch_size, bottleneck_dim]
                - 'encoder_hidden': Encoder hidden states [batch_size, seq_len, hidden_size]
                - 'decoder_hidden': Decoder hidden states [batch_size, seq_len, hidden_size]
                - 'loss_components': Dict with individual loss terms (if computing loss)
        """
        # Extract inputs from batch dictionary
        input_sequences = batch_dict['input_sequences']  # [B, T, D]
        attention_mask = batch_dict.get('attention_mask', None)  # [B, T]
        
        # Encoding phase: sequences → bottleneck
        bottleneck, encoder_hidden = self.encoder(
            input_sequences, 
            attention_mask,
            return_all_hidden=return_hidden
        )
        
        # Decoding phase: bottleneck → sequences
        # Pass target sequences for potential teacher forcing during training
        target_sequences = input_sequences if self.training else None
        reconstructed, decoder_hidden = self.decoder(
            bottleneck,
            target_sequences=target_sequences,
            mask=attention_mask
        )
        
        # Build output dictionary
        output_dict = {
            'reconstructed': reconstructed,
            'bottleneck': bottleneck
        }
        
        if return_hidden:
            output_dict['encoder_hidden'] = encoder_hidden
            output_dict['decoder_hidden'] = decoder_hidden
        
        # Add regularization term for bottleneck if needed
        if self.bottleneck_regularization > 0:
            bottleneck_reg = self.bottleneck_regularization * (bottleneck ** 2).mean()
            output_dict['bottleneck_regularization'] = bottleneck_reg
        
        return output_dict
    
    def encode(self, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode sequences to bottleneck representations only.
        
        Useful for:
        - Extracting features for downstream tasks
        - Similarity analysis between poems
        - Clustering and visualization
        
        Args:
            batch_dict: Batch dictionary from DataLoader
            
        Returns:
            Bottleneck representations [batch_size, bottleneck_dim]
        """
        input_sequences = batch_dict['input_sequences']
        attention_mask = batch_dict.get('attention_mask', None)
        
        bottleneck, _ = self.encoder(
            input_sequences, 
            attention_mask,
            return_all_hidden=False
        )
        return bottleneck
    
    def decode(
        self,
        bottleneck: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decode bottleneck representations to sequences.
        
        Useful for:
        - Generating sequences from compressed representations
        - Interpolation experiments in latent space
        - Understanding what the bottleneck captures
        
        Args:
            bottleneck: Compressed representations [batch_size, bottleneck_dim]
            seq_len: Output sequence length (defaults to max_seq_len)
            
        Returns:
            Reconstructed sequences [batch_size, seq_len, input_size]
        """
        reconstructed, _ = self.decoder(bottleneck, seq_len=seq_len)
        return reconstructed
    
    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        steps: int = 10,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Interpolate between two bottleneck representations.
        
        Useful for exploring the learned latent space and generating
        smooth transitions between different poem representations.
        
        Args:
            z1: First bottleneck representation [1, bottleneck_dim]
            z2: Second bottleneck representation [1, bottleneck_dim]
            steps: Number of interpolation steps
            seq_len: Length of generated sequences
            
        Returns:
            Interpolated sequences [steps, seq_len, input_size]
        """
        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, steps).to(z1.device)
        interpolated = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            seq = self.decode(z_interp, seq_len)
            interpolated.append(seq)
        
        return torch.cat(interpolated, dim=0)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_compression_stats(
        self,
        batch_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Analyze compression quality for a batch.
        
        Computes various metrics to understand how well the autoencoder
        preserves information through the bottleneck.
        
        Args:
            batch_dict: Batch dictionary from DataLoader
            
        Returns:
            Dictionary with compression statistics:
                - mse: Mean squared error
                - cosine_similarity: Average cosine similarity
                - bottleneck_mean: Mean bottleneck activation
                - bottleneck_std: Std of bottleneck activations
                - bottleneck_sparsity: Fraction of near-zero activations
                - compression_ratio: Input/bottleneck dimension ratio
        """
        with torch.no_grad():
            # Forward pass
            output_dict = self.forward(batch_dict, return_hidden=False)
            
            input_seq = batch_dict['input_sequences']
            recon_seq = output_dict['reconstructed']
            bottleneck = output_dict['bottleneck']
            mask = batch_dict.get('attention_mask', None)
            
            # Reconstruction error
            mse = ((input_seq - recon_seq) ** 2)
            if mask is not None:
                # Apply mask to ignore padding
                mse = mse * mask.unsqueeze(-1)
                mse = mse.sum() / (mask.sum() * self.input_size)
            else:
                mse = mse.mean()
            
            # Bottleneck statistics
            z_mean = bottleneck.mean(dim=0)
            z_std = bottleneck.std(dim=0)
            z_sparsity = (torch.abs(bottleneck) < 0.1).float().mean()
            
            # Cosine similarity between input and reconstruction
            cos_sim_per_token = F.cosine_similarity(input_seq, recon_seq, dim=-1)
            if mask is not None:
                cos_sim = (cos_sim_per_token * mask).sum() / mask.sum()
            else:
                cos_sim = cos_sim_per_token.mean()
            
            stats = {
                'mse': mse.item(),
                'cosine_similarity': cos_sim.item(),
                'bottleneck_mean': z_mean.mean().item(),
                'bottleneck_std': z_std.mean().item(),
                'bottleneck_sparsity': z_sparsity.item(),
                'compression_ratio': self.compression_ratio
            }
            
            return stats
    
    def summary(self) -> str:
        """
        Generate a summary of the model architecture.
        
        Returns:
            String summary with architecture details and parameter counts
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = self.count_parameters()
        
        summary = f"""
RNN Autoencoder Architecture Summary
=====================================
Input Dimension: {self.input_size}D
Hidden Dimension: {self.hidden_size}D
Bottleneck Dimension: {self.bottleneck_dim}D
Compression Ratio: {self.compression_ratio:.1f}x

Architecture Flow:
  {self.input_size}D → Encoder → {self.bottleneck_dim}D → Decoder → {self.input_size}D

Parameter Counts:
  Encoder: {encoder_params:,}
  Decoder: {decoder_params:,}
  Total: {total_params:,}

Theoretical Foundation:
  - Optimal compression for poetry: 300D → 15-20D
  - Complexity reduction: O(ε^-600) → O(ε^-35)
  - Effective dimensions match semantic structure
"""
        return summary
