"""
RNN Encoder Module

Encodes variable-length sequences into fixed-size bottleneck representations
for dimensionality reduction in poetry processing.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RNNEncoder(nn.Module):
    """
    RNN Encoder: Sequences → Compressed Representation
    
    Processes sequences using RNN and projects final hidden state to bottleneck
    dimension. This compression is theoretically motivated by the analysis showing
    poetry embeddings have ~15-20 effective dimensions despite using 300D GLoVe.
    
    Architecture Flow:
        Input [B, T, D] → RNN → Hidden [B, H] → Linear → Bottleneck [B, d]
        
    Where:
        B = batch size
        T = sequence length (variable, max 50)
        D = input dimension (300 for GLoVe)
        H = hidden dimension (64-128)
        d = bottleneck dimension (15-20)
    
    Args:
        input_size: Dimension of input embeddings (300 for GLoVe)
        hidden_size: RNN hidden state dimension (64-128 recommended)
        bottleneck_dim: Compressed representation size (15-20 recommended)
        rnn_type: Type of RNN cell ('vanilla', 'lstm', 'gru')
        num_layers: Number of RNN layers (1-2 recommended)
        dropout: Dropout probability for regularization
        use_batch_norm: Whether to apply batch normalization to bottleneck
    """
    
    def __init__(
        self,
        input_size: int = 300,
        hidden_size: int = 128,
        bottleneck_dim: int = 18,
        rnn_type: str = 'vanilla',
        num_layers: int = 1,
        dropout: float = 0.0,
        use_batch_norm: bool = True
    ):
        super(RNNEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Select RNN variant
        if rnn_type == 'vanilla':
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,  # Input shape: [batch, seq, features]
                nonlinearity='tanh',
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Projection layer: hidden → bottleneck
        # This is the key compression step reducing dimensionality
        self.projection = nn.Linear(hidden_size, bottleneck_dim)
        
        # Optional batch normalization for training stability
        # Helps with gradient flow through the bottleneck
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(bottleneck_dim)
        else:
            self.batch_norm = None
            
        # Dropout for regularization
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
            
        # Apply optimal initialization
        self._init_weights()
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_all_hidden: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequences to bottleneck representations.
        
        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            mask: Attention mask [batch_size, seq_len] indicating valid positions
            return_all_hidden: Whether to return all hidden states for analysis
            
        Returns:
            bottleneck: Compressed representations [batch_size, bottleneck_dim]
            hidden_states: All RNN hidden states [batch_size, seq_len, hidden_size]
                          (only if return_all_hidden=True)
                          
        Implementation Notes:
            - Mask handling: We use the last valid hidden state for sequences
              shorter than max_length, ensuring meaningful representations
            - Gradient flow: Batch norm and careful initialization help maintain
              gradients through the compression bottleneck
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize hidden state
        if self.rnn_type == 'lstm':
            # LSTM has both hidden and cell states
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            initial_state = (h0, c0)
        else:
            # Vanilla RNN and GRU only have hidden state
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            initial_state = h0
        
        # Forward pass through RNN
        output, final_state = self.rnn(x, initial_state)
        # output: [batch, seq_len, hidden_size] - all hidden states
        # final_state: final hidden (and cell) state
        
        # Extract final hidden state based on RNN type
        if self.rnn_type == 'lstm':
            # For LSTM, final_state is (h_n, c_n), we want h_n
            final_hidden = final_state[0][-1]  # [batch, hidden_size]
        else:
            # For vanilla RNN and GRU
            final_hidden = final_state[-1]  # [batch, hidden_size]
        
        # Handle variable-length sequences with masking
        if mask is not None:
            # Find the last valid position for each sequence
            # mask is [batch, seq_len] with 1s for valid positions
            lengths = (mask.sum(dim=1) - 1).long()  # [batch], zero-indexed positions as long tensor
            
            # Gather the hidden state at the last valid position
            # This ensures we use meaningful representations for variable-length sequences
            batch_indices = torch.arange(batch_size).to(device)
            final_hidden = output[batch_indices, lengths]  # [batch, hidden_size]
        
        # Apply dropout if configured
        if self.dropout is not None:
            final_hidden = self.dropout(final_hidden)
        
        # Project to bottleneck dimension
        # This is where dimensionality reduction happens
        bottleneck = self.projection(final_hidden)  # [batch, bottleneck_dim]
        
        # Apply batch normalization if configured
        # This stabilizes training and helps gradient flow
        if self.batch_norm is not None:
            bottleneck = self.batch_norm(bottleneck)
        
        if return_all_hidden:
            return bottleneck, output
        else:
            return bottleneck, None
    
    def _init_weights(self):
        """
        Apply RNN-specific optimal weight initialization.
        
        Uses appropriate initialization strategies for different layer types:
        - RNN layers: Orthogonal recurrent weights, Xavier input weights
        - Projection layer: Small gain for bottleneck stability
        - Batch norm: Default PyTorch initialization (already optimal)
        """
        # Import here to avoid circular imports
        from ..utils.initialization import init_rnn_model
        
        # Apply comprehensive initialization
        init_rnn_model(
            model=self,
            rnn_type=self.rnn_type,
            bottleneck_gain=0.1,  # Small gain for compression stability
            projection_gain=0.5   # Standard gain for other projections
        )
    
    def get_effective_compression_ratio(self) -> float:
        """
        Calculate the effective compression ratio achieved by the encoder.
        
        Returns:
            Compression ratio (input_size / bottleneck_dim)
            
        Theory Note:
            For poetry with 300D GLoVe embeddings compressed to 18D bottleneck,
            this gives ~16.7x compression, aligning with our theoretical analysis
            showing 15-20 effective dimensions in poetry semantics.
        """
        return self.input_size / self.bottleneck_dim