"""
Vanilla RNN Cell Implementation

Educational implementation of a vanilla RNN cell with clear mathematical
exposition and theoretical grounding.
"""

import numpy as np
import torch
import torch.nn as nn


class VanillaRNNCell(nn.Module):
    """
    Educational implementation of vanilla RNN cell.
    
    Mathematical formulation:
        h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_h)
    
    This implementation prioritizes clarity and educational value, showing
    exactly how RNN computations work at the cell level.
    
    Args:
        input_size: Dimension of input x_t (300 for GLoVe embeddings)
        hidden_size: Dimension of hidden state h_t (typically 64-128)
        bias: Whether to use bias terms (default: True)
        
    Theory Notes:
        The hidden size is chosen based on effective dimensionality analysis
        of the poetry dataset. Experiments suggest 64-128 hidden units provide
        sufficient expressiveness while maintaining trainability.
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super(VanillaRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weight matrices following PyTorch convention for compatibility
        # W_ih: Maps input to hidden space
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        # W_hh: Maps previous hidden to current hidden (recurrence)
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        self.init_parameters()
        
    def init_parameters(self):
        """
        Initialize parameters using RNN-specific optimal initialization.
        
        Strategy (following Saxe et al. 2014):
        - W_ih (input-to-hidden): Xavier uniform initialization
        - W_hh (hidden-to-hidden): Orthogonal initialization
        - Biases: Zero initialization
        
        Theory: Orthogonal initialization for recurrent weights prevents
        vanishing gradients by preserving gradient norms through time.
        This is crucial for training stability in vanilla RNNs.
        """
        # Input-to-hidden: Xavier uniform (maintains input signal variance)
        nn.init.xavier_uniform_(self.weight_ih, gain=1.0)
        
        # Hidden-to-hidden: Orthogonal (prevents vanishing gradients in recurrence)
        nn.init.orthogonal_(self.weight_hh, gain=1.0)
        
        if self.bias_ih is not None:
            # Initialize biases to zero (standard practice)
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)
            
    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RNN cell.
        
        Computes: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
        
        Args:
            x: Input tensor [batch_size, input_size]
            hidden: Previous hidden state [batch_size, hidden_size]
            
        Returns:
            new_hidden: Updated hidden state [batch_size, hidden_size]
            
        Implementation Notes:
            We use matrix multiplication (mm) for clarity, showing the
            mathematical operations explicitly. The transpose operations
            align with standard RNN formulation.
        """
        # Input-to-hidden transformation
        ih = torch.mm(x, self.weight_ih.t())  # [batch, hidden]
        
        # Hidden-to-hidden transformation (recurrence)
        hh = torch.mm(hidden, self.weight_hh.t())  # [batch, hidden]
        
        # Add biases if present
        if self.bias_ih is not None:
            ih = ih + self.bias_ih
            hh = hh + self.bias_hh
            
        # Combine transformations and apply activation
        # tanh constrains output to [-1, 1], preventing unbounded growth
        new_hidden = torch.tanh(ih + hh)
        
        return new_hidden
    
    def init_hidden(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        """
        Initialize hidden state with zeros.
        
        Args:
            batch_size: Number of sequences in batch
            device: Device to place tensor on ('cpu' or 'cuda')
            
        Returns:
            Initial hidden state filled with zeros
            
        Note:
            Zero initialization is standard for RNNs. Some works explore
            learned initialization, but zeros work well in practice.
        """
        return torch.zeros(batch_size, self.hidden_size, device=device)