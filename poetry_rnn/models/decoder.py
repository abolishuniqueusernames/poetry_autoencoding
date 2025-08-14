"""
RNN Decoder Module

Reconstructs sequences from compressed bottleneck representations,
implementing the generative component of the autoencoder architecture.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RNNDecoder(nn.Module):
    """
    RNN Decoder: Compressed Representation → Sequences
    
    Reconstructs full sequences from bottleneck representations through
    autoregressive generation. The decoder learns to expand the compressed
    semantic information back to the original embedding space.
    
    Architecture Flow:
        Bottleneck [B, d] → Linear → Hidden [B, H] → RNN → Output [B, T, D]
        
    Where:
        B = batch size
        d = bottleneck dimension (15-20)
        H = hidden dimension (64-128)
        T = sequence length (reconstructed to match input)
        D = output dimension (300 for GLoVe)
    
    Args:
        bottleneck_dim: Dimension of compressed representation (15-20)
        hidden_size: RNN hidden state dimension (64-128)
        output_size: Dimension of output embeddings (300 for GLoVe)
        max_seq_len: Maximum sequence length to generate (50 for poetry)
        rnn_type: Type of RNN cell ('vanilla', 'lstm', 'gru')
        num_layers: Number of RNN layers (1-2 recommended)
        dropout: Dropout probability for regularization
        use_start_token: Whether to use learned start token
        teacher_forcing_ratio: Probability of using teacher forcing during training
    """
    
    def __init__(
        self,
        bottleneck_dim: int = 18,
        hidden_size: int = 128,
        output_size: int = 300,
        max_seq_len: int = 50,
        rnn_type: str = 'vanilla',
        num_layers: int = 1,
        dropout: float = 0.0,
        use_start_token: bool = True,
        teacher_forcing_ratio: float = 0.0
    ):
        super(RNNDecoder, self).__init__()
        
        self.bottleneck_dim = bottleneck_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Initial hidden state projection: bottleneck → hidden
        # This expands the compressed representation to initialize the RNN
        self.hidden_projection = nn.Linear(bottleneck_dim, hidden_size * num_layers)
        
        # Optional separate projection for LSTM cell state
        if rnn_type == 'lstm':
            self.cell_projection = nn.Linear(bottleneck_dim, hidden_size * num_layers)
        
        # Select RNN variant
        if rnn_type == 'vanilla':
            self.rnn = nn.RNN(
                input_size=output_size,  # Uses previous output as input
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity='tanh',
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output projection: hidden → output space
        # Maps RNN hidden states back to embedding dimension
        self.output_projection = nn.Linear(hidden_size, output_size)
        
        # Learned start token for autoregressive generation
        if use_start_token:
            self.start_token = nn.Parameter(torch.randn(1, 1, output_size))
        else:
            self.start_token = None
            
        # Dropout for regularization
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
            
        # Apply optimal initialization
        self._init_weights()
    
    def init_hidden_from_bottleneck(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Initialize RNN hidden state from bottleneck representation.
        
        Args:
            z: Bottleneck representation [batch_size, bottleneck_dim]
            
        Returns:
            hidden: Initial hidden state [num_layers, batch_size, hidden_size]
            cell: Initial cell state for LSTM [num_layers, batch_size, hidden_size]
                  or None for other RNN types
                  
        Theory Note:
            The bottleneck→hidden projection is crucial for information flow.
            We use tanh activation to match RNN hidden state range [-1, 1].
        """
        batch_size = z.shape[0]
        
        # Project bottleneck to hidden dimension
        hidden = self.hidden_projection(z)  # [batch, hidden_size * num_layers]
        hidden = torch.tanh(hidden)  # Apply activation
        
        # Reshape for multi-layer RNNs
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_size)
        hidden = hidden.transpose(0, 1).contiguous()  # [num_layers, batch, hidden_size]
        
        # Handle LSTM cell state
        if self.rnn_type == 'lstm':
            cell = self.cell_projection(z)
            cell = torch.tanh(cell)
            cell = cell.view(batch_size, self.num_layers, self.hidden_size)
            cell = cell.transpose(0, 1).contiguous()
            return hidden, cell
        else:
            return hidden, None
    
    def forward(
        self,
        z: torch.Tensor,
        target_sequences: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        teacher_forcing_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode bottleneck representations to sequences.
        
        Args:
            z: Bottleneck representation [batch_size, bottleneck_dim]
            target_sequences: Target sequences for teacher forcing [batch_size, seq_len, output_size]
            mask: Attention mask [batch_size, seq_len] indicating valid positions
            seq_len: Override sequence length (defaults to max_seq_len)
            teacher_forcing_ratio: Override teacher forcing ratio (uses self.teacher_forcing_ratio if None)
            
        Returns:
            reconstructed: Output sequences [batch_size, seq_len, output_size]
            hidden_states: All RNN hidden states [batch_size, seq_len, hidden_size]
            
        Implementation Notes:
            - Scheduled sampling: Per-timestep teacher forcing decisions for stable training
            - Teacher forcing: During training, probabilistically uses target sequences vs predictions
            - Autoregressive generation: Each output depends on previous outputs through RNN state
        """
        batch_size = z.shape[0]
        device = z.device
        seq_len = seq_len or self.max_seq_len
        
        # Use provided teacher forcing ratio or default
        current_tf_ratio = teacher_forcing_ratio if teacher_forcing_ratio is not None else self.teacher_forcing_ratio
        
        # Initialize hidden state from bottleneck
        hidden, cell = self.init_hidden_from_bottleneck(z)
        
        # Prepare initial state
        if self.rnn_type == 'lstm':
            rnn_state = (hidden, cell)
        else:
            rnn_state = hidden
        
        # Check if we can use teacher forcing at all
        can_use_teacher_forcing = (
            self.training and 
            target_sequences is not None and 
            current_tf_ratio > 0.0
        )
        
        if not can_use_teacher_forcing:
            # Pure autoregressive generation (inference mode or no targets)
            outputs = []
            hidden_states = []
            
            # Initialize first input
            if self.start_token is not None:
                current_input = self.start_token.expand(batch_size, -1, -1)
            else:
                current_input = torch.zeros(batch_size, 1, self.output_size).to(device)
            
            # Generate sequence step by step
            for t in range(seq_len):
                # Run RNN for one step
                rnn_output, rnn_state = self.rnn(current_input, rnn_state)
                
                # Apply dropout if configured
                if self.dropout is not None:
                    rnn_output = self.dropout(rnn_output)
                
                # Project to output space
                predicted = self.output_projection(rnn_output)
                
                outputs.append(predicted)
                hidden_states.append(rnn_output)
                
                # Use prediction as next input
                current_input = predicted
            
            # Concatenate all outputs
            reconstructed = torch.cat(outputs, dim=1)
            all_hidden = torch.cat(hidden_states, dim=1)
            
            return reconstructed, all_hidden
        
        else:
            # Scheduled sampling: per-timestep teacher forcing decisions
            outputs = []
            hidden_states = []
            
            # Initialize first input
            if self.start_token is not None:
                current_input = self.start_token.expand(batch_size, -1, -1)
            else:
                current_input = torch.zeros(batch_size, 1, self.output_size).to(device)
            
            # Generate sequence step by step with scheduled sampling
            for t in range(seq_len):
                # Run RNN for one step
                rnn_output, rnn_state = self.rnn(current_input, rnn_state)
                
                # Apply dropout if configured
                if self.dropout is not None:
                    rnn_output = self.dropout(rnn_output)
                
                # Project to output space
                predicted = self.output_projection(rnn_output)
                
                outputs.append(predicted)
                hidden_states.append(rnn_output)
                
                # Decide input for next timestep (scheduled sampling)
                if t < seq_len - 1:  # Don't need input for last timestep
                    # Randomly choose between teacher forcing and prediction
                    use_teacher_forcing = torch.rand(1).item() < current_tf_ratio
                    
                    if use_teacher_forcing:
                        # Use target as next input
                        current_input = target_sequences[:, t:t+1, :]
                    else:
                        # Use prediction as next input
                        current_input = predicted
            
            # Concatenate all outputs
            reconstructed = torch.cat(outputs, dim=1)
            all_hidden = torch.cat(hidden_states, dim=1)
            
            return reconstructed, all_hidden
    
    def _init_weights(self):
        """
        Apply RNN-specific optimal weight initialization.
        
        Uses appropriate initialization strategies for different layer types:
        - RNN layers: Orthogonal recurrent weights, Xavier input weights
        - Output projection: Standard gain for reconstruction
        - Start token: Small random initialization for stable generation
        """
        # Import here to avoid circular imports
        from ..utils.initialization import init_rnn_model
        
        # Apply comprehensive initialization
        init_rnn_model(
            model=self,
            rnn_type=self.rnn_type,
            bottleneck_gain=0.1,  # Not used in decoder
            projection_gain=0.5   # Standard gain for output projection
        )
        
        # Initialize start token if present
        if self.start_token is not None:
            torch.nn.init.normal_(self.start_token, mean=0.0, std=0.1)
    
    def generate(
        self,
        z: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate sequences from bottleneck representations (inference mode).
        
        Args:
            z: Bottleneck representation [batch_size, bottleneck_dim]
            seq_len: Length of sequences to generate
            
        Returns:
            Generated sequences [batch_size, seq_len, output_size]
            
        Note:
            This method always uses autoregressive generation (no teacher forcing)
            and is intended for inference/evaluation.
        """
        with torch.no_grad():
            self.eval()
            reconstructed, _ = self.forward(z, seq_len=seq_len)
            self.train()
            return reconstructed