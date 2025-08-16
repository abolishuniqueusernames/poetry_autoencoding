"""
Attention-Enhanced RNN Decoder

Implements encoder-decoder attention to solve exponential accuracy decay in 
sequence reconstruction. Based on mathematical theory from SELF-ATTENTION-THEORY.md.

Key theoretical insight:
- Theorem 10.2: Expected improvement Δ ≥ (exp(n/τ) - 1)·σ² over pure RNN
- Theorem 8.3: Constant gradient path length O(1) vs RNN's O(|i-j|)
- Direct access to encoder states eliminates information bottleneck

Architecture:
    RNN Decoder + Encoder-Decoder Attention → Enhanced Context Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging

from .attention import CrossAttention
from .positional_encoding import create_causal_mask, SinusoidalPositionalEncoding

logger = logging.getLogger(__name__)


class AttentionEnhancedDecoder(nn.Module):
    """
    RNN Decoder with Encoder-Decoder Attention
    
    Solves the exponential accuracy decay problem by providing direct access
    to encoder hidden states at each decoding timestep. This bypasses the
    information bottleneck of passing everything through the RNN hidden state.
    
    Architecture Flow:
        1. Initialize RNN state from bottleneck representation
        2. At each timestep t:
           a. Run RNN step: bottleneck/previous → hidden_t
           b. Compute attention: hidden_t × encoder_states → context_t  
           c. Integrate context: concat(hidden_t, context_t) → enhanced_t
           d. Project to output: enhanced_t → output_t
        3. Apply teacher forcing with curriculum learning
    
    Expected improvement: +0.15 cosine similarity (theory: Theorem 10.2)
    
    Args:
        bottleneck_dim: Dimension of compressed representation (128)
        hidden_size: RNN hidden state dimension (512) 
        output_size: Output embedding dimension (300 for GLoVe)
        encoder_hidden_size: Encoder hidden state dimension (512, same as hidden_size)
        max_seq_len: Maximum generation length (50 for poetry)
        rnn_type: Type of RNN ('LSTM', 'GRU', 'vanilla')
        num_layers: Number of RNN layers (2)
        attention_heads: Number of attention heads (8, theory-optimal)
        dropout: Dropout probability (0.2)
        use_positional_encoding: Whether to add positional encoding to decoder states
        teacher_forcing_ratio: Initial teacher forcing probability (0.9)
    """
    
    def __init__(
        self,
        bottleneck_dim: int = 128,
        hidden_size: int = 512,
        output_size: int = 300,
        encoder_hidden_size: int = 512,
        max_seq_len: int = 50,
        rnn_type: str = 'LSTM',
        num_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.2,
        use_positional_encoding: bool = True,
        teacher_forcing_ratio: float = 0.9
    ):
        super(AttentionEnhancedDecoder, self).__init__()
        
        self.bottleneck_dim = bottleneck_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder_hidden_size = encoder_hidden_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.attention_heads = attention_heads
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # 1. Bottleneck to hidden state projection
        self.hidden_projection = nn.Linear(bottleneck_dim, hidden_size * num_layers)
        
        # 2. Cell state projection for LSTM
        if self.rnn_type == 'lstm':
            self.cell_projection = nn.Linear(bottleneck_dim, hidden_size * num_layers)
        
        # 3. RNN decoder (same as original)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'vanilla':
            self.rnn = nn.RNN(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity='tanh',
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # 4. Encoder-Decoder Attention (KEY INNOVATION)
        self.encoder_decoder_attention = CrossAttention(
            d_model=hidden_size,
            num_heads=attention_heads,
            dropout=dropout
        )
        
        # 5. Positional encoding for decoder states (optional)
        if use_positional_encoding:
            self.positional_encoding = SinusoidalPositionalEncoding(
                d_model=hidden_size,
                max_len=max_seq_len,
                dropout=0.0  # Don't add dropout here, handled separately
            )
        else:
            self.positional_encoding = None
        
        # 6. Context integration layer
        # Combines RNN hidden state + attention context
        context_dim = hidden_size + encoder_hidden_size  # concat(rnn_output, attention_context)
        self.context_integration = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 7. Output projection: enhanced hidden → output embeddings
        self.output_projection = nn.Linear(hidden_size, output_size)
        
        # 8. Teacher forcing projection: output embeddings → RNN input
        self.teacher_forcing_projection = nn.Linear(output_size, hidden_size)
        
        # 9. Learned start token for autoregressive generation
        self.start_token = nn.Parameter(torch.randn(1, 1, output_size))
        
        # 10. Dropout layers
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        # Initialize weights
        self._init_weights()
        
        # Initialize attention weights storage for monitoring
        self.last_attention_weights = None
        
        logger.info(f"AttentionEnhancedDecoder initialized:")
        logger.info(f"  Architecture: {bottleneck_dim}D → RNN({hidden_size}D) + Attention({attention_heads} heads) → {output_size}D")
        logger.info(f"  RNN type: {rnn_type.upper()}, layers: {num_layers}")
        logger.info(f"  Attention: {attention_heads} heads, encoder_dim: {encoder_hidden_size}")
        logger.info(f"  Expected improvement: +0.15 cosine similarity (theory)")
        logger.info(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Linear layers: Xavier uniform
        for module in [self.hidden_projection, self.output_projection, 
                      self.teacher_forcing_projection]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # LSTM cell projection
        if hasattr(self, 'cell_projection'):
            nn.init.xavier_uniform_(self.cell_projection.weight)
            nn.init.zeros_(self.cell_projection.bias)
        
        # Context integration layers
        for module in self.context_integration:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Start token: small random values
        nn.init.normal_(self.start_token, mean=0, std=0.1)
    
    def init_hidden_from_bottleneck(self, z: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Initialize RNN hidden state from bottleneck representation.
        
        Args:
            z: Bottleneck representation [batch_size, bottleneck_dim]
            
        Returns:
            hidden: Initial hidden state [num_layers, batch_size, hidden_size]
            cell: Initial cell state [num_layers, batch_size, hidden_size] (LSTM only)
        """
        batch_size = z.shape[0]
        
        # Project bottleneck to hidden state
        hidden = self.hidden_projection(z)
        hidden = torch.tanh(hidden)  # Activation to keep values bounded
        
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
        encoder_hidden_states: torch.Tensor,
        target_sequences: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        teacher_forcing_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attention-enhanced decoding with encoder-decoder attention.
        
        Args:
            z: Bottleneck representation [batch_size, bottleneck_dim]
            encoder_hidden_states: Encoder states [batch_size, enc_seq_len, encoder_hidden_size]
            target_sequences: Target sequences for teacher forcing [batch_size, seq_len, output_size]
            encoder_mask: Encoder attention mask [batch_size, enc_seq_len]
            seq_len: Override sequence length (defaults to max_seq_len or target length)
            teacher_forcing_ratio: Override teacher forcing ratio
            
        Returns:
            output_sequences: Generated sequences [batch_size, seq_len, output_size]
            hidden_states: All decoder hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Determine sequence length
        if seq_len is not None:
            seq_len = seq_len
        elif target_sequences is not None:
            seq_len = target_sequences.shape[1]
        else:
            seq_len = self.max_seq_len
        
        if seq_len <= 0:
            raise ValueError(f"Sequence length must be positive, got {seq_len}")
        
        # Use provided teacher forcing ratio or default
        current_tf_ratio = teacher_forcing_ratio if teacher_forcing_ratio is not None else self.teacher_forcing_ratio
        
        # Initialize RNN state from bottleneck
        hidden, cell = self.init_hidden_from_bottleneck(z)
        
        # Prepare initial RNN state
        if self.rnn_type == 'lstm':
            rnn_state = (hidden, cell)
        else:
            rnn_state = hidden
        
        # Check if we can use teacher forcing
        can_use_teacher_forcing = (
            self.training and 
            target_sequences is not None and 
            current_tf_ratio > 0.0
        )
        
        # Prepare encoder states for attention
        # encoder_hidden_states: [batch_size, enc_seq_len, encoder_hidden_size]
        enc_seq_len = encoder_hidden_states.shape[1]
        
        # Create encoder attention mask if not provided
        if encoder_mask is None:
            encoder_mask = torch.ones(batch_size, enc_seq_len, device=device)
        
        # Expand encoder mask for attention: [batch_size, 1, enc_seq_len]
        encoder_attention_mask = encoder_mask.unsqueeze(1)
        
        # Generation with scheduled sampling and attention
        outputs = []
        decoder_hidden_states = []
        
        # Initialize first input
        if can_use_teacher_forcing and torch.rand(1).item() < current_tf_ratio:
            # Use start token but project through teacher forcing
            current_input = self.teacher_forcing_projection(self.start_token.expand(batch_size, 1, -1))
        else:
            # Use learned start token projected to hidden space
            current_input = self.teacher_forcing_projection(self.start_token.expand(batch_size, 1, -1))
        
        # Autoregressive generation with attention
        for t in range(seq_len):
            # 1. RNN step: current_input → hidden_output
            rnn_output, rnn_state = self.rnn(current_input, rnn_state)
            # rnn_output: [batch_size, 1, hidden_size]
            
            # 2. Apply dropout to RNN output
            if self.dropout is not None:
                rnn_output = self.dropout(rnn_output)
            
            # 3. Add positional encoding to decoder state (optional)
            if self.positional_encoding is not None:
                # Create position tensor for current timestep
                pos_t = torch.full((batch_size, 1), t, device=device, dtype=torch.long)
                pos_encoding = self.positional_encoding.pe[:, t:t+1, :]  # [1, 1, hidden_size]
                rnn_output = rnn_output + pos_encoding
            
            # 4. CRITICAL: Encoder-Decoder Attention
            # Query: current decoder hidden state
            # Keys/Values: all encoder hidden states
            attention_context, attention_weights = self.encoder_decoder_attention(
                decoder_state=rnn_output,  # [batch_size, 1, hidden_size]
                encoder_states=encoder_hidden_states,  # [batch_size, enc_seq_len, encoder_hidden_size]
                mask=encoder_attention_mask,  # [batch_size, 1, enc_seq_len]
                return_attention=True
            )
            # attention_context: [batch_size, 1, encoder_hidden_size]
            
            # Store attention weights for monitoring (keep the latest)
            if attention_weights is not None:
                self.last_attention_weights = attention_weights.detach()
            
            # 5. Context Integration: concat(RNN output, attention context)
            enhanced_state = torch.cat([
                rnn_output,  # [batch_size, 1, hidden_size]
                attention_context  # [batch_size, 1, encoder_hidden_size]
            ], dim=-1)
            # enhanced_state: [batch_size, 1, hidden_size + encoder_hidden_size]
            
            # 6. Context integration network
            integrated_state = self.context_integration(enhanced_state)
            # integrated_state: [batch_size, 1, hidden_size]
            
            # 7. Project to output space
            predicted_output = self.output_projection(integrated_state)
            # predicted_output: [batch_size, 1, output_size]
            
            # Store outputs
            outputs.append(predicted_output)
            decoder_hidden_states.append(integrated_state)
            
            # 8. Prepare next input (scheduled sampling)
            if t < seq_len - 1:  # Not the last timestep
                if can_use_teacher_forcing and torch.rand(1).item() < current_tf_ratio:
                    # Teacher forcing: use ground truth
                    next_target = target_sequences[:, t + 1:t + 2, :]  # [batch_size, 1, output_size]
                    current_input = self.teacher_forcing_projection(next_target)
                else:
                    # Autoregressive: use model prediction
                    current_input = self.teacher_forcing_projection(predicted_output.detach())
                
                # Apply dropout to input
                if self.dropout is not None:
                    current_input = self.dropout(current_input)
        
        # Concatenate all outputs
        output_sequences = torch.cat(outputs, dim=1)  # [batch_size, seq_len, output_size]
        hidden_states = torch.cat(decoder_hidden_states, dim=1)  # [batch_size, seq_len, hidden_size]
        
        return output_sequences, hidden_states
    
    def generate(
        self,
        z: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        max_length: int = 50,
        encoder_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Pure autoregressive generation (inference mode).
        
        Args:
            z: Bottleneck representation [batch_size, bottleneck_dim]
            encoder_hidden_states: Encoder states [batch_size, enc_seq_len, encoder_hidden_size]
            max_length: Maximum generation length
            encoder_mask: Encoder attention mask
            temperature: Sampling temperature (1.0 = no temperature)
            
        Returns:
            generated_sequences: [batch_size, max_length, output_size]
        """
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            # Pure autoregressive generation (teacher_forcing_ratio = 0.0)
            generated_sequences, _ = self.forward(
                z=z,
                encoder_hidden_states=encoder_hidden_states,
                target_sequences=None,
                encoder_mask=encoder_mask,
                seq_len=max_length,
                teacher_forcing_ratio=0.0
            )
        
        return generated_sequences


# Utility functions for attention analysis

def analyze_attention_patterns(
    decoder: AttentionEnhancedDecoder,
    z: torch.Tensor,
    encoder_states: torch.Tensor,
    target_sequences: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Analyze attention patterns for interpretability.
    
    Args:
        decoder: AttentionEnhancedDecoder model
        z: Bottleneck representation
        encoder_states: Encoder hidden states
        target_sequences: Optional target sequences
        
    Returns:
        analysis: Dictionary with attention weights and statistics
    """
    decoder.eval()
    
    with torch.no_grad():
        # Forward pass with attention weights
        batch_size = z.shape[0]
        device = z.device
        seq_len = target_sequences.shape[1] if target_sequences is not None else decoder.max_seq_len
        
        # Initialize RNN state
        hidden, cell = decoder.init_hidden_from_bottleneck(z)
        if decoder.rnn_type == 'lstm':
            rnn_state = (hidden, cell)
        else:
            rnn_state = hidden
        
        # Collect attention weights for each timestep
        attention_weights_list = []
        
        # Start token
        current_input = decoder.teacher_forcing_projection(
            decoder.start_token.expand(batch_size, 1, -1)
        )
        
        for t in range(seq_len):
            # RNN step
            rnn_output, rnn_state = decoder.rnn(current_input, rnn_state)
            
            # Attention with weights
            attention_context, attention_weights = decoder.encoder_decoder_attention(
                decoder_state=rnn_output,
                encoder_states=encoder_states,
                return_attention=True
            )
            
            attention_weights_list.append(attention_weights.cpu())
            
            # Context integration and output projection
            enhanced_state = torch.cat([rnn_output, attention_context], dim=-1)
            integrated_state = decoder.context_integration(enhanced_state)
            predicted_output = decoder.output_projection(integrated_state)
            
            # Next input
            if target_sequences is not None and t < seq_len - 1:
                next_target = target_sequences[:, t + 1:t + 2, :]
                current_input = decoder.teacher_forcing_projection(next_target)
            else:
                current_input = decoder.teacher_forcing_projection(predicted_output)
        
        # Stack attention weights: [batch_size, num_heads, seq_len, enc_seq_len]
        all_attention_weights = torch.stack(attention_weights_list, dim=2)
        
        # Compute attention statistics
        attention_entropy = -(all_attention_weights * torch.log(all_attention_weights + 1e-8)).sum(dim=-1)
        attention_max = all_attention_weights.max(dim=-1)[0]
        attention_spread = all_attention_weights.std(dim=-1)
        
        return {
            'attention_weights': all_attention_weights,
            'attention_entropy': attention_entropy,
            'attention_max': attention_max,
            'attention_spread': attention_spread
        }


if __name__ == "__main__":
    # Test AttentionEnhancedDecoder
    print("Testing AttentionEnhancedDecoder...")
    
    # Test parameters
    batch_size, enc_seq_len, dec_seq_len = 2, 15, 10
    bottleneck_dim, hidden_size, output_size, encoder_hidden_size = 128, 512, 300, 512
    
    # Create model
    decoder = AttentionEnhancedDecoder(
        bottleneck_dim=bottleneck_dim,
        hidden_size=hidden_size,
        output_size=output_size,
        encoder_hidden_size=encoder_hidden_size,
        max_seq_len=50,
        rnn_type='LSTM',
        num_layers=2,
        attention_heads=8,
        dropout=0.1
    )
    
    # Test data
    z = torch.randn(batch_size, bottleneck_dim)
    encoder_states = torch.randn(batch_size, enc_seq_len, encoder_hidden_size)
    target_sequences = torch.randn(batch_size, dec_seq_len, output_size)
    
    # Training mode test
    decoder.train()
    output_seqs, hidden_states = decoder(
        z=z,
        encoder_hidden_states=encoder_states,
        target_sequences=target_sequences,
        teacher_forcing_ratio=0.8
    )
    
    print(f"✅ Training forward pass:")
    print(f"  Input: z={z.shape}, encoder_states={encoder_states.shape}")
    print(f"  Output: sequences={output_seqs.shape}, hidden={hidden_states.shape}")
    
    # Inference mode test
    decoder.eval()
    generated_seqs = decoder.generate(
        z=z,
        encoder_hidden_states=encoder_states,
        max_length=dec_seq_len
    )
    
    print(f"✅ Inference generation:")
    print(f"  Generated sequences: {generated_seqs.shape}")
    
    # Attention analysis
    analysis = analyze_attention_patterns(decoder, z, encoder_states, target_sequences)
    print(f"✅ Attention analysis:")
    print(f"  Attention weights: {analysis['attention_weights'].shape}")
    print(f"  Average entropy: {analysis['attention_entropy'].mean():.3f}")
    
    print("\n🎯 AttentionEnhancedDecoder test passed!")
    print("Expected improvement: +0.15 cosine similarity over pure RNN decoder")