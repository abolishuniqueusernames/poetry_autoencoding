"""
Denoising Autoencoder Extensions for Poetry

This module extends the poetry autoencoder system with denoising capabilities,
implementing state-of-the-art noise injection strategies for improved robustness
and generalization in poetry reconstruction tasks.

★ Insight: Denoising autoencoders force the model to learn more robust representations
by corrupting input during training, then requiring perfect reconstruction from
the corrupted input. This leads to better semantic understanding and improved
generalization to unseen poetry styles.
"""

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class NoiseType(Enum):
    """Types of noise injection for denoising autoencoders."""
    DROPOUT = "dropout"
    GAUSSIAN = "gaussian" 
    WORD_DROPOUT = "word_dropout"
    SWAP = "swap"
    MASK = "mask"
    SYNONYM = "synonym"
    SEMANTIC_EMBEDDING = "semantic_embedding"  # Gaussian noise in embedding space


@dataclass
class DenoisingConfig:
    """Configuration for denoising autoencoder training.
    
    Args:
        noise_type: Type of noise to inject (dropout, gaussian, word_dropout, etc.)
        noise_strength: Intensity of noise (0.0 = no noise, 1.0 = maximum)
        corruption_probability: Probability of corrupting each token/element
        adaptive_noise: Adjust noise level during training (curriculum learning)
        preserve_structure: Maintain poetic structure during corruption
        semantic_noise: Use semantic-aware noise (synonyms, related words)
        embedding_noise_std: Standard deviation for semantic embedding noise
        late_training_epochs: Epoch to start semantic embedding noise
    """
    noise_type: NoiseType = NoiseType.WORD_DROPOUT
    noise_strength: float = 0.15
    corruption_probability: float = 0.1
    adaptive_noise: bool = True
    preserve_structure: bool = True
    semantic_noise: bool = False
    
    # Curriculum settings for adaptive noise
    initial_noise: float = 0.05
    final_noise: float = 0.20
    warmup_epochs: int = 10
    
    # Semantic embedding noise settings
    embedding_noise_std: float = 0.1
    late_training_epochs: int = 80  # Start semantic noise after 80 epochs (when model is more stable)


class PoetryDenoiser(nn.Module):
    """
    Poetry-specific noise injection module for denoising autoencoder training.
    
    This module implements sophisticated noise injection strategies tailored
    for poetry, preserving important structural and semantic properties while
    introducing controlled corruption for robust learning.
    """
    
    def __init__(self, config: DenoisingConfig, vocabulary_size: int, embedding_matrix: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        self.vocabulary_size = vocabulary_size
        self.training_step = 0
        
        # Special token indices (should be provided by vocabulary)
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.mask_token_id = 2
        
        # Embedding matrix for semantic noise (optional)
        if embedding_matrix is not None:
            self.register_buffer('embedding_matrix', embedding_matrix)
            self.has_embeddings = True
        else:
            self.embedding_matrix = None
            self.has_embeddings = False
        
    def forward(
        self, 
        input_sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        epoch: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply noise to input sequences for denoising training.
        
        Args:
            input_sequences: Clean input sequences [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            epoch: Current training epoch (for adaptive noise)
            
        Returns:
            Tuple of (corrupted_sequences, corruption_mask)
        """
        if not self.training:
            return input_sequences, attention_mask
            
        batch_size, seq_len = input_sequences.shape
        device = input_sequences.device
        
        # Calculate current noise strength
        noise_strength = self._get_adaptive_noise_strength(epoch)
        
        # Apply noise based on configured type
        if self.config.noise_type == NoiseType.WORD_DROPOUT:
            return self._word_dropout_noise(input_sequences, attention_mask, noise_strength)
        elif self.config.noise_type == NoiseType.MASK:
            return self._mask_noise(input_sequences, attention_mask, noise_strength)
        elif self.config.noise_type == NoiseType.SWAP:
            return self._swap_noise(input_sequences, attention_mask, noise_strength)
        elif self.config.noise_type == NoiseType.GAUSSIAN:
            return self._gaussian_noise(input_sequences, attention_mask, noise_strength)
        elif self.config.noise_type == NoiseType.SEMANTIC_EMBEDDING:
            return self._semantic_embedding_noise(input_sequences, attention_mask, noise_strength, epoch)
        else:
            return input_sequences, attention_mask
            
    def _get_adaptive_noise_strength(self, epoch: Optional[int]) -> float:
        """Calculate adaptive noise strength based on training progress."""
        if not self.config.adaptive_noise or epoch is None:
            return self.config.noise_strength
            
        if epoch < self.config.warmup_epochs:
            # Linear interpolation from initial to final noise
            progress = epoch / self.config.warmup_epochs
            return self.config.initial_noise + progress * (self.config.final_noise - self.config.initial_noise)
        else:
            return self.config.final_noise
            
    def _word_dropout_noise(
        self, 
        sequences: torch.Tensor, 
        mask: torch.Tensor, 
        strength: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply word dropout noise by replacing tokens with UNK."""
        batch_size, seq_len = sequences.shape
        device = sequences.device
        
        # Create corruption mask
        corruption_probs = torch.rand(batch_size, seq_len, device=device)
        corruption_mask = (corruption_probs < strength) & mask.bool()
        
        # Don't corrupt special tokens or maintain structure if configured
        if self.config.preserve_structure:
            corruption_mask = self._preserve_poetry_structure(sequences, corruption_mask)
            
        # Apply corruption
        corrupted = sequences.clone()
        corrupted[corruption_mask] = self.unk_token_id
        
        return corrupted, mask
        
    def _mask_noise(
        self, 
        sequences: torch.Tensor, 
        mask: torch.Tensor, 
        strength: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masking noise using dedicated MASK tokens."""
        batch_size, seq_len = sequences.shape
        device = sequences.device
        
        corruption_probs = torch.rand(batch_size, seq_len, device=device)
        corruption_mask = (corruption_probs < strength) & mask.bool()
        
        if self.config.preserve_structure:
            corruption_mask = self._preserve_poetry_structure(sequences, corruption_mask)
            
        corrupted = sequences.clone()
        corrupted[corruption_mask] = self.mask_token_id
        
        return corrupted, mask
        
    def _swap_noise(
        self, 
        sequences: torch.Tensor, 
        mask: torch.Tensor, 
        strength: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply swap noise by randomly shuffling adjacent tokens."""
        corrupted = sequences.clone()
        batch_size, seq_len = sequences.shape
        
        for b in range(batch_size):
            valid_positions = torch.where(mask[b])[0]
            if len(valid_positions) < 2:
                continue
                
            num_swaps = int(len(valid_positions) * strength / 2)
            for _ in range(num_swaps):
                if len(valid_positions) < 2:
                    break
                    
                # Pick two adjacent positions to swap
                idx = torch.randint(0, len(valid_positions) - 1, (1,)).item()
                pos1, pos2 = valid_positions[idx], valid_positions[idx + 1]
                
                # Swap tokens
                corrupted[b, pos1], corrupted[b, pos2] = corrupted[b, pos2], corrupted[b, pos1]
                
        return corrupted, mask
        
    def _gaussian_noise(
        self, 
        sequences: torch.Tensor, 
        mask: torch.Tensor, 
        strength: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Gaussian noise (for embedding-space corruption)."""
        # Note: This would typically be applied in embedding space
        # For token-level sequences, we'll simulate with random token replacement
        batch_size, seq_len = sequences.shape
        device = sequences.device
        
        corruption_probs = torch.rand(batch_size, seq_len, device=device)
        corruption_mask = (corruption_probs < strength) & mask.bool()
        
        corrupted = sequences.clone()
        # Replace with random tokens from vocabulary
        random_tokens = torch.randint(
            0, self.vocabulary_size, 
            corruption_mask.sum().item(), 
            device=device
        )
        corrupted[corruption_mask] = random_tokens
        
        return corrupted, mask
        
    def _semantic_embedding_noise(
        self, 
        sequences: torch.Tensor, 
        mask: torch.Tensor, 
        strength: float,
        epoch: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply semantic embedding noise by adding Gaussian noise in embedding space
        and finding the nearest neighbor word. This forces fine-grained semantic learning.
        
        Mathematical approach:
        1. For token t_i, get embedding e_i = Embedding[t_i]
        2. Add Gaussian noise: ẽ_i = e_i + ε, where ε ~ N(0, σ²I)  
        3. Find nearest neighbor: t'_i = argmin_j ||ẽ_i - Embedding[j]||₂
        
        This should be used late in training when model has stable representations.
        """
        if not self.has_embeddings:
            # Fallback to simple random replacement if no embeddings available
            return self._gaussian_noise(sequences, mask, strength)
            
        # Only apply semantic noise after specified number of epochs
        if epoch is not None and epoch < self.config.late_training_epochs:
            return sequences, mask
            
        batch_size, seq_len = sequences.shape
        device = sequences.device
        
        # Create corruption mask
        corruption_probs = torch.rand(batch_size, seq_len, device=device)
        corruption_mask = (corruption_probs < strength) & mask.bool()
        
        if self.config.preserve_structure:
            corruption_mask = self._preserve_poetry_structure(sequences, corruption_mask)
            
        if not corruption_mask.any():
            return sequences, mask
            
        corrupted = sequences.clone()
        
        # Get embeddings for corrupted tokens
        corrupted_indices = torch.where(corruption_mask)
        if len(corrupted_indices[0]) == 0:
            return corrupted, mask
            
        # Extract tokens to corrupt
        tokens_to_corrupt = sequences[corrupted_indices]
        
        # Get their embeddings
        original_embeddings = self.embedding_matrix[tokens_to_corrupt]  # [n_corrupt, embed_dim]
        
        # Add Gaussian noise
        noise = torch.randn_like(original_embeddings) * self.config.embedding_noise_std
        noisy_embeddings = original_embeddings + noise
        
        # Find nearest neighbors in embedding space
        # Compute distances to all embeddings: ||ẽ - E||₂
        # noisy_embeddings: [n_corrupt, embed_dim]
        # embedding_matrix: [vocab_size, embed_dim]
        distances = torch.cdist(noisy_embeddings, self.embedding_matrix)  # [n_corrupt, vocab_size]
        
        # Find closest words (excluding special tokens)
        # Mask out special tokens to avoid corrupting to PAD/UNK/MASK
        special_token_mask = torch.ones(self.vocabulary_size, device=device, dtype=torch.bool)
        special_token_mask[self.pad_token_id] = False
        special_token_mask[self.unk_token_id] = False
        special_token_mask[self.mask_token_id] = False
        
        # Set distances to special tokens to infinity
        distances[:, ~special_token_mask] = float('inf')
        
        # Find nearest valid tokens
        nearest_tokens = distances.argmin(dim=1)
        
        # Replace corrupted positions with nearest neighbors
        corrupted[corrupted_indices] = nearest_tokens
        
        return corrupted, mask
        
    def _preserve_poetry_structure(
        self, 
        sequences: torch.Tensor, 
        corruption_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Preserve important poetic structure by avoiding corruption of:
        - Line endings
        - Punctuation
        - Very short words (articles, prepositions)
        """
        # This is a simplified version - in practice, you'd use vocabulary
        # information to identify these special tokens
        preserved_mask = corruption_mask.clone()
        
        # Example: Don't corrupt the last token of each sequence (line ending)
        preserved_mask[:, -1] = False
        
        return preserved_mask


class DenoisingPoetryAutoencoder(nn.Module):
    """
    Denoising extension wrapper for Poetry Autoencoder.
    
    This wrapper adds denoising capabilities to any existing poetry autoencoder
    by injecting controlled noise during training and maintaining clean
    reconstruction targets.
    """
    
    def __init__(
        self, 
        base_autoencoder: nn.Module,
        denoising_config: DenoisingConfig,
        vocabulary_size: int,
        embedding_matrix: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.autoencoder = base_autoencoder
        self.denoiser = PoetryDenoiser(denoising_config, vocabulary_size, embedding_matrix)
        self.config = denoising_config
        
    def forward(self, batch: Dict[str, torch.Tensor], epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with denoising corruption applied during training.
        
        Args:
            batch: Input batch with 'input_sequences' and 'attention_mask'
            epoch: Current training epoch for adaptive noise
            
        Returns:
            Autoencoder outputs with corrupted input, clean targets
        """
        input_sequences = batch['input_sequences']
        attention_mask = batch['attention_mask']
        
        if self.training:
            # Apply noise corruption during training
            corrupted_input, corrupted_mask = self.denoiser(
                input_sequences, attention_mask, epoch
            )
            
            # Forward pass with corrupted input
            corrupted_batch = {
                'input_sequences': corrupted_input,
                'attention_mask': corrupted_mask
            }
            outputs = self.autoencoder(corrupted_batch)
            
            # Add clean targets for loss computation
            outputs['clean_targets'] = input_sequences
            outputs['clean_mask'] = attention_mask
            
        else:
            # Clean forward pass during evaluation
            outputs = self.autoencoder(batch)
            
        return outputs


def create_denoising_autoencoder(
    base_autoencoder: nn.Module,
    noise_type: Union[str, NoiseType] = "word_dropout",
    noise_strength: float = 0.15,
    vocabulary_size: int = 10000,
    embedding_matrix: Optional[torch.Tensor] = None,
    **kwargs
) -> DenoisingPoetryAutoencoder:
    """
    Factory function to create a denoising poetry autoencoder.
    
    Args:
        base_autoencoder: Existing poetry autoencoder to wrap
        noise_type: Type of noise injection
        noise_strength: Strength of noise (0.0-1.0)
        vocabulary_size: Size of vocabulary for noise generation
        embedding_matrix: Pre-trained embeddings for semantic noise [vocab_size, embed_dim]
        **kwargs: Additional configuration options
        
    Returns:
        Denoising autoencoder wrapper
        
    Example:
        >>> from poetry_rnn.models import RNNAutoencoder
        >>> base_model = RNNAutoencoder(input_size=300, hidden_size=512, bottleneck_dim=128)
        >>> 
        >>> # Standard word dropout
        >>> denoising_model = create_denoising_autoencoder(
        ...     base_model, 
        ...     noise_type="word_dropout",
        ...     noise_strength=0.15
        ... )
        >>> 
        >>> # Semantic embedding noise (requires GLoVe embeddings)
        >>> embeddings = load_glove_embeddings()  # [vocab_size, 300]
        >>> semantic_model = create_denoising_autoencoder(
        ...     base_model,
        ...     noise_type="semantic_embedding", 
        ...     noise_strength=0.10,
        ...     embedding_matrix=embeddings,
        ...     embedding_noise_std=0.1,
        ...     late_training_epochs=30
        ... )
    """
    if isinstance(noise_type, str):
        noise_type = NoiseType(noise_type)
        
    config = DenoisingConfig(
        noise_type=noise_type,
        noise_strength=noise_strength,
        **kwargs
    )
    
    return DenoisingPoetryAutoencoder(base_autoencoder, config, vocabulary_size, embedding_matrix)