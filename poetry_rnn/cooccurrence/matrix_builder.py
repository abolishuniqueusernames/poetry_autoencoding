"""
Co-occurrence Matrix Builder for Poetry Text Analysis

This module provides the CooccurrenceMatrix class for building and managing
co-occurrence matrices from tokenized poetry text. It includes specialized
handling for poetry boundaries, multiple weighting schemes, and memory-efficient
sparse matrix operations.
"""

import numpy as np
import scipy.sparse as sp
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import pickle
from tqdm import tqdm

from ..config import CooccurrenceConfig


class CooccurrenceMatrix:
    """
    Efficient co-occurrence matrix computation for poetry text with context-aware processing.
    
    This class builds co-occurrence matrices from tokenized poetry text while respecting
    poetic structure boundaries (line breaks, stanza breaks, poem boundaries) and
    supporting multiple weighting schemes for distance-based relationships.
    
    Features:
    - Context boundary detection for poetry structure
    - Multiple weighting strategies (uniform, linear, harmonic, exponential)  
    - Memory-efficient sparse matrix operations
    - Vocabulary filtering and management
    - Progress tracking for large datasets
    - Matrix serialization and loading capabilities
    
    Args:
        config: CooccurrenceConfig object with matrix building parameters
        logger: Optional logger for progress tracking
    """
    
    def __init__(self, config: Optional[CooccurrenceConfig] = None, 
                 logger: Optional[logging.Logger] = None):
        """Initialize CooccurrenceMatrix with configuration."""
        self.config = config or CooccurrenceConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Core parameters from config
        self.window_size = self.config.window_size
        self.weighting = self.config.weighting
        self.min_count = self.config.min_count
        
        # Poetry-specific boundary tokens
        self.boundary_tokens = set(self.config.context_boundary_tokens)
        # Add essential poetry tokens if not present
        essential_boundaries = {'<LINE_BREAK>', '<STANZA_BREAK>', '<POEM_START>', '<POEM_END>'}
        self.boundary_tokens.update(essential_boundaries)
        
        # State variables
        self.vocabulary = None
        self.word_to_idx = None  
        self.idx_to_word = None
        self.matrix = None
        self.vocab_size = 0
        
    def build_vocabulary(self, all_tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Build vocabulary with frequency filtering while preserving special tokens.
        
        Creates word-to-index and index-to-word mappings from the input tokens,
        filtering by minimum count while always preserving boundary tokens and
        other special tokens that are essential for poetry processing.
        
        Args:
            all_tokens: Complete list of tokens from all documents
            
        Returns:
            Tuple of (word_to_idx, idx_to_word) dictionaries
            
        Raises:
            ValueError: If all_tokens is empty or contains only filtered tokens
        """
        if not all_tokens:
            raise ValueError("Cannot build vocabulary from empty token list")
            
        self.logger.info(f"Building vocabulary from {len(all_tokens):,} tokens...")
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        self.logger.info(f"Found {len(token_counts):,} unique tokens")
        
        # Filter by minimum count, but always keep special tokens
        vocab_tokens = []
        special_tokens = self.boundary_tokens | {'<UNK>', '<PAD>'}
        
        for token, count in token_counts.items():
            if count >= self.min_count or token in special_tokens:
                vocab_tokens.append(token)
        
        if not vocab_tokens:
            raise ValueError(f"No tokens remain after filtering with min_count={self.min_count}")
            
        # Sort for consistent indexing (special tokens first)
        special_in_vocab = [t for t in vocab_tokens if t in special_tokens]
        regular_in_vocab = [t for t in vocab_tokens if t not in special_tokens]
        vocab_tokens = sorted(special_in_vocab) + sorted(regular_in_vocab)
        
        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_tokens)}
        self.idx_to_word = {idx: word for idx, word in enumerate(vocab_tokens)}
        self.vocab_size = len(vocab_tokens)
        
        self.logger.info(f"Built vocabulary of {self.vocab_size:,} tokens "
                        f"(filtered from {len(token_counts):,})")
        
        # Store vocabulary for potential reuse
        self.vocabulary = vocab_tokens.copy()
        
        return self.word_to_idx, self.idx_to_word
    
    def compute_matrix(self, tokens_list: List[List[str]], 
                      word_to_idx: Optional[Dict[str, int]] = None,
                      show_progress: bool = True) -> sp.csr_matrix:
        """
        Compute co-occurrence matrix from tokenized documents.
        
        Builds a sparse co-occurrence matrix by processing all token sequences
        and accumulating co-occurrence counts within sliding windows. Respects
        poetry structure boundaries and applies distance-based weighting.
        
        Args:
            tokens_list: List of tokenized documents (each document is a list of tokens)
            word_to_idx: Optional vocabulary mapping (uses self.word_to_idx if None)
            show_progress: Whether to display progress bar
            
        Returns:
            Sparse CSR matrix of co-occurrences
            
        Raises:
            ValueError: If word_to_idx is None and vocabulary not built
            RuntimeError: If matrix computation fails
        """
        if word_to_idx is None:
            if self.word_to_idx is None:
                raise ValueError("Must provide word_to_idx or call build_vocabulary first")
            word_to_idx = self.word_to_idx
            
        if not tokens_list:
            raise ValueError("Cannot compute matrix from empty tokens_list")
            
        self.logger.info(f"Computing co-occurrence matrix from {len(tokens_list):,} documents...")
        
        cooccur_data = defaultdict(float)
        total_pairs = 0
        skipped_docs = 0
        
        # Progress tracking
        iterator = tqdm(tokens_list, desc="Processing documents") if show_progress else tokens_list
        
        try:
            # Process all documents to accumulate co-occurrences
            for doc_idx, tokens in enumerate(iterator):
                if not tokens or len(tokens) < 2:
                    skipped_docs += 1
                    continue
                    
                # Convert tokens to indices, handling unknown words
                token_indices = []
                filtered_tokens = []
                
                for token in tokens:
                    if token in word_to_idx:
                        token_indices.append(word_to_idx[token])
                        filtered_tokens.append(token)
                    elif '<UNK>' in word_to_idx:
                        token_indices.append(word_to_idx['<UNK>'])
                        filtered_tokens.append('<UNK>')
                    # Skip tokens not in vocabulary
                
                if len(filtered_tokens) < 2:  # Need at least 2 tokens
                    skipped_docs += 1
                    continue
                
                # Get co-occurrence pairs for this document
                doc_pairs = self._get_context_pairs(filtered_tokens, token_indices)
                
                # Accumulate pairs 
                for (i, j), weight in doc_pairs.items():
                    cooccur_data[(i, j)] += weight
                    total_pairs += 1
                    
        except Exception as e:
            raise RuntimeError(f"Error during matrix computation: {e}")
        
        self.logger.info(f"Accumulated {len(cooccur_data):,} unique co-occurrence pairs")
        self.logger.info(f"Skipped {skipped_docs:,} documents (too short or empty)")
        
        # Convert to sparse matrix
        vocab_size = len(word_to_idx)
        row_indices = []
        col_indices = []
        data = []
        
        for (i, j), count in cooccur_data.items():
            if count >= 1.0:  # Minimum threshold
                row_indices.append(i)
                col_indices.append(j)
                data.append(count)
                
                # Add symmetric entry if not diagonal
                if i != j:
                    row_indices.append(j)
                    col_indices.append(i)  
                    data.append(count)
        
        # Create sparse matrix
        try:
            matrix = sp.csr_matrix((data, (row_indices, col_indices)), 
                                 shape=(vocab_size, vocab_size))
            
            self.logger.info(f"Created sparse matrix: {matrix.shape} with {matrix.nnz:,} non-zero entries")
            self.matrix = matrix
            return matrix
            
        except Exception as e:
            raise RuntimeError(f"Error creating sparse matrix: {e}")
    
    def _get_context_pairs(self, tokens: List[str], 
                          token_indices: List[int]) -> Dict[Tuple[int, int], float]:
        """
        Extract co-occurrence pairs from a sequence respecting boundaries.
        
        Sliding window approach that respects poetry structure boundaries and
        applies distance-based weighting. Each token pair gets a weight based
        on their distance and the configured weighting scheme.
        
        Args:
            tokens: List of tokens (for boundary detection)
            token_indices: Corresponding indices in vocabulary
            
        Returns:
            Dictionary mapping (center_idx, context_idx) tuples to weights
        """
        pairs = defaultdict(float)
        
        # Ensure tokens and indices have same length
        if len(tokens) != len(token_indices):
            min_len = min(len(tokens), len(token_indices))
            tokens = tokens[:min_len]
            token_indices = token_indices[:min_len]
        
        for i, (center_token, center_idx) in enumerate(zip(tokens, token_indices)):
            if center_idx is None:  # Skip if token was filtered out
                continue
            
            # Get context boundaries if respecting boundaries
            if hasattr(self.config, 'respect_boundaries') and self.config.respect_boundaries:
                left_bound, right_bound = self._get_boundaries(tokens, i)
            else:
                left_bound, right_bound = 0, len(tokens) - 1
            
            # Define actual context window within boundaries
            context_start = max(left_bound, i - self.window_size)
            context_end = min(right_bound + 1, i + self.window_size + 1)
            
            # Collect context pairs
            for j in range(context_start, context_end):
                if j == i or j >= len(token_indices):  # Skip center word and bounds
                    continue
                    
                context_idx = token_indices[j]
                if context_idx is None:
                    continue
                    
                # Calculate distance and weight
                distance = abs(j - i)
                weight = self._calculate_weight(distance)
                
                # Add weighted pair (both directions for symmetry in undirected graph)
                pairs[(center_idx, context_idx)] += weight
        
        return pairs
    
    def _get_boundaries(self, tokens: List[str], center_idx: int) -> Tuple[int, int]:
        """
        Find left and right boundaries for context window.
        
        Searches for boundary tokens around the center position to determine
        the valid context window that doesn't cross poetry structure boundaries.
        
        Args:
            tokens: List of tokens to search
            center_idx: Index of center word
            
        Returns:
            Tuple of (left_boundary, right_boundary) indices
        """
        # Find left boundary
        left_bound = 0
        for i in range(center_idx - 1, -1, -1):
            if tokens[i] in self.boundary_tokens:
                left_bound = i + 1
                break
        
        # Find right boundary  
        right_bound = len(tokens) - 1
        for i in range(center_idx + 1, len(tokens)):
            if tokens[i] in self.boundary_tokens:
                right_bound = i - 1
                break
        
        return left_bound, right_bound
    
    def _calculate_weight(self, distance: int) -> float:
        """
        Calculate weight based on distance and weighting scheme.
        
        Supports multiple weighting strategies:
        - uniform: All context positions have equal weight (1.0)
        - linear: Weight decreases linearly with distance (1/distance)  
        - harmonic: Harmonic weighting (1/distance, same as linear)
        - exponential: Exponential decay based on window size
        - constant: Uniform weight (same as uniform)
        
        Args:
            distance: Distance between center and context word
            
        Returns:
            Weight value for the co-occurrence
        """
        if distance == 0:
            return 0.0  # No self-co-occurrence
            
        if self.weighting in ('uniform', 'constant'):
            return 1.0
        elif self.weighting in ('linear', 'harmonic'):
            return 1.0 / distance
        elif self.weighting == 'exponential':
            return np.exp(-distance / self.window_size)
        else:
            self.logger.warning(f"Unknown weighting scheme '{self.weighting}', using uniform")
            return 1.0
    
    def save_matrix(self, filepath: Union[str, Path]) -> None:
        """
        Save the co-occurrence matrix and vocabulary to disk.
        
        Saves both the sparse matrix and associated vocabulary mappings
        for later loading and analysis.
        
        Args:
            filepath: Path to save the matrix (will add .npz extension)
        """
        if self.matrix is None:
            raise ValueError("No matrix to save. Call compute_matrix first.")
            
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save sparse matrix
        sp.save_npz(str(filepath.with_suffix('.npz')), self.matrix)
        
        # Save vocabulary and metadata
        metadata = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocabulary': self.vocabulary,
            'config': {
                'window_size': self.window_size,
                'weighting': self.weighting,
                'min_count': self.min_count,
                'boundary_tokens': list(self.boundary_tokens)
            }
        }
        
        with open(filepath.with_suffix('.metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
            
        self.logger.info(f"Saved matrix and metadata to {filepath}")
    
    def load_matrix(self, filepath: Union[str, Path]) -> sp.csr_matrix:
        """
        Load a previously saved co-occurrence matrix.
        
        Loads both the sparse matrix and vocabulary mappings from disk.
        
        Args:
            filepath: Path to the saved matrix files
            
        Returns:
            Loaded sparse matrix
        """
        filepath = Path(filepath)
        
        # Load sparse matrix
        matrix_file = filepath.with_suffix('.npz')
        if not matrix_file.exists():
            raise FileNotFoundError(f"Matrix file not found: {matrix_file}")
            
        self.matrix = sp.load_npz(str(matrix_file))
        
        # Load metadata
        metadata_file = filepath.with_suffix('.metadata.pkl')
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                
            self.word_to_idx = metadata['word_to_idx']
            self.idx_to_word = metadata['idx_to_word']
            self.vocabulary = metadata.get('vocabulary', list(self.word_to_idx.keys()))
            self.vocab_size = len(self.vocabulary) if self.vocabulary else len(self.word_to_idx)
            
            # Update config if available
            if 'config' in metadata:
                config_data = metadata['config']
                self.window_size = config_data.get('window_size', self.window_size)
                self.weighting = config_data.get('weighting', self.weighting)
                self.min_count = config_data.get('min_count', self.min_count)
                self.boundary_tokens = set(config_data.get('boundary_tokens', self.boundary_tokens))
                
        self.logger.info(f"Loaded matrix {self.matrix.shape} with {self.matrix.nnz:,} non-zero entries")
        return self.matrix
    
    def get_matrix_info(self) -> Dict:
        """
        Get information about the current matrix.
        
        Returns:
            Dictionary with matrix statistics and configuration
        """
        if self.matrix is None:
            return {"status": "No matrix computed"}
            
        sparsity = 1.0 - (self.matrix.nnz / (self.matrix.shape[0] * self.matrix.shape[1]))
        
        return {
            "shape": self.matrix.shape,
            "nnz": self.matrix.nnz,
            "sparsity": sparsity,
            "vocab_size": self.vocab_size,
            "window_size": self.window_size,
            "weighting": self.weighting,
            "min_count": self.min_count,
            "boundary_tokens": list(self.boundary_tokens)
        }