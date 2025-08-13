"""
GLoVe Embedding Manager for Poetry RNN Autoencoder

Production-grade implementation of GLoVe embedding management with:
- Integration with configuration system
- Robust loading and error handling
- Memory-efficient operations
- Comprehensive logging and validation
- Support for mock embeddings for testing
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

from ..config import EmbeddingConfig


class GLoVeEmbeddingManager:
    """
    Manage GLoVe embeddings for poetry text processing.
    
    This class handles loading, aligning, and querying GLoVe embeddings with 
    support for special tokens, out-of-vocabulary handling, and integration
    with the poetry RNN configuration system.
    
    Features:
    - Configurable embedding dimensions and paths
    - Robust loading with malformed line handling
    - Vocabulary alignment with fallback strategies
    - Mock embedding generation for testing
    - Memory-efficient operations with progress tracking
    - Comprehensive error handling and validation
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None, 
                 embedding_path: Optional[str] = None, 
                 embedding_dim: int = 300):
        """
        Initialize GLoVe embedding manager.
        
        Args:
            config: EmbeddingConfig instance with settings
            embedding_path: Path to GLoVe embedding file (overrides config)
            embedding_dim: Embedding dimensions (overrides config)
        """
        # Initialize configuration
        self.config = config or EmbeddingConfig()
        
        # Override config with direct parameters if provided
        if embedding_path is not None:
            self.config.embedding_path = embedding_path
        if embedding_dim != 300:  # Only override if non-default
            self.config.embedding_dim = embedding_dim
            
        # Core attributes
        self.embedding_path = self.config.embedding_path
        self.embedding_dim = self.config.embedding_dim
        self.embeddings: Dict[str, np.ndarray] = {}
        self.embedding_matrix: Optional[np.ndarray] = None
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        
        # Special tokens (preserved from notebook implementation)
        self.special_tokens = ['<UNK>', '<LINE_BREAK>', '<STANZA_BREAK>', '<POEM_START>', '<POEM_END>']
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Validation
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration settings."""
        if self.embedding_dim <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {self.embedding_dim}")
            
        if self.config.oov_strategy not in ["random_normal", "zero", "mean"]:
            raise ValueError(f"Invalid OOV strategy: {self.config.oov_strategy}")
            
    def load_glove_embeddings(self, limit: Optional[int] = None, 
                             show_progress: bool = True) -> None:
        """
        Load GLoVe embeddings from file with robust error handling.
        
        Args:
            limit: Maximum number of embeddings to load (for testing)
            show_progress: Whether to show loading progress
            
        Raises:
            FileNotFoundError: If embedding file doesn't exist and mock creation disabled
            ValueError: If embedding file format is invalid
        """
        # Check if file exists
        if not self.embedding_path or not Path(self.embedding_path).exists():
            if self.config.create_mock_if_missing:
                self.logger.info("GLoVe embeddings not found. Creating mock embeddings.")
                self._create_mock_embeddings()
                return
            else:
                raise FileNotFoundError(f"GLoVe embedding file not found: {self.embedding_path}")
        
        self.logger.info(f"Loading GLoVe embeddings from {self.embedding_path}")
        
        # Load embeddings with robust parsing
        count = 0
        skipped_lines = 0
        
        try:
            with open(self.embedding_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if limit and count >= limit:
                        break
                    
                    try:
                        parts = line.strip().split()
                        
                        # Validate line format
                        if len(parts) != self.embedding_dim + 1:
                            skipped_lines += 1
                            if skipped_lines <= 10:  # Log first few errors
                                self.logger.warning(
                                    f"Line {line_num}: Expected {self.embedding_dim + 1} parts, "
                                    f"got {len(parts)}. Skipping."
                                )
                            continue
                        
                        # Extract word and vector
                        word = parts[0]
                        
                        try:
                            vector = np.array(parts[1:], dtype=np.float32)
                        except ValueError as e:
                            skipped_lines += 1
                            if skipped_lines <= 10:
                                self.logger.warning(f"Line {line_num}: Invalid vector data. Skipping.")
                            continue
                        
                        self.embeddings[word] = vector
                        count += 1
                        
                        # Progress reporting
                        if show_progress and count % 100000 == 0:
                            self.logger.info(f"  Loaded {count:,} embeddings...")
                            
                    except Exception as e:
                        skipped_lines += 1
                        if skipped_lines <= 10:
                            self.logger.warning(f"Line {line_num}: Error parsing line - {e}")
                        continue
                        
        except Exception as e:
            raise ValueError(f"Error reading embedding file: {e}")
        
        # Report results
        self.logger.info(f"✓ Loaded {len(self.embeddings):,} GLoVe embeddings")
        if skipped_lines > 0:
            self.logger.warning(f"Skipped {skipped_lines} malformed lines")
            
    def _create_mock_embeddings(self, vocab_size: int = 10000) -> None:
        """
        Create mock embeddings for testing and tutorial purposes.
        
        Args:
            vocab_size: Number of mock embeddings to create
        """
        self.logger.info("Creating mock embeddings for testing...")
        
        # Common poetry words that would be in GLoVe
        common_words = [
            'the', 'and', 'i', 'you', 'love', 'heart', 'like', 'feel', 'think',
            'time', 'night', 'day', 'light', 'dark', 'eyes', 'hand', 'world',
            'life', 'death', 'hope', 'dream', 'pain', 'joy', 'sad', 'happy',
            'beautiful', 'broken', 'empty', 'full', 'lonely', 'together',
            'remember', 'forget', 'always', 'never', 'maybe', 'sometimes',
            'soul', 'mind', 'body', 'spirit', 'breath', 'voice', 'silence',
            'shadow', 'memory', 'tears', 'smile', 'touch', 'kiss', 'whisper'
        ]
        
        # Create deterministic embeddings based on word
        for word in common_words:
            # Use word hash for deterministic but unique vectors
            np.random.seed(sum(ord(c) for c in word) % 2**32)
            vector = np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
            self.embeddings[word] = vector
        
        self.logger.info(f"✓ Created {len(self.embeddings)} mock embeddings")
        
    def align_vocabulary(self, vocabulary: Dict[str, int], 
                        special_tokens: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Align poetry vocabulary with GLoVe embeddings.
        
        Creates an aligned embedding matrix and vocabulary mapping that includes
        special tokens and handles out-of-vocabulary words with fallback strategies.
        
        Args:
            vocabulary: Word to index mapping from tokenized corpus
            special_tokens: List of special tokens to include (uses default if None)
            
        Returns:
            Tuple of (embedding_matrix, alignment_stats)
            
        Raises:
            ValueError: If vocabulary is empty or invalid
        """
        if not vocabulary:
            raise ValueError("Vocabulary cannot be empty")
            
        # Use default special tokens if not provided
        if special_tokens is None:
            special_tokens = self.special_tokens.copy()
            
        self.logger.info(f"Aligning vocabulary ({len(vocabulary)} words) with GLoVe embeddings...")
        
        # Initialize aligned structures
        aligned_vocab = {}
        embedding_matrix = []
        
        # Add special tokens first
        idx = 0
        for token in special_tokens:
            aligned_vocab[token] = idx
            # Initialize special tokens with small random vectors
            vector = self._create_special_token_embedding()
            embedding_matrix.append(vector)
            idx += 1
            
        # Track alignment statistics
        stats = {
            'total_words': len(vocabulary),
            'found_exact': 0,
            'found_lowercase': 0,
            'found_cleaned': 0,
            'oov_words': 0,
            'special_tokens': len(special_tokens),
            'oov_list': []
        }
        
        # Align vocabulary words
        for word, _ in sorted(vocabulary.items(), key=lambda x: x[1]):
            if word in special_tokens:
                continue  # Already handled
                
            vector = self._find_embedding_for_word(word, stats)
            
            aligned_vocab[word] = idx
            embedding_matrix.append(vector)
            idx += 1
            
        # Convert to numpy array
        self.embedding_matrix = np.array(embedding_matrix, dtype=np.float32)
        self.word_to_idx = aligned_vocab
        self.idx_to_word = {idx: word for word, idx in aligned_vocab.items()}
        
        # Log statistics
        self._log_alignment_stats(stats)
        
        return self.embedding_matrix, stats
        
    def _create_special_token_embedding(self) -> np.ndarray:
        """Create embedding vector for special tokens."""
        return np.random.normal(0, 0.01, self.embedding_dim).astype(np.float32)
        
    def _find_embedding_for_word(self, word: str, stats: Dict) -> np.ndarray:
        """
        Find embedding for word using fallback strategies.
        
        Args:
            word: Word to find embedding for
            stats: Statistics dictionary to update
            
        Returns:
            Embedding vector for the word
        """
        # Strategy 1: Exact match
        if word in self.embeddings:
            stats['found_exact'] += 1
            return self.embeddings[word]
            
        # Strategy 2: Lowercase match (if alignment_fallback enabled)
        if self.config.alignment_fallback and word.lower() in self.embeddings:
            stats['found_lowercase'] += 1
            return self.embeddings[word.lower()]
            
        # Strategy 3: Cleaned match (remove punctuation)
        if self.config.alignment_fallback:
            cleaned_word = word.strip('.,!?;:"()[]{}')
            if cleaned_word and cleaned_word in self.embeddings:
                stats['found_cleaned'] += 1
                return self.embeddings[cleaned_word]
                
        # OOV handling
        stats['oov_words'] += 1
        stats['oov_list'].append(word)
        
        return self._create_oov_embedding(word)
        
    def _create_oov_embedding(self, word: str) -> np.ndarray:
        """
        Create embedding for out-of-vocabulary word.
        
        Args:
            word: OOV word
            
        Returns:
            Embedding vector based on configured OOV strategy
        """
        if self.config.oov_strategy == "zero":
            return np.zeros(self.embedding_dim, dtype=np.float32)
            
        elif self.config.oov_strategy == "mean":
            if self.embeddings:
                # Use mean of all loaded embeddings
                all_embeddings = np.array(list(self.embeddings.values()))
                return np.mean(all_embeddings, axis=0).astype(np.float32)
            else:
                # Fallback to random if no embeddings loaded
                return np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
                
        else:  # random_normal (default)
            # Create deterministic but unique vector based on word
            seed = sum(ord(c) for c in word) % 2**32
            np.random.seed(seed)
            
            if self.embeddings:
                # Scale based on embedding statistics
                all_embeddings = np.array(list(self.embeddings.values()))
                std = np.std(all_embeddings)
                return np.random.normal(0, std * 0.1, self.embedding_dim).astype(np.float32)
            else:
                return np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
                
    def _log_alignment_stats(self, stats: Dict) -> None:
        """Log alignment statistics."""
        total = stats['total_words']
        found = stats['found_exact'] + stats['found_lowercase'] + stats['found_cleaned']
        
        self.logger.info(f"\n=== VOCABULARY ALIGNMENT RESULTS ===")
        self.logger.info(f"Total vocabulary: {total:,} words")
        self.logger.info(f"Special tokens: {stats['special_tokens']}")
        self.logger.info(f"Found in GLoVe: {found:,} ({found/total*100:.1f}%)")
        self.logger.info(f"  - Exact matches: {stats['found_exact']:,}")
        self.logger.info(f"  - Lowercase matches: {stats['found_lowercase']:,}")
        self.logger.info(f"  - Cleaned matches: {stats['found_cleaned']:,}")
        self.logger.info(f"Out-of-vocabulary: {stats['oov_words']:,} ({stats['oov_words']/total*100:.1f}%)")
        
        # Log some OOV examples
        if stats['oov_list'] and len(stats['oov_list']) <= 20:
            self.logger.info(f"OOV examples: {', '.join(stats['oov_list'][:10])}")
        elif stats['oov_list']:
            self.logger.info(f"OOV examples: {', '.join(stats['oov_list'][:10])} (and {len(stats['oov_list'])-10} more)")
            
    def get_embedding_for_word(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a specific word.
        
        Args:
            word: Word to get embedding for
            
        Returns:
            Embedding vector or None if word not found
        """
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.embedding_matrix[idx]
        return None
        
    def get_vocabulary_size(self) -> int:
        """Get size of aligned vocabulary."""
        return len(self.word_to_idx)
        
    def get_embedding_matrix(self) -> Optional[np.ndarray]:
        """Get the full embedding matrix."""
        return self.embedding_matrix
        
    def save_aligned_embeddings(self, output_path: str) -> None:
        """
        Save aligned embeddings to disk.
        
        Args:
            output_path: Base path for saving (will create .npy and .json files)
        """
        if self.embedding_matrix is None:
            raise ValueError("No aligned embeddings to save. Run align_vocabulary first.")
            
        output_path = Path(output_path)
        
        # Save embedding matrix
        matrix_path = output_path.with_suffix('.npy')
        np.save(matrix_path, self.embedding_matrix)
        
        # Save vocabulary mappings
        vocab_path = output_path.with_suffix('.json')
        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'config': {
                'embedding_dim': self.embedding_dim,
                'vocab_size': len(self.word_to_idx),
                'special_tokens': self.special_tokens
            }
        }
        
        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
            
        self.logger.info(f"Saved aligned embeddings to {matrix_path} and {vocab_path}")
        
    def load_aligned_embeddings(self, input_path: str) -> None:
        """
        Load previously saved aligned embeddings.
        
        Args:
            input_path: Base path for loading (expects .npy and .json files)
        """
        input_path = Path(input_path)
        
        # Load embedding matrix
        matrix_path = input_path.with_suffix('.npy')
        if not matrix_path.exists():
            raise FileNotFoundError(f"Embedding matrix not found: {matrix_path}")
            
        self.embedding_matrix = np.load(matrix_path)
        
        # Load vocabulary mappings
        vocab_path = input_path.with_suffix('.json')
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary mapping not found: {vocab_path}")
            
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
        
        # Validate consistency
        config = vocab_data.get('config', {})
        if config.get('embedding_dim') != self.embedding_dim:
            self.logger.warning(f"Dimension mismatch: expected {self.embedding_dim}, "
                              f"got {config.get('embedding_dim')}")
                              
        self.logger.info(f"Loaded aligned embeddings from {input_path}")
        
    def __repr__(self) -> str:
        return (f"GLoVeEmbeddingManager(dim={self.embedding_dim}, "
                f"loaded={len(self.embeddings)}, aligned={len(self.word_to_idx)})")