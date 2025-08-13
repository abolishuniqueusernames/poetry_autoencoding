"""
Embeddings module for GLoVe integration

Handles loading, vocabulary alignment, and management of 
pre-trained GLoVe embeddings for poetry text.
"""

from .glove_manager import GLoVeEmbeddingManager
from .embedding_utils import (
    EmbeddingAnalyzer, 
    create_analyzer,
    compute_pairwise_similarities,
    find_words_by_pattern
)

__all__ = [
    'GLoVeEmbeddingManager',
    'EmbeddingAnalyzer',
    'create_analyzer',
    'compute_pairwise_similarities',
    'find_words_by_pattern'
]