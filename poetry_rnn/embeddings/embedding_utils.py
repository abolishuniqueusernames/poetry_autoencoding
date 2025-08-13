"""
Embedding Utilities for Poetry RNN Autoencoder

Utility functions for working with embeddings including:
- Similarity calculations
- Analogy resolution
- Vocabulary analysis
- Embedding space exploration
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .glove_manager import GLoVeEmbeddingManager


class EmbeddingAnalyzer:
    """
    Analyzer for embedding similarity and analogy operations.
    
    Provides high-level functions for exploring embedding spaces,
    finding similar words, solving analogies, and analyzing vocabulary
    relationships in the context of poetry.
    """
    
    def __init__(self, embedding_manager: GLoVeEmbeddingManager):
        """
        Initialize analyzer with embedding manager.
        
        Args:
            embedding_manager: Configured GLoVeEmbeddingManager instance
        """
        self.embedding_manager = embedding_manager
        self.logger = logging.getLogger(__name__)
        
        # Validate that embeddings are aligned
        if embedding_manager.embedding_matrix is None:
            raise ValueError("Embedding manager must have aligned embeddings. "
                           "Call align_vocabulary() first.")
                           
    def find_similar_words(self, word: str, top_k: int = 10, 
                          exclude_special: bool = True) -> List[Tuple[str, float]]:
        """
        Find most similar words using cosine similarity.
        
        Args:
            word: Query word
            top_k: Number of similar words to return
            exclude_special: Whether to exclude special tokens from results
            
        Returns:
            List of (word, similarity_score) tuples, sorted by similarity
            
        Raises:
            ValueError: If word not found in vocabulary
        """
        if word not in self.embedding_manager.word_to_idx:
            available_similar = self._find_approximate_matches(word)
            if available_similar:
                suggestion = ', '.join(available_similar[:3])
                raise ValueError(f"Word '{word}' not in vocabulary. "
                               f"Similar words available: {suggestion}")
            else:
                raise ValueError(f"Word '{word}' not in vocabulary")
        
        word_idx = self.embedding_manager.word_to_idx[word]
        word_vector = self.embedding_manager.embedding_matrix[word_idx:word_idx+1]
        
        # Compute cosine similarities with all words
        similarities = cosine_similarity(word_vector, self.embedding_manager.embedding_matrix)[0]
        
        # Get indices sorted by similarity (excluding the word itself)
        sorted_indices = np.argsort(similarities)[::-1]
        
        similar_words = []
        for idx in sorted_indices:
            if len(similar_words) >= top_k:
                break
                
            similar_word = self.embedding_manager.idx_to_word[idx]
            similarity = similarities[idx]
            
            # Skip the query word itself
            if similar_word == word:
                continue
                
            # Skip special tokens if requested
            if exclude_special and similar_word in self.embedding_manager.special_tokens:
                continue
                
            similar_words.append((similar_word, float(similarity)))
            
        return similar_words
        
    def _find_approximate_matches(self, word: str, max_matches: int = 5) -> List[str]:
        """Find words that approximately match the query word."""
        word_lower = word.lower()
        matches = []
        
        for vocab_word in self.embedding_manager.word_to_idx.keys():
            if vocab_word in self.embedding_manager.special_tokens:
                continue
                
            # Check for substring matches
            if word_lower in vocab_word.lower() or vocab_word.lower() in word_lower:
                matches.append(vocab_word)
                if len(matches) >= max_matches:
                    break
                    
        return matches
        
    def solve_analogy(self, a: str, b: str, c: str, top_k: int = 5,
                     exclude_input: bool = True) -> List[Tuple[str, float]]:
        """
        Solve analogy: a is to b as c is to ?
        
        Uses vector arithmetic: b - a + c to find the answer.
        
        Args:
            a: First word in analogy
            b: Second word in analogy (relation target)
            c: Third word in analogy
            top_k: Number of candidate answers to return
            exclude_input: Whether to exclude input words from results
            
        Returns:
            List of (word, similarity_score) tuples for potential answers
            
        Raises:
            ValueError: If any input word not found in vocabulary
        """
        words = [a, b, c]
        missing_words = [w for w in words if w not in self.embedding_manager.word_to_idx]
        
        if missing_words:
            raise ValueError(f"Words not in vocabulary: {missing_words}")
            
        # Get embedding vectors
        vec_a = self.embedding_manager.embedding_matrix[self.embedding_manager.word_to_idx[a]]
        vec_b = self.embedding_manager.embedding_matrix[self.embedding_manager.word_to_idx[b]]
        vec_c = self.embedding_manager.embedding_matrix[self.embedding_manager.word_to_idx[c]]
        
        # Compute analogy vector: b - a + c
        analogy_vector = vec_b - vec_a + vec_c
        analogy_vector = analogy_vector.reshape(1, -1)
        
        # Find most similar words to analogy vector
        similarities = cosine_similarity(analogy_vector, self.embedding_manager.embedding_matrix)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        
        candidates = []
        for idx in sorted_indices:
            if len(candidates) >= top_k:
                break
                
            candidate_word = self.embedding_manager.idx_to_word[idx]
            similarity = similarities[idx]
            
            # Skip input words if requested
            if exclude_input and candidate_word in words:
                continue
                
            # Skip special tokens
            if candidate_word in self.embedding_manager.special_tokens:
                continue
                
            candidates.append((candidate_word, float(similarity)))
            
        return candidates
        
    def test_poetry_analogies(self) -> Dict[str, Any]:
        """
        Test poetry-specific word analogies.
        
        Returns:
            Dictionary with analogy test results
        """
        # Poetry-specific analogies to test
        poetry_analogies = [
            ("love", "heart", "pain", "soul"),
            ("night", "dark", "day", "light"),
            ("flower", "beauty", "thorn", "pain"),
            ("memory", "past", "dream", "future"),
            ("tears", "sadness", "smile", "joy"),
            ("whisper", "quiet", "shout", "loud")
        ]
        
        results = {
            'tested_analogies': [],
            'successful_tests': 0,
            'total_tests': 0,
            'missing_words': set()
        }
        
        for a, b, c, expected in poetry_analogies:
            results['total_tests'] += 1
            
            try:
                candidates = self.solve_analogy(a, b, c, top_k=5)
                
                # Check if expected word is in top candidates
                candidate_words = [word for word, _ in candidates]
                success = expected in candidate_words
                
                if success:
                    results['successful_tests'] += 1
                    rank = candidate_words.index(expected) + 1
                else:
                    rank = None
                    
                test_result = {
                    'analogy': f"{a} : {b} :: {c} : {expected}",
                    'predicted': candidate_words[0] if candidates else None,
                    'candidates': candidates,
                    'expected_found': success,
                    'expected_rank': rank
                }
                
                results['tested_analogies'].append(test_result)
                
            except ValueError as e:
                # Track missing words
                missing = str(e).split(': ')[-1]
                results['missing_words'].update(missing.strip('[]').replace("'", "").split(', '))
                
                test_result = {
                    'analogy': f"{a} : {b} :: {c} : {expected}",
                    'error': str(e),
                    'expected_found': False
                }
                
                results['tested_analogies'].append(test_result)
                
        return results
        
    def analyze_semantic_clusters(self, seed_words: List[str], 
                                cluster_size: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Analyze semantic clusters around seed words.
        
        Args:
            seed_words: Words to use as cluster centers
            cluster_size: Number of similar words per cluster
            
        Returns:
            Dictionary mapping seed words to their semantic clusters
        """
        clusters = {}
        
        for seed_word in seed_words:
            try:
                similar_words = self.find_similar_words(seed_word, top_k=cluster_size)
                clusters[seed_word] = similar_words
            except ValueError as e:
                self.logger.warning(f"Could not analyze cluster for '{seed_word}': {e}")
                clusters[seed_word] = []
                
        return clusters
        
    def compute_word_similarity(self, word1: str, word2: str) -> float:
        """
        Compute cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Cosine similarity score
            
        Raises:
            ValueError: If either word not found in vocabulary
        """
        words = [word1, word2]
        missing_words = [w for w in words if w not in self.embedding_manager.word_to_idx]
        
        if missing_words:
            raise ValueError(f"Words not in vocabulary: {missing_words}")
            
        idx1 = self.embedding_manager.word_to_idx[word1]
        idx2 = self.embedding_manager.word_to_idx[word2]
        
        vec1 = self.embedding_manager.embedding_matrix[idx1:idx1+1]
        vec2 = self.embedding_manager.embedding_matrix[idx2:idx2+1]
        
        similarity = cosine_similarity(vec1, vec2)[0, 0]
        return float(similarity)
        
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vocabulary and embeddings.
        
        Returns:
            Dictionary with vocabulary statistics
        """
        vocab_size = self.embedding_manager.get_vocabulary_size()
        special_token_count = len(self.embedding_manager.special_tokens)
        
        # Compute embedding statistics
        embedding_matrix = self.embedding_manager.embedding_matrix
        embedding_mean = np.mean(embedding_matrix, axis=0)
        embedding_std = np.std(embedding_matrix, axis=0)
        
        # Analyze norm distribution
        embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
        
        stats = {
            'vocabulary_size': vocab_size,
            'special_tokens': special_token_count,
            'regular_words': vocab_size - special_token_count,
            'embedding_dim': self.embedding_manager.embedding_dim,
            'embedding_stats': {
                'mean_norm': float(np.mean(embedding_norms)),
                'std_norm': float(np.std(embedding_norms)),
                'min_norm': float(np.min(embedding_norms)),
                'max_norm': float(np.max(embedding_norms)),
                'mean_component': embedding_mean.tolist(),
                'std_component': embedding_std.tolist()
            }
        }
        
        return stats
        
    def find_outlier_embeddings(self, threshold: float = 3.0) -> List[Tuple[str, float]]:
        """
        Find words with unusually large or small embedding norms.
        
        Args:
            threshold: Standard deviations from mean to consider outlier
            
        Returns:
            List of (word, norm) tuples for outlier embeddings
        """
        embedding_norms = np.linalg.norm(self.embedding_manager.embedding_matrix, axis=1)
        mean_norm = np.mean(embedding_norms)
        std_norm = np.std(embedding_norms)
        
        outliers = []
        
        for idx, norm in enumerate(embedding_norms):
            z_score = abs(norm - mean_norm) / std_norm
            
            if z_score > threshold:
                word = self.embedding_manager.idx_to_word[idx]
                outliers.append((word, float(norm)))
                
        # Sort by norm (most extreme first)
        outliers.sort(key=lambda x: abs(x[1] - mean_norm), reverse=True)
        
        return outliers


def create_analyzer(embedding_manager: GLoVeEmbeddingManager) -> EmbeddingAnalyzer:
    """
    Factory function to create an EmbeddingAnalyzer.
    
    Args:
        embedding_manager: Configured GLoVeEmbeddingManager instance
        
    Returns:
        EmbeddingAnalyzer instance
    """
    return EmbeddingAnalyzer(embedding_manager)


def compute_pairwise_similarities(words: List[str], 
                                embedding_manager: GLoVeEmbeddingManager) -> np.ndarray:
    """
    Compute pairwise similarities between a list of words.
    
    Args:
        words: List of words to compare
        embedding_manager: Configured GLoVeEmbeddingManager instance
        
    Returns:
        Symmetric similarity matrix
        
    Raises:
        ValueError: If any word not found in vocabulary
    """
    missing_words = [w for w in words if w not in embedding_manager.word_to_idx]
    if missing_words:
        raise ValueError(f"Words not in vocabulary: {missing_words}")
        
    # Get embedding vectors for all words
    word_vectors = []
    for word in words:
        idx = embedding_manager.word_to_idx[word]
        vector = embedding_manager.embedding_matrix[idx]
        word_vectors.append(vector)
        
    word_matrix = np.array(word_vectors)
    
    # Compute pairwise cosine similarities
    similarity_matrix = cosine_similarity(word_matrix)
    
    return similarity_matrix


def find_words_by_pattern(pattern: str, embedding_manager: GLoVeEmbeddingManager,
                         max_results: int = 20) -> List[str]:
    """
    Find words matching a pattern in the vocabulary.
    
    Args:
        pattern: Pattern to search for (substring match)
        embedding_manager: Configured GLoVeEmbeddingManager instance
        max_results: Maximum number of results to return
        
    Returns:
        List of matching words
    """
    pattern_lower = pattern.lower()
    matches = []
    
    for word in embedding_manager.word_to_idx.keys():
        if word in embedding_manager.special_tokens:
            continue
            
        if pattern_lower in word.lower():
            matches.append(word)
            if len(matches) >= max_results:
                break
                
    return sorted(matches)