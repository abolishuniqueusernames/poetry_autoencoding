"""
Co-occurrence Matrix Analysis and Visualization

This module provides comprehensive analysis and visualization functions for
co-occurrence matrices, including sparsity analysis, pattern detection,
matrix transformations, and visualization tools specifically designed for
poetry text analysis.
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import warnings


class CooccurrenceAnalyzer:
    """
    Comprehensive analysis toolkit for co-occurrence matrices.
    
    Provides methods for analyzing sparsity patterns, finding significant
    co-occurrences, applying matrix transformations, and generating
    visualizations optimized for poetry text analysis.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize analyzer with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
        
    def analyze_sparsity(self, matrix: sp.csr_matrix) -> Dict:
        """
        Comprehensive sparsity analysis of co-occurrence matrix.
        
        Analyzes the distribution of non-zero values, overall sparsity,
        and value statistics to understand matrix characteristics.
        
        Args:
            matrix: Sparse co-occurrence matrix
            
        Returns:
            Dictionary with sparsity statistics and value distributions
        """
        total_entries = matrix.shape[0] * matrix.shape[1]
        nonzero_entries = matrix.nnz
        sparsity = 1 - (nonzero_entries / total_entries)
        
        # Get non-zero values
        nonzero_values = matrix.data
        
        print("=== SPARSITY ANALYSIS ===")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Total possible entries: {total_entries:,}")
        print(f"Non-zero entries: {nonzero_entries:,}")
        print(f"Sparsity: {sparsity:.4%}")
        print(f"Density: {(1-sparsity):.4%}")
        
        if len(nonzero_values) > 0:
            print(f"\\nValue statistics:")
            print(f"  Min value: {nonzero_values.min():.4f}")
            print(f"  Max value: {nonzero_values.max():.4f}")
            print(f"  Mean value: {nonzero_values.mean():.4f}")
            print(f"  Median value: {np.median(nonzero_values):.4f}")
            print(f"  Std deviation: {nonzero_values.std():.4f}")
            
            # Value distribution
            print(f"\\nValue distribution:")
            hist, bins = np.histogram(nonzero_values, bins=10)
            for i in range(len(hist)):
                print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:,} entries")
        
        return {
            'sparsity': sparsity,
            'density': 1 - sparsity,
            'nonzero_count': nonzero_entries,
            'total_entries': total_entries,
            'value_stats': {
                'min': nonzero_values.min() if len(nonzero_values) > 0 else 0,
                'max': nonzero_values.max() if len(nonzero_values) > 0 else 0,
                'mean': nonzero_values.mean() if len(nonzero_values) > 0 else 0,
                'median': np.median(nonzero_values) if len(nonzero_values) > 0 else 0,
                'std': nonzero_values.std() if len(nonzero_values) > 0 else 0
            }
        }
    
    def find_top_cooccurrences(self, matrix: sp.csr_matrix, 
                              idx_to_word: Dict[int, str],
                              top_k: int = 30,
                              exclude_tokens: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        """
        Find the most frequent co-occurrence pairs in the matrix.
        
        Identifies the strongest co-occurrence relationships, optionally
        excluding specified tokens (like punctuation) from the results.
        
        Args:
            matrix: Sparse co-occurrence matrix
            idx_to_word: Index to word mapping
            top_k: Number of top pairs to return
            exclude_tokens: Optional list of tokens to exclude from results
            
        Returns:
            List of (word1, word2, cooccurrence_score) tuples
        """
        if exclude_tokens is None:
            exclude_tokens = ['.', ',', ';', ':', '!', '?', '"', "'", '-', '(', ')']
        
        exclude_indices = set()
        word_to_idx = {word: idx for idx, word in idx_to_word.items()}
        
        for token in exclude_tokens:
            if token in word_to_idx:
                exclude_indices.add(word_to_idx[token])
        
        # Convert to COO format for easier processing
        matrix_coo = matrix.tocoo()
        
        # Collect valid co-occurrences
        valid_cooccurrences = []
        for i, j, value in zip(matrix_coo.row, matrix_coo.col, matrix_coo.data):
            if i not in exclude_indices and j not in exclude_indices and i != j:
                word1 = idx_to_word.get(i, f"<UNK_{i}>")
                word2 = idx_to_word.get(j, f"<UNK_{j}>") 
                valid_cooccurrences.append((word1, word2, float(value)))
        
        # Sort by co-occurrence score and take top_k
        valid_cooccurrences.sort(key=lambda x: x[2], reverse=True)
        return valid_cooccurrences[:top_k]
    
    def analyze_word_cooccurrences(self, matrix: sp.csr_matrix,
                                  word_to_idx: Dict[str, int],
                                  idx_to_word: Dict[int, str], 
                                  target_word: str,
                                  top_k: int = 10,
                                  exclude_tokens: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Find top co-occurrences with a specific target word.
        
        Args:
            matrix: Sparse co-occurrence matrix
            word_to_idx: Word to index mapping
            idx_to_word: Index to word mapping  
            target_word: Word to find co-occurrences for
            top_k: Number of top co-occurrences to return
            exclude_tokens: Optional list of tokens to exclude
            
        Returns:
            List of (cooccurring_word, score) tuples
        """
        if target_word not in word_to_idx:
            self.logger.warning(f"Target word '{target_word}' not found in vocabulary")
            return []
        
        if exclude_tokens is None:
            exclude_tokens = ['.', ',', ';', ':', '!', '?', '"', "'", '-', '(', ')']
            
        target_idx = word_to_idx[target_word]
        
        # Get row for target word
        target_row = matrix.getrow(target_idx).tocoo()
        
        # Collect co-occurrences
        cooccurrences = []
        for col_idx, value in zip(target_row.col, target_row.data):
            if col_idx != target_idx:  # Skip self
                word = idx_to_word.get(col_idx, f"<UNK_{col_idx}>")
                if word not in exclude_tokens:
                    cooccurrences.append((word, float(value)))
        
        # Sort and return top_k
        cooccurrences.sort(key=lambda x: x[1], reverse=True)
        return cooccurrences[:top_k]
    
    def print_cooccurrence_analysis(self, matrix: sp.csr_matrix,
                                   word_to_idx: Dict[str, int], 
                                   idx_to_word: Dict[int, str],
                                   top_k: int = 20) -> None:
        """
        Print comprehensive co-occurrence analysis report.
        
        Generates a detailed report including overall top co-occurrences
        and specific analyses for interesting words commonly found in poetry.
        
        Args:
            matrix: Sparse co-occurrence matrix
            word_to_idx: Word to index mapping
            idx_to_word: Index to word mapping
            top_k: Number of top items to show in each category
        """
        print("=== COOCCURRENCE ANALYSIS (EXCLUDING PUNCTUATION) ===")
        print()
        
        # Overall top co-occurrences
        print(f"Top {top_k} Co-occurrences (excluding punctuation):")
        print("-" * 60)
        top_pairs = self.find_top_cooccurrences(matrix, idx_to_word, top_k)
        
        for i, (word1, word2, score) in enumerate(top_pairs, 1):
            print(f"{i:2d}. {word1} ↔ {word2}: {score:.2f}")
        print()
        
        # Analyze specific words common in poetry
        poetry_words = ['love', 'heart', 'night', 'light', 'time', 'eyes', 'soul', 'dream']
        available_words = [word for word in poetry_words if word in word_to_idx]
        
        if available_words:
            print("Co-occurrences for key poetry words:")
            print("-" * 40)
            
            for word in available_words[:5]:  # Limit to avoid too much output
                print(f"\\n'{word}' co-occurs with:")
                cooccurs = self.analyze_word_cooccurrences(matrix, word_to_idx, idx_to_word, word, top_k=8)
                for j, (cooccur_word, score) in enumerate(cooccurs, 1):
                    print(f"  {j}. {cooccur_word}: {score:.2f}")
    
    def visualize_matrix_sample(self, matrix: sp.csr_matrix,
                               idx_to_word: Dict[int, str],
                               sample_size: int = 100,
                               figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Visualize a sample of the co-occurrence matrix as a heatmap.
        
        Creates a heatmap visualization of a subset of the matrix for
        visual inspection of co-occurrence patterns.
        
        Args:
            matrix: Sparse co-occurrence matrix
            idx_to_word: Index to word mapping
            sample_size: Size of matrix sample to visualize
            figsize: Figure size for the plot
        """
        print(f"\\nVisualizing {sample_size}x{sample_size} sample of co-occurrence matrix...")
        
        # Sample a subset for visualization
        vocab_size = min(sample_size, matrix.shape[0])
        sample_matrix = matrix[:vocab_size, :vocab_size].toarray()
        
        # Get word labels for the sample
        sample_words = [idx_to_word.get(i, f"<UNK_{i}>") for i in range(vocab_size)]
        
        # Create heatmap
        plt.figure(figsize=figsize)
        
        # Use log scale for better visualization of sparse data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_matrix = np.log1p(sample_matrix)  # log(1 + x) to handle zeros
        
        sns.heatmap(log_matrix, 
                   xticklabels=sample_words, 
                   yticklabels=sample_words,
                   cmap='YlOrRd',
                   square=True,
                   cbar_kws={'label': 'log(1 + co-occurrence count)'})
        
        plt.title(f'Co-occurrence Matrix Heatmap ({vocab_size}x{vocab_size} sample)')
        plt.xlabel('Context Words')
        plt.ylabel('Center Words')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Print some statistics about the sample
        nonzero_sample = np.count_nonzero(sample_matrix)
        total_sample = sample_matrix.size
        sample_sparsity = 1 - (nonzero_sample / total_sample)
        
        print(f"Sample matrix statistics:")
        print(f"  Shape: {sample_matrix.shape}")
        print(f"  Non-zero entries: {nonzero_sample:,}")
        print(f"  Sparsity: {sample_sparsity:.4%}")
        print(f"  Max value: {sample_matrix.max():.2f}")
        print(f"  Mean value: {sample_matrix.mean():.4f}")


def preprocess_cooccurrence_matrix(matrix: sp.csr_matrix,
                                 log_transform: bool = True,
                                 normalize: bool = True,
                                 smoothing: float = 2.0) -> sp.csr_matrix:
    """
    Apply mathematical preprocessing to co-occurrence matrix.
    
    Standard preprocessing pipeline including smoothing, log transformation,
    and normalization commonly used in NLP applications.
    
    Args:
        matrix: Input sparse co-occurrence matrix
        log_transform: Whether to apply log transformation
        normalize: Whether to normalize rows
        smoothing: Smoothing parameter for log transformation
        
    Returns:
        Preprocessed sparse matrix
    """
    print("Preprocessing co-occurrence matrix...")
    processed = matrix.copy().astype(np.float32)
    
    if smoothing > 0:
        print(f"  Applying smoothing (α={smoothing})")
        processed.data = processed.data + smoothing
    
    if log_transform:
        print("  Applying log transformation")
        processed.data = np.log(processed.data)
    
    if normalize:
        print("  Normalizing rows")
        # L2 normalization of rows
        row_sums = np.array(processed.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        row_diag_inv = sp.diags(1.0 / np.sqrt(row_sums))
        processed = row_diag_inv @ processed
    
    print(f"  Result: {processed.shape} matrix with {processed.nnz:,} non-zero entries")
    return processed


def compute_pmi(matrix: sp.csr_matrix, 
               positive_only: bool = True,
               smoothing_factor: float = 0.75) -> sp.csr_matrix:
    """
    Compute Pointwise Mutual Information (PMI) or Positive PMI (PPMI).
    
    PMI addresses sparsity problems in co-occurrence matrices by normalizing
    for word frequencies and using log scaling. PPMI keeps only positive values.
    
    Formula: PMI(w1, w2) = log(P(w1, w2) / (P(w1) * P(w2)))
    
    Args:
        matrix: Input co-occurrence matrix  
        positive_only: Whether to compute PPMI (only positive values)
        smoothing_factor: Sublinear scaling factor for marginal counts
        
    Returns:
        PMI/PPMI matrix
    """
    print("Computing PMI transformation...")
    print(f"  Input matrix: {matrix.shape} with {matrix.nnz:,} non-zero entries")
    
    # Get marginal counts (row and column sums)
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    col_sums = np.array(matrix.sum(axis=0)).flatten()
    total_count = matrix.sum()
    
    print(f"  Total co-occurrence count: {total_count:,.0f}")
    
    # Apply smoothing to marginal counts
    row_sums_smooth = row_sums ** smoothing_factor
    col_sums_smooth = col_sums ** smoothing_factor
    total_smooth = row_sums_smooth.sum()
    
    # Convert to probabilities
    p_rows = row_sums_smooth / total_smooth
    p_cols = col_sums_smooth / total_smooth
    
    # Compute PMI for each non-zero entry
    matrix_coo = matrix.tocoo()
    pmi_data = []
    
    for i, j, count in zip(matrix_coo.row, matrix_coo.col, matrix_coo.data):
        # Joint probability
        p_joint = count / total_count
        
        # Independent probabilities  
        p_i = p_rows[i]
        p_j = p_cols[j]
        
        # PMI calculation
        if p_joint > 0 and p_i > 0 and p_j > 0:
            pmi_value = np.log(p_joint / (p_i * p_j))
            
            # Apply PPMI filter if requested
            if positive_only and pmi_value <= 0:
                continue
                
            pmi_data.append((i, j, pmi_value))
    
    # Create new sparse matrix
    if pmi_data:
        rows, cols, data = zip(*pmi_data)
        pmi_matrix = sp.csr_matrix((data, (rows, cols)), shape=matrix.shape)
    else:
        pmi_matrix = sp.csr_matrix(matrix.shape)
    
    matrix_type = "PPMI" if positive_only else "PMI"
    print(f"  {matrix_type} matrix: {pmi_matrix.shape} with {pmi_matrix.nnz:,} non-zero entries")
    
    return pmi_matrix


def apply_smoothing_and_scaling(matrix: sp.csr_matrix,
                               method: str = 'pmi',
                               smoothing_factor: float = 0.75,
                               min_count: int = 1) -> sp.csr_matrix:
    """
    Apply various smoothing and scaling methods to sparse co-occurrence matrix.
    
    Supports multiple transformation methods commonly used in distributional semantics.
    
    Args:
        matrix: Input sparse co-occurrence matrix
        method: Transformation method ('pmi', 'ppmi', 'log', 'sqrt', 'sublinear')
        smoothing_factor: Smoothing parameter for some methods
        min_count: Minimum count threshold
        
    Returns:
        Transformed sparse matrix
    """
    print(f"Applying {method.upper()} transformation...")
    
    # Filter by minimum count
    if min_count > 1:
        matrix.data[matrix.data < min_count] = 0
        matrix.eliminate_zeros()
        print(f"  Filtered entries below {min_count}")
    
    if method in ('pmi', 'ppmi'):
        return compute_pmi(matrix, positive_only=(method == 'ppmi'), 
                          smoothing_factor=smoothing_factor)
    
    elif method == 'log':
        result = matrix.copy().astype(np.float32)
        result.data = np.log1p(result.data)  # log(1 + x)
        return result
    
    elif method == 'sqrt':
        result = matrix.copy().astype(np.float32)  
        result.data = np.sqrt(result.data)
        return result
    
    elif method == 'sublinear':
        result = matrix.copy().astype(np.float32)
        result.data = result.data ** smoothing_factor
        return result
    
    else:
        print(f"  Unknown method '{method}', returning original matrix")
        return matrix


def compare_transformations(matrix: sp.csr_matrix,
                           word_to_idx: Dict[str, int],
                           idx_to_word: Dict[int, str],
                           sample_words: Optional[List[str]] = None) -> None:
    """
    Compare different transformation methods on sample words.
    
    Applies various matrix transformations and shows how they affect
    co-occurrence scores for a set of sample words.
    
    Args:
        matrix: Input co-occurrence matrix
        word_to_idx: Word to index mapping
        idx_to_word: Index to word mapping
        sample_words: Optional list of words to analyze
    """
    if sample_words is None:
        sample_words = ['love', 'heart', 'night', 'light', 'time']
    
    # Filter to available words
    available_words = [word for word in sample_words if word in word_to_idx]
    if not available_words:
        print("No sample words found in vocabulary")
        return
    
    methods = ['log', 'sqrt', 'pmi', 'ppmi']
    
    print("=== TRANSFORMATION COMPARISON ===")
    
    for method in methods:
        print(f"\\n--- {method.upper()} ---")
        transformed = apply_smoothing_and_scaling(matrix, method=method)
        
        for word in available_words[:3]:  # Limit output
            word_idx = word_to_idx[word]
            word_row = transformed.getrow(word_idx).tocoo()
            
            # Get top co-occurrences
            cooccurs = [(idx_to_word[col], val) for col, val in zip(word_row.col, word_row.data) if col != word_idx]
            cooccurs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  {word}: {', '.join([f'{w}({v:.2f})' for w, v in cooccurs[:5]])}")