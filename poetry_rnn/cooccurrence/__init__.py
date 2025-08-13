"""
Co-occurrence Analysis Module

This module provides comprehensive tools for building and analyzing word co-occurrence 
matrices with specialized support for poetry text processing. It includes matrix 
construction, analysis functions, and advanced dimensionality estimation capabilities.

Key Components:
- CooccurrenceMatrix: Main class for building co-occurrence matrices
- CooccurrenceAnalyzer: Analysis and visualization tools  
- DimensionalityAnalyzer: Effective dimensionality estimation
- Various transformation and preprocessing functions

Example usage:
    from poetry_rnn.cooccurrence import CooccurrenceMatrix, CooccurrenceAnalyzer
    from poetry_rnn.config import CooccurrenceConfig
    
    # Build matrix
    config = CooccurrenceConfig(window_size=5, weighting='linear')
    builder = CooccurrenceMatrix(config)
    word_to_idx, idx_to_word = builder.build_vocabulary(all_tokens)
    matrix = builder.compute_matrix(tokenized_docs, word_to_idx)
    
    # Analyze matrix
    analyzer = CooccurrenceAnalyzer()
    sparsity_stats = analyzer.analyze_sparsity(matrix)
    top_cooccurrences = analyzer.find_top_cooccurrences(matrix, idx_to_word)
"""

# Core classes
from .matrix_builder import CooccurrenceMatrix
from .matrix_analysis import (
    CooccurrenceAnalyzer, 
    preprocess_cooccurrence_matrix,
    compute_pmi,
    apply_smoothing_and_scaling,
    compare_transformations
)
from .dimensionality import DimensionalityAnalyzer

# Main exports
__all__ = [
    # Core classes
    'CooccurrenceMatrix',
    'CooccurrenceAnalyzer', 
    'DimensionalityAnalyzer',
    
    # Analysis functions
    'preprocess_cooccurrence_matrix',
    'compute_pmi',
    'apply_smoothing_and_scaling', 
    'compare_transformations',
]