"""
Visualization utilities for poetry RNN autoencoder

Provides plotting and visualization functions for embeddings, attention,
co-occurrence matrices, and model analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set default plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')


def plot_embedding_2d(embeddings: np.ndarray, 
                      labels: Optional[List[str]] = None,
                      method: str = 'pca',
                      title: str = 'Embedding Visualization',
                      figsize: Tuple[int, int] = (12, 8),
                      save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot 2D visualization of embeddings using dimensionality reduction
    
    Args:
        embeddings: Embedding matrix (n_words, embedding_dim)
        labels: Optional word labels for points
        method: Reduction method ('pca', 'tsne', 'umap')
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save plot
        
    Returns:
        matplotlib Figure object
    """
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        subtitle = f"PCA Explained Variance: {reducer.explained_variance_ratio_.sum():.1%}"
        
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
        reduced = reducer.fit_transform(embeddings)
        subtitle = "t-SNE projection"
        
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced = reducer.fit_transform(embeddings)
            subtitle = "UMAP projection"
        except ImportError:
            logger.warning("UMAP not available, falling back to PCA")
            return plot_embedding_2d(embeddings, labels, 'pca', title, figsize, save_path)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=50)
    
    # Add labels if provided
    if labels is not None:
        for i, label in enumerate(labels):
            if i < len(reduced):  # Safety check
                ax.annotate(label, (reduced[i, 0], reduced[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
    
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved embedding plot to {save_path}")
        
    return fig


def plot_attention_heatmap(attention_weights: np.ndarray,
                          token_labels: Optional[List[str]] = None,
                          title: str = 'Attention Weights',
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot attention weights as a heatmap
    
    Args:
        attention_weights: Attention matrix (seq_len, seq_len) or (batch, seq_len)
        token_labels: Optional token labels
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle different attention shapes
    if attention_weights.ndim == 1:
        # Single sequence attention
        attention_weights = attention_weights.reshape(1, -1)
        
    sns.heatmap(attention_weights, ax=ax, cmap='Blues', cbar=True)
    
    if token_labels is not None:
        if attention_weights.shape[1] == len(token_labels):
            ax.set_xticklabels(token_labels, rotation=45, ha='right')
        if attention_weights.shape[0] == len(token_labels):
            ax.set_yticklabels(token_labels, rotation=0)
    
    ax.set_title(title)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Attention Head' if attention_weights.shape[0] > 1 else 'Sequence')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention heatmap to {save_path}")
        
    return fig


def plot_cooccurrence_heatmap(cooccurrence_matrix: np.ndarray,
                             word_labels: Optional[List[str]] = None,
                             top_k: int = 50,
                             title: str = 'Co-occurrence Matrix',
                             figsize: Tuple[int, int] = (12, 10),
                             save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot co-occurrence matrix as heatmap
    
    Args:
        cooccurrence_matrix: Co-occurrence matrix
        word_labels: Word labels for axes
        top_k: Number of most frequent words to show
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save plot
        
    Returns:
        matplotlib Figure object
    """
    # Select top-k most frequent words for visualization
    if cooccurrence_matrix.shape[0] > top_k:
        # Get row sums to find most frequent words
        row_sums = np.array(cooccurrence_matrix.sum(axis=1)).flatten()
        top_indices = np.argsort(row_sums)[-top_k:]
        
        matrix_subset = cooccurrence_matrix[np.ix_(top_indices, top_indices)]
        labels_subset = [word_labels[i] for i in top_indices] if word_labels else None
    else:
        matrix_subset = cooccurrence_matrix
        labels_subset = word_labels
    
    # Convert sparse matrix if needed
    if hasattr(matrix_subset, 'toarray'):
        matrix_subset = matrix_subset.toarray()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use log scale for better visualization
    matrix_log = np.log1p(matrix_subset)
    
    sns.heatmap(matrix_log, ax=ax, cmap='YlOrRd', square=True, cbar=True)
    
    if labels_subset is not None:
        ax.set_xticklabels(labels_subset, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels_subset, rotation=0, fontsize=8)
    
    ax.set_title(f'{title} (Top {len(matrix_subset)} words, log scale)')
    ax.set_xlabel('Target Words')
    ax.set_ylabel('Context Words')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved co-occurrence heatmap to {save_path}")
        
    return fig


def plot_training_curves(train_losses: List[float],
                        val_losses: Optional[List[float]] = None,
                        train_metrics: Optional[Dict[str, List[float]]] = None,
                        val_metrics: Optional[Dict[str, List[float]]] = None,
                        title: str = 'Training Progress',
                        figsize: Tuple[int, int] = (15, 5),
                        save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot training and validation curves
    
    Args:
        train_losses: Training losses over epochs
        val_losses: Validation losses over epochs
        train_metrics: Additional training metrics
        val_metrics: Additional validation metrics
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save plot
        
    Returns:
        matplotlib Figure object
    """
    n_plots = 1
    if train_metrics:
        n_plots += len(train_metrics)
        
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Plot losses
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot additional metrics
    if train_metrics:
        for i, (metric_name, train_values) in enumerate(train_metrics.items()):
            ax = axes[i + 1]
            
            ax.plot(epochs[:len(train_values)], train_values, 'b-', 
                   label=f'Training {metric_name}', linewidth=2)
            
            if val_metrics and metric_name in val_metrics:
                val_values = val_metrics[metric_name]
                ax.plot(epochs[:len(val_values)], val_values, 'r-',
                       label=f'Validation {metric_name}', linewidth=2)
            
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
        
    return fig


def plot_reconstruction_comparison(original_tokens: List[str],
                                  reconstructed_tokens: List[str],
                                  title: str = 'Reconstruction Comparison',
                                  max_length: int = 50,
                                  figsize: Tuple[int, int] = (12, 6),
                                  save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot side-by-side comparison of original and reconstructed sequences
    
    Args:
        original_tokens: Original token sequence
        reconstructed_tokens: Reconstructed token sequence
        title: Plot title
        max_length: Maximum sequence length to display
        figsize: Figure size
        save_path: Optional path to save plot
        
    Returns:
        matplotlib Figure object
    """
    # Truncate sequences if too long
    orig_display = original_tokens[:max_length]
    recon_display = reconstructed_tokens[:max_length]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot original sequence
    ax1.text(0.02, 0.5, ' '.join(orig_display), transform=ax1.transAxes,
             fontsize=10, verticalalignment='center', wrap=True)
    ax1.set_title('Original Sequence')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Plot reconstructed sequence  
    ax2.text(0.02, 0.5, ' '.join(recon_display), transform=ax2.transAxes,
             fontsize=10, verticalalignment='center', wrap=True)
    ax2.set_title('Reconstructed Sequence')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved reconstruction comparison to {save_path}")
        
    return fig


def plot_dimensionality_analysis(singular_values: np.ndarray,
                                variance_ratios: Optional[np.ndarray] = None,
                                thresholds: List[float] = [0.8, 0.9, 0.95],
                                title: str = 'Dimensionality Analysis',
                                figsize: Tuple[int, int] = (12, 5),
                                save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot dimensionality analysis results
    
    Args:
        singular_values: Singular values from SVD
        variance_ratios: Cumulative variance ratios
        thresholds: Variance threshold lines to plot
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save plot
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot singular values
    ax1.plot(singular_values[:50], 'b-', linewidth=2)
    ax1.set_title('Singular Values')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Singular Value')
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative variance if provided
    if variance_ratios is not None:
        ax2.plot(variance_ratios, 'g-', linewidth=2)
        
        # Add threshold lines
        for threshold in thresholds:
            # Find dimension for this threshold
            dim = np.argmax(variance_ratios >= threshold) + 1
            ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
            ax2.axvline(x=dim-1, color='red', linestyle='--', alpha=0.7)
            ax2.text(dim+2, threshold-0.02, f'{threshold:.0%} at dim {dim}', 
                    fontsize=9, color='red')
        
        ax2.set_title('Cumulative Variance Ratio')
        ax2.set_xlabel('Component')
        ax2.set_ylabel('Cumulative Variance Ratio')
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dimensionality analysis to {save_path}")
        
    return fig


def plot_token_frequency_distribution(token_counts: Dict[str, int],
                                     top_k: int = 30,
                                     title: str = 'Token Frequency Distribution',
                                     figsize: Tuple[int, int] = (12, 6),
                                     save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot token frequency distribution
    
    Args:
        token_counts: Dictionary of token counts
        top_k: Number of top tokens to show
        title: Plot title  
        figsize: Figure size
        save_path: Optional path to save plot
        
    Returns:
        matplotlib Figure object
    """
    # Sort tokens by frequency
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    top_tokens = sorted_tokens[:top_k]
    
    tokens, counts = zip(*top_tokens)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot of top tokens
    ax1.bar(range(len(tokens)), counts)
    ax1.set_xticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=45, ha='right')
    ax1.set_title(f'Top {top_k} Most Frequent Tokens')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Log-log plot for Zipf's law
    all_counts = sorted(token_counts.values(), reverse=True)
    ranks = np.arange(1, len(all_counts) + 1)
    
    ax2.loglog(ranks, all_counts, 'b-', alpha=0.7)
    ax2.set_title("Zipf's Law Distribution")
    ax2.set_xlabel('Rank (log scale)')
    ax2.set_ylabel('Frequency (log scale)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved token frequency plot to {save_path}")
        
    return fig


def create_visualization_grid(plots: List[Tuple[callable, Dict[str, Any]]],
                             grid_shape: Optional[Tuple[int, int]] = None,
                             figsize: Tuple[int, int] = (16, 12),
                             title: str = 'Analysis Grid',
                             save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create a grid of multiple visualizations
    
    Args:
        plots: List of (plot_function, kwargs) tuples
        grid_shape: Grid dimensions (rows, cols)
        figsize: Figure size
        title: Overall title
        save_path: Optional path to save plot
        
    Returns:
        matplotlib Figure object
    """
    n_plots = len(plots)
    
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))
    else:
        rows, cols = grid_shape
    
    fig = plt.figure(figsize=figsize)
    
    for i, (plot_func, kwargs) in enumerate(plots):
        ax = plt.subplot(rows, cols, i + 1)
        
        # Remove save_path from individual plots
        plot_kwargs = kwargs.copy()
        plot_kwargs.pop('save_path', None)
        
        # Call the plot function
        try:
            plot_func(ax=ax, **plot_kwargs)
        except Exception as e:
            logger.warning(f"Failed to create plot {i+1}: {e}")
            ax.text(0.5, 0.5, f'Plot {i+1}\nFailed to render', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization grid to {save_path}")
        
    return fig