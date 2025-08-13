"""
Effective Dimensionality Analysis for Co-occurrence Matrices

This module provides advanced dimensionality analysis tools for co-occurrence matrices,
including SVD-based effective dimensionality estimation, spectral analysis, and
dimensionality reduction recommendations for neural network applications.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path
import warnings


class DimensionalityAnalyzer:
    """
    Advanced dimensionality analysis for co-occurrence matrices.
    
    Provides tools for estimating effective dimensionality, analyzing spectral
    properties, and making informed decisions about dimensionality reduction
    for downstream neural network applications.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize analyzer with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
        
    def compute_effective_dimensionality(self, matrix: sp.csr_matrix,
                                       variance_thresholds: List[float] = None,
                                       max_components: int = 200,
                                       sample_size: int = 2000) -> Dict:
        """
        Compute effective dimensionality using SVD/PCA analysis.
        
        Estimates the intrinsic dimensionality of the co-occurrence space by
        analyzing singular value decay and explained variance ratios. This
        helps determine optimal embedding dimensions for neural networks.
        
        Args:
            matrix: Input sparse co-occurrence matrix
            variance_thresholds: List of variance retention thresholds (e.g., [0.90, 0.95, 0.99])
            max_components: Maximum number of components to analyze
            sample_size: Maximum matrix size for dense SVD (larger matrices are sampled)
            
        Returns:
            Dictionary with dimensionality analysis results
        """
        if variance_thresholds is None:
            variance_thresholds = [0.90, 0.95, 0.99]
            
        self.logger.info("Computing effective dimensionality analysis...")
        
        # Handle large matrices by sampling
        if matrix.shape[0] > sample_size:
            self.logger.info(f"Matrix too large ({matrix.shape}), sampling {sample_size}x{sample_size}")
            # Take a representative sample
            indices = np.random.choice(matrix.shape[0], sample_size, replace=False)
            indices.sort()  # Keep indices sorted for consistency
            dense_matrix = matrix[indices, :][:, indices].toarray()
        else:
            dense_matrix = matrix.toarray()
            
        # Ensure matrix is symmetric for proper analysis
        if not np.allclose(dense_matrix, dense_matrix.T, atol=1e-8):
            self.logger.info("Making matrix symmetric for analysis")
            dense_matrix = (dense_matrix + dense_matrix.T) / 2
            
        # Compute SVD
        self.logger.info("Computing SVD decomposition...")
        try:
            U, s, Vt = np.linalg.svd(dense_matrix, full_matrices=False)
        except np.linalg.LinAlgError as e:
            self.logger.warning(f"SVD failed, trying with reduced precision: {e}")
            dense_matrix = dense_matrix.astype(np.float32)
            U, s, Vt = np.linalg.svd(dense_matrix, full_matrices=False)
        
        # Limit to max_components to avoid memory issues
        s = s[:max_components]
        U = U[:, :max_components]
        Vt = Vt[:max_components, :]
        
        # Compute explained variance
        variance_explained = s**2
        total_variance = variance_explained.sum()
        explained_variance_ratio = variance_explained / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # Find effective dimensions for different thresholds
        effective_dimensions = {}
        for threshold in variance_thresholds:
            idx = np.argmax(cumulative_variance_ratio >= threshold)
            if cumulative_variance_ratio[idx] >= threshold:
                effective_dimensions[threshold] = idx + 1
            else:
                effective_dimensions[threshold] = len(s)
                
        # Analyze singular value decay patterns
        singular_value_decay = self._analyze_singular_value_decay(s)
        
        # Estimate intrinsic dimensionality using multiple methods
        intrinsic_dim_estimates = self._estimate_intrinsic_dimensionality(s, explained_variance_ratio)
        
        # Matrix rank and numerical rank
        matrix_rank = np.linalg.matrix_rank(dense_matrix)
        numerical_rank = np.sum(s > s[0] * 1e-12) if len(s) > 0 else 0
        
        results = {
            'singular_values': s,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance_ratio': cumulative_variance_ratio,
            'effective_dimensions': effective_dimensions,
            'total_variance': total_variance,
            'matrix_rank': matrix_rank,
            'numerical_rank': numerical_rank,
            'decay_analysis': singular_value_decay,
            'intrinsic_dimensionality': intrinsic_dim_estimates,
            'matrix_shape': matrix.shape,
            'sample_shape': dense_matrix.shape
        }
        
        return results
    
    def _analyze_singular_value_decay(self, singular_values: np.ndarray) -> Dict:
        """
        Analyze the decay pattern of singular values.
        
        Different decay patterns (exponential, polynomial, etc.) suggest
        different intrinsic dimensionalities and compression possibilities.
        
        Args:
            singular_values: Array of singular values in descending order
            
        Returns:
            Dictionary with decay analysis results
        """
        if len(singular_values) < 2:
            return {'error': 'Insufficient singular values for decay analysis'}
            
        # Log-scale analysis
        log_s = np.log(singular_values + 1e-10)  # Add small epsilon for numerical stability
        indices = np.arange(len(singular_values))
        
        # Fit exponential decay: log(s_i) = a - b*i
        try:
            decay_coeffs = np.polyfit(indices, log_s, 1)
            exponential_decay_rate = -decay_coeffs[0]
        except np.linalg.LinAlgError:
            exponential_decay_rate = np.nan
            
        # Analyze decay rate changes (second derivative)
        if len(singular_values) > 4:
            second_derivative = np.diff(log_s, n=2)
            decay_acceleration = np.mean(second_derivative)
        else:
            decay_acceleration = np.nan
            
        # Find "elbow" in singular value curve
        elbow_index = self._find_elbow_point(singular_values)
        
        return {
            'exponential_decay_rate': exponential_decay_rate,
            'decay_acceleration': decay_acceleration,
            'elbow_index': elbow_index,
            'elbow_value': singular_values[elbow_index] if elbow_index >= 0 else np.nan,
            'condition_number': singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else np.inf
        }
    
    def _find_elbow_point(self, values: np.ndarray) -> int:
        """
        Find the "elbow" point in a decreasing curve using the method of maximum curvature.
        
        Args:
            values: Array of values in descending order
            
        Returns:
            Index of the elbow point
        """
        if len(values) < 3:
            return 0
            
        # Normalize values to [0, 1] for consistent analysis
        normalized = (values - values.min()) / (values.max() - values.min() + 1e-10)
        x = np.arange(len(normalized))
        
        # Compute curvature using finite differences
        dx = np.gradient(x)
        dy = np.gradient(normalized)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2 + 1e-10)**(3/2)
        
        # Find maximum curvature point
        return int(np.argmax(curvature))
    
    def _estimate_intrinsic_dimensionality(self, singular_values: np.ndarray, 
                                         explained_variance_ratio: np.ndarray) -> Dict:
        """
        Estimate intrinsic dimensionality using multiple methods.
        
        Args:
            singular_values: Array of singular values
            explained_variance_ratio: Explained variance ratio for each component
            
        Returns:
            Dictionary with different dimensionality estimates
        """
        estimates = {}
        
        # Method 1: Kaiser criterion (eigenvalues/singular values > mean)
        mean_singular_value = np.mean(singular_values)
        kaiser_dim = np.sum(singular_values > mean_singular_value)
        estimates['kaiser_criterion'] = kaiser_dim
        
        # Method 2: Broken stick model (expected distribution under null hypothesis)
        n_components = len(singular_values)
        broken_stick = np.array([np.sum(1.0 / np.arange(i, n_components)) for i in range(1, n_components + 1)])
        broken_stick = broken_stick / np.sum(broken_stick)
        broken_stick_dim = np.sum(explained_variance_ratio > broken_stick)
        estimates['broken_stick'] = broken_stick_dim
        
        # Method 3: Parallel analysis (compare to random data - simplified version)
        # This would ideally compare to eigenvalues from random matrices of same size
        percentile_95 = np.percentile(explained_variance_ratio, 95)
        parallel_analysis_dim = np.sum(explained_variance_ratio > percentile_95 * 0.1)
        estimates['parallel_analysis_approx'] = parallel_analysis_dim
        
        # Method 4: Cumulative variance threshold (practical approach)
        cum_var_90 = np.argmax(np.cumsum(explained_variance_ratio) >= 0.90) + 1
        cum_var_95 = np.argmax(np.cumsum(explained_variance_ratio) >= 0.95) + 1
        estimates['cumulative_90'] = cum_var_90
        estimates['cumulative_95'] = cum_var_95
        
        return estimates
    
    def plot_dimensionality_analysis(self, analysis_results: Dict, 
                                   figsize: Tuple[int, int] = (15, 10),
                                   save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create comprehensive plots of dimensionality analysis results.
        
        Generates multiple subplots showing singular values, explained variance,
        cumulative variance, and decay patterns to help interpret the results.
        
        Args:
            analysis_results: Results from compute_effective_dimensionality
            figsize: Figure size for the plot
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Co-occurrence Matrix Dimensionality Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        s = analysis_results['singular_values']
        var_ratio = analysis_results['explained_variance_ratio']
        cum_var = analysis_results['cumulative_variance_ratio']
        effective_dims = analysis_results['effective_dimensions']
        
        # Plot 1: Singular values (log scale)
        axes[0, 0].semilogy(s, 'b.-', alpha=0.7)
        axes[0, 0].set_title('Singular Values (Log Scale)')
        axes[0, 0].set_xlabel('Component Index')
        axes[0, 0].set_ylabel('Singular Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add elbow point if available
        if 'decay_analysis' in analysis_results and 'elbow_index' in analysis_results['decay_analysis']:
            elbow_idx = analysis_results['decay_analysis']['elbow_index']
            if 0 <= elbow_idx < len(s):
                axes[0, 0].axvline(x=elbow_idx, color='red', linestyle='--', alpha=0.7, label=f'Elbow Point ({elbow_idx})')
                axes[0, 0].legend()
        
        # Plot 2: Explained variance ratio
        axes[0, 1].plot(var_ratio, 'g.-', alpha=0.7)
        axes[0, 1].set_title('Explained Variance Ratio')
        axes[0, 1].set_xlabel('Component Index')
        axes[0, 1].set_ylabel('Variance Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative explained variance
        axes[1, 0].plot(cum_var, 'r.-', alpha=0.7)
        axes[1, 0].set_title('Cumulative Explained Variance')
        axes[1, 0].set_xlabel('Component Index')
        axes[1, 0].set_ylabel('Cumulative Variance Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add threshold lines and effective dimensions
        colors = ['orange', 'purple', 'brown']
        for i, (threshold, dim) in enumerate(effective_dims.items()):
            color = colors[i % len(colors)]
            axes[1, 0].axhline(y=threshold, color=color, linestyle='--', alpha=0.7, 
                              label=f'{threshold:.0%} threshold')
            axes[1, 0].axvline(x=dim-1, color=color, linestyle=':', alpha=0.7, 
                              label=f'Effective dim: {dim}')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 4: Dimensionality estimates comparison
        if 'intrinsic_dimensionality' in analysis_results:
            intrinsic_dims = analysis_results['intrinsic_dimensionality']
            methods = list(intrinsic_dims.keys())
            values = list(intrinsic_dims.values())
            
            axes[1, 1].bar(range(len(methods)), values, alpha=0.7, color='skyblue')
            axes[1, 1].set_title('Intrinsic Dimensionality Estimates')
            axes[1, 1].set_xlabel('Estimation Method')
            axes[1, 1].set_ylabel('Estimated Dimension')
            axes[1, 1].set_xticks(range(len(methods)))
            axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[1, 1].text(i, v + 0.5, str(int(v)), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved dimensionality analysis plot to {save_path}")
        
        plt.show()
    
    def recommend_embedding_dimensions(self, analysis_results: Dict,
                                     target_efficiency: float = 0.95) -> Dict:
        """
        Recommend embedding dimensions for neural network applications.
        
        Provides recommendations for both encoder and decoder dimensions based on
        the effective dimensionality analysis, considering computational efficiency
        and information retention trade-offs.
        
        Args:
            analysis_results: Results from compute_effective_dimensionality
            target_efficiency: Target variance retention (0.90-0.99 recommended)
            
        Returns:
            Dictionary with dimension recommendations
        """
        effective_dims = analysis_results['effective_dimensions']
        intrinsic_dims = analysis_results.get('intrinsic_dimensionality', {})
        matrix_shape = analysis_results['matrix_shape']
        
        # Primary recommendation: Use effective dimensionality at target efficiency
        if target_efficiency in effective_dims:
            primary_dim = effective_dims[target_efficiency]
        else:
            # Find closest threshold
            available_thresholds = list(effective_dims.keys())
            closest_threshold = min(available_thresholds, key=lambda x: abs(x - target_efficiency))
            primary_dim = effective_dims[closest_threshold]
        
        # Secondary recommendations based on different criteria
        conservative_dim = effective_dims.get(0.99, primary_dim)  # High information retention
        aggressive_dim = effective_dims.get(0.90, primary_dim)    # More compression
        
        # Consider intrinsic dimensionality estimates
        intrinsic_median = np.median(list(intrinsic_dims.values())) if intrinsic_dims else primary_dim
        
        # Neural network specific recommendations
        vocab_size = matrix_shape[0]
        
        # Encoder bottleneck: Should be much smaller than vocabulary
        encoder_bottleneck = max(10, min(primary_dim, vocab_size // 10))
        
        # Hidden dimensions: Often 2-4x the bottleneck
        encoder_hidden = min(primary_dim * 2, vocab_size // 5)
        decoder_hidden = encoder_hidden  # Symmetric architecture
        
        recommendations = {
            'primary_embedding_dim': int(primary_dim),
            'conservative_dim': int(conservative_dim),
            'aggressive_dim': int(aggressive_dim),
            'intrinsic_dim_estimate': int(intrinsic_median),
            'encoder_recommendations': {
                'bottleneck_dim': int(encoder_bottleneck),
                'hidden_dim': int(encoder_hidden),
                'rationale': f'Bottleneck: {encoder_bottleneck}, Hidden: {encoder_hidden} (vocab_size={vocab_size})'
            },
            'decoder_recommendations': {
                'hidden_dim': int(decoder_hidden),
                'output_dim': int(primary_dim),
                'rationale': f'Hidden: {decoder_hidden}, Output: {primary_dim}'
            },
            'compression_ratios': {
                'vocabulary_to_primary': vocab_size / primary_dim,
                'vocabulary_to_bottleneck': vocab_size / encoder_bottleneck,
                'primary_to_bottleneck': primary_dim / encoder_bottleneck
            }
        }
        
        return recommendations
    
    def print_dimensionality_report(self, analysis_results: Dict) -> None:
        """
        Print a comprehensive dimensionality analysis report.
        
        Args:
            analysis_results: Results from compute_effective_dimensionality
        """
        print("\\n=== EFFECTIVE DIMENSIONALITY ANALYSIS ===")
        print(f"Matrix shape: {analysis_results['matrix_shape']}")
        print(f"Analysis sample shape: {analysis_results['sample_shape']}")
        print(f"Matrix rank: {analysis_results['matrix_rank']}")
        print(f"Numerical rank: {analysis_results['numerical_rank']}")
        
        print("\\nEffective dimensions by variance threshold:")
        for threshold, dim in analysis_results['effective_dimensions'].items():
            print(f"  {threshold:.0%} variance retained: {dim} dimensions")
            
        if 'decay_analysis' in analysis_results:
            decay = analysis_results['decay_analysis']
            print(f"\\nSpectral decay analysis:")
            print(f"  Exponential decay rate: {decay.get('exponential_decay_rate', 'N/A'):.4f}")
            print(f"  Condition number: {decay.get('condition_number', 'N/A'):.2e}")
            if decay.get('elbow_index', -1) >= 0:
                print(f"  Elbow point: Component {decay['elbow_index']} (value: {decay['elbow_value']:.2e})")
                
        if 'intrinsic_dimensionality' in analysis_results:
            intrinsic = analysis_results['intrinsic_dimensionality']
            print(f"\\nIntrinsic dimensionality estimates:")
            for method, dim in intrinsic.items():
                print(f"  {method.replace('_', ' ').title()}: {dim} dimensions")
        
        # Generate and print recommendations
        recommendations = self.recommend_embedding_dimensions(analysis_results)
        print(f"\\n=== NEURAL NETWORK DIMENSION RECOMMENDATIONS ===")
        print(f"Primary embedding dimension: {recommendations['primary_embedding_dim']}")
        print(f"Conservative (99% variance): {recommendations['conservative_dim']}")
        print(f"Aggressive (90% variance): {recommendations['aggressive_dim']}")
        
        encoder_rec = recommendations['encoder_recommendations']
        print(f"\\nEncoder architecture:")
        print(f"  Bottleneck dimension: {encoder_rec['bottleneck_dim']}")
        print(f"  Hidden dimension: {encoder_rec['hidden_dim']}")
        
        decoder_rec = recommendations['decoder_recommendations'] 
        print(f"\\nDecoder architecture:")
        print(f"  Hidden dimension: {decoder_rec['hidden_dim']}")
        print(f"  Output dimension: {decoder_rec['output_dim']}")
        
        compression = recommendations['compression_ratios']
        print(f"\\nCompression ratios:")
        print(f"  Vocabulary → Primary: {compression['vocabulary_to_primary']:.1f}x")
        print(f"  Vocabulary → Bottleneck: {compression['vocabulary_to_bottleneck']:.1f}x")
        print(f"  Primary → Bottleneck: {compression['primary_to_bottleneck']:.1f}x")