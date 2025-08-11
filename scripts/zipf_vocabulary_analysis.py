from itertools import accumulate
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict

def analyze_zipf_fit_by_vocab_size(tokens: List[str], max_vocab_size: int = 5000) -> Dict:
    """
    Analyze how Zipf's law fit improves with vocabulary size.
    Returns optimal vocabulary size based on chi-squared goodness of fit.
    
    Args:
        tokens: List of all tokens from corpus
        max_vocab_size: Maximum vocabulary size to analyze
        
    Returns:
        Dictionary with vocab_sizes, chi_squared_values, p_values, slope, r_squared
    """
    # Get frequency distribution
    frequencies = Counter(tokens)
    sorted_items = frequencies.most_common(max_vocab_size)
    
    # Extract frequencies and compute ranks
    freq_values = np.array([count for word, count in sorted_items])
    ranks = np.arange(1, len(freq_values) + 1)
    
    # Log transform
    log_ranks = np.log(ranks)
    log_frequencies = np.log(freq_values)
    
    # Fit Zipf's law (frequency ∝ 1/rank^α)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_ranks, log_frequencies)
    
    # Compute residuals
    predicted = slope * log_ranks + intercept
    residuals = log_frequencies - predicted
    
    # Cumulative chi-squared values
    cumulative_chi_squared = list(accumulate(residuals**2, lambda acc, x: acc + x))
    
    # Degrees of freedom for each vocab size
    dfs = list(range(1, len(cumulative_chi_squared) + 1))
    
    # P-values for each vocab size (corrected chi-squared test)
    # The chi-squared statistic should be normalized by the error variance
    p_values = []
    for chi_sq, df in zip(cumulative_chi_squared, dfs):
        # Normalize by degrees of freedom and error variance
        if df > 1:  # Need at least 2 points for meaningful test
            normalized_chi_sq = chi_sq / (std_err**2)
            p_val = scipy.stats.chi2.sf(normalized_chi_sq, df-2)  # -2 for slope and intercept
            p_values.append(p_val)
        else:
            p_values.append(0.0)  # No meaningful test for df <= 1
    
    return {
        'vocab_sizes': list(range(1, len(cumulative_chi_squared) + 1)),
        'chi_squared_values': cumulative_chi_squared,
        'p_values': p_values,
        'slope': slope,
        'r_squared': r_value**2,
        'words': [word for word, count in sorted_items]
    }

def find_optimal_vocab_size(results: Dict, significance_level: float = 0.05, min_size: int = 50) -> Dict:
    """
    Find optimal vocabulary sizes based on statistical significance threshold.
    Since goodness of fit isn't monotone, find all good regions.
    
    Args:
        results: Results dictionary from analyze_zipf_fit_by_vocab_size
        significance_level: P-value threshold for goodness of fit
        min_size: Minimum vocabulary size to consider
        
    Returns:
        Dictionary with analysis of good regions
    """
    vocab_sizes = np.array(results['vocab_sizes'])
    p_values = np.array(results['p_values'])
    
    # Find all sizes that meet threshold
    good_sizes = vocab_sizes[p_values > significance_level]
    good_sizes = good_sizes[good_sizes >= min_size]  # Filter minimum size
    
    print(f"Significance level: {significance_level}")
    print(f"Number of vocab sizes meeting threshold: {len(good_sizes)}")
    
    if len(good_sizes) == 0:
        # No sizes meet threshold - find best ones
        best_idx = np.argmax(p_values)
        print(f"No size meets threshold. Best p-value: {p_values[best_idx]:.6f} at size {vocab_sizes[best_idx]}")
        return {
            'optimal_sizes': [vocab_sizes[best_idx]],
            'recommended_size': vocab_sizes[best_idx],
            'p_values_at_optimal': [p_values[best_idx]],
            'analysis': 'single_best'
        }
    
    # Find regions of good fit (consecutive ranges)
    good_p_values = p_values[vocab_sizes >= min_size][p_values[vocab_sizes >= min_size] > significance_level]
    
    # Get some representative sizes from different parts of the good region
    if len(good_sizes) > 1:
        # Small, medium, large from good region
        small_good = good_sizes[0]
        large_good = good_sizes[-1]
        medium_good = good_sizes[len(good_sizes)//2] if len(good_sizes) > 2 else None
        
        candidates = [small_good, medium_good, large_good]
        candidates = [c for c in candidates if c is not None]
        
        # Get their p-values
        candidate_p_values = []
        for size in candidates:
            idx = vocab_sizes.tolist().index(size)
            candidate_p_values.append(p_values[idx])
        
        print(f"Good vocabulary size range: {small_good} - {large_good}")
        print(f"Candidate sizes: {candidates}")
        print(f"Their p-values: {[f'{p:.4f}' for p in candidate_p_values]}")
        
        # Recommend the medium-large size for better coverage
        recommended = candidates[-2] if len(candidates) > 2 else candidates[-1]
        
        return {
            'optimal_sizes': good_sizes.tolist(),
            'recommended_size': recommended,
            'candidates': candidates,
            'p_values_at_candidates': candidate_p_values,
            'analysis': 'multiple_good_regions'
        }
    else:
        return {
            'optimal_sizes': good_sizes.tolist(),
            'recommended_size': good_sizes[0],
            'p_values_at_optimal': [p_values[vocab_sizes.tolist().index(good_sizes[0])]],
            'analysis': 'single_good_size'
        }

def plot_zipf_evolution(results: Dict, optimal_size: int = None, save_path: str = None):
    """
    Plot the evolution of Zipf's law fit with vocabulary size.
    
    Args:
        results: Results dictionary from analyze_zipf_fit_by_vocab_size
        optimal_size: Optimal vocabulary size to highlight
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 5))

    # Plot 1: Chi-squared evolution
    plt.subplot(1, 3, 1)
    plt.plot(results['vocab_sizes'], results['chi_squared_values'], 'b-', linewidth=2)
    if optimal_size:
        plt.axvline(x=optimal_size, color='r', linestyle='--', alpha=0.7, label=f'Optimal size: {optimal_size}')
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Cumulative Chi-squared')
    plt.title('Chi-squared Evolution')
    plt.grid(True, alpha=0.3)
    if optimal_size:
        plt.legend()

    # Plot 2: P-value evolution
    plt.subplot(1, 3, 2)
    plt.plot(results['vocab_sizes'], results['p_values'], 'g-', linewidth=2)
    plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='p=0.05 threshold')
    if optimal_size:
        plt.axvline(x=optimal_size, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Vocabulary Size')
    plt.ylabel('P-value')
    plt.title('Goodness of Fit Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Log-log plot for final vocabulary size
    plt.subplot(1, 3, 3)
    final_size = optimal_size if optimal_size else len(results['vocab_sizes'])
    ranks = np.arange(1, final_size + 1)
    log_ranks = np.log(ranks)
    
    # Recompute for final size
    frequencies = Counter()  # This would need the original tokens - simplified for now
    plt.xlabel('Log Rank')
    plt.ylabel('Log Frequency')
    plt.title(f'Zipf\'s Law Fit (vocab_size={final_size})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage function
def run_zipf_analysis(tokens: List[str], max_vocab_size: int = 2000):
    """
    Complete Zipf analysis pipeline.
    
    Args:
        tokens: List of all tokens from corpus
        max_vocab_size: Maximum vocabulary size to analyze
    """
    print("Analyzing Zipf's law fit across vocabulary sizes...")
    
    # Run analysis
    results = analyze_zipf_fit_by_vocab_size(tokens, max_vocab_size)
    
    # Find optimal sizes (multiple regions)
    optimal_analysis = find_optimal_vocab_size(results, significance_level=0.05)
    optimal_size = optimal_analysis['recommended_size']
    
    # Print results
    print(f"\nRecommended vocabulary size: {optimal_size}")
    print(f"Zipf's law slope (α): {-results['slope']:.3f}")
    print(f"R-squared: {results['r_squared']:.3f}")
    
    if optimal_analysis['analysis'] == 'multiple_good_regions':
        print(f"Other good candidates: {optimal_analysis['candidates']}")
        print("Consider the trade-off: smaller vocab = faster training, larger vocab = better coverage")
    
    # Plot results  
    plot_zipf_evolution(results, optimal_size, save_path='zipf_analysis.png')
    
    return results, optimal_analysis

if __name__ == "__main__":
    # This would be called from your notebook with your tokens
    print("Zipf analysis functions loaded. Use run_zipf_analysis(all_tokens) in your notebook.")