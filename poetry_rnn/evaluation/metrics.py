"""
Evaluation metrics for RNN Autoencoder

Provides quantitative metrics for assessing reconstruction quality,
compression efficiency, and semantic preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class ReconstructionMetrics:
    """
    Comprehensive metrics for evaluating reconstruction quality.
    
    Measures how well the autoencoder preserves information through
    the bottleneck, including both exact reconstruction and semantic
    similarity metrics.
    """
    
    def __init__(self, embedding_dim: int = 300):
        self.embedding_dim = embedding_dim
        self.metrics_history = defaultdict(list)
    
    def compute_mse(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute masked mean squared error.
        
        Args:
            predictions: Predicted sequences [batch, seq_len, embed_dim]
            targets: Target sequences [batch, seq_len, embed_dim]
            mask: Boolean mask for valid positions [batch, seq_len]
        
        Returns:
            Mean squared error value
        """
        mse = (predictions - targets) ** 2
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            mse = mse * mask_expanded
            valid_elements = mask_expanded.sum() * self.embedding_dim
            return (mse.sum() / (valid_elements + 1e-8)).item()
        else:
            return mse.mean().item()
    
    def compute_cosine_similarity(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute cosine similarity metrics.
        
        Args:
            predictions: Predicted sequences [batch, seq_len, embed_dim]
            targets: Target sequences [batch, seq_len, embed_dim]
            mask: Boolean mask for valid positions [batch, seq_len]
        
        Returns:
            Dictionary with cosine similarity statistics
        """
        # Token-level cosine similarity
        cos_sim = F.cosine_similarity(predictions, targets, dim=-1)
        
        if mask is not None:
            mask_float = mask.float()
            cos_sim = cos_sim * mask_float
            valid_tokens = mask_float.sum()
            
            mean_sim = (cos_sim.sum() / (valid_tokens + 1e-8)).item()
            
            # Only compute std for valid positions
            valid_sims = cos_sim[mask].cpu().numpy()
            std_sim = np.std(valid_sims) if len(valid_sims) > 0 else 0.0
        else:
            mean_sim = cos_sim.mean().item()
            std_sim = cos_sim.std().item()
        
        return {
            'mean': mean_sim,
            'std': std_sim,
            'min': cos_sim[mask].min().item() if mask is not None else cos_sim.min().item(),
            'max': cos_sim[mask].max().item() if mask is not None else cos_sim.max().item()
        }
    
    def compute_sequence_similarity(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute sequence-level similarity using mean pooling.
        
        Args:
            predictions: Predicted sequences [batch, seq_len, embed_dim]
            targets: Target sequences [batch, seq_len, embed_dim]
            mask: Boolean mask for valid positions [batch, seq_len]
        
        Returns:
            Average sequence-level cosine similarity
        """
        if mask is not None:
            # Mean pool over valid positions only
            mask_expanded = mask.unsqueeze(-1).float()
            pred_pooled = (predictions * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            target_pooled = (targets * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Mean pool over all positions
            pred_pooled = predictions.mean(dim=1)
            target_pooled = targets.mean(dim=1)
        
        # Compute cosine similarity between pooled representations
        seq_sim = F.cosine_similarity(pred_pooled, target_pooled, dim=-1)
        return seq_sim.mean().item()
    
    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Compute all reconstruction metrics.
        
        Args:
            predictions: Predicted sequences
            targets: Target sequences
            mask: Boolean mask for valid positions
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'mse': self.compute_mse(predictions, targets, mask),
            'cosine_similarity': self.compute_cosine_similarity(predictions, targets, mask),
            'sequence_similarity': self.compute_sequence_similarity(predictions, targets, mask)
        }
        
        # Add RMSE for interpretability
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Store in history
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    self.metrics_history[f"{key}_{subkey}"].append(subvalue)
            else:
                self.metrics_history[key].append(value)
        
        return metrics


class BottleneckAnalyzer:
    """
    Analyzes the learned bottleneck representations.
    
    Provides insights into what the model has learned to encode
    in the compressed representation.
    """
    
    def __init__(self, bottleneck_dim: int = 18):
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_history = []
    
    def analyze_bottleneck(
        self,
        bottleneck: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Analyze bottleneck representations.
        
        Args:
            bottleneck: Bottleneck tensor [batch_size, bottleneck_dim]
        
        Returns:
            Dictionary with bottleneck statistics
        """
        with torch.no_grad():
            # Basic statistics
            mean = bottleneck.mean(dim=0)
            std = bottleneck.std(dim=0)
            
            # Activation patterns
            sparsity = (torch.abs(bottleneck) < 0.1).float().mean().item()
            saturation = (torch.abs(bottleneck) > 0.9).float().mean().item()
            
            # Dimension utilization
            dim_variance = bottleneck.var(dim=0)
            effective_dims = (dim_variance > 0.01).sum().item()
            
            # Correlation between dimensions
            if bottleneck.shape[0] > 1:
                corr_matrix = torch.corrcoef(bottleneck.T)
                off_diagonal = corr_matrix[~torch.eye(self.bottleneck_dim, dtype=bool)]
                avg_correlation = torch.abs(off_diagonal).mean().item()
            else:
                avg_correlation = 0.0
            
            stats = {
                'mean': mean.mean().item(),
                'std': std.mean().item(),
                'sparsity': sparsity,
                'saturation': saturation,
                'effective_dimensions': effective_dims,
                'dimension_utilization': effective_dims / self.bottleneck_dim,
                'avg_correlation': avg_correlation,
                'max_activation': bottleneck.max().item(),
                'min_activation': bottleneck.min().item()
            }
            
            # Store for analysis
            self.bottleneck_history.append(bottleneck.cpu().numpy())
            
            return stats
    
    def compute_bottleneck_diversity(
        self,
        bottlenecks: List[torch.Tensor]
    ) -> float:
        """
        Compute diversity of bottleneck representations.
        
        Higher diversity indicates the model is learning distinct
        representations for different inputs.
        
        Args:
            bottlenecks: List of bottleneck tensors
        
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(bottlenecks) < 2:
            return 0.0
        
        # Stack all bottlenecks
        all_bottlenecks = torch.cat(bottlenecks, dim=0)
        
        # Compute pairwise distances
        distances = torch.cdist(all_bottlenecks, all_bottlenecks)
        
        # Get upper triangle (excluding diagonal)
        n = distances.shape[0]
        triu_indices = torch.triu_indices(n, n, offset=1)
        pairwise_distances = distances[triu_indices[0], triu_indices[1]]
        
        # Normalize by maximum possible distance
        max_distance = np.sqrt(2 * self.bottleneck_dim)  # Max L2 distance
        normalized_distances = pairwise_distances / max_distance
        
        return normalized_distances.mean().item()


def compute_reconstruction_quality(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = torch.device('cpu'),
    num_batches: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute reconstruction quality metrics over a dataset.
    
    Args:
        model: Trained autoencoder model
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        num_batches: Limit number of batches to evaluate
    
    Returns:
        Dictionary with aggregated metrics
    """
    model.eval()
    metrics_calculator = ReconstructionMetrics()
    all_metrics = defaultdict(list)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches:
                break
            
            # Move to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            output_dict = model(batch)
            
            # Compute metrics
            metrics = metrics_calculator.compute_all_metrics(
                output_dict['reconstructed'],
                batch['input_sequences'],
                batch.get('attention_mask')
            )
            
            # Aggregate
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        all_metrics[f"{key}_{subkey}"].append(subvalue)
                else:
                    all_metrics[key].append(value)
    
    # Average over batches
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    return avg_metrics


def compute_bottleneck_statistics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = torch.device('cpu'),
    num_batches: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute statistics about bottleneck representations.
    
    Args:
        model: Trained autoencoder model
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        num_batches: Limit number of batches to evaluate
    
    Returns:
        Dictionary with bottleneck statistics
    """
    model.eval()
    analyzer = BottleneckAnalyzer(model.bottleneck_dim)
    all_bottlenecks = []
    all_stats = defaultdict(list)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches:
                break
            
            # Move to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # Get bottleneck representations
            bottleneck = model.encode(batch)
            all_bottlenecks.append(bottleneck)
            
            # Analyze
            stats = analyzer.analyze_bottleneck(bottleneck)
            
            # Aggregate
            for key, value in stats.items():
                all_stats[key].append(value)
    
    # Average statistics
    avg_stats = {key: np.mean(values) for key, values in all_stats.items()}
    
    # Add diversity metric
    avg_stats['diversity'] = analyzer.compute_bottleneck_diversity(all_bottlenecks)
    
    return avg_stats