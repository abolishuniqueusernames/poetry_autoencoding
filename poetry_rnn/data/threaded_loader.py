"""
Multi-threaded Data Loading Utilities

This module provides optimized data loading with multi-threading,
memory pinning, and intelligent batching strategies.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Optional, List, Dict, Any, Iterator
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class BucketingSampler(Sampler):
    """
    Sampler that groups sequences of similar lengths together.
    
    This reduces padding overhead and improves training efficiency by
    creating batches with minimal wasted computation on padding tokens.
    
    Expected improvement: 20-30% throughput gain from reduced padding.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_boundaries: Optional[List[int]] = None
    ):
        """
        Initialize bucketing sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Batch size
            shuffle: Whether to shuffle within buckets
            drop_last: Whether to drop incomplete batches
            bucket_boundaries: Sequence length boundaries for buckets
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Get sequence lengths
        self.lengths = self._get_sequence_lengths()
        
        # Create buckets
        if bucket_boundaries is None:
            # Auto-create boundaries based on data distribution
            bucket_boundaries = self._compute_bucket_boundaries()
        self.bucket_boundaries = bucket_boundaries
        
        # Assign sequences to buckets
        self.buckets = self._create_buckets()
        
        logger.info(f"BucketingSampler created with {len(self.buckets)} buckets")
    
    def _get_sequence_lengths(self) -> List[int]:
        """Extract sequence lengths from dataset."""
        lengths = []
        
        # Try to get lengths efficiently
        if hasattr(self.dataset, 'get_sequence_lengths'):
            lengths = self.dataset.get_sequence_lengths()
        elif hasattr(self.dataset, 'attention_masks'):
            # Count non-padding tokens
            masks = self.dataset.attention_masks
            lengths = masks.sum(axis=1).tolist()
        else:
            # Fallback: iterate through dataset
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                if 'attention_mask' in item:
                    lengths.append(item['attention_mask'].sum().item())
                elif 'input_sequences' in item:
                    lengths.append(len(item['input_sequences']))
                else:
                    lengths.append(100)  # Default length
        
        return lengths
    
    def _compute_bucket_boundaries(self) -> List[int]:
        """Automatically compute bucket boundaries based on length distribution."""
        # Use percentiles to create balanced buckets
        percentiles = [20, 40, 60, 80, 100]
        boundaries = [np.percentile(self.lengths, p) for p in percentiles]
        
        # Ensure integer boundaries
        boundaries = [int(b) for b in boundaries]
        
        # Add minimum and maximum
        boundaries = [0] + boundaries
        
        return boundaries
    
    def _create_buckets(self) -> Dict[int, List[int]]:
        """Assign sequences to buckets based on length."""
        buckets = defaultdict(list)
        
        for idx, length in enumerate(self.lengths):
            # Find appropriate bucket
            bucket_id = 0
            for i, boundary in enumerate(self.bucket_boundaries[1:]):
                if length <= boundary:
                    bucket_id = i
                    break
            
            buckets[bucket_id].append(idx)
        
        # Log bucket distribution
        for bucket_id, indices in buckets.items():
            if bucket_id < len(self.bucket_boundaries) - 1:
                range_str = f"{self.bucket_boundaries[bucket_id]}-{self.bucket_boundaries[bucket_id+1]}"
            else:
                range_str = f"{self.bucket_boundaries[bucket_id]}+"
            logger.debug(f"Bucket {bucket_id} ({range_str}): {len(indices)} sequences")
        
        return dict(buckets)
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with bucketing."""
        # Collect all batches
        batches = []
        
        for bucket_id, indices in self.buckets.items():
            # Shuffle within bucket if needed
            if self.shuffle:
                indices = np.random.permutation(indices).tolist()
            
            # Create batches from bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                
                # Handle incomplete batches
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                
                batches.append(batch)
        
        # Shuffle batches if needed
        if self.shuffle:
            np.random.shuffle(batches)
        
        # Yield batches
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        """Get number of batches."""
        total_samples = sum(len(indices) for indices in self.buckets.values())
        
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


class DynamicBatchSampler(Sampler):
    """
    Dynamic batch sampler that adjusts batch size based on sequence length.
    
    Maintains approximately constant memory usage by using smaller batches
    for longer sequences and larger batches for shorter sequences.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        max_tokens: int = 4096,
        max_batch_size: int = 64,
        shuffle: bool = True
    ):
        """
        Initialize dynamic batch sampler.
        
        Args:
            dataset: Dataset to sample from
            max_tokens: Maximum tokens per batch
            max_batch_size: Maximum batch size
            shuffle: Whether to shuffle data
        """
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        
        # Get sequence lengths
        self.lengths = self._get_sequence_lengths()
        
        # Create batches
        self.batches = self._create_dynamic_batches()
        
        logger.info(f"DynamicBatchSampler created {len(self.batches)} batches")
    
    def _get_sequence_lengths(self) -> List[int]:
        """Extract sequence lengths from dataset."""
        # Similar to BucketingSampler
        lengths = []
        
        if hasattr(self.dataset, 'get_sequence_lengths'):
            lengths = self.dataset.get_sequence_lengths()
        else:
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                if 'attention_mask' in item:
                    lengths.append(item['attention_mask'].sum().item())
                else:
                    lengths.append(100)
        
        return lengths
    
    def _create_dynamic_batches(self) -> List[List[int]]:
        """Create batches with dynamic sizing based on sequence length."""
        # Sort indices by sequence length
        sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx in sorted_indices:
            seq_len = self.lengths[idx]
            
            # Check if adding this sequence exceeds limits
            batch_tokens = (len(current_batch) + 1) * max(seq_len, 
                                                          max(self.lengths[i] for i in current_batch) if current_batch else 0)
            
            if current_batch and (
                batch_tokens > self.max_tokens or 
                len(current_batch) >= self.max_batch_size
            ):
                # Start new batch
                batches.append(current_batch)
                current_batch = [idx]
                current_tokens = seq_len
            else:
                # Add to current batch
                current_batch.append(idx)
                current_tokens = batch_tokens
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate dynamic batches."""
        batches = self.batches.copy()
        
        if self.shuffle:
            np.random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        """Get number of batches."""
        return len(self.batches)


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    drop_last: bool = False,
    use_bucketing: bool = True,
    use_dynamic_batching: bool = False,
    max_tokens: Optional[int] = None,
    **kwargs
) -> DataLoader:
    """
    Create an optimized DataLoader with performance enhancements.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        drop_last: Drop incomplete last batch
        use_bucketing: Use bucketing sampler for similar-length sequences
        use_dynamic_batching: Use dynamic batch sizing
        max_tokens: Maximum tokens per batch (for dynamic batching)
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Optimized DataLoader instance
    """
    # Select appropriate sampler
    sampler = None
    batch_sampler = None
    
    if use_dynamic_batching and max_tokens:
        # Dynamic batching based on tokens
        batch_sampler = DynamicBatchSampler(
            dataset,
            max_tokens=max_tokens,
            max_batch_size=batch_size,
            shuffle=shuffle
        )
        # Disable batch_size and shuffle when using batch_sampler
        batch_size = 1
        shuffle = False
        
    elif use_bucketing:
        # Bucketing sampler for similar lengths
        sampler = BucketingSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
        # Disable shuffle when using custom sampler
        shuffle = False
    
    # Create optimized DataLoader
    dataloader_kwargs = {
        'batch_size': batch_size if batch_sampler is None else 1,
        'shuffle': shuffle if sampler is None and batch_sampler is None else False,
        'num_workers': num_workers,
        'pin_memory': pin_memory and torch.cuda.is_available(),
        'drop_last': drop_last if sampler is None and batch_sampler is None else False,
        'persistent_workers': persistent_workers and num_workers > 0,
    }
    
    # Add prefetch_factor if supported
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    
    # Add sampler or batch_sampler
    if batch_sampler is not None:
        dataloader_kwargs['batch_sampler'] = batch_sampler
    elif sampler is not None:
        dataloader_kwargs['sampler'] = sampler
    
    # Add any additional kwargs
    dataloader_kwargs.update(kwargs)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    
    # Log configuration
    logger.info(
        f"Created optimized DataLoader: "
        f"batch_size={batch_size}, "
        f"num_workers={num_workers}, "
        f"pin_memory={dataloader_kwargs['pin_memory']}, "
        f"bucketing={use_bucketing}, "
        f"dynamic_batching={use_dynamic_batching}"
    )
    
    return dataloader


class CollateFunction:
    """
    Custom collate function for efficient batch creation.
    
    Handles padding, attention masks, and tensor creation efficiently.
    """
    
    def __init__(self, padding_value: int = 0, padding_side: str = 'right'):
        """
        Initialize collate function.
        
        Args:
            padding_value: Value to use for padding
            padding_side: Side to pad ('left' or 'right')
        """
        self.padding_value = padding_value
        self.padding_side = padding_side
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Collated batch dictionary
        """
        # Find maximum sequence length in batch
        max_length = max(
            item['input_sequences'].size(0) if torch.is_tensor(item['input_sequences'])
            else len(item['input_sequences'])
            for item in batch
        )
        
        # Prepare output tensors
        batch_size = len(batch)
        collated = {}
        
        # Process each key
        for key in batch[0].keys():
            if key == 'input_sequences':
                # Handle sequence padding
                sequences = []
                for item in batch:
                    seq = item[key]
                    if not torch.is_tensor(seq):
                        seq = torch.tensor(seq)
                    
                    # Pad sequence
                    if seq.size(0) < max_length:
                        padding = torch.full(
                            (max_length - seq.size(0),) + seq.shape[1:],
                            self.padding_value,
                            dtype=seq.dtype
                        )
                        
                        if self.padding_side == 'right':
                            seq = torch.cat([seq, padding], dim=0)
                        else:
                            seq = torch.cat([padding, seq], dim=0)
                    
                    sequences.append(seq)
                
                collated[key] = torch.stack(sequences)
                
            elif key == 'attention_mask':
                # Handle attention masks
                masks = []
                for item in batch:
                    mask = item.get(key)
                    if mask is None:
                        # Create mask if not present
                        seq_len = len(item['input_sequences'])
                        mask = torch.ones(seq_len, dtype=torch.bool)
                    elif not torch.is_tensor(mask):
                        mask = torch.tensor(mask, dtype=torch.bool)
                    
                    # Pad mask
                    if mask.size(0) < max_length:
                        padding = torch.zeros(max_length - mask.size(0), dtype=torch.bool)
                        
                        if self.padding_side == 'right':
                            mask = torch.cat([mask, padding], dim=0)
                        else:
                            mask = torch.cat([padding, mask], dim=0)
                    
                    masks.append(mask)
                
                collated[key] = torch.stack(masks)
                
            else:
                # Handle other fields
                values = [item[key] for item in batch]
                
                if torch.is_tensor(values[0]):
                    collated[key] = torch.stack(values)
                else:
                    collated[key] = values
        
        return collated