"""
PyTorch Dataset classes for Poetry RNN Autoencoder training

This module provides PyTorch-compatible dataset classes for training autoencoders
on poetry data. Supports efficient loading, chunking strategies, train/validation
splits, and various sampling strategies for robust training.

Features:
- PyTorch Dataset interface with __getitem__ and __len__
- Lazy loading for memory efficiency
- Train/validation/test splits with stratification
- Chunk-aware sampling strategies
- Batch sampling with poem boundary awareness
- Memory-efficient data loading and caching
- Support for different autoencoder training objectives
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Iterator
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataset import random_split
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class AutoencoderDataset(Dataset):
    """
    PyTorch Dataset for poetry autoencoder training.
    
    This dataset handles preprocessed poetry sequences with support for:
    - Lazy loading of embedding sequences
    - Chunk metadata tracking for poem relationships
    - Train/validation splits with proper chunk isolation
    - Memory-efficient data access
    - Flexible sampling strategies
    
    The dataset is designed for autoencoder training where input and target
    sequences are identical (self-supervised reconstruction task).
    
    Attributes:
        sequences: Token index sequences [num_chunks, max_length]
        embedding_sequences: Embedding sequences [num_chunks, max_length, embedding_dim]
        attention_masks: Attention masks [num_chunks, max_length]
        metadata: List of chunk metadata dictionaries
        vocabulary: Word to index mapping
        chunk_to_poem: Mapping from chunk indices to poem indices
        poem_to_chunks: Mapping from poem indices to chunk indices
    """
    
    def __init__(self,
                 sequences: Optional[np.ndarray] = None,
                 embedding_sequences: Optional[np.ndarray] = None,
                 attention_masks: Optional[np.ndarray] = None,
                 metadata: Optional[List[Dict]] = None,
                 vocabulary: Optional[Dict[str, int]] = None,
                 artifacts_path: Optional[Union[str, Path]] = None,
                 timestamp: str = "latest",
                 split: str = "full",
                 split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                 seed: int = 42,
                 lazy_loading: bool = True,
                 device: Optional[torch.device] = None,
                 include_token_sequences: bool = False):
        """
        Initialize autoencoder dataset.
        
        Args:
            sequences: Token sequences array (if not loading from artifacts)
            embedding_sequences: Embedding sequences array
            attention_masks: Attention masks array
            metadata: Chunk metadata list
            vocabulary: Word to index mapping
            artifacts_path: Path to load artifacts from (if not providing arrays)
            timestamp: Timestamp of artifacts to load
            split: Dataset split ("full", "train", "val", "test")
            split_ratios: Train/validation/test split ratios (must sum to 1.0)
            seed: Random seed for splitting
            lazy_loading: Whether to lazy load embedding sequences
            device: PyTorch device for tensors
            
        Raises:
            ValueError: If both arrays and artifacts_path are None
            ValueError: If split ratios don't sum to 1.0
        """
        self.split = split
        self.split_ratios = split_ratios
        self.seed = seed
        self.lazy_loading = lazy_loading
        self.device = device or torch.device('cpu')
        self.include_token_sequences = include_token_sequences
        
        # Validate split ratios
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
        
        # Load data
        if artifacts_path is not None:
            self._load_from_artifacts(artifacts_path, timestamp)
        elif sequences is not None:
            self._load_from_arrays(sequences, embedding_sequences, attention_masks, metadata, vocabulary)
        else:
            raise ValueError("Must provide either arrays or artifacts_path")
        
        # Create poem-chunk mappings
        self._create_poem_chunk_mappings()
        
        # Apply data split
        if split != "full":
            self._apply_split()
        
        logger.info(f"Initialized AutoencoderDataset: {len(self)} samples, split='{split}'")
    
    def _load_from_artifacts(self, artifacts_path: Union[str, Path], timestamp: str) -> None:
        """Load data from saved preprocessing artifacts."""
        from .utils.io import ArtifactManager
        
        artifacts_path = Path(artifacts_path)
        artifact_manager = ArtifactManager(artifacts_path)
        
        try:
            data = artifact_manager.load_preprocessing_artifacts(timestamp)
            
            self.sequences = data['token_sequences']
            self.attention_masks = data['attention_masks']
            self.vocabulary = data['vocabulary']
            
            # Handle metadata
            if 'metadata' in data and 'chunk_metadata' in data['metadata']:
                self.metadata = data['metadata']['chunk_metadata']
            else:
                self.metadata = data.get('metadata', [])
            
            # Handle embedding sequences
            if 'embedding_sequences' in data:
                if self.lazy_loading:
                    # Store path for lazy loading
                    self._embedding_sequences_path = artifacts_path / f"embedding_sequences_{timestamp}.npy"
                    self.embedding_sequences = None
                else:
                    self.embedding_sequences = data['embedding_sequences']
            else:
                raise ValueError("No embedding sequences found in artifacts")
            
            logger.info(f"Loaded dataset from artifacts: {self.sequences.shape[0]} samples")
            
        except Exception as e:
            raise ValueError(f"Failed to load artifacts: {e}")
    
    def _load_from_arrays(self,
                         sequences: np.ndarray,
                         embedding_sequences: Optional[np.ndarray],
                         attention_masks: Optional[np.ndarray],
                         metadata: Optional[List[Dict]],
                         vocabulary: Optional[Dict[str, int]]) -> None:
        """Load data from provided arrays."""
        self.sequences = sequences
        self.embedding_sequences = embedding_sequences
        self.attention_masks = attention_masks
        self.metadata = metadata or []
        self.vocabulary = vocabulary or {}
        
        # Create dummy metadata if missing
        if len(self.metadata) < len(sequences):
            logger.warning("Incomplete metadata, creating dummy entries")
            for i in range(len(self.metadata), len(sequences)):
                self.metadata.append({
                    'poem_idx': i // 2,  # Rough estimate
                    'chunk_id': i % 2,
                    'chunk_length': np.sum(sequences[i] != 0),
                    'poem_title': f'Poem {i // 2}',
                    'total_chunks_in_poem': 2
                })
        
        logger.info(f"Loaded dataset from arrays: {sequences.shape[0]} samples")
    
    def _create_poem_chunk_mappings(self) -> None:
        """Create mappings between poems and chunks."""
        self.chunk_to_poem = {}
        self.poem_to_chunks = defaultdict(list)
        
        for chunk_idx, chunk_meta in enumerate(self.metadata):
            poem_idx = chunk_meta.get('poem_idx', chunk_idx // 2)
            self.chunk_to_poem[chunk_idx] = poem_idx
            self.poem_to_chunks[poem_idx].append(chunk_idx)
        
        self.num_poems = len(self.poem_to_chunks)
        logger.debug(f"Created mappings: {len(self.chunk_to_poem)} chunks from {self.num_poems} poems")
    
    def _apply_split(self) -> None:
        """Apply train/validation/test split at poem level."""
        # Split at poem level to prevent data leakage between chunks
        poem_indices = list(self.poem_to_chunks.keys())
        random.seed(self.seed)
        random.shuffle(poem_indices)
        
        # Calculate split points
        train_ratio, val_ratio, test_ratio = self.split_ratios
        n_poems = len(poem_indices)
        
        train_end = int(n_poems * train_ratio)
        val_end = int(n_poems * (train_ratio + val_ratio))
        
        # Split poems
        if self.split == "train":
            selected_poems = poem_indices[:train_end]
        elif self.split == "val":
            selected_poems = poem_indices[train_end:val_end]
        elif self.split == "test":
            selected_poems = poem_indices[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Get chunk indices for selected poems
        selected_chunks = []
        for poem_idx in selected_poems:
            selected_chunks.extend(self.poem_to_chunks[poem_idx])
        
        selected_chunks = sorted(selected_chunks)
        
        # Filter data to selected chunks
        self.sequences = self.sequences[selected_chunks]
        self.attention_masks = self.attention_masks[selected_chunks]
        self.metadata = [self.metadata[i] for i in selected_chunks]
        
        if self.embedding_sequences is not None:
            self.embedding_sequences = self.embedding_sequences[selected_chunks]
        
        # Update mappings
        self._create_poem_chunk_mappings()
        
        logger.info(f"Applied {self.split} split: {len(selected_chunks)} chunks from {len(selected_poems)} poems")
    
    def _load_embedding_sequence(self, idx: int) -> np.ndarray:
        """Lazy load embedding sequence for given index."""
        if self.embedding_sequences is not None:
            return self.embedding_sequences[idx]
        
        # Load from disk (lazy loading)
        if hasattr(self, '_embedding_sequences_path'):
            # Load single sequence from mmap
            if not hasattr(self, '_embedding_mmap'):
                self._embedding_mmap = np.load(self._embedding_sequences_path, mmap_mode='r')
            return self._embedding_mmap[idx].copy()
        
        raise ValueError("No embedding sequences available")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample data:
            {
                'input_sequences': torch.Tensor,      # Input embedding sequences
                'target_sequences': torch.Tensor,     # Target sequences (same as input for autoencoder)
                'attention_mask': torch.Tensor,       # Attention mask
                'token_sequences': torch.Tensor,      # Token index sequences
                'metadata': Dict                      # Chunk metadata
            }
        """
        # Get embedding sequence
        embedding_seq = self._load_embedding_sequence(idx)
        
        # Convert to tensors
        input_sequences = torch.from_numpy(embedding_seq).float()
        target_sequences = input_sequences.clone()  # Autoencoder target = input
        attention_mask = torch.from_numpy(self.attention_masks[idx]).long()
        token_sequences = torch.from_numpy(self.sequences[idx]).long()  # Token indices for hybrid loss
        
        # Keep tensors on CPU for now - device transfer will happen at batch level
        # This is a critical optimization: per-sample device transfer is extremely expensive
        
        return {
            'input_sequences': input_sequences,
            'target_sequences': target_sequences,
            'attention_mask': attention_mask,
            'token_sequences': token_sequences,  # Token indices for hybrid loss
            'metadata': self.metadata[idx] if isinstance(idx, int) and idx < len(self.metadata) else {}
        }
    
    def get_dataloader(self,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 0,
                      sampler: Optional[Sampler] = None,
                      **kwargs) -> DataLoader:
        """
        Create PyTorch DataLoader for this dataset.
        
        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle data (ignored if sampler provided)
            num_workers: Number of worker processes for data loading
            sampler: Custom sampler (overrides shuffle)
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            Configured PyTorch DataLoader
        """
        # Use custom sampler if provided
        if sampler is not None:
            shuffle = False
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=self._collate_fn,
            **kwargs
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Batched tensors dictionary
        """
        # Stack tensors
        input_sequences = torch.stack([item['input_sequences'] for item in batch])
        target_sequences = torch.stack([item['target_sequences'] for item in batch])
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
        
        # Collect metadata
        metadata = [item['metadata'] for item in batch]
        
        # Build return dict
        result = {
            'input_sequences': input_sequences,
            'target_sequences': target_sequences,
            'attention_mask': attention_masks,
            'metadata': metadata
        }
        
        # Include token sequences if requested (needed for hybrid loss training)
        if self.include_token_sequences:
            token_sequences = torch.stack([item['token_sequences'] for item in batch])
            result['token_sequences'] = token_sequences
        
        return result
    
    def get_sample_by_poem(self, poem_idx: int) -> List[Dict[str, torch.Tensor]]:
        """
        Get all chunks from a specific poem.
        
        Args:
            poem_idx: Index of poem to retrieve
            
        Returns:
            List of chunk samples from the poem
        """
        if poem_idx not in self.poem_to_chunks:
            raise ValueError(f"Poem {poem_idx} not found in dataset")
        
        chunk_indices = self.poem_to_chunks[poem_idx]
        return [self[i] for i in chunk_indices]
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics."""
        return {
            'vocabulary_size': len(self.vocabulary),
            'max_token_id': max(self.vocabulary.values()) if self.vocabulary else 0,
            'special_tokens': [token for token in self.vocabulary.keys() if token.startswith('<')],
            'sample_words': list(self.vocabulary.keys())[:10]
        }
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        # Calculate sequence lengths
        seq_lengths = [np.sum(seq != 0) for seq in self.sequences]
        
        # Calculate chunk distribution per poem
        chunks_per_poem = [len(chunks) for chunks in self.poem_to_chunks.values()]
        
        stats = {
            'total_samples': len(self),
            'total_poems': self.num_poems,
            'sequence_length': {
                'max': int(self.sequences.shape[1]),
                'mean': float(np.mean(seq_lengths)),
                'std': float(np.std(seq_lengths)),
                'min': int(np.min(seq_lengths)),
                'actual_max': int(np.max(seq_lengths))
            },
            'chunks_per_poem': {
                'mean': float(np.mean(chunks_per_poem)),
                'std': float(np.std(chunks_per_poem)),
                'min': int(np.min(chunks_per_poem)),
                'max': int(np.max(chunks_per_poem))
            },
            'memory_usage_mb': self.sequences.nbytes / 1024**2,
            'vocabulary_size': len(self.vocabulary),
            'split': self.split,
            'device': str(self.device)
        }
        
        # Add embedding sequence stats if available
        if self.embedding_sequences is not None:
            stats['embedding_shape'] = self.embedding_sequences.shape
            stats['embedding_memory_mb'] = self.embedding_sequences.nbytes / 1024**2
        
        return stats


class PoemAwareSampler(Sampler):
    """
    Custom sampler that ensures balanced sampling across poems.
    
    This sampler helps prevent overfitting to poems with many chunks by
    ensuring that each poem gets represented fairly in each epoch, regardless
    of how many chunks it was split into.
    """
    
    def __init__(self, dataset: AutoencoderDataset, max_chunks_per_poem: Optional[int] = None):
        """
        Initialize poem-aware sampler.
        
        Args:
            dataset: AutoencoderDataset to sample from
            max_chunks_per_poem: Maximum chunks to sample per poem per epoch
        """
        self.dataset = dataset
        self.max_chunks_per_poem = max_chunks_per_poem
        self.poem_to_chunks = dataset.poem_to_chunks
        
    def __iter__(self) -> Iterator[int]:
        """Generate sample indices with poem-aware balancing."""
        indices = []
        
        for poem_idx, chunk_indices in self.poem_to_chunks.items():
            # Shuffle chunks within poem
            poem_chunks = list(chunk_indices)
            random.shuffle(poem_chunks)
            
            # Limit chunks per poem if specified
            if self.max_chunks_per_poem:
                poem_chunks = poem_chunks[:self.max_chunks_per_poem]
            
            indices.extend(poem_chunks)
        
        # Shuffle final order
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self) -> int:
        """Return total number of samples."""
        if self.max_chunks_per_poem:
            return min(len(self.dataset), len(self.poem_to_chunks) * self.max_chunks_per_poem)
        return len(self.dataset)


class ChunkSequenceSampler(Sampler):
    """
    Sampler that preserves chunk sequences from poems.
    
    This sampler ensures that chunks from the same poem are processed
    in sequence, which can help with learning poem-level patterns.
    """
    
    def __init__(self, dataset: AutoencoderDataset, shuffle_poems: bool = True):
        """
        Initialize chunk sequence sampler.
        
        Args:
            dataset: AutoencoderDataset to sample from
            shuffle_poems: Whether to shuffle poem order
        """
        self.dataset = dataset
        self.shuffle_poems = shuffle_poems
        self.poem_to_chunks = dataset.poem_to_chunks
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices in chunk sequence order."""
        poem_indices = list(self.poem_to_chunks.keys())
        
        if self.shuffle_poems:
            random.shuffle(poem_indices)
        
        indices = []
        for poem_idx in poem_indices:
            # Add chunks in order for this poem
            chunk_indices = sorted(self.poem_to_chunks[poem_idx])
            indices.extend(chunk_indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.dataset)


def optimized_collate_fn(batch: List[Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Optimized collate function for batch-level device transfers.
    
    This is a critical performance optimization: instead of moving tensors to device
    one-by-one in __getitem__, we batch them on CPU and transfer the entire batch
    at once, which is much more efficient.
    
    Args:
        batch: List of sample dictionaries from dataset
        device: Target device for tensors
        
    Returns:
        Batched tensors on target device
    """
    # Stack tensors from all samples (on CPU)
    batched = {}
    
    # Get tensor keys from first sample
    tensor_keys = ['input_sequences', 'target_sequences', 'attention_mask', 'token_sequences']
    
    for key in tensor_keys:
        if key in batch[0]:
            # Stack all samples for this key
            stacked = torch.stack([sample[key] for sample in batch])
            
            # Single batch transfer to device (much faster than per-sample)
            batched[key] = stacked.to(device, non_blocking=True)
    
    # Handle metadata (keep on CPU)
    if 'metadata' in batch[0]:
        batched['metadata'] = [sample['metadata'] for sample in batch]
    
    return batched


def create_optimized_dataloader(
    dataset: AutoencoderDataset,
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = None,
    **kwargs
) -> DataLoader:
    """
    Create an optimized DataLoader with efficient device transfers.
    
    Args:
        dataset: AutoencoderDataset instance
        batch_size: Batch size
        device: Target device
        shuffle: Whether to shuffle data
        num_workers: Number of parallel workers
        pin_memory: Enable pin memory (auto-detect if None)
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Optimized DataLoader instance
    """
    # Auto-detect pin_memory if not specified
    if pin_memory is None:
        pin_memory = device.type == 'cuda'
    
    # Create collate function with device binding
    collate_fn = lambda batch: optimized_collate_fn(batch, device)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,  # Consistent batch sizes for better performance
        **kwargs
    )


def create_poetry_datasets(artifacts_path: Union[str, Path],
                          timestamp: str = "latest",
                          split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                          seed: int = 42,
                          lazy_loading: bool = True,
                          device: Optional[torch.device] = None,
                          include_token_sequences: bool = False) -> Tuple[AutoencoderDataset, AutoencoderDataset, AutoencoderDataset]:
    """
    Create train, validation, and test datasets from preprocessing artifacts.
    
    Args:
        artifacts_path: Path to preprocessing artifacts
        timestamp: Timestamp of artifacts to load
        split_ratios: Train/validation/test split ratios
        seed: Random seed for reproducible splits
        lazy_loading: Whether to use lazy loading for embeddings
        device: PyTorch device for tensors
        include_token_sequences: Whether to include token sequences in batches (needed for hybrid loss)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    logger.info(f"Creating poetry datasets from {artifacts_path}")
    
    # Create datasets with different splits
    train_dataset = AutoencoderDataset(
        artifacts_path=artifacts_path,
        timestamp=timestamp,
        split="train",
        split_ratios=split_ratios,
        seed=seed,
        lazy_loading=lazy_loading,
        device=device,
        include_token_sequences=include_token_sequences
    )
    
    val_dataset = AutoencoderDataset(
        artifacts_path=artifacts_path,
        timestamp=timestamp,
        split="val",
        split_ratios=split_ratios,
        seed=seed,
        lazy_loading=lazy_loading,
        device=device,
        include_token_sequences=include_token_sequences
    )
    
    test_dataset = AutoencoderDataset(
        artifacts_path=artifacts_path,
        timestamp=timestamp,
        split="test",
        split_ratios=split_ratios,
        seed=seed,
        lazy_loading=lazy_loading,
        device=device,
        include_token_sequences=include_token_sequences
    )
    
    logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_poetry_dataloaders(datasets: Tuple[AutoencoderDataset, AutoencoderDataset, AutoencoderDataset],
                             batch_size: int = 32,
                             num_workers: int = 0,
                             use_poem_aware_sampling: bool = False,
                             max_chunks_per_poem: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders from poetry datasets.
    
    Args:
        datasets: Tuple of (train, val, test) datasets
        batch_size: Batch size for training
        num_workers: Number of worker processes
        use_poem_aware_sampling: Whether to use poem-aware sampling for training
        max_chunks_per_poem: Maximum chunks per poem for balanced sampling
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset, val_dataset, test_dataset = datasets
    
    # Create samplers
    train_sampler = None
    if use_poem_aware_sampling:
        train_sampler = PoemAwareSampler(train_dataset, max_chunks_per_poem)
    
    # Create DataLoaders
    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler
    )
    
    val_loader = val_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = test_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info(f"Created DataLoaders: train={len(train_loader)} batches, "
                f"val={len(val_loader)} batches, test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
