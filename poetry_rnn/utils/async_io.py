"""
Asynchronous I/O Utilities for Non-blocking Operations

This module provides async I/O functionality for checkpointing and artifact
management, eliminating blocking operations during training.
"""

import torch
import threading
import queue
import time
import json
import pickle
import gzip
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import numpy as np
from collections import deque
import io

logger = logging.getLogger(__name__)


class AsyncCheckpointer:
    """
    Asynchronous checkpoint manager for non-blocking model saves.
    
    Features:
    - Background thread for checkpoint saving
    - Deep copying to avoid training interference
    - Compressed checkpoint support
    - Queue-based operation management
    - Graceful shutdown handling
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path], max_queue_size: int = 5):
        """
        Initialize async checkpointer.
        
        Args:
            checkpoint_dir: Directory for saving checkpoints
            max_queue_size: Maximum number of pending save operations
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Queue for checkpoint operations
        self.save_queue = queue.Queue(maxsize=max_queue_size)
        
        # Worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.save_times = deque(maxlen=100)
        self.total_saves = 0
        self.failed_saves = 0
        
        # Start worker
        self.worker_thread.start()
        logger.info(f"AsyncCheckpointer initialized with dir: {self.checkpoint_dir}")
    
    def save_checkpoint_async(self, checkpoint: Dict[str, Any], filename: str, 
                             compress: bool = False, priority: int = 0):
        """
        Queue a checkpoint for asynchronous saving.
        
        Args:
            checkpoint: Checkpoint dictionary to save
            filename: Name of checkpoint file
            compress: Whether to compress the checkpoint
            priority: Priority for save operation (higher = more important)
        """
        try:
            # Deep copy state dictionaries to avoid interference
            checkpoint_copy = self._deep_copy_checkpoint(checkpoint)
            
            # Add to queue
            self.save_queue.put(
                (priority, checkpoint_copy, filename, compress),
                block=False
            )
            
            logger.debug(f"Queued checkpoint: {filename} (queue size: {self.save_queue.qsize()})")
            
        except queue.Full:
            logger.warning(f"Checkpoint queue full, dropping save for {filename}")
            self.failed_saves += 1
    
    def _deep_copy_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a deep copy of checkpoint data.
        
        Args:
            checkpoint: Original checkpoint
            
        Returns:
            Deep copy safe for async saving
        """
        checkpoint_copy = {}
        
        for key, value in checkpoint.items():
            if key.endswith('state_dict'):
                # Deep copy state dictionaries
                if value is not None:
                    checkpoint_copy[key] = {
                        k: v.cpu().clone() if torch.is_tensor(v) else v
                        for k, v in value.items()
                    }
                else:
                    checkpoint_copy[key] = None
            elif torch.is_tensor(value):
                checkpoint_copy[key] = value.cpu().clone()
            else:
                # Simple copy for other types
                checkpoint_copy[key] = value
        
        return checkpoint_copy
    
    def _worker(self):
        """Background worker for saving checkpoints."""
        logger.info("AsyncCheckpointer worker started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get next checkpoint with timeout
                item = self.save_queue.get(timeout=1.0)
                if item is None:  # Shutdown signal
                    break
                
                priority, checkpoint, filename, compress = item
                
                # Save checkpoint
                start_time = time.perf_counter()
                self._save_checkpoint(checkpoint, filename, compress)
                save_time = time.perf_counter() - start_time
                
                self.save_times.append(save_time)
                self.total_saves += 1
                
                logger.info(f"Saved checkpoint: {filename} ({save_time:.2f}s)")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")
                self.failed_saves += 1
    
    def _save_checkpoint(self, checkpoint: Dict[str, Any], filename: str, compress: bool):
        """
        Save checkpoint to disk.
        
        Args:
            checkpoint: Checkpoint data
            filename: Output filename
            compress: Whether to compress
        """
        filepath = self.checkpoint_dir / filename
        
        if compress:
            # Save compressed checkpoint
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            
            with gzip.open(f"{filepath}.gz", 'wb') as f:
                f.write(buffer.getvalue())
        else:
            # Save regular checkpoint
            torch.save(checkpoint, filepath)
    
    def wait_for_completion(self, timeout: Optional[float] = None):
        """
        Wait for all pending saves to complete.
        
        Args:
            timeout: Maximum time to wait
        """
        start_time = time.time()
        
        while not self.save_queue.empty():
            if timeout and (time.time() - start_time) > timeout:
                logger.warning("Timeout waiting for checkpoint saves")
                break
            time.sleep(0.1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpointing statistics."""
        return {
            'total_saves': self.total_saves,
            'failed_saves': self.failed_saves,
            'pending_saves': self.save_queue.qsize(),
            'avg_save_time': np.mean(self.save_times) if self.save_times else 0,
            'max_save_time': max(self.save_times) if self.save_times else 0
        }
    
    def shutdown(self, timeout: float = 10.0):
        """
        Gracefully shutdown the checkpointer.
        
        Args:
            timeout: Maximum time to wait for pending saves
        """
        logger.info("Shutting down AsyncCheckpointer...")
        
        # Signal shutdown
        self.shutdown_event.set()
        self.save_queue.put(None)  # Sentinel value
        
        # Wait for worker to finish
        self.worker_thread.join(timeout)
        
        if self.worker_thread.is_alive():
            logger.warning("AsyncCheckpointer worker did not shutdown cleanly")
        
        # Log final statistics
        stats = self.get_statistics()
        logger.info(f"AsyncCheckpointer statistics: {stats}")


class AsyncArtifactManager:
    """
    Asynchronous manager for training artifacts (logs, metrics, plots).
    
    Features:
    - Non-blocking saves for various artifact types
    - Automatic format detection
    - Compression support
    - Batch operations
    """
    
    def __init__(self, artifact_dir: Union[str, Path], num_workers: int = 2):
        """
        Initialize artifact manager.
        
        Args:
            artifact_dir: Directory for artifacts
            num_workers: Number of worker threads
        """
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Queue and workers
        self.save_queue = queue.Queue()
        self.workers = []
        self.shutdown_event = threading.Event()
        
        # Start workers
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"AsyncArtifactManager initialized with {num_workers} workers")
    
    def save_json_async(self, data: Dict[str, Any], filename: str):
        """Save JSON data asynchronously."""
        self.save_queue.put(('json', data, filename))
    
    def save_numpy_async(self, array: np.ndarray, filename: str, compressed: bool = True):
        """Save numpy array asynchronously."""
        # Copy array to avoid modification during save
        array_copy = array.copy()
        self.save_queue.put(('numpy', array_copy, filename, compressed))
    
    def save_text_async(self, text: str, filename: str):
        """Save text file asynchronously."""
        self.save_queue.put(('text', text, filename))
    
    def save_pickle_async(self, obj: Any, filename: str):
        """Save pickled object asynchronously."""
        self.save_queue.put(('pickle', obj, filename))
    
    def save_metrics_batch_async(self, metrics: Dict[str, list], prefix: str = "metrics"):
        """
        Save a batch of metrics asynchronously.
        
        Args:
            metrics: Dictionary of metric lists
            prefix: Filename prefix
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = f"{prefix}_{timestamp}.json"
        self.save_json_async(metrics, json_file)
        
        # Save as compressed numpy if numerical
        try:
            arrays = {k: np.array(v) for k, v in metrics.items()}
            npz_file = f"{prefix}_{timestamp}.npz"
            self.save_queue.put(('npz', arrays, npz_file))
        except:
            pass  # Skip if not convertible to numpy
    
    def _worker(self):
        """Worker thread for saving artifacts."""
        while not self.shutdown_event.is_set():
            try:
                item = self.save_queue.get(timeout=1.0)
                if item is None:
                    break
                
                self._process_save(item)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error saving artifact: {e}")
    
    def _process_save(self, item: tuple):
        """Process a save operation."""
        save_type = item[0]
        
        if save_type == 'json':
            _, data, filename = item
            filepath = self.artifact_dir / filename
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif save_type == 'numpy':
            _, array, filename, compressed = item
            filepath = self.artifact_dir / filename
            if compressed:
                np.savez_compressed(filepath, data=array)
            else:
                np.save(filepath, array)
        
        elif save_type == 'npz':
            _, arrays, filename = item
            filepath = self.artifact_dir / filename
            np.savez_compressed(filepath, **arrays)
        
        elif save_type == 'text':
            _, text, filename = item
            filepath = self.artifact_dir / filename
            with open(filepath, 'w') as f:
                f.write(text)
        
        elif save_type == 'pickle':
            _, obj, filename = item
            filepath = self.artifact_dir / filename
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
    
    def shutdown(self, timeout: float = 5.0):
        """Gracefully shutdown the artifact manager."""
        logger.info("Shutting down AsyncArtifactManager...")
        
        # Signal shutdown
        self.shutdown_event.set()
        for _ in self.workers:
            self.save_queue.put(None)
        
        # Wait for workers
        for worker in self.workers:
            worker.join(timeout)


class AsyncDataPrefetcher:
    """
    Asynchronous data prefetcher for continuous data loading.
    
    This class maintains a buffer of pre-loaded batches to ensure
    the GPU never waits for data.
    """
    
    def __init__(self, dataloader, device: torch.device, buffer_size: int = 2):
        """
        Initialize data prefetcher.
        
        Args:
            dataloader: PyTorch DataLoader
            device: Target device for data
            buffer_size: Number of batches to prefetch
        """
        self.dataloader = dataloader
        self.device = device
        self.buffer_size = buffer_size
        
        # Prefetch buffer
        self.buffer = queue.Queue(maxsize=buffer_size)
        
        # Prefetch thread
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.stop_event = threading.Event()
        
        # Start prefetching
        self.prefetch_thread.start()
    
    def _prefetch_worker(self):
        """Worker thread for prefetching batches."""
        try:
            for batch in self.dataloader:
                if self.stop_event.is_set():
                    break
                
                # Move batch to device
                batch_on_device = self._move_to_device(batch)
                
                # Add to buffer
                self.buffer.put(batch_on_device)
            
            # Signal end of data
            self.buffer.put(None)
            
        except Exception as e:
            logger.error(f"Prefetch worker error: {e}")
            self.buffer.put(None)
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to target device."""
        device_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch
    
    def __iter__(self):
        """Iterate over prefetched batches."""
        while True:
            batch = self.buffer.get()
            if batch is None:
                break
            yield batch
    
    def shutdown(self):
        """Stop prefetching."""
        self.stop_event.set()
        self.prefetch_thread.join(timeout=5.0)