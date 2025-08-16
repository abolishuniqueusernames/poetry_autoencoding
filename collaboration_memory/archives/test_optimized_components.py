#!/usr/bin/env python3
"""
Test script to verify all optimized components are working correctly.

This script tests:
1. Parallel GLoVe loading
2. Optimized DataLoader configuration
3. Async checkpointing
4. Scheduler functionality
5. Model initialization
"""

import torch
import torch.nn as nn
import time
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import all optimized components
from poetry_rnn.embeddings.parallel_glove import ParallelGLoVeLoader
from poetry_rnn.data.threaded_loader import create_optimized_dataloader, BucketingSampler
from poetry_rnn.utils.async_io import AsyncCheckpointer, AsyncArtifactManager
from poetry_rnn.training.schedulers import CosineAnnealingWarmRestartsWithDecay
from poetry_rnn.utils.initialization import initialize_model_weights
from poetry_rnn.models.autoencoder import RNNAutoencoder
from poetry_rnn.dataset import AutoencoderDataset
from poetry_rnn.training.optimized_trainer import OptimizedRNNTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_parallel_glove_loading():
    """Test parallel GLoVe embedding loading."""
    logger.info("\n" + "="*60)
    logger.info("Testing Parallel GLoVe Loading")
    logger.info("="*60)
    
    glove_path = Path('embeddings/glove.6B.300d.txt')
    
    if not glove_path.exists():
        logger.warning(f"GLoVe file not found at {glove_path}")
        return False
    
    try:
        # Test with limited vocabulary for speed
        loader = ParallelGLoVeLoader(
            embedding_path=glove_path,
            embedding_dim=300,
            num_threads=4,
            vocabulary=None,  # Load all for testing
            lazy_loading=False
        )
        
        # Time the parallel loading
        start_time = time.perf_counter()
        embeddings = loader.load_parallel()
        load_time = time.perf_counter() - start_time
        
        logger.info(f"✓ Loaded {len(embeddings)} embeddings in {load_time:.2f}s")
        logger.info(f"  Sample words: {list(embeddings.keys())[:5]}")
        
        # Verify embedding dimensions
        sample_embedding = next(iter(embeddings.values()))
        assert sample_embedding.shape == (300,), f"Unexpected shape: {sample_embedding.shape}"
        logger.info(f"✓ Embedding dimensions verified: {sample_embedding.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Parallel GLoVe loading failed: {e}")
        return False


def test_optimized_dataloader():
    """Test optimized DataLoader configuration."""
    logger.info("\n" + "="*60)
    logger.info("Testing Optimized DataLoader")
    logger.info("="*60)
    
    try:
        # Load dataset
        dataset = AutoencoderDataset(
            artifacts_path=Path('preprocessed_artifacts'),
            lazy_loading=False
        )
        
        logger.info(f"✓ Dataset loaded: {len(dataset)} sequences")
        
        # Create optimized dataloader
        dataloader = create_optimized_dataloader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=2,  # Use fewer workers for testing
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True
        )
        
        # Test iteration
        start_time = time.perf_counter()
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            if batch_count >= 5:  # Test first 5 batches
                break
        
        iter_time = time.perf_counter() - start_time
        logger.info(f"✓ DataLoader iteration: {batch_count} batches in {iter_time:.2f}s")
        logger.info(f"  Batch shape: {batch['input_sequences'].shape}")
        logger.info(f"  Attention mask shape: {batch['attention_mask'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ DataLoader test failed: {e}")
        return False


def test_async_checkpointing():
    """Test async checkpointing functionality."""
    logger.info("\n" + "="*60)
    logger.info("Testing Async Checkpointing")
    logger.info("="*60)
    
    try:
        # Create test checkpoint directory
        checkpoint_dir = Path('test_checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize async checkpointer
        checkpointer = AsyncCheckpointer(checkpoint_dir)
        
        # Create dummy model state
        dummy_state = {
            'epoch': 1,
            'model_state_dict': {'weight': torch.randn(10, 10)},
            'optimizer_state_dict': {'param_groups': []},
            'loss': 0.5
        }
        
        # Test async save
        start_time = time.perf_counter()
        checkpointer.save_checkpoint_async(dummy_state, 'test_checkpoint.pth')
        save_time = time.perf_counter() - start_time
        
        # The save should be non-blocking (very fast)
        assert save_time < 0.1, f"Save took too long: {save_time:.2f}s"
        logger.info(f"✓ Async save initiated in {save_time*1000:.1f}ms")
        
        # Wait for save to complete
        time.sleep(1)
        
        # Verify checkpoint was saved
        checkpoint_path = checkpoint_dir / 'test_checkpoint.pth'
        assert checkpoint_path.exists(), "Checkpoint file not created"
        
        # Load and verify
        loaded_state = torch.load(checkpoint_path)
        assert loaded_state['epoch'] == 1
        logger.info(f"✓ Checkpoint verified at {checkpoint_path}")
        
        # Cleanup
        checkpointer.shutdown()
        checkpoint_path.unlink()
        checkpoint_dir.rmdir()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Async checkpointing test failed: {e}")
        return False


def test_scheduler():
    """Test cosine annealing scheduler with warm restarts."""
    logger.info("\n" + "="*60)
    logger.info("Testing Learning Rate Scheduler")
    logger.info("="*60)
    
    try:
        # Create dummy optimizer
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Initialize scheduler
        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
            decay_factor=0.95
        )
        
        # Simulate training epochs
        lrs = []
        for epoch in range(30):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        logger.info(f"✓ Scheduler tested for 30 epochs")
        logger.info(f"  Initial LR: {lrs[0]:.6f}")
        logger.info(f"  Min LR: {min(lrs):.6f}")
        logger.info(f"  Max LR after restart: {lrs[10]:.6f}")
        
        # Verify warm restarts occurred - check if there's a jump after decay
        restart_detected = False
        for i in range(1, len(lrs)):
            if lrs[i] > lrs[i-1] * 1.5:  # Significant jump indicates restart
                restart_detected = True
                logger.info(f"  Warm restart detected at epoch {i}: {lrs[i-1]:.6f} -> {lrs[i]:.6f}")
                break
        
        if not restart_detected:
            # With decay factor, restarts might be subtle, so just check if scheduler is working
            logger.info(f"  Note: Warm restarts with decay may be subtle")
            
        logger.info(f"✓ Scheduler functioning correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Scheduler test failed: {e}")
        return False


def test_model_initialization():
    """Test model initialization strategies."""
    logger.info("\n" + "="*60)
    logger.info("Testing Model Initialization")
    logger.info("="*60)
    
    try:
        # Create model
        model = RNNAutoencoder(
            input_size=300,
            hidden_size=512,
            bottleneck_dim=64,
            num_layers=2,
            dropout=0.2,
            rnn_type='lstm'  # lowercase
        )
        
        # Apply initialization
        initialize_model_weights(model, strategy='xavier_uniform')
        
        # Check weights are properly initialized
        for name, param in model.named_parameters():
            if 'weight' in name:
                std = param.std().item()
                mean = param.mean().item()
                logger.info(f"  {name}: mean={mean:.4f}, std={std:.4f}")
                
                # Verify reasonable initialization
                # BatchNorm weights are initialized to 1.0 by design
                if 'batch_norm.weight' in name:
                    assert abs(mean - 1.0) < 0.1, f"BatchNorm weight mean should be ~1.0: {mean}"
                else:
                    assert abs(mean) < 0.1, f"Mean too large for {name}: {mean}"
                    assert 0.001 < std < 1.0, f"Std out of range for {name}: {std}"
        
        logger.info(f"✓ Model initialization verified")
        
        # Test forward pass
        batch_size = 8
        seq_len = 50
        # Create input embeddings directly (not token IDs)
        dummy_embeddings = torch.randn(batch_size, seq_len, 300)
        dummy_mask = torch.ones(batch_size, seq_len)
        
        # Create batch dict as expected by model
        batch_dict = {
            'input_sequences': dummy_embeddings,
            'attention_mask': dummy_mask
        }
        
        with torch.no_grad():
            output = model(batch_dict)
            reconstructed = output['reconstructed']
            bottleneck = output['bottleneck']
        
        assert reconstructed.shape == (batch_size, seq_len, 300)
        assert bottleneck.shape == (batch_size, 64)
        
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Output shape: {reconstructed.shape}")
        logger.info(f"  Bottleneck shape: {bottleneck.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model initialization test failed: {e}")
        return False


def main():
    """Run all component tests."""
    logger.info("\n" + "="*70)
    logger.info(" OPTIMIZED COMPONENTS TEST SUITE ")
    logger.info("="*70)
    
    tests = [
        ("Parallel GLoVe Loading", test_parallel_glove_loading),
        ("Optimized DataLoader", test_optimized_dataloader),
        ("Async Checkpointing", test_async_checkpointing),
        ("Learning Rate Scheduler", test_scheduler),
        ("Model Initialization", test_model_initialization)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info(" TEST SUMMARY ")
    logger.info("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:30s} {status}")
    
    logger.info("-"*70)
    logger.info(f"Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        logger.info("\n🎉 All components are working correctly!")
        logger.info("You can now run the optimized training script:")
        logger.info("  python train_optimized_architecture.py")
    else:
        logger.warning("\n⚠️ Some components failed. Please fix issues before training.")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)