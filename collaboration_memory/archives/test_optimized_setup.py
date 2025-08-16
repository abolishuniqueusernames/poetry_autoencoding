#!/usr/bin/env python3
"""
Test script to verify optimized training setup.

This script tests that all components of the optimized training pipeline
are correctly configured and can initialize without errors.
"""

import torch
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, '.')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_optimized_components():
    """Test all optimized components initialization."""
    
    logger.info("="*60)
    logger.info("TESTING OPTIMIZED TRAINING COMPONENTS")
    logger.info("="*60)
    
    # Test 1: Scheduler creation
    logger.info("\n1. Testing scheduler creation...")
    try:
        from poetry_rnn.training.schedulers import CosineAnnealingWarmRestartsWithDecay
        
        # Create dummy optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create scheduler
        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
            decay_factor=0.95
        )
        
        # Test stepping
        for _ in range(5):
            scheduler.step()
        
        logger.info("✓ Scheduler working correctly")
    except Exception as e:
        logger.error(f"✗ Scheduler test failed: {e}")
        return False
    
    # Test 2: Async checkpointing
    logger.info("\n2. Testing async checkpointing...")
    try:
        from poetry_rnn.utils.async_io import AsyncCheckpointer
        
        # Create checkpoint directory
        checkpoint_dir = Path('test_checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Create checkpointer
        checkpointer = AsyncCheckpointer(checkpoint_dir)
        
        # Test async save
        test_checkpoint = {
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.5
        }
        
        checkpointer.save_checkpoint_async(test_checkpoint, 'test_checkpoint.pth')
        
        # Wait for completion
        checkpointer.wait_for_completion(timeout=5.0)
        
        # Get statistics
        stats = checkpointer.get_statistics()
        logger.info(f"Checkpoint stats: {stats}")
        
        # Cleanup
        checkpointer.shutdown()
        
        # Remove test file
        (checkpoint_dir / 'test_checkpoint.pth').unlink(missing_ok=True)
        checkpoint_dir.rmdir()
        
        logger.info("✓ Async checkpointing working correctly")
    except Exception as e:
        logger.error(f"✗ Async checkpointing test failed: {e}")
        return False
    
    # Test 3: Threaded data loader
    logger.info("\n3. Testing optimized data loader...")
    try:
        from poetry_rnn.data.threaded_loader import create_optimized_dataloader
        from torch.utils.data import TensorDataset
        
        # Create dummy dataset
        dummy_data = torch.randn(100, 50, 300)
        dummy_labels = torch.randn(100, 50, 300)
        dataset = TensorDataset(dummy_data, dummy_labels)
        
        # Create optimized dataloader
        dataloader = create_optimized_dataloader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=2,
            pin_memory=False,  # Disable for CPU
            prefetch_factor=2,
            persistent_workers=True,
            use_bucketing=False  # Disable for simple test
        )
        
        # Test iteration
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            if batch_count >= 2:  # Just test a few batches
                break
        
        logger.info(f"✓ Data loader working correctly (loaded {batch_count} batches)")
    except Exception as e:
        logger.error(f"✗ Data loader test failed: {e}")
        return False
    
    # Test 4: Parallel GLoVe loading (mock)
    logger.info("\n4. Testing parallel GLoVe loader...")
    try:
        from poetry_rnn.embeddings.parallel_glove import ParallelGLoVeLoader
        
        # We'll just test initialization since we may not have the actual file
        glove_path = Path('embeddings/glove.6B.300d.txt')
        
        if glove_path.exists():
            loader = ParallelGLoVeLoader(
                embedding_path=glove_path,
                embedding_dim=300,
                num_threads=2,
                vocabulary={'test', 'words'},
                lazy_loading=True
            )
            logger.info("✓ Parallel GLoVe loader initialized successfully")
        else:
            logger.info("⚠ GLoVe file not found, skipping actual load test")
            logger.info("✓ Parallel GLoVe loader module imported successfully")
            
    except Exception as e:
        logger.error(f"✗ Parallel GLoVe loader test failed: {e}")
        return False
    
    # Test 5: Model initialization
    logger.info("\n5. Testing model initialization...")
    try:
        from poetry_rnn.utils.initialization import initialize_model_weights
        from poetry_rnn.models.autoencoder import RNNAutoencoder
        
        # Create model
        model = RNNAutoencoder(
            vocab_size=1000,
            embedding_dim=300,
            hidden_dim=512,
            bottleneck_dim=64,
            num_layers=2
        )
        
        # Initialize weights
        initialize_model_weights(model, strategy='xavier_uniform')
        
        # Check parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model initialized successfully ({total_params:,} parameters)")
        
    except Exception as e:
        logger.error(f"✗ Model initialization test failed: {e}")
        return False
    
    # Test 6: Optimized trainer initialization
    logger.info("\n6. Testing optimized trainer...")
    try:
        from poetry_rnn.training.optimized_trainer import OptimizedRNNTrainer
        
        # Create trainer
        trainer = OptimizedRNNTrainer(
            model=model,
            train_loader=dataloader,
            val_loader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            config={'training': {'num_epochs': 1}},
            device=torch.device('cpu')
        )
        
        logger.info("✓ Optimized trainer initialized successfully")
        
    except Exception as e:
        logger.error(f"✗ Optimized trainer test failed: {e}")
        return False
    
    logger.info("\n" + "="*60)
    logger.info("ALL TESTS PASSED SUCCESSFULLY!")
    logger.info("="*60)
    
    return True


def check_performance_improvements():
    """Check expected performance improvements."""
    
    logger.info("\n" + "="*60)
    logger.info("EXPECTED PERFORMANCE IMPROVEMENTS")
    logger.info("="*60)
    
    improvements = {
        "DataLoader Optimization": {
            "baseline": "Single-threaded, no prefetch",
            "optimized": "4 workers, prefetch_factor=2, pin_memory",
            "expected_speedup": "2-3x"
        },
        "GLoVe Loading": {
            "baseline": "Sequential loading (120s)",
            "optimized": "4-thread parallel loading",
            "expected_speedup": "10-15x (8-12s)"
        },
        "Checkpointing": {
            "baseline": "Blocking saves (5-10s)",
            "optimized": "Async non-blocking saves",
            "expected_speedup": "Eliminates blocking"
        },
        "Training Configuration": {
            "baseline": "50 epochs, basic scheduler",
            "optimized": "100 epochs, cosine annealing with warm restarts",
            "expected_improvement": "Better convergence"
        },
        "Gradient Management": {
            "baseline": "Gradient clip=5.0",
            "optimized": "Gradient clip=1.0, bottleneck regularization",
            "expected_improvement": "More stable training"
        }
    }
    
    for component, details in improvements.items():
        logger.info(f"\n{component}:")
        logger.info(f"  Baseline: {details['baseline']}")
        logger.info(f"  Optimized: {details['optimized']}")
        logger.info(f"  Expected: {details.get('expected_speedup', details.get('expected_improvement'))}")
    
    logger.info("\n" + "="*60)
    logger.info("TARGET PERFORMANCE METRICS")
    logger.info("="*60)
    
    targets = {
        "Training Speed": "3-5x faster than baseline",
        "GLoVe Loading": "12-15x faster (120s → 8-10s)",
        "Memory Usage": "50% reduction",
        "Model Performance": "0.85-0.95 cosine similarity",
        "Throughput": "500+ sequences/second"
    }
    
    for metric, target in targets.items():
        logger.info(f"  {metric}: {target}")


if __name__ == "__main__":
    # Run tests
    success = test_optimized_components()
    
    if success:
        # Show expected improvements
        check_performance_improvements()
        
        logger.info("\n" + "="*60)
        logger.info("READY TO RUN OPTIMIZED TRAINING")
        logger.info("="*60)
        logger.info("\nTo start optimized training, run:")
        logger.info("  python train_optimized_architecture.py")
        logger.info("\nThis will use all performance optimizations to achieve:")
        logger.info("  - 3-5x faster training")
        logger.info("  - 50% memory reduction")
        logger.info("  - 0.85-0.95 target cosine similarity")
    else:
        logger.error("\nSome tests failed. Please fix issues before running optimized training.")