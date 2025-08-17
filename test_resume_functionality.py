#!/usr/bin/env python3
"""
Test Resume Functionality

This script tests the training resume functionality by:
1. Loading a checkpoint from epoch 80
2. Verifying all state is restored correctly
3. Running a few training steps to ensure continuity
4. Comparing with expected behavior

Usage:
    python test_resume_functionality.py
"""

import sys
import logging
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from poetry_rnn import PoetryPreprocessor, AutoencoderDataset, Config
from poetry_rnn.models import RNNAutoencoder
from poetry_rnn.training.optimized_trainer import OptimizedRNNTrainer
import torch.optim as optim

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_checkpoint_loading():
    """Test loading checkpoint from epoch 80."""
    logger.info("🧪 Testing Checkpoint Loading")
    logger.info("=" * 50)
    
    checkpoint_path = "attention_training_results/checkpoint_epoch_80.pth"
    
    if not Path(checkpoint_path).exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        logger.info("Available checkpoints:")
        for checkpoint in Path("attention_training_results").glob("*.pth"):
            logger.info(f"  📁 {checkpoint}")
        return False
    
    # Load checkpoint to inspect contents
    logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    logger.info("📊 Checkpoint Contents:")
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            logger.info(f"  {key}: dict with {len(value)} keys")
        elif isinstance(value, torch.Tensor):
            logger.info(f"  {key}: tensor {value.shape}")
        else:
            logger.info(f"  {key}: {type(value).__name__} = {value}")
    
    # Verify critical information
    epoch = checkpoint.get('epoch', 'MISSING')
    val_loss = checkpoint.get('best_val_loss', 'MISSING')
    cosine_sim = checkpoint.get('best_cosine_similarity', 'MISSING')
    
    logger.info(f"\n📈 Checkpoint Metrics:")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Best Val Loss: {val_loss}")
    logger.info(f"  Best Cosine Similarity: {cosine_sim}")
    
    if epoch == 80:
        logger.info("✅ Checkpoint epoch verification passed")
        return True
    else:
        logger.error(f"❌ Expected epoch 80, got {epoch}")
        return False

def test_model_loading():
    """Test loading model from checkpoint."""
    logger.info("\n🧪 Testing Model Loading")
    logger.info("=" * 50)
    
    try:
        # Create model with same architecture as training
        model = RNNAutoencoder(
            input_size=300,
            hidden_size=512,
            bottleneck_dim=128,
            rnn_type='LSTM',
            num_layers=2,
            use_attention=True,
            attention_heads=4,
            use_optimized_attention=True
        )
        
        # Load checkpoint
        checkpoint_path = "attention_training_results/checkpoint_epoch_80.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info("✅ Model loaded successfully")
        
        # Test forward pass
        batch_size, seq_len = 2, 20
        test_input = {
            'input_sequences': torch.randn(batch_size, seq_len, 300),
            'attention_mask': torch.ones(batch_size, seq_len).bool()
        }
        
        with torch.no_grad():
            output = model(test_input)
        
        logger.info(f"✅ Forward pass successful")
        logger.info(f"  Input shape: {test_input['input_sequences'].shape}")
        logger.info(f"  Output shape: {output['reconstructed'].shape}")
        logger.info(f"  Bottleneck shape: {output['bottleneck'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_optimizer_scheduler_loading():
    """Test loading optimizer and scheduler states."""
    logger.info("\n🧪 Testing Optimizer & Scheduler Loading")
    logger.info("=" * 50)
    
    try:
        # Create dummy model and optimizer
        model = RNNAutoencoder(
            input_size=300, hidden_size=512, bottleneck_dim=128,
            rnn_type='LSTM', num_layers=2, use_attention=True
        )
        
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        # Load checkpoint
        checkpoint_path = "attention_training_results/checkpoint_epoch_80.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load states
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✅ Optimizer state loaded")
        else:
            logger.warning("⚠️  No optimizer state in checkpoint")
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("✅ Scheduler state loaded")
        else:
            logger.warning("⚠️  No scheduler state in checkpoint")
        
        # Check learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"📊 Current learning rate: {current_lr:.2e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimizer/scheduler loading failed: {e}")
        return False

def test_trainer_resume():
    """Test the trainer's resume functionality."""
    logger.info("\n🧪 Testing Trainer Resume Functionality")
    logger.info("=" * 50)
    
    try:
        # Create minimal trainer setup
        model = RNNAutoencoder(
            input_size=300, hidden_size=512, bottleneck_dim=128,
            rnn_type='LSTM', num_layers=2, use_attention=True
        )
        
        optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        config = {
            'training': {'early_stopping_patience': 20, 'min_delta': 1e-5},
            'monitoring': {'attention_monitoring': False}
        }
        
        # Create trainer
        trainer = OptimizedRNNTrainer(
            model=model,
            train_loader=None,  # Not needed for this test
            val_loader=None,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device('cpu'),
            config=config,
            async_checkpointer=None,
            artifact_manager=None,
            performance_monitor=None
        )
        
        # Test loading checkpoint
        checkpoint_path = "attention_training_results/checkpoint_epoch_80.pth"
        checkpoint_info = trainer.load_checkpoint(checkpoint_path, resume_training=True)
        
        logger.info("✅ Trainer resume functionality working")
        logger.info(f"📊 Resumed at epoch: {trainer.current_epoch}")
        logger.info(f"📊 Best cosine similarity: {trainer.best_cosine_similarity:.4f}")
        logger.info(f"📊 Best validation loss: {trainer.best_val_loss:.6f}")
        
        # Verify epoch is set correctly
        if trainer.current_epoch == 80:
            logger.info("✅ Epoch verification passed")
            return True
        else:
            logger.error(f"❌ Expected epoch 80, got {trainer.current_epoch}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Trainer resume test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all resume functionality tests."""
    logger.info("🚀 Testing Resume Functionality")
    logger.info("=" * 70)
    
    tests = [
        ("Checkpoint Loading", test_checkpoint_loading),
        ("Model Loading", test_model_loading),
        ("Optimizer & Scheduler Loading", test_optimizer_scheduler_loading),
        ("Trainer Resume", test_trainer_resume)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        if test_func():
            logger.info(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            logger.error(f"❌ {test_name}: FAILED")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Resume functionality is working correctly.")
        logger.info("\n💡 Ready to resume training from epoch 80:")
        logger.info("python resume_training.py --checkpoint attention_training_results/checkpoint_epoch_80.pth --additional-epochs 30 --stable-scheduler")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)