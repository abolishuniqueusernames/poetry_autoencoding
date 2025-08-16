#!/usr/bin/env python3
"""
Test Attention Monitoring Fix

This script verifies that attention statistics are properly computed.
"""

import sys
import torch

# Add project root to path
sys.path.insert(0, '.')

from poetry_rnn.models import RNNAutoencoder
from poetry_rnn.training.optimized_trainer import OptimizedRNNTrainer

def test_attention_monitoring():
    """Test that attention statistics are computed correctly."""
    print("🔍 Testing Attention Monitoring Fix")
    print("=" * 50)
    
    # Create attention-enabled model
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
    
    print(f"✅ Model created with attention: {model.use_attention}")
    print(f"📊 Decoder type: {type(model.decoder).__name__}")
    print(f"🔧 Optimized attention: {model.use_optimized_attention}")
    
    # Create test batch
    batch_size = 2
    seq_len = 20
    
    test_batch = {
        'input_sequences': torch.randn(batch_size, seq_len, 300),
        'attention_mask': torch.ones(batch_size, seq_len).bool()
    }
    
    print(f"\n📋 Test batch: {batch_size} x {seq_len} x 300")
    
    # Create a trainer instance to test attention statistics
    trainer = OptimizedRNNTrainer(
        model=model,
        train_loader=None,  # Not needed for this test
        val_loader=None,
        optimizer=None,
        scheduler=None,
        device=torch.device('cpu'),
        config={},
        async_checkpointer=None,
        artifact_manager=None,
        performance_monitor=None
    )
    
    print("\n🧮 Computing attention statistics...")
    
    # Test attention statistics computation
    attention_stats = trainer._compute_attention_statistics(test_batch)
    
    print(f"\n📈 Attention Statistics:")
    print(f"  Entropy: {attention_stats['avg_attention_entropy']:.4f}")
    print(f"  Diversity: {attention_stats['attention_diversity']:.4f}")
    print(f"  Sharpness: {attention_stats['attention_sharpness']:.4f}")
    
    # Check if values are non-zero
    entropy = attention_stats['avg_attention_entropy']
    diversity = attention_stats['attention_diversity']
    sharpness = attention_stats['attention_sharpness']
    
    if entropy > 0 and diversity >= 0 and sharpness > 0:
        print("\n✅ SUCCESS: Attention statistics are working!")
        print("🎯 Non-zero values indicate attention is properly monitored")
        
        # Check if decoder has stored attention weights
        if hasattr(model.decoder, 'last_attention_weights') and model.decoder.last_attention_weights is not None:
            weights_shape = model.decoder.last_attention_weights.shape
            print(f"📊 Attention weights stored: {weights_shape}")
        else:
            print("⚠️  No attention weights stored in decoder")
        
        return True
    else:
        print("\n❌ FAILURE: Attention statistics are still zero")
        print(f"   Entropy: {entropy}")
        print(f"   Diversity: {diversity}")
        print(f"   Sharpness: {sharpness}")
        
        # Debug info
        print("\n🔧 Debug Information:")
        print(f"   Model has attention: {hasattr(model, 'use_attention') and model.use_attention}")
        if hasattr(model.decoder, 'last_attention_weights'):
            print(f"   Decoder has weights attribute: True")
            print(f"   Weights value: {model.decoder.last_attention_weights}")
        else:
            print(f"   Decoder has weights attribute: False")
        
        return False

if __name__ == "__main__":
    success = test_attention_monitoring()
    
    if success:
        print("\n🎉 Attention monitoring fix is working!")
        print("💡 Your training should now show non-zero attention metrics")
    else:
        print("\n🚨 Attention monitoring still needs debugging")
        print("💡 Check the attention computation pipeline")