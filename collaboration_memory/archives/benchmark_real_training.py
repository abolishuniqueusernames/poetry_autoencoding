#!/usr/bin/env python3
"""
Real Training Conditions Benchmark

This script tests optimizations under realistic training conditions:
- Larger batch sizes (16-32)
- Longer sequences (40-50 tokens)
- Multiple epochs to show encoder caching benefits
- Forward + backward passes

Usage:
    python benchmark_real_training.py
"""

import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, '.')

from poetry_rnn.models import RNNAutoencoder
from poetry_rnn.training.losses import EnhancedCosineLoss

def benchmark_training_scenario():
    """Benchmark attention optimizations under real training conditions."""
    print("🏋️ Real Training Conditions Benchmark")
    print("=" * 60)
    
    # Realistic training parameters
    batch_size = 16  # Real training batch size
    seq_len = 40     # Longer sequences where attention matters
    embedding_dim = 300
    num_epochs = 3   # Short but enough to show caching benefits
    
    # Model configurations to test
    configs = [
        {
            'name': 'Standard Attention (8 heads)',
            'use_attention': True,
            'attention_heads': 8,
            'use_optimized_attention': False
        },
        {
            'name': 'Quick Fix (4 heads)',
            'use_attention': True,
            'attention_heads': 4,
            'use_optimized_attention': False
        },
        {
            'name': 'Full Optimization',
            'use_attention': True,
            'attention_heads': 4,
            'use_optimized_attention': True
        }
    ]
    
    # Base model configuration
    base_config = {
        'input_size': 300,
        'hidden_size': 512,
        'bottleneck_dim': 128,
        'rnn_type': 'LSTM',
        'num_layers': 2,
        'dropout': 0.2,
        'max_seq_len': 50
    }
    
    # Generate realistic test data
    print(f"📊 Test setup: {batch_size} batch, {seq_len} seq_len, {num_epochs} epochs")
    
    test_batches = []
    for epoch in range(num_epochs):
        batch = {
            'input_sequences': torch.randn(batch_size, seq_len, embedding_dim),
            'attention_mask': torch.ones(batch_size, seq_len).bool()
        }
        test_batches.append(batch)
    
    results = []
    
    for config_info in configs:
        print(f"\n🔬 Benchmarking: {config_info['name']}")
        
        # Create model and optimizer
        model_config = {**base_config, **config_info}
        
        try:
            model = RNNAutoencoder(**model_config)
            model.train()  # Training mode
            
            # Create optimizer and loss
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            loss_fn = EnhancedCosineLoss()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Warmup (JIT compilation, etc.)
            with torch.no_grad():
                warmup_batch = test_batches[0]
                _ = model(warmup_batch)
            
            # Benchmark training loop
            start_time = time.perf_counter()
            
            total_loss = 0.0
            for epoch, batch in enumerate(test_batches):
                # Forward pass
                output = model(batch)
                
                # Compute loss
                targets = batch['input_sequences']
                predictions = output['reconstructed']
                loss = loss_fn(predictions, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            avg_loss = total_loss / num_epochs
            
            # Store results
            result = {
                'name': config_info['name'],
                'total_time_s': total_time,
                'time_per_epoch_s': total_time / num_epochs,
                'params': total_params,
                'avg_loss': avg_loss,
                'success': True
            }
            results.append(result)
            
            print(f"  ✅ Success: {total_time:.2f}s total ({total_time/num_epochs:.2f}s/epoch)")
            print(f"  📊 {total_params:,} params, avg_loss: {avg_loss:.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            result = {
                'name': config_info['name'],
                'success': False,
                'error': str(e)
            }
            results.append(result)
    
    # Performance comparison
    print("\n🚀 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) >= 2:
        baseline = successful_results[0]  # Standard attention
        
        print(f"📋 Baseline: {baseline['name']}")
        print(f"  ⏱️  Time: {baseline['total_time_s']:.2f}s ({baseline['time_per_epoch_s']:.2f}s/epoch)")
        print(f"  📏 Params: {baseline['params']:,}")
        
        for result in successful_results[1:]:
            speedup = baseline['time_per_epoch_s'] / result['time_per_epoch_s']
            param_ratio = result['params'] / baseline['params']
            loss_diff = result['avg_loss'] - baseline['avg_loss']
            
            print(f"\n🔄 {result['name']} vs Baseline:")
            print(f"  ⚡ Speedup: {speedup:.2f}x ({baseline['time_per_epoch_s']:.2f}s → {result['time_per_epoch_s']:.2f}s per epoch)")
            print(f"  📏 Parameters: {param_ratio:.3f}x")
            print(f"  📉 Loss difference: {loss_diff:+.4f}")
            
            # Performance assessment
            if 'Quick Fix' in result['name']:
                if speedup >= 1.5:
                    print(f"  ✅ Good speedup for quick fix: {speedup:.2f}x")
                elif speedup >= 1.2:
                    print(f"  🔶 Moderate speedup: {speedup:.2f}x")
                else:
                    print(f"  ⚠️  Limited speedup: {speedup:.2f}x")
            
            elif 'Full Optimization' in result['name']:
                if speedup >= 2.0:
                    print(f"  🎉 Excellent speedup: {speedup:.2f}x")
                elif speedup >= 1.5:
                    print(f"  ✅ Good speedup: {speedup:.2f}x")
                elif speedup >= 1.2:
                    print(f"  🔶 Moderate speedup: {speedup:.2f}x")
                else:
                    print(f"  ⚠️  Limited speedup: {speedup:.2f}x")
    
    # Overall assessment
    print("\n🎯 BENCHMARK SUMMARY")
    print("=" * 60)
    
    if len(successful_results) == len(configs):
        print("✅ All configurations completed successfully")
        
        if len(successful_results) >= 3:
            baseline = successful_results[0]
            quick_fix = successful_results[1]
            full_opt = successful_results[2]
            
            quick_speedup = baseline['time_per_epoch_s'] / quick_fix['time_per_epoch_s']
            full_speedup = baseline['time_per_epoch_s'] / full_opt['time_per_epoch_s']
            
            print(f"\n📈 Performance Improvements:")
            print(f"  🔸 Quick Fix (4 heads): {quick_speedup:.2f}x speedup")
            print(f"  🔹 Full Optimization: {full_speedup:.2f}x speedup")
            
            if full_speedup >= 2.0:
                print("\n🎉 EXCELLENT: Optimizations working as expected!")
                print("🚀 Ready for production training with significant speedup")
            elif full_speedup >= 1.5:
                print("\n✅ GOOD: Meaningful speedup achieved")
                print("🔧 Consider additional optimizations for maximum performance")
            else:
                print("\n🔧 NEEDS IMPROVEMENT: Limited speedup observed")
                print("💡 Check if optimizations are properly activated")
                print("💡 Consider longer sequences or larger batches for better testing")
        
    else:
        print("❌ Some configurations failed")
        failed = [r for r in results if not r.get('success', False)]
        for failure in failed:
            print(f"  💥 {failure['name']}: {failure.get('error', 'Unknown error')}")

if __name__ == "__main__":
    benchmark_training_scenario()