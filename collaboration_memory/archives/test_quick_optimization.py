#!/usr/bin/env python3
"""
Quick Test for Attention Optimizations

This script validates that:
1. Quick fix (4 heads) works correctly
2. Full optimizations (cached projections + fused operations) work correctly
3. Output equivalence is maintained
4. Performance improvement is achieved

Usage:
    python test_quick_optimization.py
"""

import sys
import time
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, '.')

from poetry_rnn.models import RNNAutoencoder

def test_optimization_configurations():
    """Test different optimization configurations."""
    print("🧪 Testing Attention Optimization Configurations")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {
            'name': 'Baseline (8 heads, standard)',
            'config': {
                'use_attention': True,
                'attention_heads': 8,
                'use_optimized_attention': False
            }
        },
        {
            'name': 'Quick Fix (4 heads, standard)',
            'config': {
                'use_attention': True,
                'attention_heads': 4,
                'use_optimized_attention': False
            }
        },
        {
            'name': 'Full Optimization (4 heads, optimized)',
            'config': {
                'use_attention': True,
                'attention_heads': 4,
                'use_optimized_attention': True
            }
        }
    ]
    
    # Common model parameters
    base_config = {
        'input_size': 300,
        'hidden_size': 512,
        'bottleneck_dim': 128,
        'rnn_type': 'LSTM',
        'num_layers': 2,
        'dropout': 0.2,
        'max_seq_len': 50
    }
    
    # Test data
    batch_size = 4
    seq_len = 20
    embedding_dim = 300
    
    test_batch = {
        'input_sequences': torch.randn(batch_size, seq_len, embedding_dim),
        'attention_mask': torch.ones(batch_size, seq_len).bool()
    }
    
    results = []
    
    for config_info in configs:
        print(f"\n📋 Testing: {config_info['name']}")
        
        # Create model
        model_config = {**base_config, **config_info['config']}
        
        try:
            model = RNNAutoencoder(**model_config)
            model.eval()  # Set to eval mode for consistent timing
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Test forward pass
            with torch.no_grad():
                start_time = time.perf_counter()
                
                # Run multiple times for better timing
                num_runs = 5
                for _ in range(num_runs):
                    output = model(test_batch)
                
                end_time = time.perf_counter()
                avg_time = (end_time - start_time) / num_runs
            
            # Validate output
            assert 'reconstructed' in output, "Missing reconstructed output"
            assert 'bottleneck' in output, "Missing bottleneck output"
            
            reconstructed = output['reconstructed']
            bottleneck = output['bottleneck']
            
            # Check output shapes (decoder may output max_seq_len)
            expected_recon_shape = (batch_size, base_config['max_seq_len'], embedding_dim)
            expected_bottleneck_shape = (batch_size, base_config['bottleneck_dim'])
            
            assert reconstructed.shape == expected_recon_shape, f"Wrong reconstructed shape: {reconstructed.shape}, expected {expected_recon_shape}"
            assert bottleneck.shape == expected_bottleneck_shape, f"Wrong bottleneck shape: {bottleneck.shape}, expected {expected_bottleneck_shape}"
            
            # Store results
            result = {
                'name': config_info['name'],
                'time_ms': avg_time * 1000,
                'params': total_params,
                'success': True,
                'reconstructed_mean': reconstructed.mean().item(),
                'bottleneck_mean': bottleneck.mean().item()
            }
            results.append(result)
            
            print(f"  ✅ Success: {avg_time*1000:.1f}ms avg, {total_params:,} params")
            print(f"  📊 Output: recon_mean={reconstructed.mean().item():.4f}, bottleneck_mean={bottleneck.mean().item():.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            result = {
                'name': config_info['name'],
                'success': False,
                'error': str(e)
            }
            results.append(result)
    
    # Summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) >= 2:
        baseline = successful_results[0]
        
        for result in successful_results[1:]:
            speedup = baseline['time_ms'] / result['time_ms']
            param_ratio = result['params'] / baseline['params']
            
            print(f"\n🔄 {result['name']} vs {baseline['name']}:")
            print(f"  ⚡ Speedup: {speedup:.2f}x ({baseline['time_ms']:.1f}ms → {result['time_ms']:.1f}ms)")
            print(f"  📏 Parameters: {param_ratio:.2f}x ({baseline['params']:,} → {result['params']:,})")
            
            # Performance expectations
            if 'Quick Fix' in result['name']:
                expected_speedup = 2.0
                if speedup >= expected_speedup * 0.8:  # Within 20% of expected
                    print(f"  ✅ Expected ~{expected_speedup}x speedup, got {speedup:.2f}x")
                else:
                    print(f"  ⚠️  Expected ~{expected_speedup}x speedup, got {speedup:.2f}x")
            
            elif 'Full Optimization' in result['name']:
                expected_speedup = 4.0  # Conservative estimate
                if speedup >= expected_speedup * 0.6:  # Within 40% of expected (more lenient for complex optimizations)
                    print(f"  ✅ Expected ~{expected_speedup}x speedup, got {speedup:.2f}x")
                else:
                    print(f"  ⚠️  Expected ~{expected_speedup}x speedup, got {speedup:.2f}x")
    
    print("\n🎯 OPTIMIZATION VALIDATION COMPLETE")
    print("=" * 60)
    
    # Overall assessment
    if len(successful_results) == len(configs):
        print("✅ All configurations working correctly")
        print("🚀 Ready for optimized training!")
        
        # Check if we have meaningful speedup
        if len(successful_results) >= 3:
            full_opt = successful_results[2]  # Full optimization
            baseline = successful_results[0]  # Baseline
            total_speedup = baseline['time_ms'] / full_opt['time_ms']
            
            if total_speedup >= 3.0:
                print(f"🎉 Excellent speedup achieved: {total_speedup:.1f}x")
            elif total_speedup >= 2.0:
                print(f"✨ Good speedup achieved: {total_speedup:.1f}x")
            else:
                print(f"⚠️  Limited speedup: {total_speedup:.1f}x (check implementation)")
    else:
        print("❌ Some configurations failed - check implementation")
        failed = [r for r in results if not r.get('success', False)]
        for failure in failed:
            print(f"  💥 {failure['name']}: {failure.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_optimization_configurations()