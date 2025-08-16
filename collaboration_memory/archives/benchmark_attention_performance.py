#!/usr/bin/env python3
"""
Benchmark script to analyze attention performance bottlenecks.

This script profiles the attention mechanism to identify specific bottlenecks
and measure the impact of various optimizations.
"""

import torch
import torch.nn as nn
import time
import sys
import numpy as np
from typing import Dict, Tuple
import torch.profiler as profiler

sys.path.insert(0, '.')

from poetry_rnn.models.attention import MultiHeadAttention
from poetry_rnn.models.attention_decoder import AttentionEnhancedDecoder


def benchmark_attention_operations(
    batch_size: int = 32,
    seq_len: int = 50,
    d_model: int = 512,
    num_heads: int = 8,
    num_iterations: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, float]:
    """
    Benchmark individual attention operations to identify bottlenecks.
    """
    print(f"\n🔬 Benchmarking Attention Operations")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}, Seq length: {seq_len}")
    print(f"   Model dim: {d_model}, Heads: {num_heads}")
    print(f"   Iterations: {num_iterations}\n")
    
    results = {}
    
    # Create test data
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # 1. Standard MultiHeadAttention
    print("Testing standard MultiHeadAttention...")
    attention = MultiHeadAttention(d_model, num_heads).to(device)
    attention.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = attention(x, x, x)
    
    # Time forward pass
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            output, _ = attention(x, x, x)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.perf_counter()
    
    results['standard_attention_ms'] = (end - start) * 1000 / num_iterations
    print(f"   Standard attention: {results['standard_attention_ms']:.2f} ms/iteration")
    
    # 2. Test individual components
    print("\nProfiling attention components...")
    
    # Linear projections
    W_q = nn.Linear(d_model, d_model, bias=False).to(device)
    W_k = nn.Linear(d_model, d_model, bias=False).to(device)
    W_v = nn.Linear(d_model, d_model, bias=False).to(device)
    W_o = nn.Linear(d_model, d_model, bias=False).to(device)
    
    with torch.no_grad():
        # Time Q, K, V projections
        start = time.perf_counter()
        for _ in range(num_iterations):
            Q = W_q(x)
            K = W_k(x)
            V = W_v(x)
        torch.cuda.synchronize() if device == 'cuda' else None
        end = time.perf_counter()
        results['qkv_projection_ms'] = (end - start) * 1000 / num_iterations
        
        # Time reshaping for multi-head
        Q = W_q(x)
        K = W_k(x) 
        V = W_v(x)
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            Q_heads = Q.view(batch_size, seq_len, num_heads, d_model // num_heads).transpose(1, 2)
            K_heads = K.view(batch_size, seq_len, num_heads, d_model // num_heads).transpose(1, 2)
            V_heads = V.view(batch_size, seq_len, num_heads, d_model // num_heads).transpose(1, 2)
        torch.cuda.synchronize() if device == 'cuda' else None
        end = time.perf_counter()
        results['reshape_ms'] = (end - start) * 1000 / num_iterations
        
        # Time attention scores computation
        Q_heads = Q.view(batch_size, seq_len, num_heads, d_model // num_heads).transpose(1, 2)
        K_heads = K.view(batch_size, seq_len, num_heads, d_model // num_heads).transpose(1, 2)
        V_heads = V.view(batch_size, seq_len, num_heads, d_model // num_heads).transpose(1, 2)
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / (d_model // num_heads) ** 0.5
            attention_weights = torch.softmax(scores, dim=-1)
            attention_output = torch.matmul(attention_weights, V_heads)
        torch.cuda.synchronize() if device == 'cuda' else None
        end = time.perf_counter()
        results['attention_compute_ms'] = (end - start) * 1000 / num_iterations
        
        # Time concatenation and output projection
        attention_output = torch.matmul(attention_weights, V_heads)
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            concat = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            output = W_o(concat)
        torch.cuda.synchronize() if device == 'cuda' else None
        end = time.perf_counter()
        results['concat_output_ms'] = (end - start) * 1000 / num_iterations
    
    print(f"   QKV projections: {results['qkv_projection_ms']:.2f} ms")
    print(f"   Reshape operations: {results['reshape_ms']:.2f} ms")
    print(f"   Attention computation: {results['attention_compute_ms']:.2f} ms")
    print(f"   Concat + output projection: {results['concat_output_ms']:.2f} ms")
    
    total_components = (results['qkv_projection_ms'] + results['reshape_ms'] + 
                       results['attention_compute_ms'] + results['concat_output_ms'])
    print(f"   Total components: {total_components:.2f} ms")
    print(f"   Overhead: {results['standard_attention_ms'] - total_components:.2f} ms")
    
    # 3. Test with different batch sizes
    print("\nTesting batch size scaling...")
    for bs in [8, 16, 32, 64]:
        x_batch = torch.randn(bs, seq_len, d_model, device=device)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(50):
                _ = attention(x_batch, x_batch, x_batch)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end = time.perf_counter()
        
        ms_per_iter = (end - start) * 1000 / 50
        results[f'batch_{bs}_ms'] = ms_per_iter
        print(f"   Batch size {bs}: {ms_per_iter:.2f} ms/iteration")
    
    # 4. Test with different sequence lengths
    print("\nTesting sequence length scaling...")
    for sl in [10, 25, 50, 100]:
        x_seq = torch.randn(batch_size, sl, d_model, device=device)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(50):
                _ = attention(x_seq, x_seq, x_seq)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end = time.perf_counter()
        
        ms_per_iter = (end - start) * 1000 / 50
        results[f'seqlen_{sl}_ms'] = ms_per_iter
        print(f"   Sequence length {sl}: {ms_per_iter:.2f} ms/iteration")
    
    return results


def compare_decoder_performance(
    batch_size: int = 32,
    seq_len: int = 50,
    num_iterations: int = 50,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, float]:
    """
    Compare standard decoder vs attention-enhanced decoder performance.
    """
    from poetry_rnn.models.decoder import RNNDecoder
    
    print(f"\n🔬 Comparing Decoder Performance")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}, Seq length: {seq_len}")
    print(f"   Iterations: {num_iterations}\n")
    
    results = {}
    
    # Test data
    bottleneck = torch.randn(batch_size, 128, device=device)
    encoder_states = torch.randn(batch_size, seq_len, 512, device=device)
    target = torch.randn(batch_size, seq_len, 300, device=device)
    
    # 1. Standard RNN Decoder
    print("Testing standard RNN decoder...")
    standard_decoder = RNNDecoder(
        bottleneck_dim=128,
        hidden_size=512,
        output_size=300,
        max_seq_len=seq_len,
        rnn_type='LSTM',
        num_layers=2
    ).to(device)
    standard_decoder.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = standard_decoder(bottleneck, target_sequences=target, teacher_forcing_ratio=0.0)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = standard_decoder(bottleneck, target_sequences=target, teacher_forcing_ratio=0.0)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.perf_counter()
    
    results['standard_decoder_ms'] = (end - start) * 1000 / num_iterations
    print(f"   Standard decoder: {results['standard_decoder_ms']:.2f} ms/iteration")
    
    # 2. Attention-Enhanced Decoder
    print("\nTesting attention-enhanced decoder...")
    attention_decoder = AttentionEnhancedDecoder(
        bottleneck_dim=128,
        hidden_size=512,
        output_size=300,
        encoder_hidden_size=512,
        max_seq_len=seq_len,
        rnn_type='LSTM',
        num_layers=2,
        attention_heads=8
    ).to(device)
    attention_decoder.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = attention_decoder(
                bottleneck, 
                encoder_hidden_states=encoder_states,
                target_sequences=target,
                teacher_forcing_ratio=0.0
            )
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = attention_decoder(
                bottleneck,
                encoder_hidden_states=encoder_states,
                target_sequences=target,
                teacher_forcing_ratio=0.0
            )
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.perf_counter()
    
    results['attention_decoder_ms'] = (end - start) * 1000 / num_iterations
    print(f"   Attention decoder: {results['attention_decoder_ms']:.2f} ms/iteration")
    
    # Calculate slowdown
    slowdown = results['attention_decoder_ms'] / results['standard_decoder_ms']
    results['slowdown_factor'] = slowdown
    print(f"\n   Slowdown factor: {slowdown:.2f}x")
    print(f"   Additional time per iteration: {results['attention_decoder_ms'] - results['standard_decoder_ms']:.2f} ms")
    
    return results


def profile_attention_memory():
    """
    Profile memory usage of attention mechanism.
    """
    print("\n🔬 Profiling Memory Usage")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test configurations
    configs = [
        (8, 50, 512, 8),   # Small batch
        (32, 50, 512, 8),  # Medium batch
        (64, 50, 512, 8),  # Large batch
        (32, 100, 512, 8), # Long sequence
        (32, 50, 1024, 8), # Larger model
    ]
    
    for batch_size, seq_len, d_model, num_heads in configs:
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        attention = MultiHeadAttention(d_model, num_heads).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # Forward pass
        output, weights = attention(x, x, x, return_attention=True)
        
        if device == 'cuda':
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"   Config (B={batch_size}, L={seq_len}, D={d_model}): {peak_memory_mb:.1f} MB")
        else:
            # Estimate memory for CPU
            param_memory = sum(p.numel() * 4 for p in attention.parameters()) / 1024 / 1024
            activation_memory = (batch_size * seq_len * d_model * 4 * 10) / 1024 / 1024  # Rough estimate
            print(f"   Config (B={batch_size}, L={seq_len}, D={d_model}): ~{param_memory + activation_memory:.1f} MB (estimated)")


if __name__ == "__main__":
    print("=" * 60)
    print("ATTENTION PERFORMANCE BENCHMARKING")
    print("=" * 60)
    
    # Run benchmarks
    attention_results = benchmark_attention_operations()
    decoder_results = compare_decoder_performance()
    profile_attention_memory()
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    print("\n📊 Key Findings:")
    print(f"   Attention overhead per forward pass: {attention_results['standard_attention_ms']:.2f} ms")
    print(f"   Decoder slowdown with attention: {decoder_results['slowdown_factor']:.2f}x")
    print(f"   Additional time per batch: {decoder_results['attention_decoder_ms'] - decoder_results['standard_decoder_ms']:.2f} ms")
    
    print("\n🎯 Bottleneck Analysis:")
    total_time = attention_results['standard_attention_ms']
    print(f"   QKV Projections: {attention_results['qkv_projection_ms']/total_time*100:.1f}%")
    print(f"   Attention Computation: {attention_results['attention_compute_ms']/total_time*100:.1f}%")
    print(f"   Reshape Operations: {attention_results['reshape_ms']/total_time*100:.1f}%")
    print(f"   Output Projection: {attention_results['concat_output_ms']/total_time*100:.1f}%")
    
    print("\n✅ Optimization Recommendations:")
    print("   1. Use Flash Attention or fused operations for attention computation")
    print("   2. Optimize reshape operations with memory-efficient views")
    print("   3. Consider reducing attention heads (8 → 4) for 2x speedup")
    print("   4. Use torch.compile() for JIT optimization")
    print("   5. Implement gradient checkpointing to reduce memory")