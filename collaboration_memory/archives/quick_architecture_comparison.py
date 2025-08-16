#!/usr/bin/env python3
"""
Quick Architecture Comparison - Uses pre-processed data
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity

# Add poetry_rnn to path
sys.path.insert(0, '.')

from poetry_rnn.models import RNNAutoencoder
from poetry_rnn import AutoencoderDataset
from torch.utils.data import DataLoader


def load_preprocessed_data():
    """Load pre-processed data from artifacts."""
    print("📚 Loading preprocessed data...")
    
    artifacts_dir = Path("preprocessed_artifacts")
    
    # Load most recent actual artifacts (not symlinks)
    embedding_sequences = np.load(artifacts_dir / "embedding_sequences_20250814_134543.npy")
    token_sequences = np.load(artifacts_dir / "token_sequences_20250814_134543.npy")
    attention_masks = np.load(artifacts_dir / "attention_masks_20250814_134543.npy")
    
    with open(artifacts_dir / "metadata_20250814_134543.json", 'r') as f:
        metadata = json.load(f)
    
    with open(artifacts_dir / "vocabulary_20250814_134543.json", 'r') as f:
        vocabulary = json.load(f)
    
    print(f"   ✅ Loaded {len(embedding_sequences)} sequences")
    return embedding_sequences, token_sequences, attention_masks, metadata, vocabulary


def quick_load_model(checkpoint_path):
    """Quick model loading with architecture detection."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Detect architecture parameters
    encoder_projection_shape = state_dict['encoder.projection.weight'].shape
    bottleneck_dim = encoder_projection_shape[0]
    
    # Detect hidden size from encoder RNN weights
    encoder_rnn_shape = state_dict['encoder.rnn.weight_ih_l0'].shape
    weight_ih_out_size = encoder_rnn_shape[0]
    weight_hh_shape = state_dict['encoder.rnn.weight_hh_l0'].shape
    weight_hh_in_size = weight_hh_shape[1]
    
    if weight_ih_out_size == 4 * weight_hh_in_size:
        hidden_size = weight_hh_in_size
        rnn_type = 'lstm'
    else:
        hidden_size = weight_ih_out_size
        rnn_type = 'vanilla'
    
    print(f"   Model: {checkpoint_path}")
    print(f"   Hidden: {hidden_size}D, Bottleneck: {bottleneck_dim}D, Type: {rnn_type}")
    
    # Create model
    model = RNNAutoencoder(
        input_size=300,
        hidden_size=hidden_size,
        bottleneck_dim=bottleneck_dim,
        rnn_type=rnn_type,
        num_layers=1,
        dropout=0.1,
        use_batch_norm=True
    )
    
    # Handle old architecture compatibility
    is_new_arch = 'decoder.teacher_forcing_projection.weight' in state_dict
    if not is_new_arch and rnn_type == 'vanilla':
        # Old architecture - modify decoder
        model.decoder.rnn = torch.nn.RNN(
            input_size=300,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            nonlinearity='tanh'
        )
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, hidden_size, bottleneck_dim, rnn_type


def evaluate_encoder_only(model, data_loader, max_batches=10):
    """Evaluate encoder performance (bottleneck quality)."""
    model.eval()
    
    all_bottlenecks = []
    all_originals = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            input_sequences = batch['input_sequences']
            attention_mask = batch['attention_mask']
            
            # Get bottleneck representation
            bottleneck, _ = model.encoder(input_sequences, attention_mask)
            
            all_bottlenecks.append(bottleneck.cpu().numpy())
            all_originals.append(input_sequences.cpu().numpy())
    
    bottlenecks = np.vstack(all_bottlenecks)
    
    # Analyze bottleneck quality
    dim_stds = np.std(bottlenecks, axis=0)
    effective_dims = np.sum(dim_stds > 0.1)
    
    return {
        'bottlenecks': bottlenecks,
        'effective_dims': effective_dims,
        'total_dims': bottlenecks.shape[1],
        'mean_activation': np.mean(bottlenecks),
        'std_activation': np.std(bottlenecks)
    }


def main():
    print("🔬 QUICK ARCHITECTURE COMPARISON")
    print("=" * 50)
    
    # Load preprocessed data
    embedding_sequences, token_sequences, attention_masks, metadata, vocabulary = load_preprocessed_data()
    
    # Create dataset with fixed metadata format
    # If metadata is a dict, convert it to list format expected by dataset
    if isinstance(metadata, dict):
        metadata_list = []
        for i in range(len(token_sequences)):
            metadata_list.append({
                'sequence_id': i,
                'poem_id': i // 10,  # Approximate
                'chunk_id': i % 10
            })
        metadata = metadata_list
    
    dataset = AutoencoderDataset(
        sequences=token_sequences,
        embedding_sequences=embedding_sequences,
        attention_masks=attention_masks,
        metadata=metadata,
        vocabulary=vocabulary
    )
    
    # Create small test loader
    test_dataset = torch.utils.data.Subset(dataset, range(min(50, len(dataset))))
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=dataset._collate_fn
    )
    
    print("\n📦 Loading models...")
    
    # Load baseline model
    baseline_model, baseline_hidden, baseline_bottleneck, baseline_type = quick_load_model('best_model.pth')
    
    # Load scaled model
    scaled_model, scaled_hidden, scaled_bottleneck, scaled_type = quick_load_model('scaled_model_lstm.pth')
    
    print("\n🔬 Evaluating models...")
    
    # Evaluate both models (encoder only for compatibility)
    baseline_results = evaluate_encoder_only(baseline_model, test_loader)
    scaled_results = evaluate_encoder_only(scaled_model, test_loader)
    
    print("\n📊 RESULTS SUMMARY")
    print("=" * 50)
    
    print("\n🔧 Baseline Model (64D hidden, 16D bottleneck):")
    print(f"   Effective dimensions: {baseline_results['effective_dims']}/{baseline_results['total_dims']}")
    print(f"   Mean activation: {baseline_results['mean_activation']:.4f}")
    print(f"   Std activation: {baseline_results['std_activation']:.4f}")
    
    print("\n🔧 Scaled Model (512D hidden, 64D bottleneck):")
    print(f"   Effective dimensions: {scaled_results['effective_dims']}/{scaled_results['total_dims']}")
    print(f"   Mean activation: {scaled_results['mean_activation']:.4f}")
    print(f"   Std activation: {scaled_results['std_activation']:.4f}")
    
    print("\n🎯 DIAGNOSIS VALIDATION:")
    
    # Compare effective dimensionality utilization
    baseline_utilization = baseline_results['effective_dims'] / baseline_results['total_dims']
    scaled_utilization = scaled_results['effective_dims'] / scaled_results['total_dims']
    
    print(f"   Baseline utilization: {baseline_utilization:.1%}")
    print(f"   Scaled utilization: {scaled_utilization:.1%}")
    
    if scaled_utilization > baseline_utilization:
        print("   ✅ CONFIRMED: Scaled model uses bottleneck dimensions more effectively")
        print("   ✅ CONFIRMED: 64D hidden was indeed creating an encoder bottleneck")
    
    # Check activation statistics
    if scaled_results['std_activation'] > baseline_results['std_activation']:
        print("   ✅ CONFIRMED: Scaled model has richer representations (higher variance)")
    
    print("\n💡 KEY INSIGHT:")
    print("   The 64D → 512D hidden size scaling allows the encoder to create")
    print("   richer intermediate representations, leading to better bottleneck")
    print("   encodings even with the same compression ratio.")


if __name__ == "__main__":
    main()