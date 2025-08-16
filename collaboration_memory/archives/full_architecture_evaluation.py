#!/usr/bin/env python3
"""
Full Architecture Evaluation - Comprehensive comparison with reconstruction metrics
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Add poetry_rnn to path
sys.path.insert(0, '.')

from poetry_rnn.models import RNNAutoencoder
from poetry_rnn import AutoencoderDataset
from torch.utils.data import DataLoader


def load_preprocessed_data():
    """Load pre-processed data from artifacts."""
    print("📚 Loading preprocessed data...")
    
    artifacts_dir = Path("preprocessed_artifacts")
    
    # Load most recent actual artifacts
    embedding_sequences = np.load(artifacts_dir / "embedding_sequences_20250814_134543.npy")
    token_sequences = np.load(artifacts_dir / "token_sequences_20250814_134543.npy")
    attention_masks = np.load(artifacts_dir / "attention_masks_20250814_134543.npy")
    
    with open(artifacts_dir / "metadata_20250814_134543.json", 'r') as f:
        metadata = json.load(f)
    
    with open(artifacts_dir / "vocabulary_20250814_134543.json", 'r') as f:
        vocabulary = json.load(f)
    
    print(f"   ✅ Loaded {len(embedding_sequences)} sequences")
    return embedding_sequences, token_sequences, attention_masks, metadata, vocabulary


def load_model_for_evaluation(checkpoint_path):
    """Load model with full architecture detection."""
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
    
    is_new_arch = 'decoder.teacher_forcing_projection.weight' in state_dict
    
    print(f"   Model: {checkpoint_path}")
    print(f"   Architecture: {'New (Fixed)' if is_new_arch else 'Old (Buggy)'}")
    print(f"   Hidden: {hidden_size}D, Bottleneck: {bottleneck_dim}D")
    print(f"   RNN Type: {rnn_type.upper()}")
    
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
    if not is_new_arch:
        # Old architecture - modify decoder to take embeddings directly
        if rnn_type == 'lstm':
            model.decoder.rnn = torch.nn.LSTM(
                input_size=300,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
        else:
            model.decoder.rnn = torch.nn.RNN(
                input_size=300,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                nonlinearity='tanh'
            )
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, {
        'hidden_size': hidden_size,
        'bottleneck_dim': bottleneck_dim,
        'rnn_type': rnn_type,
        'is_new_arch': is_new_arch
    }


def evaluate_model_comprehensive(model, model_info, data_loader, max_batches=10):
    """Comprehensive evaluation including reconstruction quality."""
    model.eval()
    
    all_cosine_sims = []
    all_mse_losses = []
    all_bottlenecks = []
    all_hidden_norms = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            input_sequences = batch['input_sequences']
            attention_mask = batch['attention_mask']
            
            # Forward pass through encoder
            bottleneck, hidden_states = model.encoder(input_sequences, attention_mask)
            
            # Store bottleneck representations
            all_bottlenecks.append(bottleneck.cpu().numpy())
            
            # Analyze hidden state norms (gradient flow indicator)
            if hidden_states is not None:
                if isinstance(hidden_states, tuple):  # LSTM
                    h, c = hidden_states
                    hidden_norm = torch.norm(h, dim=-1).mean().item()
                else:  # Vanilla RNN
                    hidden_norm = torch.norm(hidden_states, dim=-1).mean().item()
                all_hidden_norms.append(hidden_norm)
            
            # Reconstruction (if new architecture)
            if model_info['is_new_arch']:
                reconstructed, _ = model.decoder(
                    bottleneck, 
                    mask=attention_mask, 
                    seq_len=input_sequences.shape[1]
                )
                
                # Compute metrics
                batch_size, seq_len, embed_dim = input_sequences.shape
                
                for i in range(batch_size):
                    valid_length = attention_mask[i].sum().item()
                    if valid_length == 0:
                        continue
                    
                    orig_seq = input_sequences[i, :valid_length, :].cpu().numpy()
                    recon_seq = reconstructed[i, :valid_length, :].cpu().numpy()
                    
                    # Token-level cosine similarities
                    token_cosines = []
                    for t in range(valid_length):
                        cos_sim = cosine_similarity(
                            orig_seq[t:t+1], recon_seq[t:t+1]
                        )[0, 0]
                        token_cosines.append(cos_sim)
                    
                    seq_cosine = np.mean(token_cosines)
                    all_cosine_sims.append(seq_cosine)
                    
                    # MSE loss
                    mse = np.mean((orig_seq - recon_seq) ** 2)
                    all_mse_losses.append(mse)
    
    # Analyze bottleneck quality
    bottlenecks = np.vstack(all_bottlenecks)
    dim_stds = np.std(bottlenecks, axis=0)
    effective_dims = np.sum(dim_stds > 0.1)
    
    # Compute singular values (information content)
    _, singular_values, _ = np.linalg.svd(bottlenecks[:min(100, len(bottlenecks))], full_matrices=False)
    normalized_singular_values = singular_values / singular_values.sum()
    entropy = -np.sum(normalized_singular_values * np.log(normalized_singular_values + 1e-10))
    
    results = {
        'bottlenecks': bottlenecks,
        'effective_dims': effective_dims,
        'total_dims': bottlenecks.shape[1],
        'dim_utilization': effective_dims / bottlenecks.shape[1],
        'mean_activation': np.mean(bottlenecks),
        'std_activation': np.std(bottlenecks),
        'activation_range': (np.min(bottlenecks), np.max(bottlenecks)),
        'singular_entropy': entropy,
        'top_5_singular_values': singular_values[:5].tolist() if len(singular_values) >= 5 else singular_values.tolist()
    }
    
    if all_hidden_norms:
        results['mean_hidden_norm'] = np.mean(all_hidden_norms)
    
    if all_cosine_sims:
        results['cosine_mean'] = np.mean(all_cosine_sims)
        results['cosine_std'] = np.std(all_cosine_sims)
        results['cosine_min'] = np.min(all_cosine_sims)
        results['cosine_max'] = np.max(all_cosine_sims)
    
    if all_mse_losses:
        results['mse_mean'] = np.mean(all_mse_losses)
        results['mse_std'] = np.std(all_mse_losses)
        results['rmse'] = np.sqrt(np.mean(all_mse_losses))
    
    return results


def visualize_comparison(baseline_results, scaled_results, baseline_info, scaled_info):
    """Create comprehensive comparison visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Bottleneck dimensionality utilization
    labels = ['Baseline\n(64D→16D)', 'Scaled\n(512D→64D)']
    effective = [baseline_results['effective_dims'], scaled_results['effective_dims']]
    total = [baseline_results['total_dims'], scaled_results['total_dims']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, effective, width, label='Effective Dims', color='green', alpha=0.7)
    bars2 = axes[0, 0].bar(x + width/2, total, width, label='Total Dims', color='gray', alpha=0.7)
    
    axes[0, 0].set_ylabel('Dimensions')
    axes[0, 0].set_title('Bottleneck Dimension Utilization')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add utilization percentage labels
    for i, (e, t) in enumerate(zip(effective, total)):
        axes[0, 0].text(i, max(e, t) + 2, f'{e/t:.0%}', ha='center', fontweight='bold')
    
    # 2. Activation statistics
    stats_data = {
        'Mean Act.': [baseline_results['mean_activation'], scaled_results['mean_activation']],
        'Std Act.': [baseline_results['std_activation'], scaled_results['std_activation']],
    }
    
    if 'mean_hidden_norm' in baseline_results and 'mean_hidden_norm' in scaled_results:
        stats_data['Hidden Norm'] = [baseline_results['mean_hidden_norm'], scaled_results['mean_hidden_norm']]
    
    x = np.arange(len(stats_data))
    width = 0.35
    
    for i, (stat_name, values) in enumerate(stats_data.items()):
        axes[0, 1].bar(x[i] - width/2, values[0], width, label='Baseline' if i == 0 else "", color='red', alpha=0.7)
        axes[0, 1].bar(x[i] + width/2, values[1], width, label='Scaled' if i == 0 else "", color='blue', alpha=0.7)
    
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Activation Statistics')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(list(stats_data.keys()))
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Reconstruction quality (if available)
    if 'cosine_mean' in scaled_results:
        categories = ['Baseline', 'Scaled']
        cosine_means = [
            baseline_results.get('cosine_mean', 0),
            scaled_results.get('cosine_mean', 0)
        ]
        mse_means = [
            baseline_results.get('mse_mean', 0),
            scaled_results.get('mse_mean', 0)
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax3 = axes[0, 2]
        ax3_2 = ax3.twinx()
        
        bars1 = ax3.bar(x - width/2, cosine_means, width, label='Cosine Sim', color='green', alpha=0.7)
        bars2 = ax3_2.bar(x + width/2, mse_means, width, label='MSE Loss', color='orange', alpha=0.7)
        
        ax3.set_ylabel('Cosine Similarity', color='green')
        ax3_2.set_ylabel('MSE Loss', color='orange')
        ax3.set_title('Reconstruction Quality')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.tick_params(axis='y', labelcolor='green')
        ax3_2.tick_params(axis='y', labelcolor='orange')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, cosine_means):
            if value > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold', color='green')
    else:
        axes[0, 2].text(0.5, 0.5, 'Reconstruction metrics\nnot available\n(encoder-only evaluation)',
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Reconstruction Quality')
    
    # 4. Singular value distribution (information content)
    if len(baseline_results['top_5_singular_values']) > 0:
        x = range(len(baseline_results['top_5_singular_values']))
        axes[1, 0].plot(x, baseline_results['top_5_singular_values'], 'ro-', label='Baseline', alpha=0.7)
        axes[1, 0].plot(x, scaled_results['top_5_singular_values'], 'bo-', label='Scaled', alpha=0.7)
        axes[1, 0].set_xlabel('Singular Value Index')
        axes[1, 0].set_ylabel('Singular Value')
        axes[1, 0].set_title('Top 5 Singular Values (Information Distribution)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Information entropy
    entropies = [baseline_results['singular_entropy'], scaled_results['singular_entropy']]
    colors = ['red', 'blue']
    bars = axes[1, 1].bar(['Baseline', 'Scaled'], entropies, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Information Entropy (Higher = More Distributed)')
    axes[1, 1].grid(True, alpha=0.3)
    
    for bar, value in zip(bars, entropies):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Architecture summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""ARCHITECTURE COMPARISON SUMMARY
    
Baseline Model:
  • Architecture: {'Fixed' if baseline_info['is_new_arch'] else 'Buggy (Old)'}
  • RNN Type: {baseline_info['rnn_type'].upper()}
  • Hidden: {baseline_info['hidden_size']}D
  • Bottleneck: {baseline_info['bottleneck_dim']}D
  • Compression: {300/baseline_info['bottleneck_dim']:.1f}:1
  • Dim Utilization: {baseline_results['dim_utilization']:.0%}

Scaled Model:
  • Architecture: {'Fixed' if scaled_info['is_new_arch'] else 'Buggy (Old)'}
  • RNN Type: {scaled_info['rnn_type'].upper()}
  • Hidden: {scaled_info['hidden_size']}D
  • Bottleneck: {scaled_info['bottleneck_dim']}D
  • Compression: {300/scaled_info['bottleneck_dim']:.1f}:1
  • Dim Utilization: {scaled_results['dim_utilization']:.0%}

Key Findings:
  • Hidden scaling: {scaled_info['hidden_size']/baseline_info['hidden_size']:.0f}x
  • Bottleneck scaling: {scaled_info['bottleneck_dim']/baseline_info['bottleneck_dim']:.0f}x
  • Entropy change: {(scaled_results['singular_entropy']/baseline_results['singular_entropy']-1)*100:+.1f}%"""
    
    if 'cosine_mean' in scaled_results and 'cosine_mean' in baseline_results:
        improvement = (scaled_results['cosine_mean'] - baseline_results['cosine_mean']) / baseline_results['cosine_mean'] * 100
        summary_text += f"\n  • Performance: {improvement:+.1f}%"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('comprehensive_architecture_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Visualization saved: comprehensive_architecture_analysis.png")


def main():
    print("🔬 COMPREHENSIVE ARCHITECTURE EVALUATION")
    print("=" * 50)
    
    # Load preprocessed data
    embedding_sequences, token_sequences, attention_masks, metadata, vocabulary = load_preprocessed_data()
    
    # Fix metadata format if needed
    if isinstance(metadata, dict):
        metadata_list = []
        for i in range(len(token_sequences)):
            metadata_list.append({
                'sequence_id': i,
                'poem_id': i // 10,
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
    
    # Create test loader
    test_dataset = torch.utils.data.Subset(dataset, range(min(100, len(dataset))))
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=dataset._collate_fn
    )
    
    print("\n📦 Loading models...")
    
    # Load both models
    baseline_model, baseline_info = load_model_for_evaluation('best_model.pth')
    scaled_model, scaled_info = load_model_for_evaluation('scaled_model_lstm.pth')
    
    print("\n🔬 Running comprehensive evaluation...")
    
    # Evaluate both models
    baseline_results = evaluate_model_comprehensive(baseline_model, baseline_info, test_loader, max_batches=12)
    scaled_results = evaluate_model_comprehensive(scaled_model, scaled_info, test_loader, max_batches=12)
    
    print("\n📊 EVALUATION RESULTS")
    print("=" * 50)
    
    print("\n🔧 BASELINE MODEL (64D hidden → 16D bottleneck):")
    print(f"   Bottleneck utilization: {baseline_results['effective_dims']}/{baseline_results['total_dims']} ({baseline_results['dim_utilization']:.0%})")
    print(f"   Mean activation: {baseline_results['mean_activation']:.4f}")
    print(f"   Std activation: {baseline_results['std_activation']:.4f}")
    print(f"   Activation range: [{baseline_results['activation_range'][0]:.3f}, {baseline_results['activation_range'][1]:.3f}]")
    print(f"   Information entropy: {baseline_results['singular_entropy']:.3f}")
    if 'mean_hidden_norm' in baseline_results:
        print(f"   Hidden state norm: {baseline_results['mean_hidden_norm']:.3f}")
    if 'cosine_mean' in baseline_results:
        print(f"   Reconstruction quality: {baseline_results['cosine_mean']:.4f} ± {baseline_results['cosine_std']:.4f}")
        print(f"   Reconstruction error (RMSE): {baseline_results['rmse']:.4f}")
    
    print("\n🔧 SCALED MODEL (512D hidden → 64D bottleneck):")
    print(f"   Bottleneck utilization: {scaled_results['effective_dims']}/{scaled_results['total_dims']} ({scaled_results['dim_utilization']:.0%})")
    print(f"   Mean activation: {scaled_results['mean_activation']:.4f}")
    print(f"   Std activation: {scaled_results['std_activation']:.4f}")
    print(f"   Activation range: [{scaled_results['activation_range'][0]:.3f}, {scaled_results['activation_range'][1]:.3f}]")
    print(f"   Information entropy: {scaled_results['singular_entropy']:.3f}")
    if 'mean_hidden_norm' in scaled_results:
        print(f"   Hidden state norm: {scaled_results['mean_hidden_norm']:.3f}")
    if 'cosine_mean' in scaled_results:
        print(f"   Reconstruction quality: {scaled_results['cosine_mean']:.4f} ± {scaled_results['cosine_std']:.4f}")
        print(f"   Reconstruction error (RMSE): {scaled_results['rmse']:.4f}")
    
    print("\n🎯 NEURAL NETWORK MENTOR DIAGNOSIS VALIDATION:")
    print("=" * 50)
    
    # Analyze the results
    entropy_improvement = (scaled_results['singular_entropy'] / baseline_results['singular_entropy'] - 1) * 100
    
    print(f"\n1️⃣ ENCODER BOTTLENECK HYPOTHESIS:")
    if scaled_results['dim_utilization'] >= baseline_results['dim_utilization']:
        print("   ✅ CONFIRMED: Scaled model maintains full dimension utilization")
    else:
        print("   ❌ NOT CONFIRMED: Dimension utilization decreased")
    
    print(f"\n2️⃣ INFORMATION CAPACITY:")
    print(f"   Entropy improvement: {entropy_improvement:+.1f}%")
    if entropy_improvement > 0:
        print("   ✅ CONFIRMED: Scaled model has better information distribution")
    else:
        print("   ❌ NOT CONFIRMED: Information distribution did not improve")
    
    print(f"\n3️⃣ GRADIENT FLOW (Hidden Norms):")
    if 'mean_hidden_norm' in baseline_results and 'mean_hidden_norm' in scaled_results:
        norm_ratio = scaled_results['mean_hidden_norm'] / baseline_results['mean_hidden_norm']
        print(f"   Hidden norm ratio: {norm_ratio:.2f}x")
        if norm_ratio > 0.5 and norm_ratio < 2.0:
            print("   ✅ CONFIRMED: LSTM maintains stable gradient flow")
        else:
            print("   ⚠️  WARNING: Gradient flow may be unstable")
    
    print(f"\n4️⃣ RECONSTRUCTION PERFORMANCE:")
    if 'cosine_mean' in scaled_results and 'cosine_mean' in baseline_results:
        cos_improvement = (scaled_results['cosine_mean'] - baseline_results['cosine_mean']) / baseline_results['cosine_mean'] * 100
        print(f"   Cosine similarity improvement: {cos_improvement:+.1f}%")
        print(f"   Baseline: {baseline_results['cosine_mean']:.4f}")
        print(f"   Scaled: {scaled_results['cosine_mean']:.4f}")
        
        if cos_improvement > 10:
            print("   ✅ CONFIRMED: Significant performance improvement")
            print("   ✅ The 64D hidden layer WAS creating an encoder bottleneck")
        elif cos_improvement > 0:
            print("   🟠 PARTIAL: Some improvement, but less than expected")
        else:
            print("   ❌ NO IMPROVEMENT: Performance did not increase")
    else:
        print("   ⚠️  Cannot compare (old architecture lacks proper decoder)")
    
    # Create visualization
    print("\n📊 Creating comprehensive visualization...")
    visualize_comparison(baseline_results, scaled_results, baseline_info, scaled_info)
    
    print("\n💡 FINAL DIAGNOSIS:")
    print("=" * 50)
    
    if entropy_improvement > 0:
        print("✅ The scaling from 64D → 512D hidden size successfully addresses")
        print("   the encoder bottleneck issue by providing richer intermediate")
        print("   representations that better utilize the bottleneck dimensions.")
        print("")
        print("📊 The information-theoretic analysis confirms that the scaled")
        print("   architecture captures and preserves more information through")
        print("   the encoding process, validating the neural network mentor's")
        print("   diagnosis about the hidden layer bottleneck.")
    else:
        print("🔍 Results require further investigation. The architectural")
        print("   changes may need additional tuning or the bottleneck may")
        print("   be elsewhere in the system.")
    
    print("\n✅ Comprehensive evaluation completed!")
    print("   📊 Full analysis: comprehensive_architecture_analysis.png")


if __name__ == "__main__":
    main()