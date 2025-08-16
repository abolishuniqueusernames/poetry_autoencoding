#!/usr/bin/env python3
"""
Scaled Poetry RNN Autoencoder Training Script - Fixed Architecture

This script trains the scaled RNN autoencoder with 512D hidden dimensions
to fix the encoder bottleneck issue identified by neural-network-mentor.

Architecture: 300D → 512D → 64D → 512D → 300D
Expected improvement: 0.624 → 0.95+ cosine similarity

Usage:
    python train_scaled_architecture.py --rnn-type vanilla
    python train_scaled_architecture.py --rnn-type lstm

Key Changes from Baseline:
- Encoder hidden: 64D → 512D (eliminates information bottleneck)
- Decoder hidden: 64D → 512D (maintains encoder-decoder symmetry)
- Parameters: ~150K → ~1.4M (acceptable for expected performance gain)
- Expected training time: ~5-10 minutes (depending on hardware)
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# Add poetry_rnn to path
sys.path.insert(0, '.')

from poetry_rnn import (
    PoetryPreprocessor, 
    AutoencoderDataset,
    Config
)
from poetry_rnn.models import RNNAutoencoder
from poetry_rnn.training import (
    CurriculumScheduler,
    GradientMonitor,
    TrainingMonitor
)
from poetry_rnn.training.losses import MaskedMSELoss


def setup_scaled_training(config, rnn_type='vanilla'):
    """Set up scaled architecture for training."""
    print(f"🔧 Setting up SCALED architecture training components ({rnn_type.upper()})...")
    
    # Create scaled model with 512D hidden dimensions
    autoencoder = RNNAutoencoder(
        input_size=300,           # GLoVe embedding dimension
        hidden_size=512,          # SCALED: 64 → 512 (fixes bottleneck)
        bottleneck_dim=64,        # SCALED: 16D → 64D to reduce compression ratio
        rnn_type=rnn_type,        # Configurable: vanilla or lstm
        num_layers=1,             # Single layer for comparison
        dropout=0.1,              # Light regularization
        use_batch_norm=True       # Stabilizes training
    )
    
    param_count = sum(p.numel() for p in autoencoder.parameters())
    print(f"   📊 SCALED Model: {param_count:,} parameters")
    print(f"   📈 Parameter increase: {param_count / 150000:.1f}x vs baseline (150K)")
    print(f"   🗜️  Compression: {300/64:.1f}x (300D → 64D)")
    print(f"   🔧 Architecture: 300D → 512D → 64D → 512D → 300D")
    
    # Optimizer with properly scaled learning rate (1/sqrt scaling for 1.4M vs 150K params)
    optimizer = optim.Adam(autoencoder.parameters(), lr=2.5e-4, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3
    )
    
    # Loss function for variable-length sequences
    criterion = MaskedMSELoss()
    
    # Curriculum learning scheduler
    curriculum = CurriculumScheduler(adaptive=True, patience=2)
    
    # Gradient monitoring with adaptive clipping (slightly higher threshold for larger model)
    grad_monitor = GradientMonitor(
        model=autoencoder,
        clip_value=5.0,
        adaptive=True,
        vanishing_threshold=1e-6,
        exploding_threshold=15.0  # Higher threshold for 512D model
    )
    
    # Training progress monitor with scaled model suffix
    train_monitor = TrainingMonitor(log_dir='training_logs_scaled')
    
    print("✅ SCALED training setup complete!")
    print(f"   📋 Curriculum: {len(curriculum.phases)} phases")
    print(f"   📎 Adaptive gradient clipping: ON (threshold: 15.0)")
    print(f"   📊 Teacher forcing: Curriculum-integrated")
    print(f"   🎯 Target: Fix 0.624 cosine similarity → 0.95+")
    
    return autoencoder, optimizer, scheduler, criterion, curriculum, grad_monitor, train_monitor


def load_data(config):
    """Load and prepare poetry data for training."""
    print("\n📚 Loading poetry data...")
    
    # Check if we have preprocessed data
    data_path = "dataset_poetry/multi_poem_dbbc_collection.json"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("   Please ensure you have poetry data available.")
        print("   You can use any JSON file with poems in the format:")
        print("   [{'content': 'poem text here'}, ...]")
        return None, None, None
    
    # Initialize preprocessor
    preprocessor = PoetryPreprocessor(config=config)
    
    # Process poems (this may take a moment)
    print("   🔄 Processing poems with GLoVe embeddings...")
    results = preprocessor.process_poems(
        data_path, 
        save_artifacts=True,  # Save for faster reloading
    )
    
    # Create PyTorch datasets
    dataset = AutoencoderDataset(
        sequences=results['sequences'],
        embedding_sequences=results['embedding_sequences'],
        attention_masks=results['attention_masks'],
        metadata=results['metadata'],
        vocabulary=results['vocabulary']
    )
    
    # Split data: 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # Same batch size for fair comparison
        shuffle=True, 
        collate_fn=dataset._collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=dataset._collate_fn
    )
    
    print(f"✅ Data loaded successfully!")
    print(f"   📊 Total sequences: {len(dataset)}")
    print(f"   🏋️  Training: {len(train_dataset)} sequences")
    print(f"   🔍 Validation: {len(val_dataset)} sequences")
    print(f"   📏 Window size: {results['config']['window_size']}")
    print(f"   📝 Chunks processed: {results['stats']['sequences_generated']}")
    
    return train_loader, val_loader, results['stats']


def train_epoch(autoencoder, train_loader, optimizer, criterion, curriculum, 
                grad_monitor, epoch):
    """Train for one epoch with curriculum learning."""
    autoencoder.train()
    total_loss = 0.0
    batch_count = 0
    
    # Get current teacher forcing ratio from curriculum
    teacher_forcing_ratio = curriculum.get_teacher_forcing_ratio()
    
    print(f"   📚 Phase {curriculum.current_phase_idx + 1}: "
          f"max_len={curriculum.get_max_length()}, "
          f"tf_ratio={teacher_forcing_ratio:.3f}")
    
    for batch_idx, batch in enumerate(train_loader):
        # Truncate sequences according to curriculum
        batch = curriculum.truncate_batch(batch)
        
        # Extract batch components
        input_sequences = batch['input_sequences']  # [batch, seq_len, embed_dim]
        attention_mask = batch['attention_mask']    # [batch, seq_len]
        
        # Forward pass with teacher forcing
        optimizer.zero_grad()
        
        # Encode to bottleneck
        bottleneck, encoder_hidden = autoencoder.encoder(input_sequences, attention_mask)
        
        # Decode with curriculum-adaptive teacher forcing
        # Get current sequence length from truncated batch
        current_seq_len = input_sequences.shape[1]
        if current_seq_len == 0:
            print(f"⚠️  Warning: Empty sequence in batch {batch_idx}, skipping...")
            continue
            
        reconstructed, decoder_hidden = autoencoder.decoder(
            bottleneck, 
            target_sequences=input_sequences,
            mask=attention_mask,
            seq_len=current_seq_len,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        # Compute loss (handles variable lengths with masking)
        loss = criterion(reconstructed, input_sequences, attention_mask)
        
        # Backward pass with gradient monitoring
        loss.backward()
        
        # Monitor and clip gradients adaptively
        grad_norm = grad_monitor.clip_and_monitor()
        
        # Optimizer step
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # Print diagnostics every 20 batches
        if batch_idx % 20 == 0 and batch_idx > 0:
            diagnostics = grad_monitor.diagnose_gradient_issues()
            if len(diagnostics) > 0 and not diagnostics[0].startswith("✅"):
                print(f"      Batch {batch_idx}: {diagnostics[0]}")
    
    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    return avg_loss


def validate(autoencoder, val_loader, criterion, curriculum):
    """Validate the model."""
    autoencoder.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Use current curriculum length for consistency
            batch = curriculum.truncate_batch(batch)
            
            input_sequences = batch['input_sequences']
            attention_mask = batch['attention_mask']
            
            # Forward pass (no teacher forcing during validation)
            current_seq_len = input_sequences.shape[1]
            bottleneck, _ = autoencoder.encoder(input_sequences, attention_mask)
            reconstructed, _ = autoencoder.decoder(bottleneck, mask=attention_mask, seq_len=current_seq_len)
            
            loss = criterion(reconstructed, input_sequences, attention_mask)
            total_loss += loss.item()
            batch_count += 1
    
    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    return avg_loss


def save_model_checkpoint(autoencoder, optimizer, curriculum, epoch, loss, filepath):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': autoencoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'curriculum_state': {
            'current_phase_idx': curriculum.current_phase_idx,
            'epochs_in_phase': curriculum.epochs_in_phase,
            'total_epochs': curriculum.total_epochs
        },
        'loss': loss,
        'architecture': {
            'hidden_size': 512,  # Document the scaled architecture
            'bottleneck_dim': 16,
            'input_size': 300,
            'model_type': 'scaled_vanilla_rnn_autoencoder'
        },
        'timestamp': time.time()
    }
    torch.save(checkpoint, filepath)
    print(f"   💾 Scaled model checkpoint saved: {filepath}")


def plot_training_progress(train_monitor, save_path='scaled_training_progress.png'):
    """Create training progress visualization."""
    if len(train_monitor.epoch_metrics['train_loss']) < 2:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = train_monitor.epoch_metrics['epoch']
    train_losses = train_monitor.epoch_metrics['train_loss']
    val_losses = train_monitor.epoch_metrics.get('val_loss', [])
    
    # Loss curves
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', marker='o', markersize=4)
    if val_losses:
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', marker='s', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Scaled Architecture Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Curriculum phases
    phases = train_monitor.epoch_metrics.get('curriculum_phase', [])
    tf_ratios = train_monitor.epoch_metrics.get('teacher_forcing_ratio', [])
    
    if phases and tf_ratios:
        ax2 = axes[1]
        ax2.plot(epochs, tf_ratios, 'g-', marker='^', markersize=4, label='Teacher Forcing Ratio')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Teacher Forcing Ratio')
        ax2.set_title('Curriculum Learning Progress')
        ax2.grid(True, alpha=0.3)
        
        # Add phase boundaries
        for i, phase in enumerate(phases):
            if i > 0 and phase != phases[i-1]:
                ax2.axvline(x=epochs[i], color='red', linestyle='--', alpha=0.5)
        
        ax2.legend()
    else:
        axes[1].text(0.5, 0.5, 'Curriculum data\nnot available', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Curriculum Progress')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Scaled model progress plot saved: {save_path}")


def main():
    """Main training loop for scaled architecture."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Scaled RNN Autoencoder')
    parser.add_argument('--rnn-type', choices=['vanilla', 'lstm'], default='vanilla',
                       help='RNN type to use (vanilla or lstm)')
    args = parser.parse_args()
    
    rnn_type = args.rnn_type.lower()
    
    print("🎭 SCALED Poetry RNN Autoencoder Training")
    print("=" * 55)
    print(f"🔧 Architecture: 300D → 512D → 64D → 512D → 300D ({rnn_type.upper()})")
    print("🎯 Goal: Fix encoder bottleneck (0.624 → 0.95+ cosine similarity)")
    print("💡 Neural Network Mentor Diagnosis: Hidden layer too small + critical fixes")
    print(f"🧠 RNN Type: {rnn_type.upper()} ({'Better gradient flow' if rnn_type == 'lstm' else 'Educational baseline'})")
    
    # Configuration
    config = Config()
    # Override specific parameters
    config.chunking.window_size = 50  # Max sequence length
    config.chunking.overlap = 10      # Overlap for data efficiency
    config.embedding.embedding_dim = 300  # GLoVe dimension
    
    # Setup scaled training
    components = setup_scaled_training(config, rnn_type)
    if components is None:
        return
    
    autoencoder, optimizer, scheduler, criterion, curriculum, grad_monitor, train_monitor = components
    
    # Load data
    train_loader, val_loader, metadata = load_data(config)
    if train_loader is None:
        return
    
    # Training parameters
    max_epochs = 30  # Total epochs across all curriculum phases
    save_every = 5   # Save checkpoint every N epochs
    
    print(f"\n🚀 Starting SCALED architecture training for {max_epochs} epochs...")
    print(f"   📋 Curriculum phases: {curriculum.summary()}")
    print(f"   🔧 Key fix: 512D hidden eliminates encoder bottleneck")
    
    train_monitor.start_training()
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(max_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{max_epochs}")
        
        # Train one epoch
        train_loss = train_epoch(
            autoencoder, train_loader, optimizer, criterion, 
            curriculum, grad_monitor, epoch
        )
        
        # Validate
        val_loss = validate(autoencoder, val_loader, criterion, curriculum)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update curriculum (may advance to next phase)
        phase_advanced = curriculum.step(val_loss)
        if phase_advanced:
            print(f"   📈 Advanced to curriculum phase {curriculum.current_phase_idx + 1}")
        
        # Log metrics
        train_monitor.log_epoch(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            curriculum_phase=curriculum.current_phase_idx + 1,
            teacher_forcing_ratio=curriculum.get_teacher_forcing_ratio(),
            gradient_health=grad_monitor.get_statistics().get('health_indicators', {}).get('overall_health_score', 0.5)
        )
        
        # Print progress
        print(f"   📊 Train Loss: {train_loss:.6f}")
        print(f"   🔍 Val Loss: {val_loss:.6f}")
        print(f"   📎 Grad Health: {grad_monitor.get_statistics().get('health_indicators', {}).get('overall_health_score', 0.5):.3f}")
        
        # Save best model with RNN type in filename
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_filename = f'scaled_model_{rnn_type}.pth'
            save_model_checkpoint(
                autoencoder, optimizer, curriculum, epoch + 1, val_loss,
                model_filename
            )
            print(f"   ⭐ New best SCALED {rnn_type.upper()} model! Val loss: {val_loss:.6f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_model_checkpoint(
                autoencoder, optimizer, curriculum, epoch + 1, val_loss,
                f'scaled_{rnn_type}_checkpoint_epoch_{epoch + 1}.pth'
            )
        
        # Plot progress
        if (epoch + 1) % 5 == 0:
            plot_training_progress(train_monitor, f'scaled_training_progress_epoch_{epoch + 1}.png')
    
    # Final summary
    print(f"\n🎉 SCALED architecture training completed!")
    summary = train_monitor.get_summary()
    print(f"   📊 Final train loss: {summary['final_train_loss']:.6f}")
    print(f"   🔍 Final val loss: {summary['final_val_loss']:.6f}")
    print(f"   ⏰ Total time: {summary.get('total_training_time', 0) / 60:.1f} minutes")
    print(f"   💾 Best SCALED model saved as: scaled_model_{rnn_type}.pth")
    
    # Final gradient analysis
    print(f"\n📊 Final Gradient Analysis:")
    final_diagnostics = grad_monitor.diagnose_gradient_issues()
    for diag in final_diagnostics[:3]:  # Show top 3 messages
        print(f"   {diag}")
    
    # Create final visualization
    plot_training_progress(train_monitor, 'final_scaled_training_progress.png')
    
    print(f"\n✅ SCALED architecture training completed successfully!")
    print(f"   🎯 Next: Run evaluation to compare with baseline (0.624 cosine similarity)")
    print(f"   📊 Expected: Significant improvement in reconstruction quality")
    print(f"   🔍 Load scaled_model_{rnn_type}.pth to use the fixed architecture")
    print(f"   📈 Compare with best_model.pth (baseline 64D hidden)")
    print(f"   🧠 {rnn_type.upper()} Results: Expected significant improvement from neural-network-mentor fixes")


if __name__ == "__main__":
    main()