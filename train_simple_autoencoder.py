#!/usr/bin/env python3
"""
Simple Poetry RNN Autoencoder Training Script

This script demonstrates how to use the poetry_rnn library to train
an RNN autoencoder on poetry data. It includes all the improvements
from the neural-network-mentor: teacher forcing, gradient monitoring,
and proper initialization.

Usage:
    python train_simple_autoencoder.py

Features:
- Curriculum learning with adaptive teacher forcing
- Real-time gradient monitoring and diagnostics
- Comprehensive logging and visualization
- Easy to modify hyperparameters
"""

import sys
import os
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


def setup_training(config):
    """Set up all components for training."""
    print("🔧 Setting up training components...")
    
    # Create models
    autoencoder = RNNAutoencoder(
        input_size=300,           # GLoVe embedding dimension
        hidden_size=64,           # RNN hidden size
        bottleneck_dim=16,        # Compression dimension
        rnn_type='vanilla',       # Use vanilla RNN (as implemented)
        num_layers=1,             # Single layer for simplicity
        dropout=0.1,              # Light regularization
        use_batch_norm=True       # Stabilizes training
    )
    
    print(f"   📊 Model: {sum(p.numel() for p in autoencoder.parameters()):,} parameters")
    print(f"   🗜️  Compression: {300/16:.1f}x (300D → 16D)")
    
    # Optimizer with conservative learning rate
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3
    )
    
    # Loss function for variable-length sequences
    criterion = MaskedMSELoss()
    
    # Curriculum learning scheduler
    curriculum = CurriculumScheduler(adaptive=True, patience=2)
    
    # Gradient monitoring with adaptive clipping
    grad_monitor = GradientMonitor(
        model=autoencoder,
        clip_value=5.0,
        adaptive=True,
        vanishing_threshold=1e-6,
        exploding_threshold=10.0
    )
    
    # Training progress monitor
    train_monitor = TrainingMonitor(log_dir='training_logs')
    
    print("✅ Training setup complete!")
    print(f"   📋 Curriculum: {len(curriculum.phases)} phases")
    print(f"   📎 Adaptive gradient clipping: ON")
    print(f"   📊 Teacher forcing: Curriculum-integrated")
    
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
        batch_size=16,  # Small batch for stability
        shuffle=True, 
        collate_fn=dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=dataset.collate_fn
    )
    
    print(f"✅ Data loaded successfully!")
    print(f"   📊 Total sequences: {len(dataset)}")
    print(f"   🏋️  Training: {len(train_dataset)} sequences")
    print(f"   🔍 Validation: {len(val_dataset)} sequences")
    print(f"   📏 Sequence length: {results['metadata']['max_sequence_length']}")
    
    return train_loader, val_loader, results['metadata']


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
        reconstructed, decoder_hidden = autoencoder.decoder(
            bottleneck, 
            target_sequences=input_sequences,
            mask=attention_mask,
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
            bottleneck, _ = autoencoder.encoder(input_sequences, attention_mask)
            reconstructed, _ = autoencoder.decoder(bottleneck, mask=attention_mask)
            
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
        'timestamp': time.time()
    }
    torch.save(checkpoint, filepath)
    print(f"   💾 Checkpoint saved: {filepath}")


def plot_training_progress(train_monitor, save_path='training_progress.png'):
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
    axes[0].set_title('Training Progress')
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
    print(f"   📊 Progress plot saved: {save_path}")


def main():
    """Main training loop."""
    print("🎭 Poetry RNN Autoencoder Training")
    print("=" * 50)
    
    # Configuration
    config = Config(
        chunking_window_size=50,      # Max sequence length
        chunking_overlap=10,          # Overlap for data efficiency
        embedding_embedding_dim=300   # GLoVe dimension
    )
    
    # Setup
    components = setup_training(config)
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
    
    print(f"\n🚀 Starting training for {max_epochs} epochs...")
    print(f"   📋 Curriculum phases: {curriculum.summary()}")
    
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
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_checkpoint(
                autoencoder, optimizer, curriculum, epoch + 1, val_loss,
                'best_model.pth'
            )
            print(f"   ⭐ New best model! Val loss: {val_loss:.6f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_model_checkpoint(
                autoencoder, optimizer, curriculum, epoch + 1, val_loss,
                f'checkpoint_epoch_{epoch + 1}.pth'
            )
        
        # Plot progress
        if (epoch + 1) % 5 == 0:
            plot_training_progress(train_monitor, f'training_progress_epoch_{epoch + 1}.png')
    
    # Final summary
    print(f"\n🎉 Training completed!")
    summary = train_monitor.get_summary()
    print(f"   📊 Final train loss: {summary['final_train_loss']:.6f}")
    print(f"   🔍 Final val loss: {summary['final_val_loss']:.6f}")
    print(f"   ⏰ Total time: {summary.get('total_training_time', 0) / 60:.1f} minutes")
    print(f"   💾 Best model saved as: best_model.pth")
    
    # Final gradient analysis
    print(f"\n📊 Final Gradient Analysis:")
    final_diagnostics = grad_monitor.diagnose_gradient_issues()
    for diag in final_diagnostics[:3]:  # Show top 3 messages
        print(f"   {diag}")
    
    # Create final visualization
    plot_training_progress(train_monitor, 'final_training_progress.png')
    
    print(f"\n✅ Training script completed successfully!")
    print(f"   🎯 You can now experiment with the trained model")
    print(f"   📊 Check training_progress.png for learning curves")
    print(f"   🔍 Load best_model.pth to use the trained autoencoder")


if __name__ == "__main__":
    main()
