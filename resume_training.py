#!/usr/bin/env python3
"""
Resume Training Script for Attention-Enhanced RNN Autoencoder

This script allows resuming training from any checkpoint, with options to:
- Modify the learning rate scheduler to avoid instability
- Extend training for additional epochs  
- Use different training configurations
- Continue with improved stability settings

Designed specifically to address the training instability observed after epoch 80
by using more stable learning rate schedules and enhanced monitoring.

Usage:
    # Resume from epoch 80 with stable scheduler for 30 more epochs
    python resume_training.py --checkpoint attention_training_results/checkpoint_epoch_80.pth --additional-epochs 30 --stable-scheduler
    
    # Resume from best model with custom configuration
    python resume_training.py --checkpoint attention_training_results/best_model.pth --config custom_config.json
    
    # Resume with early stopping and attention monitoring
    python resume_training.py --checkpoint attention_training_results/checkpoint_epoch_80.pth --additional-epochs 50 --early-stopping --attention-monitoring
"""

import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, '.')

from poetry_rnn import PoetryPreprocessor, AutoencoderDataset, Config
from poetry_rnn.models import RNNAutoencoder
from poetry_rnn.training.optimized_trainer import OptimizedRNNTrainer
from poetry_rnn.utils.async_io import AsyncCheckpointer, AsyncArtifactManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_stable_scheduler_config(base_config: Dict[str, Any], resume_epoch: int, total_epochs: int) -> Dict[str, Any]:
    """
    Create a stable learning rate scheduler configuration to avoid training instability.
    
    Replaces the aggressive cosine annealing warm restarts with more stable alternatives.
    """
    stable_config = base_config.copy()
    
    # Calculate remaining epochs for smooth decay
    remaining_epochs = total_epochs - resume_epoch
    
    # Replace with stable cosine annealing (no warm restarts)
    stable_config['scheduler'] = {
        'type': 'cosine_annealing',
        'T_max': remaining_epochs,  # Smooth decay over remaining epochs
        'eta_min': 1e-7,  # Lower minimum for fine-tuning
        'last_epoch': resume_epoch - 1  # Account for 0-indexing
    }
    
    # Reduce learning rate for stability
    current_lr = stable_config['training']['learning_rate']
    stable_config['training']['learning_rate'] = current_lr * 0.5  # Reduce by 50% for stability
    
    logger.info(f"🔧 Stable scheduler configuration:")
    logger.info(f"   Type: Cosine Annealing (no warm restarts)")
    logger.info(f"   T_max: {remaining_epochs} epochs")
    logger.info(f"   Learning rate: {stable_config['training']['learning_rate']:.2e} (reduced for stability)")
    logger.info(f"   Last epoch: {resume_epoch - 1}")
    
    return stable_config


def create_enhanced_early_stopping_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add enhanced early stopping configuration with attention monitoring.
    """
    enhanced_config = base_config.copy()
    
    # Enhanced early stopping
    enhanced_config['training']['early_stopping_patience'] = 15  # Stop if no improvement for 15 epochs
    enhanced_config['training']['min_delta'] = 1e-4  # Smaller improvement threshold
    
    # Enhanced monitoring for attention patterns
    enhanced_config['monitoring']['attention_monitoring'] = True
    enhanced_config['monitoring']['attention_entropy_threshold'] = 0.5  # Warn if entropy > 0.5
    enhanced_config['monitoring']['attention_diversity_min'] = 0.001  # Warn if diversity < 0.001
    
    logger.info("🎯 Enhanced early stopping configuration:")
    logger.info(f"   Patience: {enhanced_config['training']['early_stopping_patience']} epochs")
    logger.info(f"   Min delta: {enhanced_config['training']['min_delta']}")
    logger.info("   Attention monitoring enabled")
    
    return enhanced_config


def load_data_for_resume(config: Dict[str, Any]) -> tuple:
    """Load and prepare data for resumed training."""
    logger.info("📚 Loading data for resumed training...")
    
    # Initialize preprocessor
    poetry_config = Config()
    poetry_config.chunking.window_size = 50
    poetry_config.chunking.overlap = 10
    poetry_config.embedding.embedding_dim = 300
    
    preprocessor = PoetryPreprocessor(config=poetry_config)
    
    # Process data
    data_path = config['data_path']
    results = preprocessor.process_poems(data_path, save_artifacts=False)  # Use existing artifacts
    
    # Create dataset
    dataset = AutoencoderDataset(
        sequences=results['sequences'],
        embedding_sequences=results['embedding_sequences'],
        attention_masks=results['attention_masks'],
        metadata=results['metadata'],
        vocabulary=results['vocabulary']
    )
    
    # Split dataset (80/20 train/val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Consistent split
    )
    
    # Create data loaders
    dataloader_config = config['dataloader']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=dataset._collate_fn,
        **dataloader_config
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=dataset._collate_fn,
        **{k: v for k, v in dataloader_config.items() if k != 'drop_last'}
    )
    
    logger.info(f"✅ Data loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    return train_loader, val_loader, results


def create_model_and_optimizer_for_resume(config: Dict[str, Any], resume_epoch: int):
    """Create model and optimizer for resumed training."""
    # Create model
    model_config = config['model']
    model = RNNAutoencoder(
        input_size=model_config['input_dim'],
        hidden_size=model_config['hidden_dim'],
        bottleneck_dim=model_config['bottleneck_dim'],
        rnn_type=model_config['rnn_type'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        use_attention=model_config['use_attention'],
        attention_heads=model_config['attention_heads'],
        use_positional_encoding=model_config['use_positional_encoding'],
        use_optimized_attention=model_config['use_optimized_attention'],
        use_batch_norm=True
    )
    
    # Create optimizer 
    training_config = config['training']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🏗️  Model and optimizer created for resumed training:")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Learning rate: {training_config['learning_rate']:.2e}")
    
    # Return without scheduler - will be created after checkpoint loading
    return model, optimizer

def create_scheduler_for_resume(optimizer, config: Dict[str, Any], resume_epoch: int):
    """Create scheduler after loading checkpoint to avoid initial_lr issues."""
    scheduler_config = config['scheduler']
    
    # Ensure initial_lr is set (required for PyTorch schedulers when resuming)
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = config['training']['learning_rate']
    
    if scheduler_config['type'] == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config['eta_min'],
            last_epoch=scheduler_config.get('last_epoch', -1)
        )
    elif scheduler_config['type'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    else:
        # Fallback to original scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get('T_0', 25),
            T_mult=scheduler_config.get('T_mult', 2),
            eta_min=scheduler_config['eta_min']
        )
    
    logger.info(f"📅 Scheduler created: {scheduler_config['type']}")
    return scheduler


def main():
    """Main function for resuming training."""
    parser = argparse.ArgumentParser(description='Resume Attention-Enhanced RNN Autoencoder Training')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file to resume from')
    parser.add_argument('--additional-epochs', type=int, default=30,
                       help='Number of additional epochs to train')
    parser.add_argument('--stable-scheduler', action='store_true',
                       help='Use stable scheduler (cosine annealing without warm restarts)')
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable enhanced early stopping with attention monitoring')
    parser.add_argument('--config', type=str, default=None,
                       help='Custom configuration file (JSON)')
    parser.add_argument('--output-dir', type=str, default='./resumed_training_results',
                       help='Output directory for resumed training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🔄 Resume Training for Attention-Enhanced RNN Autoencoder")
    logger.info("=" * 70)
    logger.info(f"📂 Checkpoint: {args.checkpoint}")
    logger.info(f"⏱️  Additional epochs: {args.additional_epochs}")
    logger.info(f"🔧 Stable scheduler: {args.stable_scheduler}")
    logger.info(f"⏹️  Early stopping: {args.early_stopping}")
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    # Load base configuration
    if args.config:
        logger.info(f"📋 Loading custom config: {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Load config from checkpoint or use default
        logger.info("📋 Loading config from checkpoint")
        checkpoint_data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'config' in checkpoint_data:
            config = checkpoint_data['config']
        else:
            logger.warning("⚠️  No config in checkpoint, using default attention config")
            # Use default from train_attention_autoencoder.py
            from train_attention_autoencoder import create_attention_config
            dummy_args = argparse.Namespace(epochs=args.additional_epochs, batch_size=32)
            config = create_attention_config(dummy_args)
    
    # Get resume epoch from checkpoint
    checkpoint_data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    resume_epoch = checkpoint_data.get('epoch', 0)
    total_epochs = resume_epoch + args.additional_epochs
    
    logger.info(f"📍 Resuming from epoch {resume_epoch}")
    logger.info(f"🎯 Target epoch: {total_epochs}")
    
    # Update configuration
    config['training']['num_epochs'] = total_epochs
    config['output_dir'] = args.output_dir
    config['logs_dir'] = f"{args.output_dir}/logs"
    
    # Override learning rate if specified
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
        logger.info(f"🔧 Learning rate override: {args.learning_rate:.2e}")
    
    # Apply stable scheduler if requested
    if args.stable_scheduler:
        config = create_stable_scheduler_config(config, resume_epoch, total_epochs)
    
    # Apply enhanced early stopping if requested
    if args.early_stopping:
        config = create_enhanced_early_stopping_config(config)
    
    # Setup directories
    output_dir = Path(config['output_dir'])
    logs_dir = Path(config['logs_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save updated configuration
    with open(output_dir / 'resumed_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    train_loader, val_loader, data_results = load_data_for_resume(config)
    
    # Create model and optimizer (scheduler created after checkpoint loading)
    model, optimizer = create_model_and_optimizer_for_resume(config, resume_epoch)
    
    # Setup training infrastructure
    device = torch.device(config['device'])
    async_checkpointer = AsyncCheckpointer(str(output_dir))
    artifact_manager = AsyncArtifactManager(str(output_dir))
    
    # Import performance monitor
    from train_attention_autoencoder import AttentionPerformanceMonitor
    performance_monitor = AttentionPerformanceMonitor(logs_dir, enable_plotting=True)
    
    # Create trainer (without scheduler initially)
    trainer = OptimizedRNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,  # Will be set after checkpoint loading
        device=device,
        config=config,
        async_checkpointer=async_checkpointer,
        artifact_manager=artifact_manager,
        performance_monitor=performance_monitor
    )
    
    # Load checkpoint and resume training state
    logger.info("🔄 Loading checkpoint and resuming training state...")
    checkpoint_info = trainer.load_checkpoint(args.checkpoint, resume_training=True)
    
    # Now create scheduler after checkpoint is loaded
    logger.info("📅 Creating scheduler for resumed training...")
    scheduler = create_scheduler_for_resume(optimizer, config, resume_epoch)
    trainer.scheduler = scheduler
    
    logger.info("📊 Checkpoint Info:")
    logger.info(f"   Epoch: {checkpoint_info['epoch']}")
    logger.info(f"   Validation Loss: {checkpoint_info['val_loss']:.6f}")
    logger.info(f"   Cosine Similarity: {checkpoint_info['cosine_similarity']:.4f}")
    
    # Start training
    logger.info("🚀 Starting resumed training...")
    performance_monitor.start_training()
    
    try:
        results = trainer.train()
        
        # Save final results
        performance_monitor.save_metrics()
        
        # Analyze results
        final_cosine = results.get('best_cosine_similarity', 0.0)
        initial_cosine = checkpoint_info['cosine_similarity']
        improvement = final_cosine - initial_cosine
        
        logger.info("=" * 70)
        logger.info("🎯 RESUMED TRAINING COMPLETED")
        logger.info(f"📈 Initial Cosine Similarity: {initial_cosine:.4f} (epoch {resume_epoch})")
        logger.info(f"📈 Final Cosine Similarity: {final_cosine:.4f}")
        logger.info(f"📊 Additional Improvement: {improvement:+.4f}")
        logger.info(f"🎲 Total Training Epochs: {resume_epoch} + {args.additional_epochs} = {total_epochs}")
        
        # Performance analysis
        if improvement >= 0.05:
            logger.info("🎉 EXCELLENT: Significant additional improvement achieved!")
        elif improvement >= 0.02:
            logger.info("✨ GOOD: Meaningful improvement from resumed training")
        elif improvement >= 0:
            logger.info("👍 STABLE: Training continued without degradation")
        else:
            logger.info("⚠️  DECLINE: Performance decreased - consider stopping earlier")
        
        # Save final model
        final_model_path = output_dir / f'resumed_model_epoch_{total_epochs}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'results': results,
            'final_cosine_similarity': final_cosine,
            'initial_cosine_similarity': initial_cosine,
            'improvement': improvement,
            'resume_info': checkpoint_info,
            'total_epochs': total_epochs
        }, final_model_path)
        
        logger.info(f"💾 Final model saved: {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        performance_monitor.save_metrics()
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    logger.info("🏁 Resumed training session complete")


if __name__ == "__main__":
    exit(main())