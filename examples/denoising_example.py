#!/usr/bin/env python3
"""
Denoising Autoencoder Example for Poetry

This example demonstrates how to use the new denoising capabilities
to train more robust poetry autoencoders that generalize better to
unseen poetry styles and handle real-world text variations.

Author: Poetry RNN Collaborative Project
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from poetry_rnn.api import (
    PoetryAutoencoder, 
    create_denoising_autoencoder,
    DenoisingConfig,
    NoiseType
)
from poetry_rnn.models import RNNAutoencoder
import torch

def demonstrate_denoising_training():
    """
    Demonstrate denoising autoencoder training with different noise strategies.
    
    ★ Insight: Denoising forces the model to learn robust semantic representations
    by requiring perfect reconstruction from corrupted input. This leads to better
    generalization and understanding of poetic structure.
    """
    print("🎭 DENOISING AUTOENCODER DEMONSTRATION")
    print("=" * 50)
    
    # 1. Create base autoencoder with proven architecture
    print("\n1. Creating base autoencoder...")
    base_model = RNNAutoencoder(
        input_size=300,        # GLoVe 300D embeddings
        hidden_size=512,       # Proven optimal size
        bottleneck_dim=128,    # 2.3x compression ratio
        rnn_type='LSTM',
        num_layers=1,
        dropout=0.1,
        use_attention=True,
        attention_heads=4
    )
    
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"✅ Base model: {total_params:,} parameters")
    
    # 2. Demonstrate different noise strategies
    noise_strategies = [
        ("word_dropout", "Replace random words with <UNK>"),
        ("mask", "Use <MASK> tokens for corruption"),
        ("swap", "Shuffle adjacent words"),
        ("gaussian", "Random token replacement")
    ]
    
    print("\n2. Available noise strategies:")
    for strategy, description in noise_strategies:
        print(f"   • {strategy}: {description}")
    
    # 3. Create denoising autoencoder with word dropout
    print("\n3. Creating denoising autoencoder...")
    denoising_config = DenoisingConfig(
        noise_type=NoiseType.WORD_DROPOUT,
        noise_strength=0.15,           # 15% of tokens corrupted
        adaptive_noise=True,           # Curriculum learning
        initial_noise=0.05,           # Start with 5% corruption
        final_noise=0.20,             # End with 20% corruption
        warmup_epochs=10,             # 10 epochs to ramp up
        preserve_structure=True       # Preserve poetic structure
    )
    
    denoising_model = create_denoising_autoencoder(
        base_autoencoder=base_model,
        noise_type="word_dropout",
        noise_strength=0.15,
        vocabulary_size=10000,
        adaptive_noise=True,
        preserve_structure=True
    )
    
    print(f"✅ Denoising model created")
    print(f"   Noise type: {denoising_model.config.noise_type.value}")
    print(f"   Adaptive schedule: {denoising_model.config.initial_noise:.2f} → {denoising_model.config.final_noise:.2f}")
    
    # 4. Demonstrate noise injection
    print("\n4. Demonstrating noise injection...")
    
    # Create sample poetry sequences (token indices)
    batch_size, seq_len = 2, 15
    sample_sequences = torch.randint(3, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len).bool()
    
    # Show original sequences
    print(f"Original sequence: {sample_sequences[0, :10].tolist()}")
    
    # Apply noise at different training epochs
    denoising_model.train()
    
    for epoch in [1, 5, 15]:
        sample_batch = {
            'input_sequences': sample_sequences,
            'attention_mask': attention_mask
        }
        
        # Get corrupted input
        corrupted_sequences, _ = denoising_model.denoiser(
            sample_sequences, attention_mask, epoch=epoch
        )
        
        corrupted_count = (sample_sequences[0] != corrupted_sequences[0]).sum().item()
        noise_level = denoising_model.denoiser._get_adaptive_noise_strength(epoch)
        
        print(f"Epoch {epoch:2d}: {corrupted_count:2d}/{seq_len} tokens corrupted (noise={noise_level:.3f})")
    
    return denoising_model

def expected_benefits():
    """
    Explain the theoretical and practical benefits of denoising training.
    """
    print("\n🎯 EXPECTED BENEFITS OF DENOISING TRAINING")
    print("=" * 50)
    
    benefits = [
        ("Improved Generalization", 
         "Model learns to reconstruct from partial information, improving performance on unseen poetry styles"),
        
        ("Robust Representations", 
         "Forced to capture essential semantic structure rather than surface patterns"),
        
        ("Better Semantic Understanding", 
         "Must infer meaning and context from corrupted input, leading to deeper comprehension"),
        
        ("Real-World Robustness", 
         "Handles typos, OCR errors, and text variations naturally"),
        
        ("Enhanced Creativity", 
         "Learning to 'fill in the gaps' may improve generation quality and creativity")
    ]
    
    for benefit, explanation in benefits:
        print(f"\n• {benefit}:")
        print(f"  {explanation}")

def integration_with_hybrid_loss():
    """
    Explain how denoising integrates with the proven HybridTokenEmbeddingLoss.
    """
    print("\n🚀 INTEGRATION WITH HYBRID LOSS (99.7% ACCURACY)")
    print("=" * 50)
    
    print("""
The denoising approach seamlessly integrates with your breakthrough 
HybridTokenEmbeddingLoss methodology:

1. TRAINING PROCESS:
   • Input: Clean poetry sequences
   • Corruption: Apply noise (word dropout, masking, etc.)
   • Forward: Process corrupted input through autoencoder
   • Loss: Compute hybrid loss against CLEAN targets
   
2. MATHEMATICAL FORMULATION:
   • Standard: L = αL_token(f(x), x) + βL_embedding(f(x), x)
   • Denoising: L = αL_token(f(corrupt(x)), x) + βL_embedding(f(corrupt(x)), x)
   
   Where corrupt(x) applies controlled noise but targets remain clean.

3. EXPECTED IMPROVEMENTS:
   • Current: 99.7% token accuracy on clean text
   • Denoising: Potential for 99%+ accuracy on corrupted text
   • Generalization: Better performance on real-world poetry variations
   
4. CURRICULUM LEARNING:
   • Start with minimal noise (5%) for stable learning
   • Gradually increase to 20% for robust representations
   • Maintain proven 70% token + 30% embedding loss weights
""")

def main():
    """Main demonstration function."""
    try:
        denoising_model = demonstrate_denoising_training()
        expected_benefits()
        integration_with_hybrid_loss()
        
        print("\n✅ DENOISING FOUNDATION READY")
        print("=" * 50)
        print("Next steps:")
        print("1. Integrate with existing hybrid loss training pipeline")
        print("2. Train on poetry dataset with noise curriculum")
        print("3. Evaluate on clean vs corrupted test sets")
        print("4. Compare generalization to unseen poetry styles")
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()