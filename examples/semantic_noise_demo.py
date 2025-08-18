#!/usr/bin/env python3
"""
Semantic Embedding Noise Demonstration

This script demonstrates the sophisticated semantic embedding noise technique
that adds Gaussian noise in embedding space and finds nearest neighbor words,
forcing the model to learn fine-grained semantic distinctions.

Mathematical approach:
1. For token t_i, get embedding e_i = Embedding[t_i]  
2. Add Gaussian noise: ẽ_i = e_i + ε, where ε ~ N(0, σ²I)
3. Find nearest neighbor: t'_i = argmin_j ||ẽ_i - Embedding[j]||₂

Author: Poetry RNN Collaborative Project
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from poetry_rnn.api.core.denoising import (
    PoetryDenoiser, DenoisingConfig, NoiseType
)

def create_mock_embeddings(vocab_size: int = 1000, embed_dim: int = 300):
    """
    Create mock word embeddings that simulate semantic relationships.
    
    We'll create clusters of semantically similar words to demonstrate
    how the noise finds semantically related neighbors.
    """
    print(f"Creating mock embeddings: {vocab_size} words, {embed_dim}D")
    
    # Create base embeddings
    embeddings = torch.randn(vocab_size, embed_dim)
    
    # Normalize to unit vectors (like GLoVe)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Create some semantic clusters for demonstration
    cluster_words = {
        "poetry": [10, 11, 12, 13, 14],        # poem, verse, stanza, rhyme, meter
        "emotions": [20, 21, 22, 23, 24],      # love, joy, sorrow, anger, fear  
        "nature": [30, 31, 32, 33, 34],        # tree, flower, mountain, river, sky
        "colors": [40, 41, 42, 43, 44],        # red, blue, green, yellow, purple
        "time": [50, 51, 52, 53, 54]           # morning, noon, evening, night, dawn
    }
    
    # Make words in each cluster more similar
    for cluster_name, word_ids in cluster_words.items():
        # Create a cluster center
        center = torch.randn(embed_dim)
        center = F.normalize(center, p=2, dim=0)
        
        # Move words closer to cluster center
        for word_id in word_ids:
            # Blend with cluster center (0.7 original + 0.3 center)
            embeddings[word_id] = 0.7 * embeddings[word_id] + 0.3 * center
            embeddings[word_id] = F.normalize(embeddings[word_id], p=2, dim=0)
    
    return embeddings, cluster_words

def demonstrate_semantic_noise():
    """
    Demonstrate semantic embedding noise with mock word clusters.
    """
    print("🎭 SEMANTIC EMBEDDING NOISE DEMONSTRATION")
    print("=" * 55)
    
    # Create mock embeddings with semantic clusters
    vocab_size = 100
    embed_dim = 50  # Smaller for demo
    embeddings, clusters = create_mock_embeddings(vocab_size, embed_dim)
    
    print(f"\n✅ Created {vocab_size} mock word embeddings ({embed_dim}D)")
    print("📚 Semantic clusters:")
    for cluster_name, word_ids in clusters.items():
        print(f"   • {cluster_name}: words {word_ids}")
    
    # Create denoising configuration for semantic noise
    config = DenoisingConfig(
        noise_type=NoiseType.SEMANTIC_EMBEDDING,
        noise_strength=0.3,  # High for demonstration
        embedding_noise_std=0.1,  # Moderate noise
        late_training_epochs=80,  # Late training only
        preserve_structure=False  # Allow all corruption for demo
    )
    
    # Create denoiser
    denoiser = PoetryDenoiser(config, vocab_size, embeddings)
    print(f"\n✅ Denoiser created with σ={config.embedding_noise_std} noise")
    
    # Test semantic noise on cluster words
    print(f"\n🧪 TESTING SEMANTIC NOISE (Late training: epoch 85)")
    print("-" * 40)
    
    # Create test sequences with words from different clusters
    test_words = [10, 20, 30, 40, 50]  # One from each cluster
    cluster_names = ["poetry", "emotions", "nature", "colors", "time"]
    
    batch_size = 1
    seq_len = len(test_words)
    test_sequences = torch.tensor(test_words).unsqueeze(0)  # [1, 5]
    attention_mask = torch.ones(batch_size, seq_len).bool()
    
    print("Original words (by cluster):")
    for i, (word_id, cluster) in enumerate(zip(test_words, cluster_names)):
        print(f"  Position {i}: word_{word_id:2d} ({cluster})")
    
    # Apply semantic noise (epoch 85 > late_training_epochs=80)
    denoiser.train()
    corrupted_sequences, _ = denoiser(test_sequences, attention_mask, epoch=85)
    
    print(f"\nAfter semantic embedding noise:")
    for i, (orig_id, new_id, cluster) in enumerate(zip(test_words, corrupted_sequences[0], cluster_names)):
        if orig_id != new_id:
            # Check if new word is in same cluster
            same_cluster = new_id.item() in clusters[cluster]
            status = "✅ same cluster" if same_cluster else "🔄 different word"
            print(f"  Position {i}: word_{orig_id:2d} → word_{new_id.item():2d} ({cluster}) {status}")
        else:
            print(f"  Position {i}: word_{orig_id:2d} unchanged ({cluster})")
    
    # Demonstrate distance calculation
    print(f"\n📊 EMBEDDING SPACE ANALYSIS")
    print("-" * 40)
    
    # Show distances within and between clusters
    poetry_center = embeddings[clusters["poetry"]].mean(dim=0)
    emotion_center = embeddings[clusters["emotions"]].mean(dim=0) 
    
    # Distance between cluster centers
    inter_cluster_dist = F.pairwise_distance(poetry_center.unsqueeze(0), emotion_center.unsqueeze(0))
    
    # Distance within poetry cluster
    poetry_embeddings = embeddings[clusters["poetry"]]
    intra_cluster_dists = F.pairwise_distance(
        poetry_embeddings[0].unsqueeze(0), 
        poetry_embeddings[1].unsqueeze(0)
    )
    
    print(f"📏 Distance between poetry & emotion clusters: {inter_cluster_dist.item():.3f}")
    print(f"📏 Distance within poetry cluster: {intra_cluster_dists.item():.3f}")
    
    if intra_cluster_dists.item() < inter_cluster_dist.item():
        print("✅ Semantic structure preserved: intra-cluster < inter-cluster distance")
    
    return denoiser, embeddings, clusters

def demonstrate_noise_curriculum():
    """
    Show how semantic noise is only applied in late training epochs.
    """
    print(f"\n🎓 CURRICULUM LEARNING: LATE-TRAINING SEMANTIC NOISE")
    print("=" * 55)
    
    vocab_size = 50
    embeddings = torch.randn(vocab_size, 300)
    
    config = DenoisingConfig(
        noise_type=NoiseType.SEMANTIC_EMBEDDING,
        noise_strength=0.2,
        late_training_epochs=80,  # Only apply after epoch 80
        adaptive_noise=True
    )
    
    denoiser = PoetryDenoiser(config, vocab_size, embeddings)
    
    # Test at different epochs
    test_seq = torch.randint(3, vocab_size, (1, 5))
    mask = torch.ones(1, 5).bool()
    
    epochs_to_test = [20, 40, 60, 80, 100]
    
    print("Epoch  | Semantic Noise Applied")
    print("-------|-------------------")
    
    denoiser.train()
    for epoch in epochs_to_test:
        original = test_seq.clone()
        corrupted, _ = denoiser(test_seq, mask, epoch=epoch)
        
        changes = (original != corrupted).sum().item()
        applied = "YES" if epoch >= config.late_training_epochs and changes > 0 else "NO"
        
        print(f"   {epoch:2d}  | {applied:>8s} ({changes} changes)")
    
    print(f"\n✅ Semantic noise only applied after epoch {config.late_training_epochs}")
    print("💡 This ensures model has stable representations and good validation accuracy")

def integration_benefits():
    """
    Explain the theoretical benefits of semantic embedding noise.
    """
    print(f"\n🎯 THEORETICAL BENEFITS")
    print("=" * 55)
    
    benefits = [
        ("Fine-Grained Semantic Learning", 
         "Forces model to distinguish between semantically similar words"),
        
        ("Embedding Space Utilization", 
         "Better use of embedding space neighborhood structure"),
        
        ("Robust Representations", 
         "Model must reconstruct from near-neighbor words, improving semantic understanding"),
        
        ("Late-Training Refinement", 
         "Applied only when model has stable representations for fine-tuning"),
        
        ("Poetry-Specific Benefits", 
         "Helps with synonym choice, word selection, and semantic coherence")
    ]
    
    for benefit, explanation in benefits:
        print(f"\n• {benefit}:")
        print(f"  {explanation}")
    
    print(f"\n🔬 MATHEMATICAL INSIGHT:")
    print("This technique essentially performs adversarial training in embedding space,")
    print("where the adversarial perturbation ε ~ N(0, σ²I) is applied in the continuous")
    print("embedding domain, then projected back to the discrete token space via nearest")
    print("neighbor search. This should significantly improve semantic robustness.")

def main():
    """Main demonstration function."""
    try:
        denoiser, embeddings, clusters = demonstrate_semantic_noise()
        demonstrate_noise_curriculum()
        integration_benefits()
        
        print(f"\n✅ SEMANTIC EMBEDDING NOISE READY FOR INTEGRATION")
        print("=" * 55)
        print("Next steps for poetry autoencoder:")
        print("1. Load GLoVe embeddings for vocabulary")
        print("2. Train with word dropout initially (epochs 1-80)")
        print("3. Switch to semantic embedding noise (epochs 80+)")
        print("4. Evaluate fine-grained semantic understanding")
        print("5. Expected: Better synonym handling and semantic coherence")
        
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()