#!/usr/bin/env python3
"""
Poem Reconstruction Comparison Script

This script loads a trained autoencoder model and compares original poems
with their reconstructed versions to evaluate semantic preservation quality.

Features:
- Load trained model from checkpoint
- Process poems through encode -> decode pipeline
- Display side-by-side comparison with similarity metrics
- Support for multiple poems and interactive selection
- Semantic similarity analysis using cosine similarity
- Word-level and line-level comparison

Usage:
    python compare_poem_reconstruction.py [--model MODEL_PATH] [--poem-index INDEX]
"""

import sys
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, '.')

from poetry_rnn import PoetryPreprocessor, AutoencoderDataset, Config
from poetry_rnn.models import RNNAutoencoder

def load_trained_model(model_path: str) -> Tuple[RNNAutoencoder, Dict]:
    """Load a trained autoencoder model from checkpoint."""
    print(f"📂 Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract model configuration
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
    else:
        # Fallback configuration for older checkpoints
        print("⚠️  No config found in checkpoint, using default configuration")
        model_config = {
            'input_dim': 300,
            'hidden_dim': 512,
            'bottleneck_dim': 128,
            'rnn_type': 'LSTM',
            'num_layers': 2,
            'dropout': 0.2,
            'use_attention': True,
            'attention_heads': 4,
            'use_positional_encoding': True,
            'use_optimized_attention': True
        }
    
    # Create model
    model = RNNAutoencoder(
        input_size=model_config.get('input_dim', 300),
        hidden_size=model_config.get('hidden_dim', 512),
        bottleneck_dim=model_config.get('bottleneck_dim', 128),
        rnn_type=model_config.get('rnn_type', 'LSTM'),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.2),
        use_attention=model_config.get('use_attention', True),
        attention_heads=model_config.get('attention_heads', 4),
        use_positional_encoding=model_config.get('use_positional_encoding', True),
        use_optimized_attention=model_config.get('use_optimized_attention', True)
    )
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded successfully")
    print(f"   Architecture: {model_config.get('input_dim', 300)}D → {model_config.get('hidden_dim', 512)}D → {model_config.get('bottleneck_dim', 128)}D")
    print(f"   Features: {model_config.get('rnn_type', 'LSTM')}, {model_config.get('attention_heads', 4)} attention heads")
    print(f"   Parameters: {total_params:,}")
    
    if 'final_cosine_similarity' in checkpoint:
        print(f"   Training performance: {checkpoint['final_cosine_similarity']:.4f} cosine similarity")
    
    return model, checkpoint

def load_poems(data_path: str = None) -> List[Dict]:
    """Load poems from the dataset."""
    if data_path is None:
        data_path = 'dataset_poetry/multi_poem_dbbc_collection.json'
    
    print(f"📚 Loading poems from: {data_path}")
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        poems = json.load(f)
    
    print(f"✅ Loaded {len(poems)} poems")
    return poems

def setup_preprocessor() -> Tuple[PoetryPreprocessor, AutoencoderDataset]:
    """Setup the poetry preprocessor and dataset."""
    print("🔧 Setting up preprocessor...")
    
    # Initialize preprocessor
    poetry_config = Config()
    poetry_config.chunking.window_size = 50
    poetry_config.chunking.overlap = 10
    poetry_config.embedding.embedding_dim = 300
    
    preprocessor = PoetryPreprocessor(config=poetry_config)
    
    # Process data (this loads embeddings and creates dataset)
    data_path = 'dataset_poetry/multi_poem_dbbc_collection.json'
    results = preprocessor.process_poems(data_path, save_artifacts=False)
    
    # Create dataset
    dataset = AutoencoderDataset(
        sequences=results['sequences'],
        embedding_sequences=results['embedding_sequences'],
        attention_masks=results['attention_masks'],
        metadata=results['metadata'],
        vocabulary=results['vocabulary']
    )
    
    print("✅ Preprocessor ready")
    return preprocessor, dataset

def compute_similarity_metrics(original: torch.Tensor, reconstructed: torch.Tensor, 
                             attention_mask: torch.Tensor) -> Dict[str, float]:
    """Compute various similarity metrics between original and reconstructed embeddings."""
    # Apply attention mask
    mask = attention_mask.unsqueeze(-1).float()
    orig_masked = original * mask
    recon_masked = reconstructed * mask
    
    # Cosine similarity (average over valid tokens)
    cosine_sim = F.cosine_similarity(orig_masked, recon_masked, dim=-1)
    valid_tokens = attention_mask.sum().item()
    avg_cosine = (cosine_sim * attention_mask.float()).sum().item() / valid_tokens
    
    # MSE (mean squared error)
    mse = F.mse_loss(recon_masked, orig_masked, reduction='none').mean(dim=-1)
    avg_mse = (mse * attention_mask.float()).sum().item() / valid_tokens
    
    # L2 distance
    l2_dist = torch.norm(recon_masked - orig_masked, dim=-1)
    avg_l2 = (l2_dist * attention_mask.float()).sum().item() / valid_tokens
    
    return {
        'cosine_similarity': avg_cosine,
        'mse': avg_mse,
        'l2_distance': avg_l2,
        'valid_tokens': valid_tokens
    }

def find_closest_words(embedding: torch.Tensor, vocabulary: Dict[str, int], 
                      embeddings_dict: Dict[str, torch.Tensor], top_k: int = 3) -> List[str]:
    """Find the closest words to a given embedding."""
    similarities = []
    
    for word, word_embedding in embeddings_dict.items():
        sim = F.cosine_similarity(embedding.unsqueeze(0), word_embedding.unsqueeze(0))
        similarities.append((word, sim.item()))
    
    # Sort by similarity and return top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in similarities[:top_k]]

def reconstruct_text_approximation(embeddings: torch.Tensor, attention_mask: torch.Tensor,
                                 vocabulary: Dict[str, int], embeddings_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Approximate text reconstruction by finding closest words to embeddings."""
    reconstructed_words = []
    
    for i in range(embeddings.size(0)):
        if attention_mask[i]:
            embedding = embeddings[i]
            closest_words = find_closest_words(embedding, vocabulary, embeddings_dict, top_k=1)
            if closest_words:
                reconstructed_words.append(closest_words[0])
            else:
                reconstructed_words.append("[UNK]")
        else:
            break  # Stop at padding
    
    return reconstructed_words

def compare_poem_reconstruction(model: RNNAutoencoder, dataset: AutoencoderDataset, 
                              poem_index: int, poems: List[Dict],
                              vocabulary: Dict[str, int], embeddings_dict: Dict[str, torch.Tensor]):
    """Compare original poem with its reconstruction."""
    
    # Get the data sample
    if poem_index >= len(dataset):
        raise ValueError(f"Poem index {poem_index} out of range (0-{len(dataset)-1})")
    
    sample = dataset[poem_index]
    original_embeddings = sample['input_sequences'].unsqueeze(0)  # Add batch dimension
    attention_mask = sample['attention_mask'].unsqueeze(0)
    metadata = sample['metadata']
    
    print(f"\n🎭 POEM COMPARISON #{poem_index}")
    print("=" * 80)
    
    # Show original poem info
    poem_info = poems[metadata['poem_idx']]
    print(f"📖 Original Poem: \"{poem_info['title']}\" by {poem_info['author']}")
    print(f"📊 Chunk: {metadata['chunk_id'] + 1}/{metadata['total_chunks_in_poem']} (tokens {metadata['start_position']}-{metadata['end_position']})")
    
    # Extract original text chunk
    original_text = poem_info['text']
    words = original_text.split()
    chunk_words = words[metadata['start_position']:metadata['end_position']]
    original_chunk = ' '.join(chunk_words)
    
    print(f"\n📝 ORIGINAL TEXT:")
    print("-" * 40)
    print(f"{original_chunk}")
    
    # Run through autoencoder
    with torch.no_grad():
        batch = {
            'input_sequences': original_embeddings,
            'attention_mask': attention_mask
        }
        
        output = model(batch)
        reconstructed_embeddings = output['reconstructed']
        bottleneck = output['bottleneck']
    
    # Compute metrics
    metrics = compute_similarity_metrics(
        original_embeddings.squeeze(0), 
        reconstructed_embeddings.squeeze(0), 
        attention_mask.squeeze(0)
    )
    
    # Approximate reconstruction (find closest words)
    reconstructed_words = reconstruct_text_approximation(
        reconstructed_embeddings.squeeze(0),
        attention_mask.squeeze(0),
        vocabulary,
        embeddings_dict
    )
    reconstructed_text = ' '.join(reconstructed_words)
    
    print(f"\n🔄 RECONSTRUCTED TEXT:")
    print("-" * 40)
    print(f"{reconstructed_text}")
    
    print(f"\n📊 SIMILARITY METRICS:")
    print("-" * 40)
    print(f"Cosine Similarity: {metrics['cosine_similarity']:.4f}")
    print(f"MSE Loss:          {metrics['mse']:.6f}")
    print(f"L2 Distance:       {metrics['l2_distance']:.4f}")
    print(f"Valid Tokens:      {int(metrics['valid_tokens'])}")
    
    # Bottleneck analysis
    print(f"\n🗜️  COMPRESSION ANALYSIS:")
    print("-" * 40)
    print(f"Bottleneck Dimension: {bottleneck.shape[-1]}D")
    print(f"Compression Ratio:    {original_embeddings.shape[-1]}D → {bottleneck.shape[-1]}D ({original_embeddings.shape[-1]/bottleneck.shape[-1]:.1f}x)")
    print(f"Bottleneck Range:     [{bottleneck.min().item():.3f}, {bottleneck.max().item():.3f}]")
    print(f"Bottleneck Std:       {bottleneck.std().item():.4f}")
    
    # Performance assessment
    print(f"\n🎯 QUALITY ASSESSMENT:")
    print("-" * 40)
    cosine_sim = metrics['cosine_similarity']
    if cosine_sim >= 0.90:
        quality = "Excellent"
        emoji = "🟢"
    elif cosine_sim >= 0.80:
        quality = "Good"
        emoji = "🟡"
    elif cosine_sim >= 0.70:
        quality = "Fair"
        emoji = "🟠"
    else:
        quality = "Needs Improvement"
        emoji = "🔴"
    
    print(f"{emoji} Quality: {quality} ({cosine_sim:.4f} similarity)")
    
    return metrics

def interactive_poem_browser(model: RNNAutoencoder, dataset: AutoencoderDataset, 
                           poems: List[Dict], vocabulary: Dict[str, int], 
                           embeddings_dict: Dict[str, torch.Tensor]):
    """Interactive browser for comparing multiple poems."""
    print(f"\n🎭 INTERACTIVE POEM BROWSER")
    print("=" * 50)
    print(f"Dataset contains {len(dataset)} poem chunks from {len(poems)} poems")
    print("Commands:")
    print("  [number] - View poem chunk by index")
    print("  'random' - View random poem chunk")
    print("  'list'   - List available poems")
    print("  'quit'   - Exit browser")
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd in ['quit', 'exit', 'q']:
                break
            elif cmd == 'random':
                poem_index = np.random.randint(0, len(dataset))
                compare_poem_reconstruction(model, dataset, poem_index, poems, vocabulary, embeddings_dict)
            elif cmd == 'list':
                print("\n📚 Available Poems:")
                for i, poem in enumerate(poems[:10]):  # Show first 10
                    print(f"  {i}: \"{poem['title']}\" by {poem['author']}")
                if len(poems) > 10:
                    print(f"  ... and {len(poems) - 10} more")
            elif cmd.isdigit():
                poem_index = int(cmd)
                compare_poem_reconstruction(model, dataset, poem_index, poems, vocabulary, embeddings_dict)
            else:
                print("❌ Unknown command. Try a number, 'random', 'list', or 'quit'")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thanks for exploring poem reconstructions!")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare original poems with autoencoder reconstructions')
    parser.add_argument('--model', type=str, default='attention_optimized_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--poem-index', type=int, default=None,
                       help='Specific poem chunk index to analyze')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to poems dataset')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive poem browser')
    
    args = parser.parse_args()
    
    print("🎭 Poem Reconstruction Comparison")
    print("=" * 50)
    
    try:
        # Load model
        model, checkpoint = load_trained_model(args.model)
        
        # Load poems and setup preprocessor
        poems = load_poems(args.data_path)
        preprocessor, dataset = setup_preprocessor()
        
        # Get vocabulary and embeddings for text reconstruction
        vocabulary = dataset.vocabulary
        # Simple embeddings dict (in practice, you'd want the full GLoVe embeddings)
        embeddings_dict = {}
        
        print(f"\n🔍 Ready to compare reconstructions!")
        
        if args.interactive:
            # Interactive browser
            interactive_poem_browser(model, dataset, poems, vocabulary, embeddings_dict)
        elif args.poem_index is not None:
            # Single poem comparison
            compare_poem_reconstruction(model, dataset, args.poem_index, poems, vocabulary, embeddings_dict)
        else:
            # Default: show a few examples
            print("\n📋 Showing sample reconstructions...")
            sample_indices = [0, len(dataset)//4, len(dataset)//2, 3*len(dataset)//4, len(dataset)-1]
            
            for i, idx in enumerate(sample_indices[:3]):  # Show first 3
                compare_poem_reconstruction(model, dataset, idx, poems, vocabulary, embeddings_dict)
                if i < 2:  # Add separator
                    print("\n" + "="*80)
            
            print(f"\n💡 Use --interactive to explore more poems")
            print(f"💡 Use --poem-index N to see specific chunk")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())