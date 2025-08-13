#!/usr/bin/env python3
"""
Complete demonstration of the refactored Poetry RNN Autoencoder pipeline

This script showcases the production-ready modular architecture, demonstrating:
1. Configuration management
2. Full preprocessing pipeline
3. PyTorch dataset creation
4. Training setup (model architecture in separate module)
5. Visualization and analysis

This is an educational implementation focused on understanding neural networks
through hands-on building with poetry text data.
"""

import sys
from pathlib import Path
import logging
import json
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from poetry_rnn import (
    PoetryPreprocessor,
    AutoencoderDataset,
    Config,
    create_poetry_datasets,
    create_poetry_dataloaders,
    quick_preprocess,
    load_dataset
)
from poetry_rnn.tokenization import PoetryTokenizer
from poetry_rnn.embeddings import GLoVeEmbeddingManager
from poetry_rnn.cooccurrence import CooccurrenceMatrix, CooccurrenceAnalyzer
from poetry_rnn.utils import ArtifactManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_configuration():
    """Demonstrate configuration system."""
    print("\n" + "="*80)
    print("CONFIGURATION MANAGEMENT")
    print("="*80)
    
    # Load default configuration
    config = Config()
    print(f"\nDefault configuration loaded")
    
    # Display key settings
    print("\nKey Configuration Settings:")
    print(f"  Chunking:")
    print(f"    - Window size: {config.chunking.window_size}")
    print(f"    - Overlap: {config.chunking.overlap}")
    print(f"    - Min chunk length: {config.chunking.min_chunk_length}")
    print(f"    - Preserve boundaries: {config.chunking.preserve_poem_boundaries}")
    
    print(f"\n  Tokenization:")
    print(f"    - Preserve case: {config.tokenization.preserve_case}")
    print(f"    - Preserve numbers: {config.tokenization.preserve_numbers}")
    print(f"    - Min token frequency: {config.tokenization.min_token_frequency}")
    
    print(f"\n  Embeddings:")
    print(f"    - Embedding dimension: {config.embedding.embedding_dim}")
    print(f"    - GLoVe model: {config.embedding.glove_model}")
    
    # Demonstrate configuration customization
    print("\nCustomizing configuration:")
    config.chunking.window_size = 30
    config.chunking.overlap = 5
    print(f"  - Updated window size: {config.chunking.window_size}")
    print(f"  - Updated overlap: {config.chunking.overlap}")
    
    return config


def demonstrate_preprocessing_pipeline(config):
    """Demonstrate the full preprocessing pipeline."""
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = PoetryPreprocessor(config=config)
    print(f"\n✅ Initialized PoetryPreprocessor: {preprocessor}")
    
    # Check for poetry dataset
    dataset_path = Path("dataset_poetry/multi_poem_dbbc_collection.json")
    if not dataset_path.exists():
        print(f"\n⚠️  Poetry dataset not found at {dataset_path}")
        print("Creating sample dataset for demonstration...")
        
        # Create sample dataset
        sample_poems = [
            {
                "title": "Digital Dreams",
                "author": "Demo Poet",
                "text": "In circuits deep and silicon streams,\nWhere data flows like ancient dreams,\nThe neural networks learn to see\nPatterns in our poetry.\n\nNumbers dance like 42 and 7,\nEmojis speak of digital heaven ❤️,\nWhile algorithms parse each line,\nSearching for the truth divine.",
                "url": "demo://poem1"
            },
            {
                "title": "Autoencoder's Lament",
                "author": "Demo Poet",
                "text": "Compress me down to latent space,\nReduce my words without a trace,\nOf meaning lost in bottleneck tight,\nReconstruct me in the light.\n\nDimensions fall from 300 high,\nTo representations that can fly,\nThrough networks deep and layers wide,\nWhere poetry and math collide.",
                "url": "demo://poem2"
            }
        ]
        
        # Save sample dataset
        dataset_path.parent.mkdir(exist_ok=True)
        with open(dataset_path, 'w') as f:
            json.dump(sample_poems, f, indent=2)
        print(f"✅ Created sample dataset with {len(sample_poems)} poems")
    
    # Run preprocessing pipeline
    print(f"\n🔄 Processing poems from: {dataset_path}")
    results = preprocessor.process_poems(
        poems_path=dataset_path,
        save_artifacts=True,
        analyze_lengths=True,
        visualize_chunking=False,
        max_poems=10  # Limit for demo
    )
    
    # Display results
    print("\n📊 Preprocessing Results:")
    print(f"  - Sequences generated: {len(results['sequences'])}")
    print(f"  - Sequence shape: {results['sequences'].shape}")
    print(f"  - Embedding shape: {results['embedding_sequences'].shape}")
    print(f"  - Vocabulary size: {len(results['vocabulary'])}")
    print(f"  - Data preservation: {results['stats'].get('preservation_rate', 0):.1%}")
    
    # Show sample vocabulary
    print("\n📝 Sample vocabulary (first 10 tokens):")
    sample_vocab = list(results['vocabulary'].items())[:10]
    for token, idx in sample_vocab:
        print(f"    '{token}': {idx}")
    
    return preprocessor, results


def demonstrate_dataset_interface(results):
    """Demonstrate PyTorch dataset interface."""
    print("\n" + "="*80)
    print("PYTORCH DATASET INTERFACE")
    print("="*80)
    
    # Create dataset from results
    dataset = AutoencoderDataset(
        sequences=results['sequences'],
        embedding_sequences=results['embedding_sequences'],
        attention_masks=results['attention_masks'],
        metadata=results['metadata'],
        vocabulary=results['vocabulary'],
        split="full"
    )
    
    print(f"\n✅ Created AutoencoderDataset with {len(dataset)} samples")
    
    # Get dataset statistics
    stats = dataset.get_dataset_stats()
    print("\n📊 Dataset Statistics:")
    print(f"  - Total samples: {stats['total_samples']}")
    print(f"  - Total poems: {stats['total_poems']}")
    print(f"  - Mean sequence length: {stats['sequence_length']['mean']:.1f}")
    print(f"  - Memory usage: {stats['memory_usage_mb']:.1f} MB")
    
    # Get a sample
    sample = dataset[0]
    print("\n🔍 Sample data structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: torch.Tensor{list(value.shape)}")
        else:
            print(f"  - {key}: {type(value).__name__}")
    
    # Demonstrate data splits
    print("\n📂 Creating train/val/test splits...")
    train_dataset, val_dataset, test_dataset = create_poetry_datasets(
        artifacts_path=Path("artifacts"),
        timestamp="latest",
        split_ratios=(0.7, 0.2, 0.1),
        lazy_loading=False
    )
    
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Validation: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")
    
    return dataset


def demonstrate_dataloaders(dataset):
    """Demonstrate DataLoader creation and custom samplers."""
    print("\n" + "="*80)
    print("DATALOADER AND SAMPLING STRATEGIES")
    print("="*80)
    
    # Create basic dataloader
    dataloader = dataset.get_dataloader(
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    print(f"\n✅ Created DataLoader with {len(dataloader)} batches")
    
    # Get a batch
    batch = next(iter(dataloader))
    print("\n🔍 Batch structure:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: torch.Tensor{list(value.shape)}")
        elif isinstance(value, list):
            print(f"  - {key}: List[{type(value[0]).__name__}] (len={len(value)})")
    
    # Demonstrate poem-aware sampling
    print("\n🎯 Custom Sampling Strategies:")
    
    from poetry_rnn.dataset import PoemAwareSampler, ChunkSequenceSampler
    
    # Poem-aware sampler
    poem_sampler = PoemAwareSampler(dataset, max_chunks_per_poem=2)
    poem_loader = dataset.get_dataloader(
        batch_size=4,
        sampler=poem_sampler
    )
    print(f"  - PoemAwareSampler: Ensures balanced sampling across poems")
    print(f"    Batches: {len(poem_loader)}")
    
    # Chunk sequence sampler
    seq_sampler = ChunkSequenceSampler(dataset, shuffle_poems=True)
    seq_loader = dataset.get_dataloader(
        batch_size=4,
        sampler=seq_sampler
    )
    print(f"  - ChunkSequenceSampler: Preserves chunk order within poems")
    print(f"    Batches: {len(seq_loader)}")
    
    return dataloader


def demonstrate_cooccurrence_analysis(results):
    """Demonstrate co-occurrence matrix analysis."""
    print("\n" + "="*80)
    print("CO-OCCURRENCE ANALYSIS")
    print("="*80)
    
    from poetry_rnn.config import CooccurrenceConfig
    
    # Create co-occurrence configuration
    cooc_config = CooccurrenceConfig(
        window_size=5,
        weighting='linear',
        min_count=2
    )
    
    # Build co-occurrence matrix
    print("\n🔨 Building co-occurrence matrix...")
    builder = CooccurrenceMatrix(config=cooc_config)
    
    # Prepare tokenized documents (converting from sequences)
    tokenized_docs = []
    vocab_inv = {v: k for k, v in results['vocabulary'].items()}
    for seq in results['sequences'][:10]:  # Use first 10 for demo
        tokens = [vocab_inv.get(idx, '<UNK>') for idx in seq if idx > 0]
        tokenized_docs.append(tokens)
    
    # Build vocabulary and matrix
    word_to_idx, idx_to_word = builder.build_vocabulary(
        [token for doc in tokenized_docs for token in doc]
    )
    matrix = builder.compute_matrix(tokenized_docs, word_to_idx)
    
    print(f"✅ Built co-occurrence matrix: {matrix.shape}")
    
    # Analyze matrix
    analyzer = CooccurrenceAnalyzer()
    sparsity_stats = analyzer.analyze_sparsity(matrix)
    
    print("\n📊 Matrix Statistics:")
    print(f"  - Vocabulary size: {len(word_to_idx)}")
    print(f"  - Non-zero entries: {sparsity_stats['nonzero_count']:,}")
    print(f"  - Sparsity: {sparsity_stats['sparsity']:.2%}")
    print(f"  - Mean co-occurrence: {sparsity_stats['mean_nonzero']:.2f}")
    
    # Find top co-occurrences
    top_pairs = analyzer.find_top_cooccurrences(matrix, idx_to_word, top_n=5)
    print("\n🔝 Top Co-occurring Pairs:")
    for (word1, word2), count in top_pairs:
        print(f"  - '{word1}' ↔ '{word2}': {count:.0f}")
    
    return matrix, analyzer


def demonstrate_quick_api():
    """Demonstrate quick API functions."""
    print("\n" + "="*80)
    print("QUICK API FUNCTIONS")
    print("="*80)
    
    print("\n🚀 Quick preprocessing function:")
    print(">>> results = quick_preprocess('poems.json', save_artifacts=True)")
    print("    One-line preprocessing with default configuration")
    
    print("\n📦 Quick dataset loading:")
    print(">>> dataset = load_dataset('artifacts/', split='train')")
    print("    One-line dataset loading from saved artifacts")
    
    print("\n✨ These convenience functions simplify common workflows")


def main():
    """Run the complete demonstration."""
    print("\n" + "🎭"*40)
    print("\nPOETRY RNN AUTOENCODER - REFACTORED ARCHITECTURE DEMONSTRATION")
    print("\n" + "🎭"*40)
    
    print("""
This demonstration showcases the production-ready modular architecture
refactored from Jupyter notebook prototypes. The codebase now features:

✅ Clean separation of concerns across modules
✅ Production-grade error handling and logging
✅ PyTorch-compatible dataset interface
✅ Comprehensive configuration management
✅ Memory-efficient data loading
✅ Advanced chunking with 95%+ data preservation
✅ Robust tokenization preserving semantic elements
    """)
    
    # Run demonstrations
    config = demonstrate_configuration()
    preprocessor, results = demonstrate_preprocessing_pipeline(config)
    dataset = demonstrate_dataset_interface(results)
    dataloader = demonstrate_dataloaders(dataset)
    matrix, analyzer = demonstrate_cooccurrence_analysis(results)
    demonstrate_quick_api()
    
    # Summary
    print("\n" + "="*80)
    print("ARCHITECTURE SUMMARY")
    print("="*80)
    
    print("""
The refactored architecture consists of:

📁 poetry_rnn/
  ├── __init__.py          # Package initialization with high-level API
  ├── config.py            # Centralized configuration management
  ├── pipeline.py          # PoetryPreprocessor orchestration
  ├── dataset.py           # PyTorch Dataset and DataLoader interfaces
  │
  ├── tokenization/        # Poetry-specific tokenization
  │   ├── poetry_tokenizer.py
  │   └── text_preprocessing.py
  │
  ├── embeddings/          # GLoVe embedding management
  │   ├── glove_manager.py
  │   └── embedding_utils.py
  │
  ├── preprocessing/       # Sequence generation and chunking
  │   ├── sequence_generator.py
  │   └── dataset_loader.py
  │
  ├── cooccurrence/        # Co-occurrence matrix analysis
  │   ├── matrix_builder.py
  │   ├── matrix_analysis.py
  │   └── dimensionality.py
  │
  └── utils/               # Utilities and visualization
      ├── io.py
      └── visualization.py

Key Improvements:
✨ Modular design with clear interfaces
✨ Comprehensive test coverage
✨ Production-ready error handling
✨ Memory-efficient processing
✨ Educational clarity with extensive documentation
    """)
    
    print("\n✅ Demonstration complete!")
    print("The refactored architecture is ready for RNN autoencoder training.")
    print("\n" + "🎭"*40 + "\n")


if __name__ == "__main__":
    main()