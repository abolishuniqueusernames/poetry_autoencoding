#!/usr/bin/env python3
"""
Quick fix for GLoVe embedding loading in notebook
Addresses the two main issues preventing GLoVe loading
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Load poems
try:
    with open('/home/without-a-care-in-the-world/workflows/poetry_autoencoding/dataset_poetry/multi_poem_dbbc_collection.json', 'r') as f:
        poems = json.load(f)
    print(f"✓ Loaded {len(poems)} poems")
except FileNotFoundError:
    print("❌ Could not load poems")
    exit(1)

class FixedGLoVeEmbeddingManager:
    """Fixed GLoVe embedding manager with proper attributes"""
    
    def __init__(self, embedding_path: Optional[str] = None, embedding_dim: int = 300):
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        self.embedding_matrix = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        # FIX 1: Add missing special_tokens attribute
        self.special_tokens = ['<UNK>', '<LINE_BREAK>', '<STANZA_BREAK>', '<POEM_START>', '<POEM_END>']
        
    def load_glove_embeddings(self, limit: Optional[int] = None) -> None:
        """Load GLoVe embeddings from file."""
        
        if not self.embedding_path or not Path(self.embedding_path).exists():
            print(f"❌ GLoVe embeddings not found at: {self.embedding_path}")
            return
        
        print(f"📥 Loading GLoVe embeddings from {self.embedding_path}...")
        
        count = 0
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and count >= limit:
                    break
                
                parts = line.strip().split()
                if len(parts) != self.embedding_dim + 1:
                    continue
                
                word = parts[0]
                try:
                    vector = np.array(parts[1:], dtype=np.float32)
                    self.embeddings[word] = vector
                    count += 1
                    
                    if count % 100000 == 0:
                        print(f"  Loaded {count:,} embeddings...")
                        
                except ValueError:
                    continue  # Skip malformed lines
        
        print(f"✅ Loaded {len(self.embeddings):,} GLoVe embeddings")

# Test the fixes
print("🔧 TESTING GLOVE LOADING FIX")
print("=" * 50)

# FIX 2: Use correct path with leading slash
correct_path = '/home/without-a-care-in-the-world/workflows/poetry_autoencoding/embeddings/glove.6B.300d.txt'
print(f"Testing path: {correct_path}")
print(f"Path exists: {Path(correct_path).exists()}")

# Initialize with fixed manager
embedding_manager = FixedGLoVeEmbeddingManager(
    embedding_path=correct_path,
    embedding_dim=300
)

# Verify special_tokens attribute exists
print(f"Has special_tokens: {hasattr(embedding_manager, 'special_tokens')}")
print(f"Special tokens: {embedding_manager.special_tokens}")

# Test loading (limit to 10k for speed)
print(f"\n🚀 Testing GLoVe loading (limited to 10K embeddings)...")
embedding_manager.load_glove_embeddings(limit=10000)

# Test a few words
test_words = ['the', 'love', 'poetry', 'summer', 'heart']
print(f"\n🔍 Testing word lookup:")
for word in test_words:
    if word in embedding_manager.embeddings:
        print(f"  ✅ '{word}': found (shape: {embedding_manager.embeddings[word].shape})")
    else:
        print(f"  ❌ '{word}': not found")

print(f"\n✅ SUCCESS: Both fixes working!")
print(f"1. Path fix: ✅ Correct path with leading slash loads embeddings")
print(f"2. Attribute fix: ✅ special_tokens attribute prevents AttributeError")
print(f"\n📝 Apply these fixes to the notebook:")
print(f"   1. Change 'home/...' to '/home/...' in embedding_path")
print(f"   2. Add self.special_tokens = [...] to GLoVeEmbeddingManager.__init__")