#!/usr/bin/env python3
"""
Regenerate chunked sequences with fixed tokenizer
This script applies the tokenizer fix and regenerates preprocessed artifacts
"""

import json
import numpy as np
import re
import spacy
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os

print("🔧 REGENERATING SEQUENCES WITH FIXED TOKENIZER")
print("=" * 60)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ Loaded en_core_web_sm")
except OSError:
    from spacy.lang.en import English
    nlp = English()
    print("✓ Using fallback English tokenizer")

# Load poems
try:
    with open('/home/without-a-care-in-the-world/workflows/poetry_autoencoding/dataset_poetry/multi_poem_dbbc_collection.json', 'r') as f:
        poems = json.load(f)
    print(f"✓ Loaded {len(poems)} poems")
except FileNotFoundError:
    print("❌ Could not load poems")
    exit(1)

# Load GLoVe embeddings
print("\n📥 Loading GLoVe embeddings...")
embeddings_path = "/home/without-a-care-in-the-world/workflows/poetry_autoencoding/embeddings/glove.6B.300d.txt"
embeddings = {}
embedding_dim = 300

try:
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num % 50000 == 0:
                print(f"  Loaded {line_num:,} embeddings...")
            parts = line.strip().split()
            if len(parts) != 301:  # word + 300 dimensions
                continue  # Skip malformed lines
            word = parts[0]
            try:
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                embeddings[word] = vector
            except ValueError:
                continue  # Skip lines with non-numeric values
    print(f"✓ Loaded {len(embeddings):,} GLoVe embeddings")
except FileNotFoundError:
    print("❌ GLoVe embeddings not found")
    exit(1)

def clean_poetry_text(text):
    """Simple text cleaning"""
    return re.sub(r'\s+', ' ', text.strip())

class FixedPoetryTokenizer:
    """Fixed tokenizer that preserves numbers, Unicode, and case"""
    
    def __init__(self):
        self.nlp = nlp
        self.special_tokens = {'<LINE_BREAK>', '<STANZA_BREAK>', '<POEM_START>', '<POEM_END>'}
        
    def preprocess_text(self, text: str) -> str:
        """Fixed preprocessing - preserves aesthetic elements"""
        processed = re.sub(r'\n\s*\n', ' <STANZA_BREAK> ', text)
        processed = re.sub(r'\n', ' <LINE_BREAK> ', processed)
        processed = re.sub(r'\s+', ' ', processed)
        processed = clean_poetry_text(processed)
        # FIXED: Don't force lowercase - preserves numbers, Unicode, aesthetic casing
        return processed.strip()
        
    def tokenize(self, text: str) -> List[str]:
        """Fixed tokenization"""
        processed_text = self.preprocess_text(text)
        doc = self.nlp(processed_text)
        
        tokens = []
        for token in doc:
            # Keep special tokens
            if token.text in self.special_tokens:
                tokens.append(token.text)
            # Skip pure whitespace
            elif not token.is_space and token.text.strip():
                tokens.append(token.text.strip())
        
        return tokens

def get_embedding(word: str, embeddings: dict, embedding_dim: int = 300) -> np.ndarray:
    """Get embedding for word with fallback strategies"""
    # Try exact match
    if word in embeddings:
        return embeddings[word]
    
    # Try lowercase
    if word.lower() in embeddings:
        return embeddings[word.lower()]
    
    # Try without punctuation
    clean_word = re.sub(r'[^\w]', '', word.lower())
    if clean_word and clean_word in embeddings:
        return embeddings[clean_word]
    
    # Return random vector for OOV - FIXED to match GLoVe distribution
    vec = np.random.normal(0, 0.4, embedding_dim)  # Larger std
    vec = vec / np.linalg.norm(vec) * 7.0  # Normalize to typical GLoVe norm
    return vec.astype(np.float32)

def create_chunked_sequences(poems: List[Dict], tokenizer: FixedPoetryTokenizer, 
                           embeddings: dict, max_length: int = 50, 
                           overlap: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Create chunked sequences with fixed tokenizer"""
    
    all_token_sequences = []
    all_embedding_sequences = []
    all_attention_masks = []
    chunk_metadata = []
    
    stats = {
        'total_poems': len(poems),
        'total_chunks': 0,
        'tokens_preserved': 0,
        'tokens_total': 0,
        'oov_count': 0,
        'numbers_preserved': 0,
        'unicode_preserved': 0
    }
    
    stride = max_length - overlap
    
    print(f"\n🔄 Processing {len(poems)} poems with chunking...")
    print(f"   Window size: {max_length}, Overlap: {overlap}, Stride: {stride}")
    
    for poem_idx, poem in enumerate(poems):
        if poem_idx % 25 == 0:
            print(f"   Processing poem {poem_idx+1}/{len(poems)}...")
            
        # Tokenize with fixed tokenizer
        tokens = tokenizer.tokenize(poem['text'])
        stats['tokens_total'] += len(tokens)
        
        # Count preserved elements
        numbers = sum(1 for t in tokens if any(c.isdigit() for c in t))
        unicode_chars = sum(1 for t in tokens if any(ord(c) > 127 for c in t))
        stats['numbers_preserved'] += numbers
        stats['unicode_preserved'] += unicode_chars
        
        if len(tokens) < 5:  # Skip very short poems
            continue
            
        # Create chunks with sliding window
        chunks_in_poem = 0
        for start_idx in range(0, len(tokens), stride):
            end_idx = min(start_idx + max_length, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            if len(chunk_tokens) < 5:  # Skip very short chunks
                continue
                
            # Pad to max_length
            padded_tokens = chunk_tokens + ['<PAD>'] * (max_length - len(chunk_tokens))
            
            # Create embeddings
            chunk_embeddings = []
            attention_mask = []
            
            for token in padded_tokens:
                if token == '<PAD>':
                    chunk_embeddings.append(np.zeros(300, dtype=np.float32))
                    attention_mask.append(0)
                else:
                    embedding = get_embedding(token, embeddings)
                    if token not in embeddings and token.lower() not in embeddings:
                        stats['oov_count'] += 1
                    chunk_embeddings.append(embedding)
                    attention_mask.append(1)
            
            all_token_sequences.append(padded_tokens)
            all_embedding_sequences.append(np.array(chunk_embeddings))
            all_attention_masks.append(attention_mask)
            
            chunk_metadata.append({
                'poem_idx': poem_idx,
                'chunk_id': chunks_in_poem,
                'start_pos': start_idx,
                'end_pos': end_idx,
                'original_length': len(chunk_tokens),
                'author': poem.get('author', 'Unknown'),
                'title': poem.get('title', 'Untitled')[:50] + ('...' if len(poem.get('title', '')) > 50 else '')
            })
            
            chunks_in_poem += 1
            stats['total_chunks'] += 1
            
    stats['tokens_preserved'] = stats['tokens_total'] - stats['oov_count']
    
    return (
        np.array(all_embedding_sequences, dtype=np.float32),
        np.array(all_attention_masks, dtype=np.int32), 
        all_token_sequences,
        {'chunk_metadata': chunk_metadata, 'processing_stats': stats}
    )

# Initialize fixed tokenizer
tokenizer = FixedPoetryTokenizer()
print("\n✅ Initialized fixed tokenizer (preserves numbers, Unicode, case)")

# Test the tokenizer on problematic example
test_text = poems[0]['text'][:120]
test_tokens = tokenizer.tokenize(test_text)
print(f"\n🔍 TOKENIZER TEST:")
print(f"Original: {repr(test_text)}")
print(f"Tokens ({len(test_tokens)}): {test_tokens[:10]}...")

# Check preservation
numbers = [t for t in test_tokens if any(c.isdigit() for c in t)]
uppercase = [t for t in test_tokens if any(c.isupper() for c in t)]
print(f"Numbers preserved: {numbers}")
print(f"Uppercase preserved: {uppercase}")

# Generate chunked sequences
print("\n🚀 GENERATING CHUNKED SEQUENCES...")
embedding_sequences, attention_masks, token_sequences, metadata = create_chunked_sequences(
    poems, tokenizer, embeddings, max_length=50, overlap=10
)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
artifacts_dir = "preprocessed_artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

print(f"\n💾 SAVING RESULTS (timestamp: {timestamp})...")

# Save sequences
np.save(f"{artifacts_dir}/embedding_sequences_{timestamp}.npy", embedding_sequences)
np.save(f"{artifacts_dir}/attention_masks_{timestamp}.npy", attention_masks)
np.save(f"{artifacts_dir}/token_sequences_{timestamp}.npy", token_sequences)

# Save metadata
with open(f"{artifacts_dir}/chunk_metadata_{timestamp}.json", 'w') as f:
    json.dump(metadata, f, indent=2)

# Also save as "latest" for easy access
np.save(f"{artifacts_dir}/embedding_sequences_latest.npy", embedding_sequences)
np.save(f"{artifacts_dir}/attention_masks_latest.npy", attention_masks)
np.save(f"{artifacts_dir}/token_sequences_latest.npy", token_sequences)

with open(f"{artifacts_dir}/chunk_metadata_latest.json", 'w') as f:
    json.dump(metadata, f, indent=2)

# Print results
stats = metadata['processing_stats']
print(f"\n📊 FINAL RESULTS:")
print(f"✅ Total poems processed: {stats['total_poems']}")
print(f"✅ Total chunks generated: {stats['total_chunks']}")
print(f"✅ Total tokens: {stats['tokens_total']:,}")
print(f"✅ Numbers preserved: {stats['numbers_preserved']}")
print(f"✅ Unicode characters preserved: {stats['unicode_preserved']}")
print(f"✅ Out-of-vocabulary tokens: {stats['oov_count']}")
print(f"✅ Embedding sequences shape: {embedding_sequences.shape}")
print(f"✅ Attention masks shape: {attention_masks.shape}")

print(f"\n🎉 SUCCESS: Fixed tokenizer regenerated {stats['total_chunks']} sequences!")
print(f"📁 Saved to preprocessed_artifacts/ with timestamp {timestamp}")
print("Ready for RNN autoencoder training with properly preserved poetry elements!")