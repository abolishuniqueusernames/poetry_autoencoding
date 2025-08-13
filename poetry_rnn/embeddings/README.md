# GLoVe Embeddings Module

This module provides production-grade GLoVe embedding management for the Poetry RNN Autoencoder project. It was extracted and refactored from the Jupyter notebook implementation with significant enhancements.

## Components

### `GLoVeEmbeddingManager`

Core class for loading, aligning, and managing GLoVe embeddings.

**Key Features:**
- Integration with the configuration system (`EmbeddingConfig`)
- Robust loading with malformed line handling
- Vocabulary alignment with fallback strategies (lowercase, cleaned words)
- Mock embedding generation for testing
- Configurable OOV (out-of-vocabulary) handling strategies
- Memory-efficient operations with progress tracking
- Comprehensive error handling and validation

**Basic Usage:**
```python
from poetry_rnn.config import Config
from poetry_rnn.embeddings import GLoVeEmbeddingManager

# Using default configuration
config = Config()
manager = GLoVeEmbeddingManager(config=config.embedding)

# Load embeddings (will create mocks if file not found)
manager.load_glove_embeddings()

# Align with your vocabulary
vocabulary = {'love': 0, 'heart': 1, 'pain': 2}  # word -> index mapping
embedding_matrix, stats = manager.align_vocabulary(vocabulary)

print(f"Aligned {len(vocabulary)} words with {stats['found_exact']} exact matches")
```

### `EmbeddingAnalyzer`

High-level analysis tools for exploring embedding spaces.

**Features:**
- Word similarity search
- Analogy resolution (king : queen :: man : ?)
- Semantic cluster analysis
- Vocabulary statistics
- Outlier detection

**Basic Usage:**
```python
from poetry_rnn.embeddings import create_analyzer

# Create analyzer (requires aligned embeddings)
analyzer = create_analyzer(manager)

# Find similar words
similar = analyzer.find_similar_words('love', top_k=5)
print(f"Words similar to 'love': {[word for word, score in similar]}")

# Solve analogies
candidates = analyzer.solve_analogy('day', 'light', 'night', top_k=3)
print(f"day : light :: night : {candidates[0][0]}")

# Test poetry analogies
results = analyzer.test_poetry_analogies()
print(f"Analogy tests: {results['successful_tests']}/{results['total_tests']} passed")
```

## Configuration

The module integrates with the existing configuration system:

```python
from poetry_rnn.config import EmbeddingConfig

config = EmbeddingConfig(
    embedding_dim=300,
    embedding_path="/path/to/glove.6B.300d.txt",
    create_mock_if_missing=True,
    oov_strategy="random_normal",  # "random_normal", "zero", "mean"
    alignment_fallback=True  # Try lowercase/cleaned versions
)

manager = GLoVeEmbeddingManager(config=config)
```

## Special Tokens

The module preserves special tokens for poetry processing:
- `<UNK>`: Unknown/out-of-vocabulary words
- `<LINE_BREAK>`: Line breaks in poems
- `<STANZA_BREAK>`: Stanza boundaries
- `<POEM_START>`: Beginning of poem
- `<POEM_END>`: End of poem

## Error Handling

The module provides robust error handling:
- Validates configuration parameters
- Handles missing embedding files (with mock fallback)
- Skips malformed lines in embedding files
- Provides informative error messages
- Logs progress and statistics

## Performance Features

- **Memory-efficient loading**: Progress tracking for large files
- **Robust parsing**: Handles malformed lines gracefully
- **Batch operations**: Efficient vocabulary alignment
- **Caching support**: Save/load aligned embeddings

## Preserved Fixes

This implementation preserves all recent fixes from the notebook:
- ✅ Path handling with leading slash fix
- ✅ `special_tokens` attribute properly initialized
- ✅ Robust loading with malformed line handling
- ✅ Mock embedding fallback for testing
- ✅ Configuration system integration

## Testing

The module includes comprehensive validation and can be tested with:

```python
# Test with mock embeddings
config = EmbeddingConfig(embedding_path=None, create_mock_if_missing=True)
manager = GLoVeEmbeddingManager(config=config)
manager.load_glove_embeddings()

# Test vocabulary alignment
test_vocab = {'love': 0, 'heart': 1, 'pain': 2}
embedding_matrix, stats = manager.align_vocabulary(test_vocab)

# Test analyzer
analyzer = create_analyzer(manager)
similar_words = analyzer.find_similar_words('love')
```

The module has been thoroughly tested and all functionality verified to work correctly.