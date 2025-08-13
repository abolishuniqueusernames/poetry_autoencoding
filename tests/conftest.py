"""
pytest configuration and shared fixtures for poetry RNN tests
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import json

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from poetry_rnn.config import Config, TokenizationConfig, EmbeddingConfig, CooccurrenceConfig


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_poems() -> List[Dict[str, Any]]:
    """Sample poetry dataset for testing"""
    return [
        {
            "url": "test://poem1",
            "author": "Test Poet",
            "title": "Test Poem One",
            "text": "Love is like the summer rain,\nFalling softly on my heart.\nNumbers like 6 and 80 remain,\nWhen we are far apart.\n\n❤️ forever ❤️",
            "content_type": "poetry",
            "length": 95,
            "line_count": 5,
            "source": "Test Collection",
            "dbbc_score": 25
        },
        {
            "url": "test://poem2", 
            "author": "Test Poet",
            "title": "＊✿❀ Digital Dreams ❀✿＊",
            "text": "i write in lowercase\nbecause CAPITALS are for\nSERIOUS THINGS\n\nlike paying taxes\nand forgetting passwords",
            "content_type": "poetry",
            "length": 78,
            "line_count": 6,
            "source": "Test Collection",
            "dbbc_score": 18
        },
        {
            "url": "test://poem3",
            "author": "Another Poet", 
            "title": "Short Poem",
            "text": "brief\npoetic\nmoment",
            "content_type": "poetry",
            "length": 19,
            "line_count": 3,
            "source": "Test Collection", 
            "dbbc_score": 10
        }
    ]


@pytest.fixture
def sample_tokens() -> List[str]:
    """Sample tokenized text for testing"""
    return [
        'Love', 'is', 'like', 'the', 'summer', 'rain', ',',
        '<LINE_BREAK>', 'Falling', 'softly', 'on', 'my', 'heart', '.',
        '<LINE_BREAK>', 'Numbers', 'like', '6', 'and', '80', 'remain', ',',
        '<STANZA_BREAK>', 'i', 'write', 'in', 'lowercase', 
        '<LINE_BREAK>', 'because', 'CAPITALS', 'are', 'for',
        '<LINE_BREAK>', 'SERIOUS', 'THINGS'
    ]


@pytest.fixture  
def sample_vocabulary(sample_tokens) -> Dict[str, int]:
    """Sample vocabulary mapping for testing"""
    unique_tokens = sorted(list(set(sample_tokens)))
    return {token: idx for idx, token in enumerate(unique_tokens)}


@pytest.fixture
def mock_embeddings() -> Dict[str, np.ndarray]:
    """Small set of mock embeddings for testing"""
    words = ['love', 'heart', 'summer', 'rain', 'like', 'the', 'and', 'is', 'on', 'my']
    embeddings = {}
    
    np.random.seed(42)  # For reproducible tests
    for word in words:
        embeddings[word] = np.random.normal(0, 1, 50).astype(np.float32)  # 50-dim for speed
    
    return embeddings


@pytest.fixture
def mock_embedding_file(temp_dir, mock_embeddings) -> Path:
    """Create a mock GLoVe embedding file for testing"""
    embedding_file = temp_dir / "mock_glove.txt"
    
    with open(embedding_file, 'w') as f:
        for word, vector in mock_embeddings.items():
            vector_str = ' '.join(f'{x:.6f}' for x in vector)
            f.write(f'{word} {vector_str}\n')
    
    return embedding_file


@pytest.fixture
def test_config(temp_dir, mock_embedding_file) -> Config:
    """Test configuration with temporary paths"""
    config = Config()
    
    # Override paths for testing
    config.project_root = temp_dir
    config.data_dir = temp_dir / "data"
    config.embeddings_dir = temp_dir / "embeddings" 
    config.artifacts_dir = temp_dir / "artifacts"
    
    # Create directories
    for dir_path in [config.data_dir, config.embeddings_dir, config.artifacts_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Set mock embedding path
    config.embedding.embedding_path = str(mock_embedding_file)
    config.embedding.embedding_dim = 50  # Match mock embeddings
    
    return config


@pytest.fixture
def tokenization_config() -> TokenizationConfig:
    """Test tokenization configuration"""
    return TokenizationConfig(
        preserve_case=True,
        preserve_numbers=True,
        max_sequence_length=20,
        min_sequence_length=3
    )


@pytest.fixture
def embedding_config(mock_embedding_file) -> EmbeddingConfig:
    """Test embedding configuration"""
    return EmbeddingConfig(
        embedding_dim=50,
        embedding_path=str(mock_embedding_file),
        create_mock_if_missing=True
    )


@pytest.fixture
def cooccurrence_config() -> CooccurrenceConfig:
    """Test co-occurrence configuration"""
    return CooccurrenceConfig(
        window_size=3,
        weighting="linear",
        min_count=1
    )


@pytest.fixture
def sample_cooccurrence_matrix() -> np.ndarray:
    """Small co-occurrence matrix for testing"""
    # Create a 10x10 symmetric matrix with some structure
    np.random.seed(42)
    matrix = np.random.poisson(2, (10, 10))
    matrix = (matrix + matrix.T) / 2  # Make symmetric
    np.fill_diagonal(matrix, 0)  # No self-cooccurrence
    return matrix


@pytest.fixture
def sample_embedding_matrix() -> np.ndarray:
    """Sample embedding matrix for testing"""
    np.random.seed(42)
    return np.random.normal(0, 1, (100, 50)).astype(np.float32)


@pytest.fixture
def sample_sequences() -> np.ndarray:
    """Sample token sequences for testing"""
    return np.array([
        [1, 2, 3, 4, 5, 0, 0, 0],  # Padded sequences
        [6, 7, 8, 9, 10, 11, 0, 0],
        [12, 13, 14, 0, 0, 0, 0, 0]
    ])


@pytest.fixture  
def sample_attention_masks(sample_sequences) -> np.ndarray:
    """Sample attention masks corresponding to sequences"""
    masks = np.zeros_like(sample_sequences)
    for i, seq in enumerate(sample_sequences):
        # Set mask to 1 for non-padded positions
        masks[i, :np.count_nonzero(seq)] = 1
    return masks


@pytest.fixture
def sample_poem_dataset(temp_dir, sample_poems) -> Path:
    """Create sample poem dataset file"""
    dataset_file = temp_dir / "sample_poems.json"
    
    with open(dataset_file, 'w', encoding='utf-8') as f:
        json.dump(sample_poems, f, indent=2, ensure_ascii=False)
    
    return dataset_file


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )


# Skip integration tests by default unless specifically requested
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip integration tests by default"""
    if config.getoption("--run-integration"):
        return
    
    skip_integration = pytest.mark.skip(reason="integration test (use --run-integration to run)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="run slow tests"
    )