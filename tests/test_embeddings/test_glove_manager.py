"""
Unit tests for GLoVeEmbeddingManager

These tests validate the GLoVe embedding management functionality,
including loading, alignment, vocabulary mapping, and the recent
improvements to error handling and mock embedding generation.

Test Coverage:
- GLoVe embedding loading and parsing
- Vocabulary alignment with fallback strategies
- Out-of-vocabulary (OOV) handling
- Mock embedding generation for testing
- Special token embedding creation
- Error handling for malformed files
- Memory efficiency and performance
- Saving and loading aligned embeddings
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, List

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from poetry_rnn.embeddings.glove_manager import GLoVeEmbeddingManager
from poetry_rnn.config import EmbeddingConfig


@pytest.fixture
def basic_manager():
    """Basic embedding manager with default configuration."""
    config = EmbeddingConfig(create_mock_if_missing=True)
    return GLoVeEmbeddingManager(config=config)


@pytest.fixture
def mock_glove_file_content():
    """Mock GLoVe file content for testing."""
    return [
        "the 0.1 0.2 0.3 0.4 0.5",
        "and 0.2 0.3 0.4 0.5 0.6", 
        "love 0.3 0.4 0.5 0.6 0.7",
        "heart 0.4 0.5 0.6 0.7 0.8",
        "poetry 0.5 0.6 0.7 0.8 0.9",
        "beautiful 0.1 0.3 0.5 0.7 0.9",
        "rain 0.2 0.4 0.6 0.8 1.0",
        "summer 0.3 0.5 0.7 0.9 0.1",
        "flowers 0.4 0.6 0.8 0.1 0.3",
        "dreams 0.5 0.7 0.9 0.2 0.4"
    ]


@pytest.fixture
def mock_glove_file_with_errors():
    """Mock GLoVe file with various error conditions."""
    return [
        "the 0.1 0.2 0.3 0.4 0.5",  # Good line
        "malformed 0.1 0.2",  # Too few dimensions
        "and 0.2 0.3 0.4 0.5 0.6",  # Good line
        "bad_vector 0.1 bad 0.3 0.4 0.5",  # Non-numeric vector
        "",  # Empty line
        "love 0.3 0.4 0.5 0.6 0.7",  # Good line
        "too_many 0.1 0.2 0.3 0.4 0.5 0.6 0.7",  # Too many dimensions
        "heart 0.4 0.5 0.6 0.7 0.8"  # Good line
    ]


@pytest.fixture
def create_mock_glove_file(temp_dir, mock_glove_file_content):
    """Create a mock GLoVe file for testing."""
    glove_file = temp_dir / "mock_glove_5d.txt"
    
    with open(glove_file, 'w') as f:
        for line in mock_glove_file_content:
            f.write(line + '\n')
    
    return glove_file


@pytest.fixture
def create_glove_file_with_errors(temp_dir, mock_glove_file_with_errors):
    """Create a mock GLoVe file with errors for testing error handling."""
    glove_file = temp_dir / "mock_glove_errors.txt"
    
    with open(glove_file, 'w') as f:
        for line in mock_glove_file_with_errors:
            f.write(line + '\n')
    
    return glove_file


@pytest.fixture
def sample_vocabulary():
    """Sample vocabulary for testing alignment."""
    return {
        'the': 0,
        'and': 1,
        'love': 2,
        'heart': 3,
        'poetry': 4,
        'beautiful': 5,
        'rain': 6,
        'unknown_word': 7,
        'Another_Case': 8,
        'punctuation!': 9,
        '<UNK>': 10,
        '<LINE_BREAK>': 11,
        '<STANZA_BREAK>': 12
    }


class TestEmbeddingManagerInitialization:
    """Test embedding manager initialization."""
    
    def test_basic_initialization(self):
        """Test basic initialization with default config."""
        manager = GLoVeEmbeddingManager()
        
        assert manager is not None
        assert manager.config is not None
        assert manager.embedding_dim == 300  # Default
        assert manager.embeddings == {}
        assert manager.embedding_matrix is None
        assert len(manager.special_tokens) > 0
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = EmbeddingConfig(
            embedding_dim=100,
            oov_strategy="zero",
            alignment_fallback=False
        )
        
        manager = GLoVeEmbeddingManager(config=config)
        
        assert manager.embedding_dim == 100
        assert manager.config.oov_strategy == "zero"
        assert manager.config.alignment_fallback is False
    
    def test_initialization_with_direct_parameters(self):
        """Test initialization with direct parameter overrides."""
        manager = GLoVeEmbeddingManager(
            embedding_path="/test/path",
            embedding_dim=200
        )
        
        assert manager.embedding_dim == 200
        assert manager.config.embedding_path == "/test/path"
    
    def test_config_validation(self):
        """Test that invalid configuration raises errors."""
        with pytest.raises(ValueError):
            config = EmbeddingConfig(embedding_dim=-1)
            GLoVeEmbeddingManager(config=config)
        
        with pytest.raises(ValueError):
            config = EmbeddingConfig(oov_strategy="invalid_strategy")
            GLoVeEmbeddingManager(config=config)


class TestGLoVeLoading:
    """Test GLoVe embedding loading functionality."""
    
    def test_load_embeddings_success(self, temp_dir, create_mock_glove_file):
        """Test successful loading of GLoVe embeddings."""
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        
        manager.load_glove_embeddings()
        
        assert len(manager.embeddings) > 0
        assert 'the' in manager.embeddings
        assert 'love' in manager.embeddings
        assert 'heart' in manager.embeddings
        
        # Check vector dimensions
        for word, vector in manager.embeddings.items():
            assert vector.shape == (5,)
            assert vector.dtype == np.float32
    
    def test_load_embeddings_with_errors(self, temp_dir, create_glove_file_with_errors):
        """Test loading embeddings with malformed lines."""
        config = EmbeddingConfig(
            embedding_path=str(create_glove_file_with_errors),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        
        # Should not raise exception, but should skip malformed lines
        manager.load_glove_embeddings()
        
        # Should have loaded good lines
        assert 'the' in manager.embeddings
        assert 'and' in manager.embeddings
        assert 'love' in manager.embeddings
        assert 'heart' in manager.embeddings
        
        # Should not have malformed entries
        assert 'malformed' not in manager.embeddings
        assert 'bad_vector' not in manager.embeddings
        assert 'too_many' not in manager.embeddings
    
    def test_load_embeddings_with_limit(self, temp_dir, create_mock_glove_file):
        """Test loading embeddings with a limit."""
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        
        manager.load_glove_embeddings(limit=3)
        
        assert len(manager.embeddings) == 3
    
    def test_load_embeddings_missing_file(self, temp_dir):
        """Test handling of missing embedding file."""
        config = EmbeddingConfig(
            embedding_path=str(temp_dir / "nonexistent.txt"),
            embedding_dim=5,
            create_mock_if_missing=False
        )
        manager = GLoVeEmbeddingManager(config=config)
        
        with pytest.raises(FileNotFoundError):
            manager.load_glove_embeddings()
    
    def test_create_mock_embeddings(self, basic_manager):
        """Test mock embedding creation."""
        basic_manager._create_mock_embeddings(vocab_size=50)
        
        assert len(basic_manager.embeddings) > 0
        
        # Should have common words
        common_words = ['the', 'love', 'heart', 'beautiful']
        found_words = [word for word in common_words if word in basic_manager.embeddings]
        assert len(found_words) > 0
        
        # Check vector properties
        for word, vector in basic_manager.embeddings.items():
            assert vector.shape == (300,)  # Default dimension
            assert vector.dtype == np.float32
    
    def test_mock_embeddings_deterministic(self, basic_manager):
        """Test that mock embeddings are deterministic."""
        basic_manager._create_mock_embeddings()
        embeddings1 = basic_manager.embeddings.copy()
        
        basic_manager.embeddings = {}
        basic_manager._create_mock_embeddings()
        embeddings2 = basic_manager.embeddings
        
        # Should have same words
        assert set(embeddings1.keys()) == set(embeddings2.keys())
        
        # Vectors should be identical (deterministic)
        for word in embeddings1:
            np.testing.assert_array_equal(embeddings1[word], embeddings2[word])


class TestVocabularyAlignment:
    """Test vocabulary alignment functionality."""
    
    def test_basic_alignment(self, temp_dir, create_mock_glove_file, sample_vocabulary):
        """Test basic vocabulary alignment."""
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        manager.load_glove_embeddings()
        
        embedding_matrix, stats = manager.align_vocabulary(sample_vocabulary)
        
        # Check results
        assert embedding_matrix is not None
        assert embedding_matrix.shape[0] == len(sample_vocabulary)
        assert embedding_matrix.shape[1] == 5
        
        # Check statistics
        assert 'total_words' in stats
        assert 'found_exact' in stats
        assert 'oov_words' in stats
        assert stats['total_words'] == len(sample_vocabulary)
        assert stats['found_exact'] > 0  # Should find some exact matches
    
    def test_alignment_with_special_tokens(self, temp_dir, create_mock_glove_file, sample_vocabulary):
        """Test alignment with special tokens."""
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        manager.load_glove_embeddings()
        
        special_tokens = ['<UNK>', '<START>', '<END>']
        embedding_matrix, stats = manager.align_vocabulary(
            sample_vocabulary, 
            special_tokens=special_tokens
        )
        
        # Special tokens should be included
        assert '<UNK>' in manager.word_to_idx
        assert '<START>' in manager.word_to_idx
        assert '<END>' in manager.word_to_idx
        
        # Should have right number of special tokens
        assert stats['special_tokens'] == len(special_tokens)
    
    def test_alignment_fallback_strategies(self, temp_dir, create_mock_glove_file):
        """Test alignment fallback strategies."""
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5,
            alignment_fallback=True
        )
        manager = GLoVeEmbeddingManager(config=config)
        manager.load_glove_embeddings()
        
        # Vocabulary with case variations and punctuation
        vocab_with_variations = {
            'THE': 0,  # Should match 'the' in lowercase
            'Love': 1,  # Should match 'love' in lowercase
            'heart!': 2,  # Should match 'heart' after cleaning
            'unknown': 3  # Should not match
        }
        
        embedding_matrix, stats = manager.align_vocabulary(vocab_with_variations)
        
        # Should find lowercase and cleaned matches
        assert stats['found_lowercase'] > 0 or stats['found_cleaned'] > 0
        assert stats['oov_words'] > 0  # Should have some OOV words
    
    def test_oov_strategies(self, temp_dir, create_mock_glove_file):
        """Test different OOV handling strategies."""
        vocab = {'unknown_word_xyz': 0, '<UNK>': 1}
        
        # Test zero strategy
        config_zero = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5,
            oov_strategy="zero"
        )
        manager_zero = GLoVeEmbeddingManager(config=config_zero)
        manager_zero.load_glove_embeddings()
        matrix_zero, _ = manager_zero.align_vocabulary(vocab)
        
        # Unknown word should have zero vector
        unknown_idx = manager_zero.word_to_idx['unknown_word_xyz']
        np.testing.assert_array_equal(matrix_zero[unknown_idx], np.zeros(5))
        
        # Test random_normal strategy
        config_random = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5,
            oov_strategy="random_normal"
        )
        manager_random = GLoVeEmbeddingManager(config=config_random)
        manager_random.load_glove_embeddings()
        matrix_random, _ = manager_random.align_vocabulary(vocab)
        
        # Unknown word should have non-zero vector
        unknown_idx = manager_random.word_to_idx['unknown_word_xyz']
        assert not np.allclose(matrix_random[unknown_idx], np.zeros(5))
        
        # Test mean strategy
        config_mean = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5,
            oov_strategy="mean"
        )
        manager_mean = GLoVeEmbeddingManager(config=config_mean)
        manager_mean.load_glove_embeddings()
        matrix_mean, _ = manager_mean.align_vocabulary(vocab)
        
        # Unknown word should have vector close to mean of loaded embeddings
        unknown_idx = manager_mean.word_to_idx['unknown_word_xyz']
        all_embeddings = np.array(list(manager_mean.embeddings.values()))
        expected_mean = np.mean(all_embeddings, axis=0)
        np.testing.assert_allclose(matrix_mean[unknown_idx], expected_mean, rtol=1e-5)
    
    def test_empty_vocabulary(self, basic_manager):
        """Test handling of empty vocabulary."""
        with pytest.raises(ValueError):
            basic_manager.align_vocabulary({})


class TestEmbeddingAccess:
    """Test embedding access methods."""
    
    def test_get_embedding_for_word(self, temp_dir, create_mock_glove_file, sample_vocabulary):
        """Test getting embedding for specific word."""
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        manager.load_glove_embeddings()
        manager.align_vocabulary(sample_vocabulary)
        
        # Test existing word
        love_embedding = manager.get_embedding_for_word('love')
        assert love_embedding is not None
        assert love_embedding.shape == (5,)
        
        # Test non-existent word
        nonexistent_embedding = manager.get_embedding_for_word('nonexistent')
        assert nonexistent_embedding is None
    
    def test_get_vocabulary_size(self, temp_dir, create_mock_glove_file, sample_vocabulary):
        """Test getting vocabulary size."""
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        manager.load_glove_embeddings()
        manager.align_vocabulary(sample_vocabulary)
        
        vocab_size = manager.get_vocabulary_size()
        assert vocab_size == len(sample_vocabulary)
    
    def test_get_embedding_matrix(self, temp_dir, create_mock_glove_file, sample_vocabulary):
        """Test getting embedding matrix."""
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        manager.load_glove_embeddings()
        matrix, _ = manager.align_vocabulary(sample_vocabulary)
        
        retrieved_matrix = manager.get_embedding_matrix()
        np.testing.assert_array_equal(matrix, retrieved_matrix)


class TestSavingAndLoading:
    """Test saving and loading aligned embeddings."""
    
    def test_save_and_load_aligned_embeddings(self, temp_dir, create_mock_glove_file, sample_vocabulary):
        """Test saving and loading aligned embeddings."""
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        manager.load_glove_embeddings()
        original_matrix, _ = manager.align_vocabulary(sample_vocabulary)
        
        # Save embeddings
        save_path = temp_dir / "test_embeddings"
        manager.save_aligned_embeddings(str(save_path))
        
        # Check files were created
        assert (temp_dir / "test_embeddings.npy").exists()
        assert (temp_dir / "test_embeddings.json").exists()
        
        # Create new manager and load
        new_manager = GLoVeEmbeddingManager(config=config)
        new_manager.load_aligned_embeddings(str(save_path))
        
        # Check that data was loaded correctly
        np.testing.assert_array_equal(original_matrix, new_manager.embedding_matrix)
        assert manager.word_to_idx == new_manager.word_to_idx
        assert manager.idx_to_word == new_manager.idx_to_word
    
    def test_save_without_alignment(self, basic_manager):
        """Test that saving without alignment raises error."""
        with pytest.raises(ValueError):
            basic_manager.save_aligned_embeddings("test_path")
    
    def test_load_nonexistent_files(self, temp_dir, basic_manager):
        """Test loading from nonexistent files."""
        with pytest.raises(FileNotFoundError):
            basic_manager.load_aligned_embeddings(str(temp_dir / "nonexistent"))


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_malformed_glove_file(self, temp_dir):
        """Test handling of completely malformed GLoVe file."""
        malformed_file = temp_dir / "malformed.txt"
        with open(malformed_file, 'w') as f:
            f.write("This is not a valid GLoVe file\n")
            f.write("Neither is this line\n")
        
        config = EmbeddingConfig(
            embedding_path=str(malformed_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        
        # Should not crash, but might load nothing
        manager.load_glove_embeddings()
        
        # Should have empty embeddings
        assert len(manager.embeddings) == 0
    
    def test_inconsistent_dimensions(self, temp_dir):
        """Test handling of inconsistent embedding dimensions."""
        inconsistent_file = temp_dir / "inconsistent.txt"
        with open(inconsistent_file, 'w') as f:
            f.write("word1 0.1 0.2 0.3\n")  # 3 dimensions
            f.write("word2 0.1 0.2 0.3 0.4 0.5\n")  # 5 dimensions
        
        config = EmbeddingConfig(
            embedding_path=str(inconsistent_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        
        # Should skip lines with wrong dimensions
        manager.load_glove_embeddings()
        
        # Should only load the 5-dimensional word
        assert 'word2' in manager.embeddings
        assert 'word1' not in manager.embeddings
    
    def test_unicode_words_in_glove(self, temp_dir):
        """Test handling of Unicode words in GLoVe file."""
        unicode_file = temp_dir / "unicode.txt"
        with open(unicode_file, 'w', encoding='utf-8') as f:
            f.write("café 0.1 0.2 0.3 0.4 0.5\n")
            f.write("naïve 0.2 0.3 0.4 0.5 0.6\n")
            f.write("résumé 0.3 0.4 0.5 0.6 0.7\n")
        
        config = EmbeddingConfig(
            embedding_path=str(unicode_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        manager.load_glove_embeddings()
        
        # Should handle Unicode words correctly
        assert 'café' in manager.embeddings
        assert 'naïve' in manager.embeddings
        assert 'résumé' in manager.embeddings
    
    def test_very_large_vocabulary(self, temp_dir, create_mock_glove_file):
        """Test handling of very large vocabulary."""
        # Create large vocabulary
        large_vocab = {}
        for i in range(10000):
            large_vocab[f'word_{i}'] = i
        
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        manager.load_glove_embeddings()
        
        # Should handle large vocabulary without errors
        embedding_matrix, stats = manager.align_vocabulary(large_vocab)
        
        assert embedding_matrix.shape[0] == len(large_vocab)
        assert stats['total_words'] == len(large_vocab)
        assert stats['oov_words'] > 0  # Most words will be OOV
    
    def test_alignment_without_loaded_embeddings(self, basic_manager, sample_vocabulary):
        """Test vocabulary alignment without loaded embeddings."""
        # Don't load embeddings, just try to align
        embedding_matrix, stats = basic_manager.align_vocabulary(sample_vocabulary)
        
        # Should still work (will use OOV strategy for all words)
        assert embedding_matrix.shape[0] == len(sample_vocabulary)
        assert stats['oov_words'] == len(sample_vocabulary) - len(basic_manager.special_tokens)


class TestSpecialTokenHandling:
    """Test special token handling."""
    
    def test_special_token_embeddings(self, temp_dir, create_mock_glove_file, sample_vocabulary):
        """Test that special tokens get proper embeddings."""
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        manager.load_glove_embeddings()
        
        embedding_matrix, _ = manager.align_vocabulary(sample_vocabulary)
        
        # Special tokens should have embeddings
        for token in manager.special_tokens:
            if token in manager.word_to_idx:
                idx = manager.word_to_idx[token]
                embedding = embedding_matrix[idx]
                
                # Should not be zero vector (unless using zero OOV strategy)
                assert embedding.shape == (5,)
                assert not np.all(embedding == 0) or manager.config.oov_strategy == "zero"
    
    def test_custom_special_tokens(self, temp_dir, create_mock_glove_file):
        """Test using custom special tokens."""
        custom_tokens = ['<CUSTOM>', '<SPECIAL>', '<TOKEN>']
        vocab = {'word1': 0, 'word2': 1}
        
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        manager.load_glove_embeddings()
        
        embedding_matrix, stats = manager.align_vocabulary(
            vocab,
            special_tokens=custom_tokens
        )
        
        # Custom tokens should be in vocabulary
        for token in custom_tokens:
            assert token in manager.word_to_idx
        
        assert stats['special_tokens'] == len(custom_tokens)


class TestPerformanceAndMemory:
    """Test performance and memory efficiency."""
    
    @pytest.mark.slow
    def test_loading_performance(self, temp_dir):
        """Test loading performance with larger file."""
        # Create larger mock file
        large_file = temp_dir / "large_glove.txt"
        with open(large_file, 'w') as f:
            for i in range(1000):
                vector = ' '.join([f'{np.random.random():.6f}' for _ in range(50)])
                f.write(f'word_{i} {vector}\n')
        
        config = EmbeddingConfig(
            embedding_path=str(large_file),
            embedding_dim=50
        )
        manager = GLoVeEmbeddingManager(config=config)
        
        import time
        start_time = time.time()
        manager.load_glove_embeddings()
        end_time = time.time()
        
        loading_time = end_time - start_time
        
        # Should load reasonably quickly
        assert len(manager.embeddings) == 1000
        assert loading_time < 10  # Should load in under 10 seconds
        
        print(f"Loaded 1000 embeddings in {loading_time:.2f} seconds")
    
    def test_memory_efficiency(self, temp_dir, create_mock_glove_file, sample_vocabulary):
        """Test memory usage is reasonable."""
        config = EmbeddingConfig(
            embedding_path=str(create_mock_glove_file),
            embedding_dim=5
        )
        manager = GLoVeEmbeddingManager(config=config)
        manager.load_glove_embeddings()
        
        # Check memory usage of embeddings
        total_vectors = len(manager.embeddings)
        expected_memory = total_vectors * 5 * 4  # 5 dims * 4 bytes per float32
        
        actual_memory = sum(embedding.nbytes for embedding in manager.embeddings.values())
        
        assert actual_memory == expected_memory
        
        # Check aligned matrix memory
        embedding_matrix, _ = manager.align_vocabulary(sample_vocabulary)
        matrix_memory = embedding_matrix.nbytes
        expected_matrix_memory = len(sample_vocabulary) * 5 * 4
        
        assert matrix_memory == expected_matrix_memory


if __name__ == "__main__":
    # Run unit tests
    pytest.main([__file__, "-v"])