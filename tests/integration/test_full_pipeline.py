"""
Comprehensive integration tests for the full poetry preprocessing pipeline

These tests validate the complete end-to-end pipeline from raw poetry JSON
to training-ready PyTorch datasets, ensuring all components work together
correctly and that recent improvements (tokenization fixes, chunking, etc.)
function properly.

Test Coverage:
- Full pipeline with PoetryPreprocessor
- AutoencoderDataset functionality
- Data splits and sampling strategies
- Artifact saving and loading
- Error handling and edge cases
- Performance benchmarks
- Memory usage validation
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import json
from typing import Dict, List, Any
import time

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from poetry_rnn.pipeline import PoetryPreprocessor
from poetry_rnn.dataset import AutoencoderDataset, create_poetry_datasets, create_poetry_dataloaders
from poetry_rnn.dataset import PoemAwareSampler, ChunkSequenceSampler
from poetry_rnn.config import Config


@pytest.fixture
def extended_sample_poems():
    """Extended sample poetry dataset for comprehensive testing."""
    return [
        {
            "url": "test://poem1",
            "author": "Test Poet Alpha",
            "title": "Digital Hearts in Summer Rain",
            "text": "Love is like the summer rain,\nFalling softly on my heart ❤️.\nNumbers like 6 and 80 remain,\nWhen we are far apart.\n\n❤️ forever beats in digital dreams ❤️\nwhere algorithms learn to feel\nand data streams become\nthe poetry we never knew we needed.\n\nIn this age of artificial minds,\nwe search for authentic connection,\nfinding beauty in the intersection\nof human souls and silicon binds.",
            "content_type": "poetry",
            "length": 287,
            "line_count": 12,
            "source": "Test Collection",
            "dbbc_score": 25
        },
        {
            "url": "test://poem2", 
            "author": "Test Poet Beta",
            "title": "lowercase manifesto",
            "text": "i write in lowercase\nbecause CAPITALS are for\nSERIOUS THINGS\n\nlike paying taxes\nand forgetting passwords\nand SCREAMING AT THE VOID\n\nbut poetry is gentle\nlike whispers in the dark\nlike secrets shared between\ntwo hearts that understand\n\nthe revolutionary act\nof being soft\nin a hard world ☁️",
            "content_type": "poetry",
            "length": 198,
            "line_count": 14,
            "source": "Test Collection",
            "dbbc_score": 18
        },
        {
            "url": "test://poem3",
            "author": "Test Poet Gamma", 
            "title": "Brief Moments",
            "text": "brief\npoetic\nmoment\n\ncaptured\nin\nthree\nlines",
            "content_type": "poetry",
            "length": 35,
            "line_count": 7,
            "source": "Test Collection", 
            "dbbc_score": 10
        },
        {
            "url": "test://poem4",
            "author": "Test Poet Delta",
            "title": "Extended Journey Through Memory Lane",
            "text": "I remember walking down that old familiar street\nwhere every crack in the sidewalk\ntold a story of summers past\nand winters that seemed to last forever.\n\nThe oak tree on the corner,\nwith its branches reaching toward heaven,\nheld secrets in its leaves\nthat only children understand.\n\nWe carved our names in bark,\nthinking permanence was possible,\nnot knowing that even trees\ncan forget the promises we make.\n\nBut memory persists\nlike echoes in empty halls,\nlike photographs that fade\nbut never quite disappear.\n\nAnd now I walk these digital streets,\nwhere algorithms predict my path\nand data points replace\nthe mysteries we used to love.\n\nYet something in the human heart\nresists quantification,\ninsists on wonder,\ndemands that poetry survive\nin whatever form it takes.\n\nSo here I am,\nwriting verses on a screen,\nhoping someone somewhere\nwill remember what it means\nto be gloriously,\nmessily,\nbeautifully human.",
            "content_type": "poetry",
            "length": 567,
            "line_count": 31,
            "source": "Test Collection",
            "dbbc_score": 32
        },
        {
            "url": "test://poem5",
            "author": "Test Poet Epsilon",
            "title": "Fragments of the Digital Age",
            "text": "notification pings\nat 3 AM—\nsomeone liked\nyour existential crisis\n\nscrolling through feeds\nof curated happiness,\nwe forget\nthe weight of silence\n\n🌟 trending now: authentic sadness 🌟\n#realness #nofilter #truth\nbut even our pain\ngets monetized\n\nstill,\nbetween the algorithms\nand targeted ads,\nsomething real persists:\n\nthe way morning light\nfalls across your coffee cup,\nthe sound of rain\non windows,\n\nthe inexplicable joy\nof finding the perfect word\nfor an imperfect feeling\nin an imperfect world.",
            "content_type": "poetry",
            "length": 345,
            "line_count": 24,
            "source": "Test Collection",
            "dbbc_score": 28
        }
    ]


@pytest.fixture
def extended_poem_dataset_file(temp_dir, extended_sample_poems):
    """Create extended poem dataset file for testing."""
    dataset_file = temp_dir / "extended_poems.json"
    
    with open(dataset_file, 'w', encoding='utf-8') as f:
        json.dump(extended_sample_poems, f, indent=2, ensure_ascii=False)
    
    return dataset_file


@pytest.mark.integration
class TestFullPipelineIntegration:
    """Integration tests for the complete preprocessing pipeline."""
    
    def test_pipeline_initialization(self, test_config):
        """Test that pipeline initializes correctly with configuration."""
        # Test default initialization
        preprocessor = PoetryPreprocessor()
        assert preprocessor.config is not None
        assert preprocessor.tokenizer is not None
        assert preprocessor.embedding_manager is not None
        assert preprocessor.sequence_generator is not None
        
        # Test initialization with custom config
        preprocessor_custom = PoetryPreprocessor(config=test_config)
        assert preprocessor_custom.config == test_config
        assert preprocessor_custom.config.embedding.embedding_dim == 50
    
    def test_pipeline_validation(self, extended_poem_dataset_file, test_config):
        """Test pipeline validation with sample data."""
        preprocessor = PoetryPreprocessor(config=test_config)
        
        # Run validation
        validation_results = preprocessor.validate_pipeline(
            extended_poem_dataset_file, 
            sample_size=3
        )
        
        assert validation_results['status'] == 'passed'
        assert validation_results['samples_processed'] > 0
        assert validation_results['vocabulary_size'] > 0
        assert 'sequence_shape' in validation_results
        assert 'embedding_shape' in validation_results
        assert validation_results['preservation_rate'] > 0
    
    def test_end_to_end_processing(self, extended_poem_dataset_file, test_config):
        """Test complete end-to-end processing pipeline."""
        preprocessor = PoetryPreprocessor(config=test_config, enable_logging=True)
        
        # Process poems
        results = preprocessor.process_poems(
            poems_path=extended_poem_dataset_file,
            save_artifacts=True,
            analyze_lengths=True,
            visualize_chunking=False,  # Disable for cleaner test output
            seed=42
        )
        
        # Validate results structure
        required_keys = [
            'sequences', 'embedding_sequences', 'attention_masks', 
            'metadata', 'vocabulary', 'stats', 'config', 'artifacts'
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
        
        # Validate data shapes and consistency
        sequences = results['sequences']
        embeddings = results['embedding_sequences']
        masks = results['attention_masks']
        metadata = results['metadata']
        
        assert sequences.shape[0] == embeddings.shape[0] == masks.shape[0]
        assert sequences.shape[1] == embeddings.shape[1] == masks.shape[1]
        assert embeddings.shape[2] == test_config.embedding.embedding_dim
        assert len(metadata) == len(sequences)
        
        # Validate statistics
        stats = results['stats']
        assert stats['poems_processed'] == 5  # Extended sample has 5 poems
        assert stats['sequences_generated'] > 5  # Should have chunks
        assert stats['preservation_rate'] > 0.8  # Should preserve most data
        assert stats['vocabulary_size'] > 20
        
        # Validate chunking worked correctly
        assert 'chunking_preservation' in stats
        assert 'data_amplification' in stats
        assert stats['data_amplification'] > 1.0  # Should create multiple chunks
        
        # Validate artifacts were saved
        artifacts = results['artifacts']
        assert 'token_sequences' in artifacts
        assert 'embedding_sequences' in artifacts
        assert 'attention_masks' in artifacts
        
        return results
    
    def test_autoencoder_dataset_creation(self, extended_poem_dataset_file, test_config):
        """Test AutoencoderDataset creation from pipeline results."""
        # First process poems
        preprocessor = PoetryPreprocessor(config=test_config)
        results = preprocessor.process_poems(
            extended_poem_dataset_file,
            save_artifacts=True,
            seed=42
        )
        
        # Create dataset from results
        dataset = AutoencoderDataset(
            sequences=results['sequences'],
            embedding_sequences=results['embedding_sequences'],
            attention_masks=results['attention_masks'],
            metadata=results['metadata'],
            vocabulary=results['vocabulary'],
            split="full",
            seed=42
        )
        
        assert len(dataset) == len(results['sequences'])
        assert dataset.num_poems == 5  # 5 poems in extended sample
        
        # Test data access
        sample = dataset[0]
        assert 'input_sequences' in sample
        assert 'target_sequences' in sample
        assert 'attention_mask' in sample
        assert 'token_sequences' in sample
        assert 'metadata' in sample
        
        # Validate tensor shapes
        input_seq = sample['input_sequences']
        assert input_seq.shape[0] == test_config.chunking.window_size
        assert input_seq.shape[1] == test_config.embedding.embedding_dim
        
        # Test autoencoder property (input == target)
        torch.testing.assert_close(sample['input_sequences'], sample['target_sequences'])
    
    def test_data_splits(self, extended_poem_dataset_file, test_config):
        """Test train/validation/test splitting functionality."""
        # Process poems and save artifacts
        preprocessor = PoetryPreprocessor(config=test_config)
        preprocessor.process_poems(
            extended_poem_dataset_file,
            save_artifacts=True,
            seed=42
        )
        
        # Create split datasets
        train_dataset, val_dataset, test_dataset = create_poetry_datasets(
            artifacts_path=test_config.artifacts_dir,
            split_ratios=(0.6, 0.3, 0.1),
            seed=42
        )
        
        # Validate splits
        total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert total_samples > 0
        
        # Check that all splits have data (with 5 poems, even small splits should have something)
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        assert len(test_dataset) >= 0  # Might be 0 with very small dataset
        
        # Validate no data leakage (check poem-level splits)
        train_poems = set(train_dataset.chunk_to_poem.values())
        val_poems = set(val_dataset.chunk_to_poem.values())
        test_poems = set(test_dataset.chunk_to_poem.values())
        
        # No overlap between splits
        assert len(train_poems & val_poems) == 0
        assert len(train_poems & test_poems) == 0
        assert len(val_poems & test_poems) == 0
    
    def test_dataloader_creation(self, extended_poem_dataset_file, test_config):
        """Test DataLoader creation with various configurations."""
        # Process poems
        preprocessor = PoetryPreprocessor(config=test_config)
        preprocessor.process_poems(
            extended_poem_dataset_file,
            save_artifacts=True,
            seed=42
        )
        
        # Create datasets
        datasets = create_poetry_datasets(
            artifacts_path=test_config.artifacts_dir,
            split_ratios=(0.7, 0.2, 0.1),
            seed=42
        )
        
        # Test standard DataLoaders
        train_loader, val_loader, test_loader = create_poetry_dataloaders(
            datasets,
            batch_size=2,
            use_poem_aware_sampling=False
        )
        
        # Test batch loading
        for batch in train_loader:
            assert 'input_sequences' in batch
            assert 'target_sequences' in batch
            assert 'attention_mask' in batch
            assert 'token_sequences' in batch
            assert 'metadata' in batch
            
            # Check batch shapes
            batch_size = batch['input_sequences'].shape[0]
            assert batch_size <= 2
            assert batch['input_sequences'].shape[1] == test_config.chunking.window_size
            assert batch['input_sequences'].shape[2] == test_config.embedding.embedding_dim
            
            # Test autoencoder property
            torch.testing.assert_close(batch['input_sequences'], batch['target_sequences'])
            break  # Just test first batch
        
        # Test poem-aware sampling
        train_loader_balanced, _, _ = create_poetry_dataloaders(
            datasets,
            batch_size=2,
            use_poem_aware_sampling=True,
            max_chunks_per_poem=2
        )
        
        # Should still work
        batch = next(iter(train_loader_balanced))
        assert batch['input_sequences'].shape[0] <= 2
    
    def test_custom_samplers(self, extended_poem_dataset_file, test_config):
        """Test custom sampling strategies."""
        # Create dataset
        preprocessor = PoetryPreprocessor(config=test_config)
        results = preprocessor.process_poems(
            extended_poem_dataset_file,
            save_artifacts=False,
            seed=42
        )
        
        dataset = AutoencoderDataset(
            sequences=results['sequences'],
            embedding_sequences=results['embedding_sequences'],
            attention_masks=results['attention_masks'],
            metadata=results['metadata'],
            vocabulary=results['vocabulary'],
            split="full"
        )
        
        # Test PoemAwareSampler
        poem_sampler = PoemAwareSampler(dataset, max_chunks_per_poem=2)
        poem_indices = list(poem_sampler)
        assert len(poem_indices) <= len(dataset)
        assert len(poem_indices) <= dataset.num_poems * 2
        
        # Test ChunkSequenceSampler
        sequence_sampler = ChunkSequenceSampler(dataset, shuffle_poems=False)
        sequence_indices = list(sequence_sampler)
        assert len(sequence_indices) == len(dataset)
        
        # Test that it works with DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, sampler=poem_sampler)
        batch = next(iter(loader))
        assert batch['input_sequences'].shape[0] <= 2
    
    def test_artifact_save_and_load(self, extended_poem_dataset_file, test_config):
        """Test saving and loading preprocessing artifacts."""
        # Process and save
        preprocessor = PoetryPreprocessor(config=test_config)
        original_results = preprocessor.process_poems(
            extended_poem_dataset_file,
            save_artifacts=True,
            seed=42
        )
        
        # Load from artifacts
        loaded_data = preprocessor.load_preprocessed_data(
            artifacts_dir=test_config.artifacts_dir,
            timestamp="latest"
        )
        
        # Validate loaded data
        assert 'token_sequences' in loaded_data
        assert 'embedding_sequences' in loaded_data
        assert 'attention_masks' in loaded_data
        assert 'vocabulary' in loaded_data
        assert 'metadata' in loaded_data
        
        # Check shapes match
        np.testing.assert_array_equal(
            original_results['sequences'], 
            loaded_data['token_sequences']
        )
        np.testing.assert_array_equal(
            original_results['embedding_sequences'],
            loaded_data['embedding_sequences']
        )
        
        # Test dataset creation from loaded data
        dataset = AutoencoderDataset(
            artifacts_path=test_config.artifacts_dir,
            timestamp="latest",
            split="full"
        )
        assert len(dataset) == len(original_results['sequences'])
    
    def test_memory_efficiency(self, extended_poem_dataset_file, test_config):
        """Test memory efficiency features."""
        # Test lazy loading
        preprocessor = PoetryPreprocessor(config=test_config)
        preprocessor.process_poems(
            extended_poem_dataset_file,
            save_artifacts=True,
            seed=42
        )
        
        # Create dataset with lazy loading
        dataset_lazy = AutoencoderDataset(
            artifacts_path=test_config.artifacts_dir,
            timestamp="latest",
            lazy_loading=True,
            split="full"
        )
        
        # Should still be able to access data
        sample = dataset_lazy[0]
        assert 'input_sequences' in sample
        assert sample['input_sequences'].shape[1] == test_config.embedding.embedding_dim
        
        # Create dataset without lazy loading
        dataset_full = AutoencoderDataset(
            artifacts_path=test_config.artifacts_dir,
            timestamp="latest",
            lazy_loading=False,
            split="full"
        )
        
        # Data should be the same
        sample_full = dataset_full[0]
        torch.testing.assert_close(sample['input_sequences'], sample_full['input_sequences'])
    
    def test_error_handling(self, temp_dir, test_config):
        """Test error handling in various failure scenarios."""
        preprocessor = PoetryPreprocessor(config=test_config)
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            preprocessor.process_poems("nonexistent.json")
        
        # Test with invalid JSON
        invalid_json_file = temp_dir / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises((json.JSONDecodeError, RuntimeError)):
            preprocessor.process_poems(invalid_json_file)
        
        # Test with empty poems list
        empty_poems_file = temp_dir / "empty.json"
        with open(empty_poems_file, 'w') as f:
            json.dump([], f)
        
        # Should handle gracefully or raise meaningful error
        try:
            results = preprocessor.process_poems(empty_poems_file)
            assert len(results['sequences']) == 0
        except (ValueError, RuntimeError) as e:
            assert "empty" in str(e).lower() or "no" in str(e).lower()
    
    def test_configuration_validation(self, temp_dir):
        """Test configuration validation."""
        # Test invalid configuration
        config = Config()
        config.chunking.window_size = -1  # Invalid
        
        with pytest.raises(ValueError):
            PoetryPreprocessor(config=config)
        
        # Test configuration from file
        config_file = temp_dir / "test_config.json"
        config_data = {
            "chunking": {
                "window_size": 30,
                "overlap": 5
            },
            "embedding": {
                "embedding_dim": 100
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Should load configuration correctly
        preprocessor = PoetryPreprocessor(config=str(config_file))
        assert preprocessor.config.chunking.window_size == 30
        assert preprocessor.config.chunking.overlap == 5
        assert preprocessor.config.embedding.embedding_dim == 100
    
    def test_preprocessing_statistics(self, extended_poem_dataset_file, test_config):
        """Test that preprocessing statistics are accurate."""
        preprocessor = PoetryPreprocessor(config=test_config)
        results = preprocessor.process_poems(
            extended_poem_dataset_file,
            analyze_lengths=True,
            seed=42
        )
        
        stats = results['stats']
        
        # Basic counts
        assert stats['poems_processed'] == 5
        assert stats['sequences_generated'] > 5  # Should have chunks
        
        # Preservation statistics
        assert 'preservation_rate' in stats
        assert 'chunking_preservation' in stats
        assert stats['preservation_rate'] > 0.5  # Should preserve most data
        
        # Vocabulary statistics
        assert stats['vocabulary_size'] > 10
        assert 'found_exact' in stats  # From embedding alignment
        
        # Configuration tracking
        config_info = results['config']
        assert config_info['window_size'] == test_config.chunking.window_size
        assert config_info['overlap'] == test_config.chunking.overlap
        
        # Processing summary
        summary = preprocessor.get_processing_summary()
        assert summary['status'] == 'completed'
        assert summary['vocabulary_built'] is True
        assert summary['embeddings_aligned'] is True


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance and benchmark tests for the pipeline."""
    
    def test_processing_speed(self, extended_poem_dataset_file, test_config):
        """Benchmark processing speed."""
        preprocessor = PoetryPreprocessor(config=test_config)
        
        start_time = time.time()
        results = preprocessor.process_poems(
            extended_poem_dataset_file,
            save_artifacts=False,
            analyze_lengths=False,
            visualize_chunking=False
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        poems_per_second = 5 / processing_time  # 5 poems in test set
        
        # Should process reasonably quickly (adjust threshold as needed)
        assert processing_time < 30, f"Processing took {processing_time:.2f}s, may be too slow"
        assert poems_per_second > 0.1, f"Only {poems_per_second:.3f} poems/second"
        
        print(f"Processing speed: {poems_per_second:.2f} poems/second")
        print(f"Total time: {processing_time:.2f} seconds")
    
    def test_memory_usage(self, extended_poem_dataset_file, test_config):
        """Test memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024**2  # MB
        
        preprocessor = PoetryPreprocessor(config=test_config)
        results = preprocessor.process_poems(
            extended_poem_dataset_file,
            save_artifacts=False
        )
        
        final_memory = process.memory_info().rss / 1024**2  # MB
        memory_increase = final_memory - initial_memory
        
        # Check that memory usage is reasonable
        data_size_mb = results['embedding_sequences'].nbytes / 1024**2
        
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Data size: {data_size_mb:.1f} MB")
        print(f"Memory efficiency: {data_size_mb / max(memory_increase, 0.1):.2f}")
        
        # Memory increase should be roughly proportional to data size
        # Allow for some overhead but not excessive
        assert memory_increase < data_size_mb * 5, "Memory usage seems excessive"
    
    def test_scalability_simulation(self, temp_dir, test_config):
        """Simulate processing larger datasets."""
        # Create a larger synthetic dataset
        large_poems = []
        for i in range(20):  # 20 poems
            poem_text = f"This is poem number {i+1}.\n" + \
                       "Line after line of poetry flows,\n" * 10 + \
                       f"Each poem has unique content {i}.\n" + \
                       "❤️ With decorative elements ☁️\n" * 3
            
            large_poems.append({
                "url": f"test://poem{i+1}",
                "author": f"Test Poet {i+1}",
                "title": f"Poem Number {i+1}",
                "text": poem_text,
                "content_type": "poetry",
                "length": len(poem_text),
                "source": "Generated Test Collection"
            })
        
        large_dataset_file = temp_dir / "large_poems.json"
        with open(large_dataset_file, 'w') as f:
            json.dump(large_poems, f)
        
        # Process larger dataset
        preprocessor = PoetryPreprocessor(config=test_config)
        
        start_time = time.time()
        results = preprocessor.process_poems(
            large_dataset_file,
            save_artifacts=True,
            analyze_lengths=False,
            visualize_chunking=False
        )
        processing_time = time.time() - start_time
        
        # Validate results
        assert results['stats']['poems_processed'] == 20
        assert results['stats']['sequences_generated'] > 20  # Should have chunks
        
        # Performance should scale reasonably
        poems_per_second = 20 / processing_time
        print(f"Large dataset: {poems_per_second:.2f} poems/second")
        
        # Create datasets and test data loading
        train_dataset, val_dataset, test_dataset = create_poetry_datasets(
            artifacts_path=test_config.artifacts_dir,
            timestamp="latest"
        )
        
        # Should handle larger datasets
        assert len(train_dataset) > 10
        
        # Test DataLoader with larger dataset
        train_loader, _, _ = create_poetry_dataloaders(
            (train_dataset, val_dataset, test_dataset),
            batch_size=4
        )
        
        # Should be able to iterate through batches
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            if batch_count >= 3:  # Test a few batches
                break
        
        assert batch_count > 0


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--run-integration"])