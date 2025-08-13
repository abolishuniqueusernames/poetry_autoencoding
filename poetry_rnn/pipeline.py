"""
High-level pipeline API for Poetry RNN Autoencoder preprocessing

This module provides the PoetryPreprocessor class - a unified, production-ready
API that orchestrates all preprocessing components (tokenization, embedding
alignment, sequence generation) into a seamless pipeline for autoencoder training.

Features:
- One-line setup with configuration management
- End-to-end processing from raw poems to training data
- Progress tracking and comprehensive logging
- Artifact management and reproducibility
- Error recovery and robust error handling
- Memory optimization and performance monitoring
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import numpy as np

from .config import Config, default_config
from .tokenization.poetry_tokenizer import PoetryTokenizer
from .embeddings.glove_manager import GLoVeEmbeddingManager
from .preprocessing.sequence_generator import SequenceGenerator
from .preprocessing.dataset_loader import PoetryDatasetLoader
from .utils.io import ArtifactManager

logger = logging.getLogger(__name__)


class PoetryPreprocessor:
    """
    High-level pipeline for poetry preprocessing and autoencoder sequence preparation.
    
    This class provides a unified interface to all preprocessing components,
    handling the complete pipeline from raw poetry JSON files to training-ready
    sequences with comprehensive error handling and artifact management.
    
    Example Usage:
        # Simple setup
        >>> preprocessor = PoetryPreprocessor()
        >>> dataset = preprocessor.process_poems("poems.json")
        
        # Custom configuration
        >>> config = Config()
        >>> config.chunking.window_size = 30
        >>> preprocessor = PoetryPreprocessor(config=config)
        >>> dataset = preprocessor.process_poems("poems.json")
        
        # With progress tracking
        >>> with PoetryPreprocessor() as pp:
        ...     dataset = pp.process_poems("poems.json", save_artifacts=True)
        
    Attributes:
        config: Configuration instance managing all preprocessing settings
        tokenizer: Poetry tokenizer for text processing
        embedding_manager: GLoVe embedding manager for vocabulary alignment
        sequence_generator: Sequence generator for chunking and preparation
        dataset_loader: Dataset loader for poetry JSON files
        artifact_manager: Manager for saving/loading preprocessing artifacts
    """
    
    def __init__(self,
                 config: Optional[Union[Config, str, Path]] = None,
                 tokenizer: Optional[PoetryTokenizer] = None,
                 embedding_manager: Optional[GLoVeEmbeddingManager] = None,
                 sequence_generator: Optional[SequenceGenerator] = None,
                 artifacts_dir: Optional[Union[str, Path]] = None,
                 enable_logging: bool = True,
                 log_level: str = "INFO"):
        """
        Initialize the poetry preprocessing pipeline.
        
        Args:
            config: Configuration instance, file path, or None for default
            tokenizer: Pre-initialized tokenizer (created if None)
            embedding_manager: Pre-initialized embedding manager (created if None)
            sequence_generator: Pre-initialized sequence generator (created if None)
            artifacts_dir: Directory for saving artifacts (auto-detected if None)
            enable_logging: Whether to enable detailed logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file path doesn't exist
        """
        # Setup logging
        if enable_logging:
            self._setup_logging(log_level)
        
        # Load configuration
        self.config = self._load_configuration(config)
        
        # Validate configuration
        try:
            self.config.validate()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        # Initialize components
        self.tokenizer = tokenizer or self._create_tokenizer()
        self.embedding_manager = embedding_manager or self._create_embedding_manager()
        self.sequence_generator = sequence_generator or self._create_sequence_generator(artifacts_dir)
        self.dataset_loader = PoetryDatasetLoader()
        
        # Initialize artifact manager
        artifacts_path = artifacts_dir or self.config.artifacts_dir
        self.artifact_manager = ArtifactManager(artifacts_path)
        
        # Track processing state
        self._last_processing_time = None
        self._last_artifacts = None
        self._vocabulary_built = False
        self._embeddings_aligned = False
        
        logger.info(f"PoetryPreprocessor initialized successfully")
        logger.info(f"  Project root: {self.config.project_root}")
        logger.info(f"  Artifacts dir: {artifacts_path}")
        logger.info(f"  Window size: {self.config.chunking.window_size}")
        logger.info(f"  Overlap: {self.config.chunking.overlap}")
    
    def _setup_logging(self, log_level: str) -> None:
        """Setup logging configuration."""
        level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Configure logger for this module
        logger.setLevel(level)
        
        # Add console handler if not exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    def _load_configuration(self, config: Optional[Union[Config, str, Path]]) -> Config:
        """Load configuration from various sources."""
        if config is None:
            return default_config
        elif isinstance(config, Config):
            return config
        elif isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            return Config(config_path=str(config_path))
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
    
    def _create_tokenizer(self) -> PoetryTokenizer:
        """Create tokenizer with current configuration."""
        try:
            tokenizer = PoetryTokenizer(
                config=self.config.tokenization,
                project_root=self.config.project_root
            )
            logger.debug("Created PoetryTokenizer")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to create tokenizer: {e}")
            raise
    
    def _create_embedding_manager(self) -> GLoVeEmbeddingManager:
        """Create embedding manager with current configuration."""
        try:
            manager = GLoVeEmbeddingManager(config=self.config.embedding)
            logger.debug("Created GLoVeEmbeddingManager")
            return manager
        except Exception as e:
            logger.error(f"Failed to create embedding manager: {e}")
            raise
    
    def _create_sequence_generator(self, artifacts_dir: Optional[Union[str, Path]]) -> SequenceGenerator:
        """Create sequence generator with current configuration."""
        try:
            generator = SequenceGenerator(
                config=self.config,
                tokenizer=self.tokenizer,
                embedding_manager=self.embedding_manager,
                artifacts_dir=artifacts_dir
            )
            logger.debug("Created SequenceGenerator")
            return generator
        except Exception as e:
            logger.error(f"Failed to create sequence generator: {e}")
            raise
    
    def process_poems(self,
                     poems_path: Union[str, Path],
                     save_artifacts: bool = True,
                     load_existing_embeddings: bool = True,
                     analyze_lengths: bool = True,
                     visualize_chunking: bool = False,
                     chunk_overlap: Optional[int] = None,
                     max_poems: Optional[int] = None,
                     seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Process poems from JSON file into training-ready sequences.
        
        This is the main entry point for the preprocessing pipeline. It handles
        the complete workflow: loading → tokenizing → embedding alignment →
        chunking → sequence generation → artifact saving.
        
        Args:
            poems_path: Path to poetry JSON file
            save_artifacts: Whether to save preprocessing artifacts
            load_existing_embeddings: Try to load existing GLoVe embeddings
            analyze_lengths: Perform poem length analysis
            visualize_chunking: Show chunking visualization example
            chunk_overlap: Override config chunk overlap
            max_poems: Limit number of poems processed (for testing)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with complete preprocessing results:
            {
                'sequences': np.ndarray,           # Token sequences
                'embedding_sequences': np.ndarray, # Embedding sequences  
                'attention_masks': np.ndarray,     # Attention masks
                'metadata': List[Dict],            # Chunk metadata
                'vocabulary': Dict[str, int],      # Word to index mapping
                'stats': Dict,                     # Processing statistics
                'config': Dict,                    # Configuration used
                'artifacts': Dict                  # Saved artifact paths (if save_artifacts=True)
            }
            
        Raises:
            FileNotFoundError: If poems file doesn't exist
            ValueError: If poems data is invalid
            RuntimeError: If processing fails
        """
        start_time = datetime.now()
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Set random seed to {seed} for reproducibility")
        
        # Override chunk overlap if provided
        if chunk_overlap is not None:
            original_overlap = self.config.chunking.overlap
            self.config.chunking.overlap = chunk_overlap
            logger.info(f"Overriding chunk overlap: {original_overlap} → {chunk_overlap}")
        
        logger.info(f"{'='*80}")
        logger.info(f"POETRY PREPROCESSING PIPELINE - STARTING")
        logger.info(f"{'='*80}")
        logger.info(f"Input file: {poems_path}")
        logger.info(f"Max poems: {max_poems or 'all'}")
        logger.info(f"Save artifacts: {save_artifacts}")
        
        try:
            # Step 1: Load poems
            logger.info(f"\n📖 Step 1: Loading poems from {poems_path}")
            poems_path = Path(poems_path)
            if not poems_path.exists():
                raise FileNotFoundError(f"Poems file not found: {poems_path}")
            
            # Set dataset path and load the specific file
            self.dataset_loader.dataset_path = poems_path.parent
            poems = self.dataset_loader.load_dataset(filename=poems_path.name)
            
            if max_poems:
                poems = poems[:max_poems]
                logger.info(f"Limited to first {max_poems} poems")
            
            logger.info(f"✅ Loaded {len(poems)} poems")
            
            # Step 2: Build vocabulary
            logger.info(f"\n🔤 Step 2: Building vocabulary")
            if not self._vocabulary_built:
                poem_texts = [poem['text'] for poem in poems]
                self.tokenizer.build_vocabulary(poem_texts)
                self._vocabulary_built = True
                
                vocab_stats = self.tokenizer.get_vocabulary_stats()
                logger.info(f"✅ Built vocabulary: {vocab_stats['total_tokens']} tokens")
                logger.info(f"  Special tokens: {vocab_stats['special_tokens']}")
                logger.info(f"  Regular tokens: {vocab_stats['regular_tokens']}")
                logger.info(f"  Decoration tokens: {vocab_stats['decoration_tokens']}")
            else:
                logger.info("✅ Using existing vocabulary")
            
            # Step 3: Load and align embeddings
            logger.info(f"\n🎯 Step 3: Loading and aligning embeddings")
            if not self._embeddings_aligned:
                # Load GLoVe embeddings
                if load_existing_embeddings and self.config.embedding.embedding_path:
                    try:
                        self.embedding_manager.load_glove_embeddings(show_progress=True)
                        logger.info(f"✅ Loaded {len(self.embedding_manager.embeddings):,} GLoVe embeddings")
                    except Exception as e:
                        logger.warning(f"Could not load GLoVe embeddings: {e}")
                        if self.config.embedding.create_mock_if_missing:
                            logger.info("Creating mock embeddings for testing...")
                        else:
                            raise
                
                # Align vocabulary with embeddings
                embedding_matrix, alignment_stats = self.embedding_manager.align_vocabulary(
                    vocabulary=self.tokenizer.vocabulary,
                    special_tokens=list(self.tokenizer.special_tokens.keys())
                )
                self._embeddings_aligned = True
                logger.info(f"✅ Aligned embeddings: {embedding_matrix.shape}")
            else:
                logger.info("✅ Using existing aligned embeddings")
                embedding_matrix = self.embedding_manager.get_embedding_matrix()
                alignment_stats = {}
            
            # Step 4: Analyze poem lengths
            length_stats = {}
            if analyze_lengths:
                logger.info(f"\n📊 Step 4: Analyzing poem lengths")
                length_stats = self.sequence_generator.analyze_poem_lengths(poems)
                logger.info(f"✅ Length analysis complete")
                logger.info(f"  Mean length: {length_stats['mean_length']:.1f} tokens")
                logger.info(f"  Chunking preservation: {length_stats['chunking_preservation']:.1%}")
            
            # Step 5: Generate sequences with chunking
            logger.info(f"\n⚙️ Step 5: Generating sequences with chunking")
            sequences, attention_masks, metadata, chunking_stats = (
                self.sequence_generator.prepare_autoencoder_sequences_with_chunking(
                    poems=poems,
                    max_length=self.config.chunking.window_size,
                    min_length=self.config.chunking.min_chunk_length,
                    chunk_overlap=self.config.chunking.overlap,
                    respect_boundaries=self.config.chunking.preserve_poem_boundaries,
                    analyze_preservation=True
                )
            )
            logger.info(f"✅ Generated {len(sequences)} sequences")
            
            # Step 6: Create embedding sequences
            logger.info(f"\n🎯 Step 6: Creating embedding sequences")
            embedding_sequences = self.sequence_generator.create_embedding_sequences(sequences)
            logger.info(f"✅ Created embedding sequences: {embedding_sequences.shape}")
            
            # Step 7: Optional chunking visualization
            if visualize_chunking:
                logger.info(f"\n👁️ Step 7: Visualizing chunking example")
                self.sequence_generator.visualize_chunking_example(
                    poems=poems,
                    max_length=self.config.chunking.window_size,
                    overlap=self.config.chunking.overlap
                )
            
            # Step 8: Save artifacts
            artifact_paths = {}
            if save_artifacts:
                logger.info(f"\n💾 Step 8: Saving artifacts")
                artifact_paths = self.sequence_generator.save_preprocessing_artifacts(
                    sequences=sequences,
                    embedding_sequences=embedding_sequences,
                    attention_masks=attention_masks,
                    metadata=metadata,
                    chunking_stats=chunking_stats,
                    save_latest=True
                )
                self._last_artifacts = artifact_paths
                logger.info(f"✅ Saved artifacts to {self.artifact_manager.artifacts_dir}")
            
            # Calculate processing time
            processing_time = datetime.now() - start_time
            self._last_processing_time = processing_time
            
            # Compile results
            results = {
                'sequences': sequences,
                'embedding_sequences': embedding_sequences,
                'attention_masks': attention_masks,
                'metadata': metadata,
                'vocabulary': self.tokenizer.vocabulary,
                'inverse_vocabulary': self.tokenizer.inverse_vocabulary,
                'embedding_matrix': embedding_matrix,
                'stats': {
                    'poems_processed': len(poems),
                    'sequences_generated': len(sequences),
                    'processing_time_seconds': processing_time.total_seconds(),
                    'vocabulary_size': len(self.tokenizer.vocabulary),
                    'embedding_dim': self.config.embedding.embedding_dim,
                    **length_stats,
                    **chunking_stats,
                    **alignment_stats
                },
                'config': {
                    'window_size': self.config.chunking.window_size,
                    'overlap': self.config.chunking.overlap,
                    'min_chunk_length': self.config.chunking.min_chunk_length,
                    'preserve_case': self.config.tokenization.preserve_case,
                    'embedding_dim': self.config.embedding.embedding_dim
                }
            }
            
            if save_artifacts:
                results['artifacts'] = artifact_paths
            
            logger.info(f"\n✅ PIPELINE COMPLETE")
            logger.info(f"{'='*80}")
            logger.info(f"Processing time: {processing_time}")
            logger.info(f"Sequences generated: {len(sequences):,}")
            logger.info(f"Data preservation rate: {chunking_stats.get('preservation_rate', 0):.1%}")
            logger.info(f"Memory usage: {embedding_sequences.nbytes / 1024**2:.1f} MB")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise RuntimeError(f"Poetry preprocessing pipeline failed: {e}") from e
    
    def load_preprocessed_data(self, 
                              artifacts_dir: Optional[Union[str, Path]] = None,
                              timestamp: str = "latest") -> Dict[str, Any]:
        """
        Load previously saved preprocessing artifacts.
        
        Args:
            artifacts_dir: Directory containing artifacts (uses config default if None)
            timestamp: Timestamp of artifacts to load ("latest" for most recent)
            
        Returns:
            Dictionary with loaded preprocessing data
            
        Raises:
            FileNotFoundError: If artifacts not found
        """
        logger.info(f"Loading preprocessed data (timestamp: {timestamp})")
        
        artifacts_path = artifacts_dir or self.config.artifacts_dir
        artifact_manager = ArtifactManager(artifacts_path)
        
        try:
            # Load artifacts
            data = artifact_manager.load_preprocessing_artifacts(timestamp)
            
            # Rebuild vocabulary in tokenizer
            if 'vocabulary' in data:
                self.tokenizer.vocabulary = data['vocabulary']
                self.tokenizer.inverse_vocabulary = {v: k for k, v in data['vocabulary'].items()}
                self._vocabulary_built = True
            
            # Rebuild embedding manager state
            if 'embedding_matrix' in data:
                self.embedding_manager.embedding_matrix = data['embedding_matrix']
                self.embedding_manager.word_to_idx = data['vocabulary']
                self.embedding_manager.idx_to_word = {v: k for k, v in data['vocabulary'].items()}
                self._embeddings_aligned = True
            
            logger.info(f"✅ Loaded preprocessed data")
            logger.info(f"  Sequences: {data['token_sequences'].shape}")
            logger.info(f"  Vocabulary size: {len(data['vocabulary'])}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load preprocessed data: {e}")
            raise
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of the last processing run.
        
        Returns:
            Dictionary with processing statistics and metadata
        """
        if not hasattr(self, '_last_processing_time'):
            return {'status': 'no_processing_completed'}
        
        summary = {
            'status': 'completed',
            'processing_time': self._last_processing_time,
            'vocabulary_built': self._vocabulary_built,
            'embeddings_aligned': self._embeddings_aligned,
            'config': {
                'window_size': self.config.chunking.window_size,
                'overlap': self.config.chunking.overlap,
                'embedding_dim': self.config.embedding.embedding_dim,
                'preserve_case': self.config.tokenization.preserve_case
            }
        }
        
        if self._vocabulary_built:
            summary['vocabulary_stats'] = self.tokenizer.get_vocabulary_stats()
        
        if self._last_artifacts:
            summary['artifacts'] = self._last_artifacts
            
        return summary
    
    def validate_pipeline(self, poems_path: Union[str, Path], sample_size: int = 5) -> Dict[str, Any]:
        """
        Validate the preprocessing pipeline with a small sample.
        
        Args:
            poems_path: Path to poetry JSON file
            sample_size: Number of poems to process for validation
            
        Returns:
            Validation results dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        logger.info(f"Validating pipeline with {sample_size} sample poems")
        
        try:
            # Process small sample
            results = self.process_poems(
                poems_path=poems_path,
                max_poems=sample_size,
                save_artifacts=False,
                analyze_lengths=False,
                visualize_chunking=False
            )
            
            # Validate results structure
            required_keys = ['sequences', 'embedding_sequences', 'attention_masks', 'metadata', 'vocabulary']
            for key in required_keys:
                if key not in results:
                    raise ValueError(f"Missing required key in results: {key}")
            
            # Validate data shapes
            sequences = results['sequences']
            embeddings = results['embedding_sequences']
            masks = results['attention_masks']
            
            if sequences.shape[0] != embeddings.shape[0] != masks.shape[0]:
                raise ValueError("Inconsistent sequence counts across arrays")
            
            if sequences.shape[1] != embeddings.shape[1] != masks.shape[1]:
                raise ValueError("Inconsistent sequence lengths across arrays")
            
            if embeddings.shape[2] != self.config.embedding.embedding_dim:
                raise ValueError("Incorrect embedding dimension")
            
            # Validate metadata
            metadata = results['metadata']
            if len(metadata) != len(sequences):
                raise ValueError("Metadata count doesn't match sequence count")
            
            validation_results = {
                'status': 'passed',
                'samples_processed': len(sequences),
                'vocabulary_size': len(results['vocabulary']),
                'sequence_shape': sequences.shape,
                'embedding_shape': embeddings.shape,
                'preservation_rate': results['stats'].get('preservation_rate', 0),
                'processing_time': results['stats']['processing_time_seconds']
            }
            
            logger.info("✅ Pipeline validation passed")
            return validation_results
            
        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type:
            logger.error(f"Pipeline exited with error: {exc_val}")
        else:
            logger.info("Pipeline completed successfully")
        return False
    
    def __repr__(self) -> str:
        vocab_size = len(self.tokenizer.vocabulary) if self._vocabulary_built else 0
        return (f"PoetryPreprocessor("
                f"vocab_size={vocab_size}, "
                f"window_size={self.config.chunking.window_size}, "
                f"embedding_dim={self.config.embedding.embedding_dim})")