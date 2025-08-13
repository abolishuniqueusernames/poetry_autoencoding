"""
Configuration management for Poetry RNN Autoencoder

Centralized configuration handling with environment detection,
default parameters, and validation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, asdict


@dataclass
class TokenizationConfig:
    """Configuration for poetry tokenization"""
    preserve_case: bool = True
    preserve_numbers: bool = True
    special_tokens: list = None
    max_sequence_length: int = 50
    min_sequence_length: int = 5
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = ['<UNK>', '<LINE_BREAK>', '<STANZA_BREAK>', '<POEM_START>', '<POEM_END>']


@dataclass  
class EmbeddingConfig:
    """Configuration for GLoVe embeddings"""
    embedding_dim: int = 300
    embedding_path: Optional[str] = None
    create_mock_if_missing: bool = True
    oov_strategy: str = "random_normal"  # random_normal, zero, mean
    alignment_fallback: bool = True  # Try lowercase, cleaned versions


@dataclass
class ChunkingConfig:
    """Configuration for sequence chunking"""
    window_size: int = 50
    overlap: int = 10
    min_chunk_length: int = 5
    preserve_poem_boundaries: bool = True
    

@dataclass
class CooccurrenceConfig:
    """Configuration for co-occurrence analysis"""
    window_size: int = 5
    context_boundary_tokens: list = None
    weighting: str = "linear"  # linear, harmonic, constant
    min_count: int = 1
    
    def __post_init__(self):
        if self.context_boundary_tokens is None:
            self.context_boundary_tokens = ['<STANZA_BREAK>']


class Config:
    """Main configuration class with environment detection and validation"""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize configuration with optional config file and overrides
        
        Args:
            config_path: Path to JSON/YAML config file  
            **kwargs: Direct configuration overrides
        """
        # Set default paths based on current working directory
        self.project_root = self._detect_project_root()
        self.data_dir = self.project_root / "dataset_poetry"
        self.embeddings_dir = self.project_root / "embeddings" 
        self.artifacts_dir = self.project_root / "GLoVe preprocessing" / "preprocessed_artifacts"
        
        # Initialize sub-configurations
        self.tokenization = TokenizationConfig()
        self.embedding = EmbeddingConfig()
        self.chunking = ChunkingConfig()
        self.cooccurrence = CooccurrenceConfig()
        
        # Load config file if provided
        if config_path:
            self.load_from_file(config_path)
            
        # Apply any direct overrides
        self.update_from_dict(kwargs)
        
        # Set default embedding path if not specified
        if self.embedding.embedding_path is None:
            default_glove = self.embeddings_dir / "glove.6B.300d.txt"
            if default_glove.exists():
                self.embedding.embedding_path = str(default_glove)
                
    def _detect_project_root(self) -> Path:
        """Auto-detect project root directory"""
        current_dir = Path.cwd()
        
        # Look for project markers
        project_markers = ["CLAUDE.md", "poetry_rnn", "dataset_poetry"]
        
        # Check current directory and parents
        for path in [current_dir] + list(current_dir.parents):
            if any((path / marker).exists() for marker in project_markers):
                return path
                
        # Fallback to current directory
        return current_dir
        
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path) as f:
            config_dict = json.load(f)
            
        self.update_from_dict(config_dict)
        
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (TokenizationConfig, EmbeddingConfig, 
                                                   ChunkingConfig, CooccurrenceConfig)):
                    # Update sub-configuration
                    sub_config = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
                else:
                    setattr(self, key, value)
                    
    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to JSON file"""
        config_dict = {
            "tokenization": asdict(self.tokenization),
            "embedding": asdict(self.embedding), 
            "chunking": asdict(self.chunking),
            "cooccurrence": asdict(self.cooccurrence),
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "embeddings_dir": str(self.embeddings_dir),
            "artifacts_dir": str(self.artifacts_dir)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    def validate(self) -> None:
        """Validate configuration settings"""
        errors = []
        
        # Check critical paths
        if not self.project_root.exists():
            errors.append(f"Project root does not exist: {self.project_root}")
            
        # Check embedding configuration
        if self.embedding.embedding_path:
            embedding_path = Path(self.embedding.embedding_path)
            if not embedding_path.exists() and not self.embedding.create_mock_if_missing:
                errors.append(f"Embedding file not found: {embedding_path}")
                
        # Check dimension consistency
        if self.chunking.window_size <= self.chunking.overlap:
            errors.append("Chunking window_size must be larger than overlap")
            
        if self.tokenization.max_sequence_length < self.tokenization.min_sequence_length:
            errors.append("Max sequence length must be >= min sequence length")
            
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
            
    def __repr__(self) -> str:
        return f"Config(project_root='{self.project_root}')"


# Default global configuration instance  
default_config = Config()