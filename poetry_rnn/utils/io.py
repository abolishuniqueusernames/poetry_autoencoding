"""
I/O utilities for poetry RNN autoencoder

Handles file operations, artifact management, and data serialization
for preprocessing results and model artifacts.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages saving and loading of preprocessing artifacts"""
    
    def __init__(self, artifacts_dir: Union[str, Path]):
        """
        Initialize artifact manager
        
        Args:
            artifacts_dir: Directory to store artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    def save_preprocessing_artifacts(
        self,
        token_sequences: np.ndarray,
        embedding_sequences: np.ndarray, 
        attention_masks: np.ndarray,
        vocabulary: Dict[str, int],
        metadata: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save preprocessing artifacts with timestamp
        
        Args:
            token_sequences: Tokenized sequences
            embedding_sequences: Embedded sequences  
            attention_masks: Attention masks
            vocabulary: Word to index mapping
            metadata: Processing metadata
            timestamp: Optional timestamp string
            
        Returns:
            Dictionary of saved file paths
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        saved_files = {}
        
        # Save numpy arrays
        token_path = self.artifacts_dir / f"token_sequences_{timestamp}.npy"
        embedding_path = self.artifacts_dir / f"embedding_sequences_{timestamp}.npy" 
        mask_path = self.artifacts_dir / f"attention_masks_{timestamp}.npy"
        
        np.save(token_path, token_sequences)
        np.save(embedding_path, embedding_sequences)
        np.save(mask_path, attention_masks)
        
        saved_files.update({
            'token_sequences': str(token_path),
            'embedding_sequences': str(embedding_path),
            'attention_masks': str(mask_path)
        })
        
        # Save vocabulary
        vocab_path = self.artifacts_dir / f"vocabulary_{timestamp}.json"
        with open(vocab_path, 'w') as f:
            json.dump(vocabulary, f, indent=2)
        saved_files['vocabulary'] = str(vocab_path)
        
        # Save metadata
        metadata_path = self.artifacts_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = str(metadata_path)
        
        # Create symlinks to latest versions
        self._create_latest_symlinks(saved_files, timestamp)
        
        logger.info(f"Saved preprocessing artifacts with timestamp {timestamp}")
        return saved_files
        
    def load_preprocessing_artifacts(
        self, 
        timestamp: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[str, Any]]:
        """
        Load preprocessing artifacts
        
        Args:
            timestamp: Specific timestamp to load, or None for latest
            
        Returns:
            Tuple of (token_sequences, embedding_sequences, attention_masks, vocabulary, metadata)
        """
        if timestamp is None:
            # Load latest versions
            token_sequences = np.load(self.artifacts_dir / "token_sequences_latest.npy", allow_pickle=True)
            embedding_sequences = np.load(self.artifacts_dir / "embedding_sequences_latest.npy")
            attention_masks = np.load(self.artifacts_dir / "attention_masks_latest.npy")
            
            with open(self.artifacts_dir / "vocabulary_latest.json") as f:
                vocabulary = json.load(f)
                
            with open(self.artifacts_dir / "metadata_latest.json") as f:
                metadata = json.load(f)
        else:
            # Load specific timestamp
            token_sequences = np.load(self.artifacts_dir / f"token_sequences_{timestamp}.npy", allow_pickle=True)
            embedding_sequences = np.load(self.artifacts_dir / f"embedding_sequences_{timestamp}.npy")
            attention_masks = np.load(self.artifacts_dir / f"attention_masks_{timestamp}.npy")
            
            with open(self.artifacts_dir / f"vocabulary_{timestamp}.json") as f:
                vocabulary = json.load(f)
                
            with open(self.artifacts_dir / f"metadata_{timestamp}.json") as f:
                metadata = json.load(f)
                
        return token_sequences, embedding_sequences, attention_masks, vocabulary, metadata
    
    def _create_latest_symlinks(self, saved_files: Dict[str, str], timestamp: str) -> None:
        """Create symlinks to latest versions"""
        for file_type, file_path in saved_files.items():
            latest_path = self.artifacts_dir / f"{file_type}_latest{Path(file_path).suffix}"
            
            # Remove existing symlink
            if latest_path.is_symlink():
                latest_path.unlink()
            elif latest_path.exists():
                latest_path.unlink()
                
            # Create new symlink
            latest_path.symlink_to(Path(file_path).name)
            
    def list_artifacts(self) -> Dict[str, List[str]]:
        """List all available artifacts by type"""
        artifacts = {
            'token_sequences': [],
            'embedding_sequences': [], 
            'attention_masks': [],
            'vocabulary': [],
            'metadata': []
        }
        
        for file_path in self.artifacts_dir.glob("*"):
            if file_path.is_file() and not file_path.is_symlink():
                for artifact_type in artifacts.keys():
                    if file_path.name.startswith(artifact_type):
                        artifacts[artifact_type].append(file_path.name)
                        
        return artifacts
    
    def cleanup_old_artifacts(self, keep_latest: int = 5) -> None:
        """Clean up old artifacts, keeping only the most recent"""
        artifacts = self.list_artifacts()
        
        for artifact_type, files in artifacts.items():
            if len(files) > keep_latest:
                # Sort by modification time, keep most recent
                files_with_time = [(f, (self.artifacts_dir / f).stat().st_mtime) for f in files]
                files_with_time.sort(key=lambda x: x[1], reverse=True)
                
                for file_to_remove, _ in files_with_time[keep_latest:]:
                    (self.artifacts_dir / file_to_remove).unlink()
                    logger.info(f"Removed old artifact: {file_to_remove}")


def save_model_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save model configuration to JSON file"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    logger.info(f"Saved model config to {config_path}")


def load_model_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load model configuration from JSON file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path) as f:
        config = json.load(f)
        
    logger.info(f"Loaded model config from {config_path}")
    return config


def save_poetry_dataset(poems: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """Save poetry dataset to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(poems, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Saved {len(poems)} poems to {output_path}")


def load_poetry_dataset(dataset_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load poetry dataset from JSON file"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
    with open(dataset_path, 'r', encoding='utf-8') as f:
        poems = json.load(f)
        
    logger.info(f"Loaded {len(poems)} poems from {dataset_path}")
    return poems


def export_embeddings(embedding_matrix: np.ndarray, vocabulary: Dict[str, int], 
                     output_path: Union[str, Path], format: str = 'txt') -> None:
    """
    Export embeddings in various formats
    
    Args:
        embedding_matrix: Embedding vectors
        vocabulary: Word to index mapping
        output_path: Output file path
        format: Export format ('txt', 'binary', 'json')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create reverse mapping
    idx_to_word = {idx: word for word, idx in vocabulary.items()}
    
    if format == 'txt':
        # GLoVe-style text format
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx in range(len(embedding_matrix)):
                if idx in idx_to_word:
                    word = idx_to_word[idx]
                    vector = embedding_matrix[idx]
                    vector_str = ' '.join(f'{x:.6f}' for x in vector)
                    f.write(f'{word} {vector_str}\n')
                    
    elif format == 'binary':
        # Pickle format for fast loading
        embedding_data = {
            'embedding_matrix': embedding_matrix,
            'vocabulary': vocabulary,
            'idx_to_word': idx_to_word
        }
        with open(output_path, 'wb') as f:
            pickle.dump(embedding_data, f)
            
    elif format == 'json':
        # JSON format (slower but human readable)
        embedding_data = {
            'embeddings': {
                idx_to_word[idx]: embedding_matrix[idx].tolist()
                for idx in range(len(embedding_matrix))
                if idx in idx_to_word
            },
            'vocabulary': vocabulary,
            'embedding_dim': embedding_matrix.shape[1]
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_data, f, indent=2, ensure_ascii=False)
            
    else:
        raise ValueError(f"Unsupported export format: {format}")
        
    logger.info(f"Exported embeddings to {output_path} in {format} format")


def create_backup(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """Create timestamped backup of a file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File to backup not found: {file_path}")
        
    if backup_dir is None:
        backup_dir = file_path.parent / "backups"
    else:
        backup_dir = Path(backup_dir)
        
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    # Copy file
    import shutil
    shutil.copy2(file_path, backup_path)
    
    logger.info(f"Created backup: {backup_path}")
    return backup_path