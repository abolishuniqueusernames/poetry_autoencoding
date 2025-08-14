"""
Utility functions for the Poetry RNN API.

These functions provide auto-detection capabilities, file system utilities,
and other helper functions that make the API more intelligent and
user-friendly.

Features:
- Auto-detection of data formats and embedding files
- Device detection and optimization recommendations
- Path resolution and project structure detection
- Validation helpers and diagnostic functions
"""

import json
import torch
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import os
import re


def find_glove_embeddings(search_paths: Optional[List[str]] = None) -> Optional[str]:
    """
    Auto-locate GLoVe embedding files in common locations.
    
    Searches for GLoVe embedding files in:
    1. Project embeddings directory
    2. Standard system locations
    3. User-specified search paths
    4. Current working directory and subdirectories
    
    Args:
        search_paths: Additional paths to search
        
    Returns:
        Path to GLoVe embedding file, or None if not found
        
    Example:
        >>> glove_path = find_glove_embeddings()
        >>> if glove_path:
        ...     print(f"Found GLoVe embeddings: {glove_path}")
    """
    # Common GLoVe filename patterns
    glove_patterns = [
        r'glove\.6B\.300d\.txt',
        r'glove\.6B\.200d\.txt', 
        r'glove\.6B\.100d\.txt',
        r'glove\.6B\.50d\.txt',
        r'glove\.twitter\.27B\.200d\.txt',
        r'glove\.840B\.300d\.txt',
        r'glove.*\.txt'
    ]
    
    # Default search paths
    default_paths = [
        Path.cwd() / "embeddings",
        Path.cwd() / "GloVe_preprocessing" / "embeddings",
        Path.cwd() / "data" / "embeddings",
        Path.cwd(),
        Path.home() / "embeddings",
        Path.home() / "data" / "glove",
        Path("/data/embeddings"),
        Path("/usr/local/share/embeddings")
    ]
    
    # Add user-specified paths
    if search_paths:
        default_paths.extend([Path(p) for p in search_paths])
    
    # Search for files
    for search_path in default_paths:
        if not search_path.exists():
            continue
            
        for pattern in glove_patterns:
            for file_path in search_path.rglob("*"):
                if file_path.is_file() and re.match(pattern, file_path.name, re.IGNORECASE):
                    print(f"Found GLoVe embeddings: {file_path}")
                    return str(file_path)
    
    return None


def detect_data_format(data_path: str) -> str:
    """
    Auto-detect the format of poetry data file.
    
    Analyzes file structure to determine format:
    - JSON with poems list
    - JSON with nested structure
    - Text file with poem markers
    - CSV format
    - Other formats
    
    Args:
        data_path: Path to data file
        
    Returns:
        Detected format string
        
    Example:
        >>> format_type = detect_data_format("poems.json")
        >>> print(f"Detected format: {format_type}")
    """
    data_file = Path(data_path)
    
    if not data_file.exists():
        return "file_not_found"
    
    file_ext = data_file.suffix.lower()
    
    try:
        if file_ext == '.json':
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Check if it's a list of poems
                if data and isinstance(data[0], dict):
                    if any(key in data[0] for key in ['text', 'content', 'poem', 'body']):
                        return "json_poem_list"
                    else:
                        return "json_object_list"
                return "json_list"
            
            elif isinstance(data, dict):
                if 'poems' in data:
                    return "json_nested_poems"
                elif any(key in data for key in ['text', 'content', 'body']):
                    return "json_single_poem"
                else:
                    return "json_object"
            
            return "json_unknown"
        
        elif file_ext in ['.txt', '.text']:
            with open(data_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for poem markers
            if '<POEM_START>' in content and '<POEM_END>' in content:
                return "text_with_markers"
            elif content.count('\n\n') > 10:  # Multiple paragraph breaks
                return "text_separated_poems"
            else:
                return "text_single_poem"
        
        elif file_ext == '.csv':
            return "csv_format"
        
        else:
            return f"unknown_{file_ext[1:]}"
    
    except Exception as e:
        return f"error_reading_file: {str(e)}"


def auto_detect_device() -> str:
    """
    Auto-detect optimal device for training.
    
    Checks available hardware and returns recommendations:
    - CUDA GPU with memory info
    - MPS (Apple Silicon)
    - CPU with core count
    
    Returns:
        Device string ('cuda:0', 'mps', 'cpu')
        
    Example:
        >>> device = auto_detect_device()
        >>> print(f"Recommended device: {device}")
    """
    if torch.cuda.is_available():
        # Get GPU information
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
        gpu_memory_gb = gpu_memory / (1024**3)
        
        print(f"CUDA GPU detected: {gpu_name}")
        print(f"GPU memory: {gpu_memory_gb:.1f} GB")
        
        if gpu_memory_gb < 4:
            print("Warning: Limited GPU memory. Consider reducing batch size.")
        
        return f"cuda:{current_device}" if gpu_count > 1 else "cuda"
    
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) acceleration detected")
        return "mps"
    
    else:
        # CPU information
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            print(f"Using CPU with {cpu_count} cores")
        except:
            print("Using CPU")
        
        return "cpu"


def estimate_memory_requirements(
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    embedding_dim: int,
    vocab_size: int = 10000
) -> Dict[str, float]:
    """
    Estimate memory requirements for training.
    
    Provides rough estimates of GPU/CPU memory needed
    for model parameters, activations, and gradients.
    
    Args:
        batch_size: Training batch size
        sequence_length: Maximum sequence length
        hidden_size: RNN hidden dimension
        embedding_dim: Embedding dimension
        vocab_size: Vocabulary size
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Rough parameter count estimates
    embedding_params = vocab_size * embedding_dim
    rnn_params = 4 * (embedding_dim * hidden_size + hidden_size**2)  # LSTM approximation
    linear_params = hidden_size * embedding_dim
    total_params = embedding_params + rnn_params + linear_params
    
    # Memory per parameter (4 bytes for float32, *3 for param + grad + optimizer state)
    param_memory_mb = total_params * 4 * 3 / (1024**2)
    
    # Activation memory (rough estimate)
    activation_memory_mb = batch_size * sequence_length * hidden_size * 4 / (1024**2)
    
    # Total memory estimate
    total_memory_mb = param_memory_mb + activation_memory_mb
    
    return {
        'parameters_mb': param_memory_mb,
        'activations_mb': activation_memory_mb,
        'total_mb': total_memory_mb,
        'total_gb': total_memory_mb / 1024,
        'recommended_batch_size': max(1, int(2048 / total_memory_mb * batch_size))  # For 2GB available
    }


def validate_project_structure() -> Dict[str, bool]:
    """
    Validate that the project structure is set up correctly.
    
    Checks for required directories, files, and configurations
    needed for the poetry RNN pipeline.
    
    Returns:
        Dictionary mapping component names to validation status
    """
    checks = {}
    current_dir = Path.cwd()
    
    # Check for key directories
    checks['poetry_rnn_module'] = (current_dir / 'poetry_rnn').exists()
    checks['dataset_directory'] = (current_dir / 'dataset_poetry').exists()
    checks['embeddings_directory'] = any([
        (current_dir / 'embeddings').exists(),
        (current_dir / 'GloVe_preprocessing' / 'embeddings').exists()
    ])
    
    # Check for key files
    checks['claude_instructions'] = (current_dir / 'CLAUDE.md').exists()
    checks['poetry_data'] = any((current_dir / 'dataset_poetry').glob('*.json')) if checks['dataset_directory'] else False
    checks['glove_embeddings'] = find_glove_embeddings() is not None
    
    # Check Python environment
    try:
        import torch
        import numpy as np
        checks['pytorch_available'] = True
    except ImportError:
        checks['pytorch_available'] = False
    
    try:
        import poetry_rnn
        checks['poetry_rnn_importable'] = True
    except ImportError:
        checks['poetry_rnn_importable'] = False
    
    return checks


def print_system_info():
    """Print comprehensive system information for diagnostics."""
    print("=== Poetry RNN System Information ===")
    
    # Python and package info
    import sys
    print(f"Python version: {sys.version}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch: Not installed")
    
    # Hardware info
    device = auto_detect_device()
    print(f"Recommended device: {device}")
    
    # Project structure
    print("\n=== Project Structure Validation ===")
    structure = validate_project_structure()
    for component, status in structure.items():
        status_str = "✅" if status else "❌"
        print(f"{status_str} {component.replace('_', ' ').title()}: {'OK' if status else 'Missing'}")
    
    # Data info
    print("\n=== Data Availability ===")
    glove_path = find_glove_embeddings()
    if glove_path:
        print(f"✅ GLoVe embeddings: {glove_path}")
    else:
        print("❌ GLoVe embeddings: Not found")
    
    # Look for poetry data
    dataset_dir = Path.cwd() / 'dataset_poetry'
    if dataset_dir.exists():
        json_files = list(dataset_dir.glob('*.json'))
        if json_files:
            for json_file in json_files[:3]:  # Show first 3
                format_type = detect_data_format(str(json_file))
                print(f"✅ Poetry data: {json_file.name} ({format_type})")
            if len(json_files) > 3:
                print(f"     ... and {len(json_files) - 3} more files")
        else:
            print("❌ Poetry data: No JSON files found")
    else:
        print("❌ Poetry data: dataset_poetry directory not found")


def get_optimal_batch_size(
    model_size: str = 'medium',
    sequence_length: int = 50,
    available_memory_gb: float = 8.0
) -> int:
    """
    Suggest optimal batch size based on model size and available memory.
    
    Args:
        model_size: Model size preset ('tiny', 'small', 'medium', 'large', 'xlarge')
        sequence_length: Expected sequence length
        available_memory_gb: Available GPU/system memory in GB
        
    Returns:
        Recommended batch size
    """
    # Model size to parameter count mapping (rough estimates)
    model_params = {
        'tiny': 100_000,
        'small': 500_000,
        'medium': 2_000_000,
        'large': 8_000_000,
        'xlarge': 20_000_000
    }
    
    params = model_params.get(model_size, 2_000_000)
    
    # Memory usage estimation
    param_memory_gb = params * 4 * 3 / (1024**3)  # params + grads + optimizer
    
    # Available memory for activations
    available_for_activations = available_memory_gb - param_memory_gb - 1.0  # 1GB buffer
    
    if available_for_activations <= 0:
        return 1  # Model too large for available memory
    
    # Estimate activation memory per sample
    memory_per_sample_mb = sequence_length * 512 * 4 / (1024**2)  # Rough estimate
    memory_per_sample_gb = memory_per_sample_mb / 1024
    
    # Calculate batch size
    optimal_batch_size = max(1, int(available_for_activations / memory_per_sample_gb))
    
    # Common batch sizes (powers of 2)
    common_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Find largest common size that fits
    for size in reversed(common_sizes):
        if size <= optimal_batch_size:
            return size
    
    return 1


def suggest_architecture_for_data(data_path: str) -> Dict[str, Any]:
    """
    Analyze data and suggest optimal architecture configuration.
    
    Args:
        data_path: Path to poetry dataset
        
    Returns:
        Dictionary with suggested architecture parameters
    """
    try:
        # Analyze data characteristics
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            poems = data
        elif isinstance(data, dict) and 'poems' in data:
            poems = data['poems']
        else:
            poems = [data]  # Single poem
        
        # Analyze text characteristics
        total_words = 0
        total_chars = 0
        unique_words = set()
        
        for poem in poems[:100]:  # Sample first 100 poems
            text = poem.get('text', '') or poem.get('content', '') or poem.get('body', '')
            words = text.lower().split()
            total_words += len(words)
            total_chars += len(text)
            unique_words.update(words)
        
        avg_words_per_poem = total_words / len(poems[:100])
        vocab_diversity = len(unique_words) / total_words if total_words > 0 else 0
        
        # Suggest architecture based on analysis
        if len(poems) < 100:
            # Small dataset
            suggestion = {
                'preset': 'small',
                'hidden_size': 256,
                'bottleneck_size': 32,
                'reason': 'Small dataset - using smaller model to prevent overfitting'
            }
        elif avg_words_per_poem > 100:
            # Long poems
            suggestion = {
                'preset': 'large',
                'hidden_size': 768,
                'bottleneck_size': 96,
                'reason': 'Long poems detected - using larger model for better representation'
            }
        elif vocab_diversity > 0.5:
            # High vocabulary diversity
            suggestion = {
                'preset': 'large',
                'hidden_size': 512,
                'bottleneck_size': 64,
                'reason': 'High vocabulary diversity - using larger model'
            }
        else:
            # Standard case
            suggestion = {
                'preset': 'medium',
                'hidden_size': 512,
                'bottleneck_size': 64,
                'reason': 'Standard configuration for balanced performance'
            }
        
        suggestion.update({
            'data_stats': {
                'num_poems': len(poems),
                'avg_words_per_poem': avg_words_per_poem,
                'unique_words': len(unique_words),
                'vocab_diversity': vocab_diversity
            }
        })
        
        return suggestion
        
    except Exception as e:
        # Fallback suggestion
        return {
            'preset': 'medium',
            'hidden_size': 512,
            'bottleneck_size': 64,
            'reason': f'Default configuration (data analysis failed: {str(e)})'
        }