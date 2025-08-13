"""
Dataset loader module for poetry RNN autoencoder

Handles loading, validation, and analysis of poetry datasets from JSON files.
Provides comprehensive dataset statistics, filtering capabilities, and integration
with the configuration system for reproducible experiments.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from collections import Counter

from ..config import Config, default_config

logger = logging.getLogger(__name__)


class PoetryDatasetLoader:
    """
    Load and manage poetry datasets for autoencoder training.
    
    This class handles the complete dataset pipeline from JSON files to
    preprocessed poetry data, with comprehensive validation, statistics,
    and filtering capabilities.
    
    Key Features:
    - Multi-format dataset loading with fallback strategies
    - Comprehensive dataset validation and statistics
    - Filtering by quality scores and content criteria
    - Memory-efficient loading with progress tracking
    - Integration with configuration system
    - Extensible metadata handling
    
    Attributes:
        config: Configuration instance with dataset settings
        dataset_path: Path to dataset directory
        poems: Loaded poem data
        dataset_stats: Comprehensive dataset statistics
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 dataset_path: Optional[Union[str, Path]] = None):
        """
        Initialize poetry dataset loader.
        
        Args:
            config: Configuration instance with dataset settings
            dataset_path: Path to dataset directory (overrides config)
        """
        self.config = config or default_config
        
        # Set dataset path
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        elif hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'path'):
            self.dataset_path = Path(self.config.dataset.path)
        else:
            self.dataset_path = Path.cwd() / "dataset_poetry"
        
        self.poems = None
        self.dataset_stats = None
        
        logger.info(f"Initialized PoetryDatasetLoader with path: {self.dataset_path}")
    
    def load_dataset(self, 
                    filename: Optional[str] = None,
                    validate: bool = True,
                    compute_stats: bool = True) -> List[Dict[str, Any]]:
        """
        Load poetry dataset from JSON file with fallback options.
        
        Args:
            filename: Specific filename to load (searches common names if None)
            validate: Whether to validate loaded data
            compute_stats: Whether to compute comprehensive statistics
            
        Returns:
            List of poem dictionaries
            
        Raises:
            FileNotFoundError: If no valid dataset file found
            ValueError: If dataset format is invalid
        """
        logger.info(f"Loading poetry dataset from {self.dataset_path}")
        
        # Define possible dataset filenames in order of preference
        if filename:
            possible_files = [filename]
        else:
            possible_files = [
                "multi_poem_dbbc_collection.json",
                "improved_dbbc_collection.json", 
                "expanded_contemporary_poetry.json",
                "poetry_dataset.json",
                "poems.json"
            ]
        
        # Try to load from each possible file
        poems = None
        loaded_filename = None
        
        for filename in possible_files:
            filepath = self.dataset_path / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        poems = json.load(f)
                    loaded_filename = filename
                    logger.info(f"Successfully loaded {len(poems)} poems from {filename}")
                    print(f"✓ Loaded {len(poems)} poems from {filename}")
                    break
                except (json.JSONDecodeError, UnicodeDecodeError, Exception) as e:
                    logger.warning(f"Failed to load {filename}: {e}")
                    continue
        
        if poems is None:
            error_msg = f"No valid poetry dataset found in {self.dataset_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Validate dataset if requested
        if validate:
            self._validate_dataset(poems, loaded_filename)
        
        # Compute statistics if requested
        if compute_stats:
            self.dataset_stats = self.compute_dataset_statistics(poems)
            self._display_dataset_overview(poems, self.dataset_stats)
        
        self.poems = poems
        return poems
    
    def _validate_dataset(self, poems: List[Dict], filename: str) -> None:
        """
        Validate dataset format and content.
        
        Args:
            poems: List of poem dictionaries
            filename: Name of loaded file for error reporting
            
        Raises:
            ValueError: If dataset format is invalid
        """
        if not isinstance(poems, list):
            raise ValueError(f"Dataset must be a list, got {type(poems)}")
        
        if len(poems) == 0:
            raise ValueError("Dataset is empty")
        
        # Check required fields
        required_fields = ['text']
        optional_fields = ['title', 'author', 'alt_lit_score', 'url', 'date']
        
        for i, poem in enumerate(poems[:10]):  # Check first 10 poems
            if not isinstance(poem, dict):
                raise ValueError(f"Poem {i} must be a dictionary, got {type(poem)}")
            
            for field in required_fields:
                if field not in poem:
                    raise ValueError(f"Poem {i} missing required field: {field}")
            
            # Validate text content
            if not isinstance(poem['text'], str) or len(poem['text'].strip()) == 0:
                raise ValueError(f"Poem {i} has invalid text content")
        
        logger.info(f"Dataset validation passed for {filename}")
    
    def compute_dataset_statistics(self, poems: List[Dict]) -> Dict[str, Any]:
        """
        Compute comprehensive dataset statistics.
        
        Args:
            poems: List of poem dictionaries
            
        Returns:
            Dictionary with detailed dataset statistics
        """
        logger.info(f"Computing statistics for {len(poems)} poems")
        
        # Basic counts
        total_poems = len(poems)
        
        # Text statistics
        text_lengths = [len(poem['text']) for poem in poems]
        word_counts = [len(poem['text'].split()) for poem in poems]
        line_counts = [poem['text'].count('\n') + 1 for poem in poems]
        
        # Alt-lit score statistics (if available)
        alt_lit_scores = []
        for poem in poems:
            score = poem.get('alt_lit_score', None)
            if score is not None:
                try:
                    alt_lit_scores.append(float(score))
                except (ValueError, TypeError):
                    pass
        
        # Author and title statistics
        authors = [poem.get('author', 'Unknown') for poem in poems]
        titles = [poem.get('title', 'Untitled') for poem in poems]
        
        author_counts = Counter(authors)
        unique_authors = len(author_counts)
        
        # Character analysis
        all_text = ' '.join(poem['text'] for poem in poems)
        total_characters = len(all_text)
        unique_characters = len(set(all_text))
        
        # Metadata completeness
        fields_completeness = {}
        optional_fields = ['title', 'author', 'alt_lit_score', 'url', 'date']
        
        for field in optional_fields:
            count = sum(1 for poem in poems if field in poem and poem[field] is not None)
            fields_completeness[field] = {
                'count': count,
                'percentage': count / total_poems * 100
            }
        
        stats = {
            'basic_info': {
                'total_poems': total_poems,
                'unique_authors': unique_authors,
                'most_prolific_authors': author_counts.most_common(5)
            },
            
            'text_statistics': {
                'total_characters': total_characters,
                'unique_characters': unique_characters,
                'character_lengths': {
                    'mean': np.mean(text_lengths),
                    'median': np.median(text_lengths),
                    'std': np.std(text_lengths),
                    'min': np.min(text_lengths),
                    'max': np.max(text_lengths),
                    'percentiles': {
                        '25th': np.percentile(text_lengths, 25),
                        '75th': np.percentile(text_lengths, 75),
                        '90th': np.percentile(text_lengths, 90),
                        '95th': np.percentile(text_lengths, 95)
                    }
                },
                'word_counts': {
                    'mean': np.mean(word_counts),
                    'median': np.median(word_counts),
                    'std': np.std(word_counts),
                    'min': np.min(word_counts),
                    'max': np.max(word_counts)
                },
                'line_counts': {
                    'mean': np.mean(line_counts),
                    'median': np.median(line_counts),
                    'std': np.std(line_counts),
                    'min': np.min(line_counts),
                    'max': np.max(line_counts)
                }
            },
            
            'quality_scores': {
                'alt_lit_scores_available': len(alt_lit_scores),
                'alt_lit_scores_percentage': len(alt_lit_scores) / total_poems * 100,
            },
            
            'metadata_completeness': fields_completeness
        }
        
        # Add alt-lit score statistics if available
        if alt_lit_scores:
            stats['quality_scores'].update({
                'alt_lit_score_stats': {
                    'mean': np.mean(alt_lit_scores),
                    'median': np.median(alt_lit_scores),
                    'std': np.std(alt_lit_scores),
                    'min': np.min(alt_lit_scores),
                    'max': np.max(alt_lit_scores),
                    'distribution': {
                        'low_0_10': sum(1 for s in alt_lit_scores if 0 <= s <= 10),
                        'medium_11_20': sum(1 for s in alt_lit_scores if 11 <= s <= 20),
                        'high_21_plus': sum(1 for s in alt_lit_scores if s >= 21)
                    }
                }
            })
        
        return stats
    
    def _display_dataset_overview(self, poems: List[Dict], stats: Dict[str, Any]) -> None:
        """
        Display comprehensive dataset overview.
        
        Args:
            poems: List of poem dictionaries
            stats: Dataset statistics dictionary
        """
        print(f"\nDataset Overview:")
        print(f"  Total poems: {stats['basic_info']['total_poems']}")
        print(f"  Unique authors: {stats['basic_info']['unique_authors']}")
        
        # Alt-lit scores if available
        if 'alt_lit_score_stats' in stats['quality_scores']:
            score_stats = stats['quality_scores']['alt_lit_score_stats']
            print(f"  Alt-lit score range: {score_stats['min']:.0f} - {score_stats['max']:.0f}")
            print(f"  Average score: {score_stats['mean']:.1f}")
        else:
            print(f"  Alt-lit scores: Not available")
        
        # Text statistics
        text_stats = stats['text_statistics']
        print(f"  Total characters: {text_stats['total_characters']:,}")
        print(f"  Average poem length: {text_stats['character_lengths']['mean']:.0f} characters")
        print(f"  Average words per poem: {text_stats['word_counts']['mean']:.1f}")
        print(f"  Average lines per poem: {text_stats['line_counts']['mean']:.1f}")
        
        # Metadata completeness
        print(f"\\n  Metadata completeness:")
        for field, info in stats['metadata_completeness'].items():
            print(f"    {field}: {info['count']}/{stats['basic_info']['total_poems']} ({info['percentage']:.1f}%)")
    
    def filter_poems(self,
                    poems: Optional[List[Dict]] = None,
                    min_alt_lit_score: Optional[float] = None,
                    max_alt_lit_score: Optional[float] = None,
                    min_length: Optional[int] = None,
                    max_length: Optional[int] = None,
                    min_words: Optional[int] = None,
                    max_words: Optional[int] = None,
                    required_fields: Optional[List[str]] = None,
                    exclude_authors: Optional[List[str]] = None,
                    include_authors: Optional[List[str]] = None) -> List[Dict]:
        """
        Filter poems based on various criteria.
        
        Args:
            poems: List of poems to filter (uses self.poems if None)
            min_alt_lit_score: Minimum alt-lit score threshold
            max_alt_lit_score: Maximum alt-lit score threshold
            min_length: Minimum character length
            max_length: Maximum character length
            min_words: Minimum word count
            max_words: Maximum word count
            required_fields: List of fields that must be present
            exclude_authors: List of authors to exclude
            include_authors: List of authors to include (exclusive with exclude)
            
        Returns:
            Filtered list of poems
        """
        if poems is None:
            if self.poems is None:
                raise ValueError("No poems loaded. Call load_dataset() first.")
            poems = self.poems
        
        original_count = len(poems)
        filtered_poems = poems.copy()
        
        logger.info(f"Filtering {original_count} poems with criteria...")
        
        # Filter by alt-lit score
        if min_alt_lit_score is not None or max_alt_lit_score is not None:
            def score_filter(poem):
                score = poem.get('alt_lit_score')
                if score is None:
                    return False
                try:
                    score = float(score)
                    if min_alt_lit_score is not None and score < min_alt_lit_score:
                        return False
                    if max_alt_lit_score is not None and score > max_alt_lit_score:
                        return False
                    return True
                except (ValueError, TypeError):
                    return False
            
            filtered_poems = [p for p in filtered_poems if score_filter(p)]
            logger.info(f"After alt-lit score filter: {len(filtered_poems)} poems")
        
        # Filter by text length
        if min_length is not None or max_length is not None:
            def length_filter(poem):
                length = len(poem['text'])
                if min_length is not None and length < min_length:
                    return False
                if max_length is not None and length > max_length:
                    return False
                return True
            
            filtered_poems = [p for p in filtered_poems if length_filter(p)]
            logger.info(f"After length filter: {len(filtered_poems)} poems")
        
        # Filter by word count
        if min_words is not None or max_words is not None:
            def word_filter(poem):
                word_count = len(poem['text'].split())
                if min_words is not None and word_count < min_words:
                    return False
                if max_words is not None and word_count > max_words:
                    return False
                return True
            
            filtered_poems = [p for p in filtered_poems if word_filter(p)]
            logger.info(f"After word count filter: {len(filtered_poems)} poems")
        
        # Filter by required fields
        if required_fields:
            def field_filter(poem):
                for field in required_fields:
                    if field not in poem or poem[field] is None:
                        return False
                    if isinstance(poem[field], str) and len(poem[field].strip()) == 0:
                        return False
                return True
            
            filtered_poems = [p for p in filtered_poems if field_filter(p)]
            logger.info(f"After required fields filter: {len(filtered_poems)} poems")
        
        # Filter by authors
        if exclude_authors:
            filtered_poems = [p for p in filtered_poems if p.get('author') not in exclude_authors]
            logger.info(f"After author exclusion filter: {len(filtered_poems)} poems")
        
        if include_authors:
            filtered_poems = [p for p in filtered_poems if p.get('author') in include_authors]
            logger.info(f"After author inclusion filter: {len(filtered_poems)} poems")
        
        final_count = len(filtered_poems)
        logger.info(f"Filtering complete: {final_count}/{original_count} poems retained ({final_count/original_count*100:.1f}%)")
        
        return filtered_poems
    
    def get_sample_poems(self, 
                        count: int = 5,
                        random_state: Optional[int] = None) -> List[Dict]:
        """
        Get a random sample of poems for inspection.
        
        Args:
            count: Number of poems to sample
            random_state: Random seed for reproducibility
            
        Returns:
            List of sampled poems
        """
        if self.poems is None:
            raise ValueError("No poems loaded. Call load_dataset() first.")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        if count >= len(self.poems):
            return self.poems
        
        indices = np.random.choice(len(self.poems), size=count, replace=False)
        return [self.poems[i] for i in indices]
    
    def analyze_text_patterns(self, poems: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze text patterns in the poetry dataset.
        
        Args:
            poems: List of poems to analyze (uses self.poems if None)
            
        Returns:
            Dictionary with pattern analysis results
        """
        if poems is None:
            if self.poems is None:
                raise ValueError("No poems loaded. Call load_dataset() first.")
            poems = self.poems
        
        logger.info(f"Analyzing text patterns in {len(poems)} poems")
        
        # Collect all text
        all_texts = [poem['text'] for poem in poems]
        combined_text = ' '.join(all_texts)
        
        # Character frequency analysis
        char_counter = Counter(combined_text)
        most_common_chars = char_counter.most_common(20)
        
        # Line structure analysis
        line_patterns = {
            'empty_lines': sum(text.count('\n\n') for text in all_texts),
            'single_line_poems': sum(1 for text in all_texts if '\n' not in text),
            'very_short_lines': 0,  # Lines with <= 3 characters
            'very_long_lines': 0,   # Lines with >= 100 characters
        }
        
        for text in all_texts:
            lines = text.split('\n')
            line_patterns['very_short_lines'] += sum(1 for line in lines if len(line.strip()) <= 3)
            line_patterns['very_long_lines'] += sum(1 for line in lines if len(line.strip()) >= 100)
        
        # Unicode and special character analysis
        unicode_chars = set()
        for text in all_texts:
            for char in text:
                if ord(char) > 127:  # Non-ASCII characters
                    unicode_chars.add(char)
        
        return {
            'character_frequency': {
                'most_common': most_common_chars,
                'unique_characters': len(char_counter),
                'unicode_characters': {
                    'count': len(unicode_chars),
                    'examples': sorted(list(unicode_chars))[:20]
                }
            },
            'line_patterns': line_patterns,
            'text_diversity': {
                'total_unique_words': len(set(combined_text.lower().split())),
                'vocabulary_richness': len(set(combined_text.lower().split())) / len(combined_text.split()),
            }
        }
    
    def save_filtered_dataset(self, 
                             poems: List[Dict],
                             output_path: Union[str, Path],
                             include_stats: bool = True) -> None:
        """
        Save filtered dataset to JSON file.
        
        Args:
            poems: List of poems to save
            output_path: Output file path
            include_stats: Whether to include statistics in metadata
        """
        output_path = Path(output_path)
        
        # Prepare data to save
        data_to_save = {
            'poems': poems,
            'metadata': {
                'created_by': 'PoetryDatasetLoader',
                'original_count': len(self.poems) if self.poems else len(poems),
                'filtered_count': len(poems),
                'creation_date': str(np.datetime64('now')),
            }
        }
        
        # Add statistics if requested
        if include_stats and len(poems) > 0:
            data_to_save['statistics'] = self.compute_dataset_statistics(poems)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(poems)} poems to {output_path}")
        print(f"💾 Saved filtered dataset: {len(poems)} poems to {output_path.name}")
    
    def create_sample_dataset(self, output_path: Union[str, Path]) -> List[Dict]:
        """
        Create a small sample dataset for testing purposes.
        
        Args:
            output_path: Path to save sample dataset
            
        Returns:
            List of sample poems
        """
        sample_poems = [
            {
                "title": "Digital Intimacy",
                "author": "Sample Author",
                "text": "i think about love\nlike a distant star\nshining in darkness\n\nyour messages arrive\nlike small gifts\nwrapped in blue light",
                "alt_lit_score": 15,
                "url": "sample_url_1"
            },
            {
                "title": "Urban Loneliness",
                "author": "Another Author", 
                "text": "walking through the city\nat 3am\neveryone else is sleeping\n\nbut i'm here\ncounting streetlights\nlike rosary beads",
                "alt_lit_score": 18,
                "url": "sample_url_2"
            },
            {
                "title": "Internet Feelings",
                "author": "Sample Author",
                "text": "refresh\nrefresh\nrefresh\n\nwaiting for something\nthat might never come\nbut hoping anyway\n\n❤️ 0 likes",
                "alt_lit_score": 22,
                "url": "sample_url_3"
            }
        ]
        
        self.save_filtered_dataset(sample_poems, output_path, include_stats=True)
        return sample_poems