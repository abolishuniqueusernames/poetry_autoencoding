"""
Poetry-specialized tokenizer for RNN autoencoder training

Provides a custom tokenizer optimized for contemporary poetry and alt-lit aesthetic,
with special handling for Unicode decorations, line structure, and semantic elements
while preserving case sensitivity and poetic formatting.
"""

import re
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import Counter

try:
    import spacy
    from spacy.lang.en import English
except ImportError:
    spacy = None
    English = None

from ..config import TokenizationConfig, Config, default_config
from .text_preprocessing import clean_poetry_text, normalize_unicode_for_poetry


logger = logging.getLogger(__name__)


class PoetryTokenizer:
    """
    Custom tokenizer optimized for contemporary poetry and alt-lit aesthetic.
    
    This tokenizer handles the unique characteristics of contemporary poetry:
    - Preserves decorative Unicode elements (❤, ☁️, etc.) as semantic tokens
    - Respects line breaks and stanza structure
    - Maintains case sensitivity for aesthetic elements
    - Handles Japanese emoticons and alt-lit decorations
    - Filters web scraping artifacts while preserving meaningful content
    
    The tokenizer integrates with the configuration system and provides
    both preprocessing and analysis capabilities for poetry datasets.
    
    Attributes:
        config: TokenizationConfig instance with tokenization settings
        nlp: spaCy language model for base tokenization  
        special_tokens: Dict mapping special tokens to descriptions
        unicode_patterns: Compiled regex patterns for Unicode handling
        vocabulary: Built vocabulary mapping tokens to indices
        
    Example:
        >>> from poetry_rnn.tokenization import PoetryTokenizer
        >>> tokenizer = PoetryTokenizer()
        >>> tokens = tokenizer.tokenize("love ❤\\nflows like rain")
        >>> print(tokens)
        ['love', '<HEART>', '<LINE_BREAK>', 'flows', 'like', 'rain']
        
        >>> # Analyze poem structure
        >>> analysis = tokenizer.analyze_poem_structure("first stanza\\n\\nsecond stanza")
        >>> print(f"Stanzas: {analysis['stanza_breaks']}")
    """
    
    def __init__(self, 
                 config: Optional[TokenizationConfig] = None,
                 nlp_model: Optional[Any] = None,
                 project_root: Optional[Path] = None):
        """
        Initialize the poetry tokenizer.
        
        Args:
            config: TokenizationConfig instance. If None, uses default config
            nlp_model: Pre-loaded spaCy model. If None, loads English model
            project_root: Project root path for loading emoji files. Auto-detected if None
            
        Raises:
            ImportError: If spaCy is not available and nlp_model is None
            FileNotFoundError: If emoji files are missing and cannot be loaded
        """
        
        # Set up configuration
        if config is None:
            if hasattr(default_config, 'tokenization'):
                self.config = default_config.tokenization
            else:
                self.config = TokenizationConfig()
        else:
            self.config = config
        
        # Set up project root for resource loading
        if project_root is None:
            if hasattr(default_config, 'project_root'):
                self.project_root = default_config.project_root
            else:
                self.project_root = self._detect_project_root()
        else:
            self.project_root = Path(project_root)
            
        # Initialize spaCy model
        self.nlp = self._initialize_nlp_model(nlp_model)
        
        # Initialize tokenization components
        self._initialize_special_tokens()
        self._initialize_unicode_patterns()
        
        # Vocabulary will be built when needed
        self.vocabulary: Dict[str, int] = {}
        self.inverse_vocabulary: Dict[int, str] = {}
        
        logger.info(f"PoetryTokenizer initialized with {len(self.special_tokens)} special tokens")
        
    def _detect_project_root(self) -> Path:
        """Auto-detect project root directory."""
        current_dir = Path.cwd()
        
        # Look for project markers
        project_markers = ["CLAUDE.md", "poetry_rnn", "dataset_poetry", "GLoVe preprocessing"]
        
        # Check current directory and parents
        for path in [current_dir] + list(current_dir.parents):
            if any((path / marker).exists() for marker in project_markers):
                return path
                
        # Fallback to current directory
        return current_dir
        
    def _initialize_nlp_model(self, nlp_model: Optional[Any]) -> Any:
        """Initialize spaCy model for tokenization."""
        
        if nlp_model is not None:
            return nlp_model
            
        if spacy is None:
            raise ImportError(
                "spaCy is required for tokenization. Install with: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )
        
        try:
            # Try to load the full model first
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded en_core_web_sm spaCy model")
        except (OSError, IOError):
            try:
                # Fall back to basic English tokenizer
                nlp = English()
                logger.warning("Full spaCy model not found, using basic English tokenizer")
            except Exception as e:
                raise ImportError(f"Could not initialize spaCy model: {e}")
                
        return nlp
        
    def _initialize_special_tokens(self) -> None:
        """Initialize special tokens from configuration."""
        
        # Start with configuration tokens
        if hasattr(self.config, 'special_tokens') and self.config.special_tokens:
            special_tokens_list = self.config.special_tokens
        else:
            special_tokens_list = ['<UNK>', '<LINE_BREAK>', '<STANZA_BREAK>', '<POEM_START>', '<POEM_END>']
        
        # Create special tokens dictionary with descriptions
        self.special_tokens = {
            '<UNK>': 'unknown word token',
            '<LINE_BREAK>': 'line break marker',
            '<STANZA_BREAK>': 'stanza break marker', 
            '<POEM_START>': 'poem start marker',
            '<POEM_END>': 'poem end marker',
            '<PAD>': 'padding token',
            '<MASK>': 'masked token for training'
        }
        
        # Ensure all config tokens are included
        for token in special_tokens_list:
            if token not in self.special_tokens:
                self.special_tokens[token] = f'special token {token}'
                
        logger.debug(f"Initialized {len(self.special_tokens)} special tokens")
        
    def _initialize_unicode_patterns(self) -> None:
        """Initialize Unicode pattern mappings for semantic token conversion."""
        
        # Base Unicode patterns for common poetry elements
        base_patterns = {
            r'[❤♥💕💖💗💘💙💚💛💜🧡]': '<HEART>',
            r'[☁️⛅⛈️🌤️🌥️🌦️🌧️🌨️🌩️]': '<CLOUD>',
            r'[🌟⭐✨💫⚡]': '<STAR>',
            r'[🌸🌺🌻🌷🌹💐🌼]': '<FLOWER>',
            r'[😢😭💧]': '<TEAR>',
            r'[🔥💥]': '<FIRE>'
        }
        
        self.unicode_patterns = base_patterns.copy()
        
        # Load Japanese emoticons if available
        japanese_emoji_path = self.project_root / "GLoVe preprocessing" / "japaneseemojis"
        
        if japanese_emoji_path.exists():
            try:
                self._load_japanese_emoticons(japanese_emoji_path)
                logger.info(f"Loaded {len([k for k in self.unicode_patterns.keys() if '<WEEB_SHIT>' in self.unicode_patterns[k]])} Japanese emoticons")
            except Exception as e:
                logger.warning(f"Could not load Japanese emoticons: {e}")
        else:
            logger.debug(f"Japanese emoticons file not found at {japanese_emoji_path}")
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for regex_pattern, token in self.unicode_patterns.items():
            try:
                self.compiled_patterns[token] = re.compile(regex_pattern)
            except re.error as e:
                logger.warning(f"Could not compile regex pattern '{regex_pattern}': {e}")
                
        logger.debug(f"Compiled {len(self.compiled_patterns)} Unicode patterns")
        
    def _load_japanese_emoticons(self, emoji_file_path: Path) -> None:
        """Load Japanese emoticons from file and add to patterns."""
        
        with open(emoji_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                emoji = line.strip()
                if emoji:  # Skip empty lines
                    # Escape special regex characters in emoticons
                    escaped_emoji = re.escape(emoji)
                    self.unicode_patterns[escaped_emoji] = '<WEEB_SHIT>'
                    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text while preserving poetic structure.
        
        Applies semantic Unicode token replacement, normalizes line breaks,
        and cleans web artifacts while maintaining poetic formatting intent.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text with semantic tokens and normalized structure
            
        Example:
            >>> tokenizer.preprocess_text("love flows ❤\\n\\nlike rain")
            'love flows <HEART> <STANZA_BREAK> like rain'
        """
        
        if not text or not isinstance(text, str):
            return ""
        
        processed = text
        
        # Replace Unicode decorations with semantic tokens
        for token, pattern in self.compiled_patterns.items():
            processed = pattern.sub(f' {token} ', processed)
        
        # Normalize line breaks to semantic tokens
        # Double/triple line breaks = stanza breaks
        processed = re.sub(r'\n\s*\n+', ' <STANZA_BREAK> ', processed)
        # Single line breaks = line breaks  
        processed = re.sub(r'\n', ' <LINE_BREAK> ', processed)
        
        # Clean extra whitespace but preserve intentional spacing
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
        
    def tokenize(self, text: str, clean_text: bool = True) -> List[str]:
        """
        Tokenize poetry text with spaCy and custom processing.
        
        Performs complete tokenization pipeline including preprocessing,
        spaCy tokenization, and poetry-specific filtering and processing.
        
        Args:
            text: Text to tokenize
            clean_text: Whether to apply text cleaning preprocessing
            
        Returns:
            List of tokens with special tokens and cleaned content
            
        Example:
            >>> tokens = tokenizer.tokenize("The year 2023\\nwas different ❤")
            >>> print(tokens)
            ['The', 'year', '2023', '<LINE_BREAK>', 'was', 'different', '<HEART>']
        """
        
        if not text or not isinstance(text, str):
            return []
        
        # Optional text cleaning
        if clean_text:
            text = clean_poetry_text(text, preserve_numbers=self.config.preserve_numbers)
        
        # Preprocess for poetry-specific patterns
        processed_text = self.preprocess_text(text)
        
        # Use spaCy for initial tokenization
        doc = self.nlp(processed_text)
        
        tokens = []
        for token in doc:
            # Handle special tokens
            if token.text in self.special_tokens:
                tokens.append(token.text)
            # Handle semantic Unicode tokens
            elif token.text.startswith('<') and token.text.endswith('>'):
                tokens.append(token.text)
            # Skip unwanted tokens
            elif self._should_skip_token(token):
                continue
            # Process regular words
            else:
                processed_token = self._process_token(token)
                if processed_token:
                    tokens.append(processed_token)
        
        return tokens
        
    def _should_skip_token(self, token) -> bool:
        """
        Determine if a token should be skipped during tokenization.
        
        Filters out whitespace, very short noise tokens, and website metadata
        while preserving meaningful single characters and punctuation.
        
        Args:
            token: spaCy token to evaluate
            
        Returns:
            True if token should be skipped, False otherwise
        """
        
        # Skip pure whitespace
        if token.is_space:
            return True
        
        # Skip empty tokens
        text = token.text.strip()
        if not text:
            return True
            
        # Skip website metadata patterns
        metadata_patterns = [
            'continue', 'reading', 'posted', 'tags', 'category',
            'www', '.com', 'http', 'onlylovepoetry', 'wordpress',
            'tumblr', 'instagram', 'twitter', 'facebook'
        ]
        
        if text.lower() in metadata_patterns:
            return True
            
        # Skip very short tokens unless they're meaningful
        if len(text) == 1:
            # Keep meaningful single characters
            meaningful_singles = {'.', ',', '!', '?', ';', ':', '-', '"', "'", 
                                '(', ')', 'i', 'a', 'I', 'A', 'o', 'O'}
            if text not in meaningful_singles:
                return True
        
        # Skip tokens that are likely noise (multiple punctuation, weird combinations)
        if re.match(r'^[^\w\s]+$', text) and len(text) > 2:
            return True
            
        return False
        
    def _process_token(self, token) -> Optional[str]:
        """
        Process individual tokens with poetry-specific handling.
        
        Applies case preservation settings and handles contractions
        and hyphenated words appropriately for poetry context.
        
        Args:
            token: spaCy token to process
            
        Returns:
            Processed token string, or None if token should be discarded
        """
        
        text = token.text.strip()
        if not text:
            return None
        
        # Preserve case for poetic effect if enabled
        if not self.config.preserve_case:
            text = text.lower()
        
        # Handle contractions and hyphenated words
        # Keep them as single units to preserve meaning in poetry context
        # (e.g., "don't", "twenty-one", "self-aware")
        
        # Remove leading/trailing punctuation for some cases, but be careful
        # Poetry often uses punctuation meaningfully
        
        return text
        
    def build_vocabulary(self, texts: List[str], min_count: int = None) -> None:
        """
        Build vocabulary from a collection of texts.
        
        Tokenizes all texts and builds vocabulary mappings with frequency filtering.
        Special tokens are always included regardless of frequency.
        
        Args:
            texts: List of texts to build vocabulary from
            min_count: Minimum token frequency (uses config default if None)
            
        Example:
            >>> texts = ["love flows", "rain falls ❤"]
            >>> tokenizer.build_vocabulary(texts)
            >>> print(len(tokenizer.vocabulary))
            7  # includes special tokens
        """
        
        if min_count is None:
            min_count = getattr(self.config, 'min_count', 1)
        
        # Collect all tokens
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        
        logger.info(f"Collected {len(all_tokens)} total tokens from {len(texts)} texts")
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Filter by minimum count, but always keep special tokens
        vocab_tokens = []
        for token, count in token_counts.items():
            if count >= min_count or token in self.special_tokens:
                vocab_tokens.append(token)
        
        # Sort for consistent indexing (special tokens first, then alphabetical)
        special_tokens_list = [t for t in vocab_tokens if t in self.special_tokens]
        regular_tokens = sorted([t for t in vocab_tokens if t not in self.special_tokens])
        vocab_tokens = special_tokens_list + regular_tokens
        
        # Create mappings
        self.vocabulary = {token: idx for idx, token in enumerate(vocab_tokens)}
        self.inverse_vocabulary = {idx: token for token, idx in self.vocabulary.items()}
        
        logger.info(f"Built vocabulary of size {len(self.vocabulary)} (min_count={min_count})")
        logger.debug(f"Token frequency range: {min(token_counts.values())}-{max(token_counts.values())}")
        
    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to vocabulary indices.
        
        Args:
            tokens: List of tokens to convert
            
        Returns:
            List of vocabulary indices, with <UNK> for out-of-vocabulary tokens
        """
        
        if not self.vocabulary:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        unk_idx = self.vocabulary.get('<UNK>', 0)
        return [self.vocabulary.get(token, unk_idx) for token in tokens]
        
    def indices_to_tokens(self, indices: List[int]) -> List[str]:
        """
        Convert vocabulary indices back to tokens.
        
        Args:
            indices: List of vocabulary indices
            
        Returns:
            List of tokens
        """
        
        if not self.inverse_vocabulary:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        return [self.inverse_vocabulary.get(idx, '<UNK>') for idx in indices]
        
    def analyze_poem_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze structural features of a poem.
        
        Provides detailed analysis of poem structure including line counts,
        stanza breaks, decorative elements, and vocabulary statistics.
        
        Args:
            text: Poem text to analyze
            
        Returns:
            Dictionary with structural analysis metrics
            
        Example:
            >>> analysis = tokenizer.analyze_poem_structure("love flows\\n\\nlike rain ❤")
            >>> print(f"Lines: {analysis['line_breaks']}, Stanzas: {analysis['stanza_breaks']}")
            Lines: 0, Stanzas: 1
        """
        
        if not text:
            return self._empty_analysis()
        
        tokens = self.tokenize(text)
        
        if not tokens:
            return self._empty_analysis()
        
        # Count structural elements
        line_breaks = tokens.count('<LINE_BREAK>')
        stanza_breaks = tokens.count('<STANZA_BREAK>')
        
        # Count Unicode decorations (semantic tokens that aren't structural)
        decoration_tokens = set()
        for token in tokens:
            if (token.startswith('<') and token.endswith('>') 
                and token not in self.special_tokens):
                decoration_tokens.add(token)
        
        # Regular word tokens (not special or decoration tokens)
        regular_tokens = [t for t in tokens if not t.startswith('<') or t.endswith('>')]
        
        # Calculate statistics
        analysis = {
            'total_tokens': len(tokens),
            'regular_tokens': len(regular_tokens),
            'line_breaks': line_breaks,
            'stanza_breaks': stanza_breaks,
            'unicode_decorations': len(decoration_tokens),
            'decoration_types': list(decoration_tokens),
            'unique_tokens': len(set(tokens)),
            'unique_regular_tokens': len(set(regular_tokens)),
            'type_token_ratio': len(set(tokens)) / len(tokens) if tokens else 0,
            'avg_token_length': np.mean([len(t) for t in regular_tokens]) if regular_tokens else 0,
            'poem_lines': line_breaks + 1,  # Lines = line breaks + 1
            'poem_stanzas': stanza_breaks + 1 if stanza_breaks > 0 else 1
        }
        
        return analysis
        
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure for invalid/empty texts."""
        return {
            'total_tokens': 0,
            'regular_tokens': 0,
            'line_breaks': 0,
            'stanza_breaks': 0,
            'unicode_decorations': 0,
            'decoration_types': [],
            'unique_tokens': 0,
            'unique_regular_tokens': 0,
            'type_token_ratio': 0,
            'avg_token_length': 0,
            'poem_lines': 0,
            'poem_stanzas': 0
        }
        
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the built vocabulary.
        
        Returns:
            Dictionary with vocabulary statistics
            
        Example:
            >>> stats = tokenizer.get_vocabulary_stats()
            >>> print(f"Vocab size: {stats['total_tokens']}")
        """
        
        if not self.vocabulary:
            return {
                'total_tokens': 0,
                'special_tokens': 0,
                'regular_tokens': 0,
                'decoration_tokens': 0
            }
        
        special_count = sum(1 for token in self.vocabulary.keys() 
                          if token in self.special_tokens)
        decoration_count = sum(1 for token in self.vocabulary.keys()
                             if token.startswith('<') and token.endswith('>')
                             and token not in self.special_tokens)
        regular_count = len(self.vocabulary) - special_count - decoration_count
        
        return {
            'total_tokens': len(self.vocabulary),
            'special_tokens': special_count,
            'regular_tokens': regular_count,
            'decoration_tokens': decoration_count,
            'special_token_list': list(self.special_tokens.keys()),
            'decoration_token_list': [t for t in self.vocabulary.keys()
                                    if t.startswith('<') and t.endswith('>')
                                    and t not in self.special_tokens]
        }
        
    def save_vocabulary(self, path: Path) -> None:
        """Save vocabulary to file."""
        
        if not self.vocabulary:
            raise ValueError("No vocabulary to save")
            
        vocab_data = {
            'vocabulary': self.vocabulary,
            'config': {
                'preserve_case': self.config.preserve_case,
                'preserve_numbers': self.config.preserve_numbers,
                'special_tokens': list(self.special_tokens.keys())
            }
        }
        
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved vocabulary of size {len(self.vocabulary)} to {path}")
        
    def load_vocabulary(self, path: Path) -> None:
        """Load vocabulary from file."""
        
        import json
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            
        self.vocabulary = vocab_data['vocabulary']
        self.inverse_vocabulary = {v: k for k, v in self.vocabulary.items()}
        
        logger.info(f"Loaded vocabulary of size {len(self.vocabulary)} from {path}")
        
    def __repr__(self) -> str:
        vocab_size = len(self.vocabulary) if self.vocabulary else 0
        return f"PoetryTokenizer(vocab_size={vocab_size}, preserve_case={self.config.preserve_case})"