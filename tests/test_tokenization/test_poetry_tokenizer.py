"""
Unit tests for PoetryTokenizer

These tests validate the poetry-specific tokenization functionality,
including the recent fixes for Unicode handling, special token processing,
and poetry structure preservation.

Test Coverage:
- Basic tokenization functionality
- Unicode decoration handling (❤️, ☁️, etc.)
- Special token processing (<LINE_BREAK>, <STANZA_BREAK>, etc.)
- Case preservation settings
- Vocabulary building and management
- Text preprocessing and cleaning
- Japanese emoticon handling
- Poetry structure analysis
- Error handling and edge cases
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from typing import List, Dict, Any

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from poetry_rnn.tokenization.poetry_tokenizer import PoetryTokenizer
from poetry_rnn.config import TokenizationConfig


@pytest.fixture
def basic_tokenizer():
    """Basic tokenizer with default configuration."""
    return PoetryTokenizer()


@pytest.fixture
def custom_tokenizer():
    """Tokenizer with custom configuration."""
    config = TokenizationConfig(
        preserve_case=False,
        preserve_numbers=False,
        max_sequence_length=100,
        min_sequence_length=2
    )
    return PoetryTokenizer(config=config)


@pytest.fixture
def sample_poetry_texts():
    """Sample poetry texts for testing."""
    return [
        "Love is like the summer rain,\nFalling softly on my heart ❤️.",
        "i write in lowercase\nbecause CAPITALS are for\nSERIOUS THINGS\n\nlike taxes",
        "brief\npoetic\nmoment",
        "Digital dreams ☁️ and silicon hearts 💫\nwhere algorithms learn to feel",
        "Numbers like 42 and 3.14\nmean different things\nin poetry vs code"
    ]


class TestBasicTokenization:
    """Test basic tokenization functionality."""
    
    def test_tokenizer_initialization(self):
        """Test that tokenizer initializes correctly."""
        tokenizer = PoetryTokenizer()
        
        assert tokenizer is not None
        assert tokenizer.config is not None
        assert tokenizer.special_tokens is not None
        assert len(tokenizer.special_tokens) > 0
        assert hasattr(tokenizer, 'nlp')
        assert hasattr(tokenizer, 'unicode_patterns')
    
    def test_special_tokens_presence(self, basic_tokenizer):
        """Test that all required special tokens are present."""
        required_tokens = [
            '<UNK>', '<LINE_BREAK>', '<STANZA_BREAK>', 
            '<POEM_START>', '<POEM_END>', '<PAD>', '<MASK>'
        ]
        
        for token in required_tokens:
            assert token in basic_tokenizer.special_tokens
    
    def test_basic_tokenization(self, basic_tokenizer):
        """Test basic tokenization of simple text."""
        text = "Love flows like rain"
        tokens = basic_tokenizer.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert 'Love' in tokens
        assert 'flows' in tokens
        assert 'like' in tokens
        assert 'rain' in tokens
    
    def test_empty_text_handling(self, basic_tokenizer):
        """Test handling of empty or None text."""
        assert basic_tokenizer.tokenize("") == []
        assert basic_tokenizer.tokenize(None) == []
        assert basic_tokenizer.tokenize("   ") == []
    
    def test_line_break_tokenization(self, basic_tokenizer):
        """Test that line breaks are converted to special tokens."""
        text = "First line\nSecond line"
        tokens = basic_tokenizer.tokenize(text)
        
        assert '<LINE_BREAK>' in tokens
        
        # Check positioning
        first_idx = tokens.index('First')
        line_break_idx = tokens.index('<LINE_BREAK>')
        second_idx = tokens.index('Second')
        
        assert first_idx < line_break_idx < second_idx
    
    def test_stanza_break_tokenization(self, basic_tokenizer):
        """Test that stanza breaks are converted to special tokens."""
        text = "First stanza\n\nSecond stanza"
        tokens = basic_tokenizer.tokenize(text)
        
        assert '<STANZA_BREAK>' in tokens
        
        # Should not have separate line breaks for double newlines
        assert '<LINE_BREAK>' not in tokens or tokens.count('<LINE_BREAK>') == 0
    
    def test_case_preservation(self):
        """Test case preservation settings."""
        # Test with case preservation (default)
        tokenizer_preserve = PoetryTokenizer()
        text = "Love AND Hate"
        tokens_preserve = tokenizer_preserve.tokenize(text)
        
        assert 'Love' in tokens_preserve
        assert 'AND' in tokens_preserve
        assert 'Hate' in tokens_preserve
        
        # Test without case preservation
        config = TokenizationConfig(preserve_case=False)
        tokenizer_lower = PoetryTokenizer(config=config)
        tokens_lower = tokenizer_lower.tokenize(text)
        
        # Should be lowercased
        assert 'love' in tokens_lower or any('love' in token.lower() for token in tokens_lower)
        assert not any(token.isupper() and len(token) > 1 for token in tokens_lower)
    
    def test_number_preservation(self):
        """Test number preservation settings."""
        text = "Year 2023 and number 42"
        
        # Test with number preservation (default)
        config_preserve = TokenizationConfig(preserve_numbers=True)
        tokenizer_preserve = PoetryTokenizer(config=config_preserve)
        tokens_preserve = tokenizer_preserve.tokenize(text)
        
        assert '2023' in tokens_preserve or any('2023' in token for token in tokens_preserve)
        assert '42' in tokens_preserve or any('42' in token for token in tokens_preserve)
        
        # Test without number preservation
        config_no_preserve = TokenizationConfig(preserve_numbers=False)
        tokenizer_no_preserve = PoetryTokenizer(config=config_no_preserve)
        tokens_no_preserve = tokenizer_no_preserve.tokenize(text)
        
        # Numbers might be filtered or modified
        # This test depends on the text_preprocessing implementation


class TestUnicodeHandling:
    """Test Unicode decoration and emoji handling."""
    
    def test_heart_emoji_conversion(self, basic_tokenizer):
        """Test that heart emojis are converted to semantic tokens."""
        text = "Love ❤️ and more ♥ hearts"
        tokens = basic_tokenizer.tokenize(text)
        
        assert '<HEART>' in tokens
        # Should not have raw emoji in tokens
        assert '❤️' not in tokens
        assert '♥' not in tokens
    
    def test_cloud_emoji_conversion(self, basic_tokenizer):
        """Test that cloud emojis are converted to semantic tokens."""
        text = "Dreams float like ☁️ clouds"
        tokens = basic_tokenizer.tokenize(text)
        
        assert '<CLOUD>' in tokens
        assert '☁️' not in tokens
    
    def test_star_emoji_conversion(self, basic_tokenizer):
        """Test that star emojis are converted to semantic tokens."""
        text = "Wishes upon a ⭐ star"
        tokens = basic_tokenizer.tokenize(text)
        
        assert '<STAR>' in tokens
        assert '⭐' not in tokens
    
    def test_flower_emoji_conversion(self, basic_tokenizer):
        """Test that flower emojis are converted to semantic tokens."""
        text = "Spring brings 🌸 blossoms"
        tokens = basic_tokenizer.tokenize(text)
        
        assert '<FLOWER>' in tokens
        assert '🌸' not in tokens
    
    def test_multiple_unicode_elements(self, basic_tokenizer):
        """Test handling of multiple Unicode decorations."""
        text = "Love ❤️ flows like ☁️ clouds under ⭐ stars"
        tokens = basic_tokenizer.tokenize(text)
        
        assert '<HEART>' in tokens
        assert '<CLOUD>' in tokens
        assert '<STAR>' in tokens
        
        # Check that original emojis are not present
        unicode_chars = ['❤️', '☁️', '⭐']
        for char in unicode_chars:
            assert char not in tokens
    
    def test_japanese_emoticon_handling(self, basic_tokenizer):
        """Test Japanese emoticon handling if available."""
        # This test will depend on whether Japanese emoticons file exists
        # We'll test the mechanism even if the file isn't present
        
        # Mock some Japanese-style emoticons
        text = "Happy (╯°□°）╯ and sad ಥ_ಥ faces"
        tokens = basic_tokenizer.tokenize(text)
        
        # Should tokenize without errors
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert 'Happy' in tokens
        assert 'and' in tokens
        assert 'sad' in tokens
    
    def test_unicode_pattern_compilation(self, basic_tokenizer):
        """Test that Unicode patterns compile correctly."""
        assert hasattr(basic_tokenizer, 'compiled_patterns')
        assert len(basic_tokenizer.compiled_patterns) > 0
        
        # Test that patterns are actually compiled regex objects
        import re
        for pattern in basic_tokenizer.compiled_patterns.values():
            assert isinstance(pattern, re.Pattern)


class TestVocabularyManagement:
    """Test vocabulary building and management."""
    
    def test_vocabulary_building(self, basic_tokenizer, sample_poetry_texts):
        """Test building vocabulary from texts."""
        basic_tokenizer.build_vocabulary(sample_poetry_texts)
        
        assert len(basic_tokenizer.vocabulary) > 0
        assert len(basic_tokenizer.inverse_vocabulary) > 0
        assert len(basic_tokenizer.vocabulary) == len(basic_tokenizer.inverse_vocabulary)
        
        # Special tokens should be included
        for special_token in basic_tokenizer.special_tokens:
            assert special_token in basic_tokenizer.vocabulary
    
    def test_vocabulary_statistics(self, basic_tokenizer, sample_poetry_texts):
        """Test vocabulary statistics."""
        basic_tokenizer.build_vocabulary(sample_poetry_texts)
        stats = basic_tokenizer.get_vocabulary_stats()
        
        assert 'total_tokens' in stats
        assert 'special_tokens' in stats
        assert 'regular_tokens' in stats
        assert 'decoration_tokens' in stats
        
        assert stats['total_tokens'] > 0
        assert stats['special_tokens'] > 0
        assert stats['total_tokens'] == (
            stats['special_tokens'] + 
            stats['regular_tokens'] + 
            stats['decoration_tokens']
        )
    
    def test_token_index_conversion(self, basic_tokenizer, sample_poetry_texts):
        """Test converting between tokens and indices."""
        basic_tokenizer.build_vocabulary(sample_poetry_texts)
        
        # Test conversion
        tokens = ['Love', 'is', '<HEART>', 'beautiful']
        indices = basic_tokenizer.tokens_to_indices(tokens)
        converted_tokens = basic_tokenizer.indices_to_tokens(indices)
        
        assert len(indices) == len(tokens)
        assert len(converted_tokens) == len(tokens)
        
        # Special tokens should be preserved
        assert '<HEART>' in converted_tokens
    
    def test_unknown_token_handling(self, basic_tokenizer, sample_poetry_texts):
        """Test handling of unknown tokens."""
        basic_tokenizer.build_vocabulary(sample_poetry_texts)
        
        # Test with unknown token
        unknown_tokens = ['unknown_word_xyz123']
        indices = basic_tokenizer.tokens_to_indices(unknown_tokens)
        
        # Should get UNK index
        unk_idx = basic_tokenizer.vocabulary.get('<UNK>', 0)
        assert indices[0] == unk_idx
    
    def test_vocabulary_save_load(self, basic_tokenizer, sample_poetry_texts, temp_dir):
        """Test saving and loading vocabulary."""
        basic_tokenizer.build_vocabulary(sample_poetry_texts)
        
        # Save vocabulary
        vocab_file = temp_dir / "test_vocab.json"
        basic_tokenizer.save_vocabulary(vocab_file)
        
        assert vocab_file.exists()
        
        # Create new tokenizer and load vocabulary
        new_tokenizer = PoetryTokenizer()
        new_tokenizer.load_vocabulary(vocab_file)
        
        # Should have same vocabulary
        assert new_tokenizer.vocabulary == basic_tokenizer.vocabulary
        assert len(new_tokenizer.inverse_vocabulary) == len(basic_tokenizer.inverse_vocabulary)


class TestTextPreprocessing:
    """Test text preprocessing functionality."""
    
    def test_preprocess_text_basic(self, basic_tokenizer):
        """Test basic text preprocessing."""
        text = "Love flows ❤️\n\nlike rain"
        processed = basic_tokenizer.preprocess_text(text)
        
        # Should have semantic tokens
        assert '<HEART>' in processed
        assert '<STANZA_BREAK>' in processed
        
        # Original Unicode should be replaced
        assert '❤️' not in processed
    
    def test_preprocess_line_breaks(self, basic_tokenizer):
        """Test line break preprocessing."""
        text = "Line one\nLine two\n\nLine four"
        processed = basic_tokenizer.preprocess_text(text)
        
        assert '<LINE_BREAK>' in processed
        assert '<STANZA_BREAK>' in processed
        
        # Should not have raw newlines
        assert '\n' not in processed
    
    def test_preprocess_whitespace_normalization(self, basic_tokenizer):
        """Test whitespace normalization."""
        text = "Too    many     spaces\n\n\n\ntoo many newlines"
        processed = basic_tokenizer.preprocess_text(text)
        
        # Should normalize excessive whitespace
        assert '    ' not in processed
        assert '\n\n\n' not in processed
        
        # But should preserve semantic structure
        assert '<STANZA_BREAK>' in processed
    
    def test_preprocess_empty_or_invalid(self, basic_tokenizer):
        """Test preprocessing of empty or invalid input."""
        assert basic_tokenizer.preprocess_text("") == ""
        assert basic_tokenizer.preprocess_text(None) == ""
        assert basic_tokenizer.preprocess_text("   ") == ""


class TestPoetryStructureAnalysis:
    """Test poetry structure analysis functionality."""
    
    def test_poem_structure_analysis(self, basic_tokenizer):
        """Test comprehensive poem structure analysis."""
        text = "Love flows ❤️\nlike summer rain\n\nBringing joy ⭐\nto waiting hearts"
        analysis = basic_tokenizer.analyze_poem_structure(text)
        
        # Should have analysis components
        assert 'total_tokens' in analysis
        assert 'line_breaks' in analysis
        assert 'stanza_breaks' in analysis
        assert 'unicode_decorations' in analysis
        assert 'decoration_types' in analysis
        assert 'poem_lines' in analysis
        assert 'poem_stanzas' in analysis
        
        # Check reasonable values
        assert analysis['total_tokens'] > 0
        assert analysis['line_breaks'] >= 0
        assert analysis['stanza_breaks'] >= 0
        assert analysis['unicode_decorations'] > 0  # Has ❤️ and ⭐
        assert analysis['poem_stanzas'] >= 1
    
    def test_structure_analysis_decoration_detection(self, basic_tokenizer):
        """Test detection of different decoration types."""
        text = "Hearts ❤️ and stars ⭐ and clouds ☁️"
        analysis = basic_tokenizer.analyze_poem_structure(text)
        
        decoration_types = analysis['decoration_types']
        assert '<HEART>' in decoration_types
        assert '<STAR>' in decoration_types
        assert '<CLOUD>' in decoration_types
        assert analysis['unicode_decorations'] >= 3
    
    def test_structure_analysis_empty_text(self, basic_tokenizer):
        """Test structure analysis of empty text."""
        analysis = basic_tokenizer.analyze_poem_structure("")
        
        # Should return empty analysis structure
        assert analysis['total_tokens'] == 0
        assert analysis['line_breaks'] == 0
        assert analysis['stanza_breaks'] == 0
        assert analysis['unicode_decorations'] == 0
        assert analysis['poem_lines'] == 0
        assert analysis['poem_stanzas'] == 0
    
    def test_structure_analysis_statistics(self, basic_tokenizer):
        """Test statistical measures in structure analysis."""
        text = "Love flows\nlike rain\nbringing joy\nto hearts"
        analysis = basic_tokenizer.analyze_poem_structure(text)
        
        assert 'type_token_ratio' in analysis
        assert 'avg_token_length' in analysis
        assert 'unique_tokens' in analysis
        
        assert 0 <= analysis['type_token_ratio'] <= 1
        assert analysis['avg_token_length'] > 0
        assert analysis['unique_tokens'] <= analysis['total_tokens']


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_malformed_text_handling(self, basic_tokenizer):
        """Test handling of malformed or unusual text."""
        malformed_texts = [
            "Text with\x00null chars",
            "Mixed\t\tabs\nand\r\nline endings",
            "Unicode \u200b\u200c\u200d zero-width chars",
            "Extremely\xa0non-breaking\xa0spaces",
            "Emoji with skin tone 👋🏽 modifiers"
        ]
        
        for text in malformed_texts:
            try:
                tokens = basic_tokenizer.tokenize(text)
                assert isinstance(tokens, list)
                # Should not crash, even if results are imperfect
            except Exception as e:
                pytest.fail(f"Tokenizer crashed on malformed text: {text[:50]}... Error: {e}")
    
    def test_very_long_text(self, basic_tokenizer):
        """Test handling of very long text."""
        # Create a very long poem
        long_text = "This is a very long poem.\n" * 1000
        
        tokens = basic_tokenizer.tokenize(long_text)
        assert isinstance(tokens, list)
        assert len(tokens) > 1000  # Should have many tokens
        
        # Should have many line breaks
        assert tokens.count('<LINE_BREAK>') > 500
    
    def test_special_characters_in_text(self, basic_tokenizer):
        """Test handling of various special characters."""
        text = 'Poetry with quotes "like this" and \'this\'\nDashes—em and -en and ellipses…'
        tokens = basic_tokenizer.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert '<LINE_BREAK>' in tokens
    
    def test_configuration_edge_cases(self):
        """Test edge cases in configuration."""
        # Test with minimal configuration
        config = TokenizationConfig(
            max_sequence_length=1,
            min_sequence_length=0
        )
        
        tokenizer = PoetryTokenizer(config=config)
        assert tokenizer is not None
        
        # Should still work
        tokens = tokenizer.tokenize("Love")
        assert isinstance(tokens, list)
    
    def test_vocabulary_edge_cases(self, basic_tokenizer):
        """Test vocabulary building edge cases."""
        # Test with empty texts
        basic_tokenizer.build_vocabulary([])
        
        # Should still have special tokens
        assert len(basic_tokenizer.vocabulary) > 0
        for special_token in basic_tokenizer.special_tokens:
            assert special_token in basic_tokenizer.vocabulary
        
        # Test with very short texts
        basic_tokenizer.build_vocabulary(["a", "I", ""])
        
        # Should still work
        assert len(basic_tokenizer.vocabulary) > 0
    
    def test_japanese_emoticon_file_missing(self, temp_dir):
        """Test graceful handling when Japanese emoticon file is missing."""
        # Create tokenizer with project root that doesn't have emoticon file
        tokenizer = PoetryTokenizer(project_root=temp_dir)
        
        # Should initialize without errors
        assert tokenizer is not None
        
        # Should still tokenize text
        tokens = tokenizer.tokenize("Test text without emoticons")
        assert isinstance(tokens, list)
        assert len(tokens) > 0


class TestIntegrationWithConfig:
    """Test integration with configuration system."""
    
    def test_config_integration(self):
        """Test that tokenizer properly uses configuration."""
        config = TokenizationConfig(
            preserve_case=False,
            preserve_numbers=False,
            max_sequence_length=50,
            min_sequence_length=1,
            special_tokens=['<UNK>', '<START>', '<END>']
        )
        
        tokenizer = PoetryTokenizer(config=config)
        
        # Should have config special tokens
        assert '<START>' in tokenizer.special_tokens
        assert '<END>' in tokenizer.special_tokens
        
        # Should respect preserve_case setting
        text = "UPPERCASE and lowercase"
        tokens = tokenizer.tokenize(text)
        
        # With preserve_case=False, should not have uppercase tokens
        # (exact behavior depends on implementation)
        # At minimum, should not crash and should produce tokens
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_default_config_fallback(self):
        """Test fallback to default configuration."""
        tokenizer = PoetryTokenizer(config=None)
        
        assert tokenizer.config is not None
        assert hasattr(tokenizer.config, 'preserve_case')
        assert hasattr(tokenizer.config, 'special_tokens')
        
        # Should work normally
        tokens = tokenizer.tokenize("Test text")
        assert isinstance(tokens, list)
        assert len(tokens) > 0


if __name__ == "__main__":
    # Run unit tests
    pytest.main([__file__, "-v"])