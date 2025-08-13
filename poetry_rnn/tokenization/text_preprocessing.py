"""
Text preprocessing utilities for poetry data

Provides specialized preprocessing functions for contemporary poetry
and alt-lit aesthetic text, focusing on preserving semantic elements
while cleaning noise from web scraping artifacts.
"""

import re
import logging
from typing import List, Optional


logger = logging.getLogger(__name__)


def clean_poetry_text(text: str, preserve_numbers: bool = True) -> str:
    """
    Clean poetry text while preserving semantic elements and aesthetic structure.
    
    This function removes web scraping artifacts and metadata while being careful
    to preserve numbers, decorative elements, and intentional formatting that
    are semantically meaningful in contemporary poetry.
    
    Args:
        text: Raw text to clean
        preserve_numbers: Whether to preserve numeric elements (default: True)
                         When False, removes user handles and metadata with numbers
    
    Returns:
        Cleaned text with preserved semantic elements
        
    Examples:
        >>> clean_poetry_text("user123 posted: beautiful poem ❤")
        'beautiful poem ❤'
        
        >>> clean_poetry_text("the year 2023\\nLINE_BREAK\\nwas different")
        'the year 2023 <LINE_BREAK> was different'
        
        >>> clean_poetry_text("she was 25 years old", preserve_numbers=True)
        'she was 25 years old'
    """
    
    if not text or not isinstance(text, str):
        return ""
    
    # Create a copy to work with
    cleaned = text
    
    # Remove usernames and handles (alphanumeric with numbers) only if not preserving numbers
    # or if they match specific metadata patterns
    if not preserve_numbers:
        # Remove usernames that are clearly metadata (alphanumeric with numbers)
        cleaned = re.sub(r'\b[a-zA-Z]*\d+[a-zA-Z]*\b', '', cleaned)
    else:
        # More selective removal - only remove obvious usernames/handles
        # Keep meaningful numbers like ages, years, quantities
        cleaned = re.sub(r'\b(?:user|account|post|id)\d+\b', '', cleaned, flags=re.IGNORECASE)
    
    # Remove @mentions and handles
    cleaned = re.sub(r'@\w+', '', cleaned)
    
    # Remove website artifacts and navigation elements
    website_artifacts = [
        r'\b(?:continue|reading|posted|tags?|categor(?:y|ies))\b',
        r'\b(?:www|\.com|http[s]?|onlylovepoetry)\b',
        r'\b(?:next|previous|home|about|contact)\b',
        r'\bpage\s*\d+\b',
        r'\b(?:share|like|comment|subscribe)\b'
    ]
    
    for pattern in website_artifacts:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Handle special tokens properly - normalize to standardized format
    cleaned = cleaned.replace('LINE_BREAK', '<LINE_BREAK>')
    cleaned = cleaned.replace('STANZA_BREAK', '<STANZA_BREAK>')
    cleaned = cleaned.replace('POEM_START', '<POEM_START>')
    cleaned = cleaned.replace('POEM_END', '<POEM_END>')
    
    # Remove very short "words" that are likely noise, but preserve punctuation
    words = cleaned.split()
    meaningful_words = []
    
    for word in words:
        # Always keep special tokens
        if word.startswith('<') and word.endswith('>'):
            meaningful_words.append(word)
        # Keep meaningful punctuation
        elif word in ['.', ',', '!', '?', ';', ':', '-', '"', "'", '(', ')']:
            meaningful_words.append(word)
        # Keep words with meaningful length or containing numbers/unicode
        elif len(word) > 1 or any(c.isdigit() or ord(c) > 127 for c in word):
            meaningful_words.append(word)
        # Keep single meaningful characters (not whitespace or random punctuation)
        elif len(word) == 1 and (word.isalnum() or word in "iayo"):  # Common single letters in poetry
            meaningful_words.append(word)
    
    cleaned = ' '.join(meaningful_words)
    
    # Final cleanup - normalize whitespace but don't over-normalize
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    logger.debug(f"Cleaned text: '{text[:50]}...' -> '{cleaned[:50]}...'")
    
    return cleaned


def normalize_unicode_for_poetry(text: str, 
                                  preserve_decorative: bool = True,
                                  convert_to_tokens: bool = False) -> str:
    """
    Normalize Unicode characters for poetry processing.
    
    Contemporary poetry often uses Unicode characters for aesthetic effect.
    This function can either preserve them or convert them to semantic tokens
    for downstream processing.
    
    Args:
        text: Input text with Unicode characters
        preserve_decorative: Whether to keep decorative Unicode (default: True)
        convert_to_tokens: Whether to convert to semantic tokens like <HEART> (default: False)
    
    Returns:
        Text with normalized Unicode handling
        
    Examples:
        >>> normalize_unicode_for_poetry("love ❤ poetry", convert_to_tokens=True)
        'love <HEART> poetry'
        
        >>> normalize_unicode_for_poetry("storm ⛈️ clouds", preserve_decorative=False)
        'storm clouds'
    """
    
    if not text or not isinstance(text, str):
        return ""
    
    if convert_to_tokens:
        # Define Unicode to token mappings
        unicode_mappings = {
            r'[❤♥💕💖💗💘💙💚💛💜🧡]': '<HEART>',
            r'[☁️⛅⛈️🌤️🌥️🌦️🌧️🌨️🌩️]': '<CLOUD>',
            r'[🌟⭐✨💫⚡]': '<STAR>',
            r'[🌸🌺🌻🌷🌹💐🌼]': '<FLOWER>',
            r'[😢😭💧]': '<TEAR>',
            r'[🔥💥]': '<FIRE>'
        }
        
        normalized = text
        for pattern, token in unicode_mappings.items():
            normalized = re.sub(pattern, f' {token} ', normalized)
            
        # Clean up extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    elif not preserve_decorative:
        # Remove decorative Unicode but keep essential characters
        # Remove emoji and symbols but keep letters and basic punctuation
        normalized = ''.join(char for char in text 
                           if ord(char) < 256 or char.isalpha())
        return normalized
    
    else:
        # Just normalize the text without removing Unicode
        return text.strip()


def extract_poetry_metadata(text: str) -> dict:
    """
    Extract metadata from poetry text while cleaning it.
    
    Identifies and extracts common metadata patterns found in scraped
    poetry data, such as author information, publication details, etc.
    
    Args:
        text: Raw poetry text with potential metadata
        
    Returns:
        Dictionary with extracted metadata and cleaned text
        
    Examples:
        >>> extract_poetry_metadata("By John Doe\\nThis is a poem\\nPublished 2023")
        {'author': 'John Doe', 'year': '2023', 'clean_text': 'This is a poem'}
    """
    
    metadata = {
        'author': None,
        'title': None,
        'year': None,
        'source': None,
        'clean_text': text
    }
    
    if not text or not isinstance(text, str):
        return metadata
    
    lines = text.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for author patterns
        author_patterns = [
            r'^(?:by|author):?\s*(.+)$',
            r'^(.+?)\s*[-–—]\s*poet$',
            r'^poem\s+by\s+(.+)$'
        ]
        
        for pattern in author_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match and not metadata['author']:
                metadata['author'] = match.group(1).strip()
                continue
        
        # Check for year patterns
        year_match = re.search(r'\b(19|20)\d{2}\b', line)
        if year_match and not metadata['year']:
            metadata['year'] = year_match.group()
            
        # Check for title patterns (often first line or in quotes)
        if not metadata['title'] and (line.startswith('"') or line.isupper()):
            metadata['title'] = line.strip('"').title()
            continue
            
        # If we didn't extract metadata from this line, keep it as content
        if not any([
            re.match(r'^(?:by|author)', line, re.IGNORECASE),
            re.search(r'\b(?:published|posted|source)', line, re.IGNORECASE),
            len(line) < 3  # Very short lines often metadata
        ]):
            clean_lines.append(line)
    
    metadata['clean_text'] = '\n'.join(clean_lines)
    
    return metadata


def split_into_lines_and_stanzas(text: str, 
                                  min_line_length: int = 1,
                                  preserve_empty_lines: bool = True) -> List[List[str]]:
    """
    Split poetry text into structured stanzas and lines.
    
    Handles the structural elements of poetry by identifying stanza breaks
    and line breaks, returning a nested structure that preserves the
    original formatting intent.
    
    Args:
        text: Poetry text to structure
        min_line_length: Minimum characters for a valid line (default: 1)
        preserve_empty_lines: Whether to keep empty lines as stanza breaks (default: True)
        
    Returns:
        List of stanzas, where each stanza is a list of lines
        
    Examples:
        >>> split_into_lines_and_stanzas("line 1\\nline 2\\n\\nstanza 2\\nline 2")
        [['line 1', 'line 2'], ['stanza 2', 'line 2']]
    """
    
    if not text or not isinstance(text, str):
        return [[]]
    
    # First, normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Split into potential stanzas (separated by double newlines)
    stanza_candidates = re.split(r'\n\s*\n', text)
    
    stanzas = []
    
    for stanza_text in stanza_candidates:
        if not stanza_text.strip():
            continue
            
        # Split stanza into lines
        lines = []
        for line in stanza_text.split('\n'):
            line = line.strip()
            
            # Filter by minimum length if specified
            if len(line) >= min_line_length:
                lines.append(line)
            elif preserve_empty_lines and len(line) == 0:
                lines.append("")  # Preserve intentional empty lines
        
        if lines:  # Only add non-empty stanzas
            stanzas.append(lines)
    
    # If no stanzas were found, treat entire text as one stanza
    if not stanzas and text.strip():
        lines = [line.strip() for line in text.split('\n') 
                if len(line.strip()) >= min_line_length]
        if lines:
            stanzas.append(lines)
    
    return stanzas if stanzas else [[]]