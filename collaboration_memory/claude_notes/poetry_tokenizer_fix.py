#!/usr/bin/env python3
"""
Poetry-Specific Tokenization Fix

Addresses issues with current tokenizer that loses critical information:
- Numbers (6, 80, etc.) being filtered out
- Forced lowercase destroying aesthetic casing  
- Unicode/emoji being lost
- Over-aggressive metadata filtering
"""

import re
import spacy
from typing import List, Dict, Optional
from spacy.lang.en import English

class ImprovedPoetryTokenizer:
    """Improved tokenizer that preserves poetry-specific elements."""
    
    def __init__(self, nlp_model=None, preserve_case=True, preserve_numbers=True):
        """
        Args:
            nlp_model: spaCy model (defaults to en_core_web_sm)
            preserve_case: Keep original casing for aesthetic elements
            preserve_numbers: Keep numbers as semantic units
        """
        try:
            self.nlp = nlp_model or spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: en_core_web_sm not found, using basic English")
            self.nlp = English()
            
        self.preserve_case = preserve_case
        self.preserve_numbers = preserve_numbers
        
        # Special tokens for poetry structure
        self.special_tokens = {
            '<LINE_BREAK>',  # Single line breaks
            '<STANZA_BREAK>',  # Double line breaks
            '<POEM_START>',
            '<POEM_END>',
        }
        
        # Alt-lit aesthetic patterns to preserve
        self.aesthetic_patterns = [
            r'[❤️💕💖🖤💜💙💚💛🧡]',  # Hearts
            r'[☁️🌙⭐️✨💫⚡️]',  # Celestial
            r'[🍒🍓🥝🍑🍊🍋]',  # Fruit emojis
            r'[＊✿❀☆★♡♥]',  # Decorative symbols  
            r'[𝓜𝓮𝓵𝓲𝓷𝓭𝓪|𝔰𝔱𝔯𝔞𝔴𝔟𝔢𝔯𝔯𝔶|𝖚𝖓𝖌𝖔𝖉𝖑𝖞]',  # Unicode fonts
        ]
        
    def preprocess_text(self, text: str) -> str:
        """Minimal preprocessing that preserves poetry elements."""
        
        # Preserve line structure
        processed = re.sub(r'\n\s*\n', ' <STANZA_BREAK> ', text)  # Double breaks = stanza
        processed = re.sub(r'\n', ' <LINE_BREAK> ', processed)    # Single breaks = line
        
        # Clean excessive whitespace but preserve intentional spacing
        processed = re.sub(r'[ \t]+', ' ', processed)  # Multiple spaces/tabs -> single space
        
        # DON'T force lowercase - this was the key problem!
        # DON'T remove numbers - preserve them as semantic units
        # DON'T remove Unicode - critical for alt-lit aesthetic
        
        return processed.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize poetry text preserving aesthetic and semantic elements."""
        
        # Minimal preprocessing
        processed_text = self.preprocess_text(text)
        
        # Use spaCy for intelligent tokenization
        doc = self.nlp(processed_text)
        
        tokens = []
        for token in doc:
            # Handle special structure tokens
            if token.text in self.special_tokens:
                tokens.append(token.text)
                continue
                
            # Skip pure whitespace
            if token.is_space or len(token.text.strip()) == 0:
                continue
                
            # Skip obvious website metadata (but be less aggressive)
            if self._is_metadata_token(token):
                continue
                
            # Process and keep the token
            processed_token = self._process_token(token)
            if processed_token:
                tokens.append(processed_token)
                
        return tokens
    
    def _is_metadata_token(self, token) -> bool:
        """Conservative metadata filtering - only obvious website elements."""
        text_lower = token.text.lower()
        
        # Only filter clear website metadata, not legitimate words
        strict_metadata = [
            'www.', '.com', '.org', 'http', 'https',
            'posted on', 'continue reading', 'tags:', 'category:'
        ]
        
        return any(pattern in text_lower for pattern in strict_metadata)
    
    def _process_token(self, token) -> Optional[str]:
        """Process individual token with minimal modification."""
        text = token.text.strip()
        
        if not text:
            return None
            
        # Preserve case for aesthetic elements by default
        if self.preserve_case:
            return text
        else:
            # Only lowercase if explicitly requested
            return text.lower()
    
    def analyze_tokenization_quality(self, original_text: str, tokens: List[str]) -> Dict:
        """Analyze how well tokenization preserved important elements."""
        
        # Count preserved elements
        numbers_in_original = len(re.findall(r'\d+', original_text))
        numbers_in_tokens = sum(1 for token in tokens if re.match(r'\d+', token))
        
        unicode_in_original = len(re.findall(r'[^\x00-\x7F]', original_text))
        unicode_in_tokens = sum(len(re.findall(r'[^\x00-\x7F]', token)) for token in tokens)
        
        uppercase_in_original = len(re.findall(r'[A-Z]', original_text))
        uppercase_in_tokens = sum(len(re.findall(r'[A-Z]', token)) for token in tokens)
        
        return {
            'total_tokens': len(tokens),
            'numbers_preserved': numbers_in_tokens / max(numbers_in_original, 1),
            'unicode_preserved': unicode_in_tokens / max(unicode_in_original, 1),
            'case_preserved': uppercase_in_tokens / max(uppercase_in_original, 1),
            'original_length': len(original_text),
            'token_coverage': sum(len(token) for token in tokens) / len(original_text.replace(' ', ''))
        }


def test_tokenizer_improvements():
    """Test the improved tokenizer against problematic examples."""
    
    # Test cases from the actual dataset
    test_cases = [
        # Numbers and contractions
        "Over summer, I do ketamine 6 times in one week with an 80 year old anesthesiologist. He says we're fixing my hard drive.",
        
        # Alt-lit aesthetic with Unicode
        "＊✿❀ ☁️ SUMMER KETAMINE RITUAL 🍒 ❀✿＊",
        
        # Mixed case with decorative elements
        "❤•.¸♥ 𝓜𝓮𝓵𝓲𝓷𝓭𝓪 ♥¸.•❤",
        
        # Poetry with line breaks
        "i love you\nlike stars love darkness\n\n❤️ forever ❤️"
    ]
    
    print("=== TOKENIZER COMPARISON ===\n")
    
    # Original tokenizer (simulated problems)
    class ProblematicTokenizer:
        def __init__(self):
            self.nlp = spacy.load("en_core_web_sm")
            
        def tokenize(self, text):
            # Simulate current problems
            text = text.lower()  # Force lowercase - THE PROBLEM
            text = re.sub(r'[^\x00-\x7F]', '', text)  # Remove Unicode - PROBLEM
            doc = self.nlp(text)
            return [token.text for token in doc if not token.is_space and len(token.text.strip()) > 0]
    
    old_tokenizer = ProblematicTokenizer()
    new_tokenizer = ImprovedPoetryTokenizer()
    
    for i, test_text in enumerate(test_cases):
        print(f"TEST CASE {i+1}:")
        print(f"Original: {repr(test_text)}")
        print()
        
        old_tokens = old_tokenizer.tokenize(test_text)
        new_tokens = new_tokenizer.tokenize(test_text)
        
        print(f"OLD tokenizer ({len(old_tokens)} tokens):")
        print(old_tokens[:15])  # Show first 15
        print()
        
        print(f"NEW tokenizer ({len(new_tokens)} tokens):")
        print(new_tokens[:15])
        print()
        
        # Quality analysis
        quality = new_tokenizer.analyze_tokenization_quality(test_text, new_tokens)
        print(f"QUALITY METRICS:")
        for key, value in quality.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
        
        print("="*60)
        print()


if __name__ == "__main__":
    test_tokenizer_improvements()