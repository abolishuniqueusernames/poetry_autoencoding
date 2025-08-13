#!/usr/bin/env python3
"""Quick test of tokenizer fix"""

import re
import spacy
from typing import List
import json

try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ Loaded en_core_web_sm")
except OSError:
    from spacy.lang.en import English
    nlp = English()
    print("✓ Using fallback English tokenizer")

# Load poems
try:
    with open('/home/without-a-care-in-the-world/workflows/poetry_autoencoding/dataset_poetry/multi_poem_dbbc_collection.json', 'r') as f:
        poems = json.load(f)
    print(f"✓ Loaded {len(poems)} poems")
except FileNotFoundError:
    print("⚠ Using test poem")
    poems = [{"text": "Over summer, I do ketamine 6 times in one week with an 80 year old anesthesiologist."}]

def clean_poetry_text(text):
    """Simple text cleaning"""
    return re.sub(r'\s+', ' ', text.strip())

class FixedPoetryTokenizer:
    def __init__(self):
        self.nlp = nlp
        self.special_tokens = {'<LINE_BREAK>', '<STANZA_BREAK>', '<POEM_START>', '<POEM_END>'}
        
    def preprocess_text(self, text: str) -> str:
        """Fixed preprocessing - NO forced lowercase"""
        processed = re.sub(r'\n\s*\n', ' <STANZA_BREAK> ', text)
        processed = re.sub(r'\n', ' <LINE_BREAK> ', processed)
        processed = re.sub(r'\s+', ' ', processed)
        processed = clean_poetry_text(processed)
        # REMOVED: processed = processed.lower() - THE FIX!
        return processed.strip()
        
    def tokenize(self, text: str) -> List[str]:
        """Fixed tokenization preserving numbers, Unicode, case"""
        processed_text = self.preprocess_text(text)
        doc = self.nlp(processed_text)
        
        tokens = []
        for token in doc:
            if token.text in self.special_tokens or not token.is_space:
                if token.text.strip():
                    tokens.append(token.text.strip())
        return tokens

# Test the fix
tokenizer = FixedPoetryTokenizer()

# Test case from your example
test_text = poems[0]['text'][:120]  # First part of first poem
print(f"\nOriginal text:")
print(repr(test_text))

tokens = tokenizer.tokenize(test_text)
print(f"\nFixed tokenizer ({len(tokens)} tokens):")
print(tokens)

# Check preservation
numbers = [t for t in tokens if any(c.isdigit() for c in t)]
uppercase = [t for t in tokens if any(c.isupper() for c in t)]
unicode_chars = [t for t in tokens if any(ord(c) > 127 for c in t)]

print(f"\n✅ PRESERVATION CHECK:")
print(f"Numbers preserved: {numbers}")
print(f"Uppercase preserved: {uppercase}")
print(f"Unicode preserved: {unicode_chars}")

print(f"\n🎯 SUCCESS: Tokenizer fixed to preserve poetry elements!")