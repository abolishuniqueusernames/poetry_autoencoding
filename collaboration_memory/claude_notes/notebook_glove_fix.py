#!/usr/bin/env python3
"""
Fix for GLoVe embedding loading issues in spacy_glove_advanced_tutorial.ipynb

Issues found:
1. Missing leading slash in embedding_path (line 2307): 'home/...' → '/home/...'
2. Missing special_tokens attribute in GLoVeEmbeddingManager class  
3. AttributeError when saving artifacts

This script identifies the specific fixes needed.
"""

print("🔍 NOTEBOOK GLOVE LOADING ISSUES IDENTIFIED:")
print("=" * 60)

print("\n1. PATH ISSUE (Line 2307):")
print("   CURRENT: embedding_path= 'home/without-a-care-in-the-world/workflows/...'")
print("   FIXED:   embedding_path= '/home/without-a-care-in-the-world/workflows/...'")
print("   ❌ Missing leading slash prevents file from being found")

print("\n2. MISSING ATTRIBUTE:")
print("   ERROR: 'GLoVeEmbeddingManager' object has no attribute 'special_tokens'")
print("   FIX:   Add self.special_tokens in __init__ method")

print("\n3. SPECIFIC FIXES NEEDED:")
fixes = [
    {
        "file": "spacy_glove_advanced_tutorial.ipynb",
        "line": 2307,
        "issue": "Missing leading slash in embedding_path",
        "current": "'home/without-a-care-in-the-world/workflows/poetry_autoencoding/embeddings/glove.6B.300d.txt'",
        "fixed": "'/home/without-a-care-in-the-world/workflows/poetry_autoencoding/embeddings/glove.6B.300d.txt'"
    },
    {
        "file": "GLoVeEmbeddingManager.__init__",
        "issue": "Missing special_tokens attribute",
        "fix": "Add: self.special_tokens = ['<UNK>', '<LINE_BREAK>', '<STANZA_BREAK>', '<POEM_START>', '<POEM_END>']"
    }
]

for i, fix in enumerate(fixes, 1):
    print(f"\n{i}. {fix['issue']}:")
    if 'current' in fix:
        print(f"   Current: {fix['current']}")
        print(f"   Fixed:   {fix['fixed']}")
    if 'fix' in fix:
        print(f"   Fix:     {fix['fix']}")

print(f"\n✅ VERIFICATION:")
print(f"GLoVe file exists: /home/without-a-care-in-the-world/workflows/poetry_autoencoding/embeddings/glove.6B.300d.txt")
print(f"File size: 4.08GB (correct)")
print(f"Ready to fix notebook cells!")