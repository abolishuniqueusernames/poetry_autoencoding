#!/usr/bin/env python3
"""
Merge DBBC poetry collection with expanded contemporary poetry collection.
Harmonizes field names and structures for unified dataset.
"""

import json
from typing import List, Dict, Any

def harmonize_poem_fields(poem: Dict[str, Any], source_dataset: str) -> Dict[str, Any]:
    """
    Harmonize field names and structures between datasets.
    
    Args:
        poem: Original poem dictionary
        source_dataset: 'dbbc' or 'expanded' to indicate source
        
    Returns:
        Harmonized poem dictionary
    """
    harmonized = {
        'text': poem.get('text', ''),
        'title': poem.get('title', ''),
        'source': poem.get('source', ''),
        'url': poem.get('url', ''),
        'length': poem.get('length', 0),
        'line_count': poem.get('line_count', 0),
        'dataset_source': source_dataset  # Track which dataset this came from
    }
    
    # Handle author field (only in DBBC)
    if 'author' in poem:
        harmonized['author'] = poem['author']
    
    # Handle scoring fields - normalize to common structure
    if source_dataset == 'dbbc':
        harmonized['alt_lit_score'] = poem.get('dbbc_score', 0)
        harmonized['extraction_method'] = poem.get('extraction_method', '')
        harmonized['score_factors'] = poem.get('score_factors', {})
        # Convert dbbc score_factors to style_indicators format
        if 'score_factors' in poem:
            harmonized['style_indicators'] = {
                'dbbc_aesthetic': True,  # Flag indicating DBBC aesthetic
                **poem['score_factors']  # Include original factors
            }
    else:  # expanded dataset
        harmonized['alt_lit_score'] = poem.get('alt_lit_score', 0)
        harmonized['extraction_method'] = poem.get('method', '')
        harmonized['style_indicators'] = poem.get('style_indicators', {})
        harmonized['style_indicators']['dbbc_aesthetic'] = False
    
    return harmonized

def merge_poetry_datasets(dbbc_path: str, expanded_path: str, output_path: str) -> Dict[str, Any]:
    """
    Merge DBBC and expanded poetry datasets into unified collection.
    
    Args:
        dbbc_path: Path to DBBC poetry JSON file
        expanded_path: Path to expanded poetry JSON file  
        output_path: Path for merged output JSON file
        
    Returns:
        Dictionary with merge statistics
    """
    # Load datasets
    with open(dbbc_path, 'r', encoding='utf-8') as f:
        dbbc_poems = json.load(f)
    
    with open(expanded_path, 'r', encoding='utf-8') as f:
        expanded_poems = json.load(f)
    
    print(f"Loading {len(dbbc_poems)} poems from DBBC dataset")
    print(f"Loading {len(expanded_poems)} poems from expanded dataset")
    
    # Harmonize and merge
    merged_poems = []
    
    # Process DBBC poems
    for poem in dbbc_poems:
        harmonized = harmonize_poem_fields(poem, 'dbbc')
        merged_poems.append(harmonized)
    
    # Process expanded poems  
    for poem in expanded_poems:
        harmonized = harmonize_poem_fields(poem, 'expanded')
        merged_poems.append(harmonized)
    
    # Sort by alt_lit_score (highest first) for easy analysis
    merged_poems.sort(key=lambda p: p['alt_lit_score'], reverse=True)
    
    # Add metadata
    merged_dataset = {
        'metadata': {
            'total_poems': len(merged_poems),
            'dbbc_poems': len(dbbc_poems),
            'expanded_poems': len(expanded_poems),
            'merge_date': '2025-08-11',  # Today's date from env
            'description': 'Merged DBBC and expanded contemporary poetry collections',
            'fields_harmonized': True,
            'alt_lit_score_range': [
                min(p['alt_lit_score'] for p in merged_poems),
                max(p['alt_lit_score'] for p in merged_poems)
            ],
            'total_tokens_approx': sum(len(p['text'].split()) for p in merged_poems)
        },
        'poems': merged_poems
    }
    
    # Save merged dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_dataset, f, indent=2, ensure_ascii=False)
    
    # Generate summary statistics
    stats = {
        'total_poems': len(merged_poems),
        'dbbc_count': len(dbbc_poems),
        'expanded_count': len(expanded_poems),
        'avg_alt_lit_score': sum(p['alt_lit_score'] for p in merged_poems) / len(merged_poems),
        'high_score_poems': len([p for p in merged_poems if p['alt_lit_score'] > 50]),
        'total_length': sum(p['length'] for p in merged_poems),
        'avg_length': sum(p['length'] for p in merged_poems) / len(merged_poems),
        'dbbc_poems_titles': [p['title'] for p in merged_poems if p['dataset_source'] == 'dbbc']
    }
    
    return stats

def create_training_format(merged_path: str, training_output_path: str):
    """
    Create training format version of merged dataset.
    Adds <POEM_START> and <POEM_END> tokens for RNN training.
    """
    with open(merged_path, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    
    training_texts = []
    
    for poem in merged_data['poems']:
        # Format for RNN autoencoder training
        formatted_text = f"<POEM_START>\n{poem['text']}\n<POEM_END>"
        training_texts.append(formatted_text)
    
    # Save as simple text file for easy loading
    with open(training_output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(training_texts))
    
    print(f"Created training format with {len(training_texts)} poems")

if __name__ == "__main__":
    # Merge datasets
    print("Merging poetry datasets...")
    
    stats = merge_poetry_datasets(
        '../dataset_poetry/improved_dbbc_collection.json',
        '../dataset_poetry/expanded_contemporary_poetry.json', 
        '../dataset_poetry/merged_poetry_collection.json'
    )
    
    # Print statistics
    print(f"\n=== MERGE COMPLETE ===")
    print(f"Total poems: {stats['total_poems']}")
    print(f"  - DBBC poems: {stats['dbbc_count']}")
    print(f"  - Expanded poems: {stats['expanded_count']}")
    print(f"Average alt-lit score: {stats['avg_alt_lit_score']:.1f}")
    print(f"High-scoring poems (>50): {stats['high_score_poems']}")
    print(f"Average poem length: {stats['avg_length']:.0f} characters")
    print(f"Total corpus length: {stats['total_length']:,} characters")
    
    print(f"\nDBBC poems included:")
    for title in stats['dbbc_poems_titles']:
        print(f"  - {title}")
    
    # Create training format
    create_training_format(
        'dataset_poetry/merged_poetry_collection.json',
        'dataset_poetry/merged_poetry_training.txt'
    )
    
    print(f"\nFiles created:")
    print(f"  - dataset_poetry/merged_poetry_collection.json (full dataset)")
    print(f"  - dataset_poetry/merged_poetry_training.txt (RNN training format)")