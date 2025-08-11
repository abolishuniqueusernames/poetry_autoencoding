# GLoVe Preprocessing & Environment Session - August 11, 2025

## Session Summary
Major progress transitioning from theoretical preparation to hands-on implementation tools. All foundational work now complete.

## Key Technical Insights Discovered

### spaCy vs Manual Tokenization
- **Problem**: spaCy aggressively splits Unicode emoji like `(っ◔◡◔)っ` into separate tokens
- **Impact**: Loses semantic meaning of emoticons important in alt-lit poetry  
- **Decision**: Manual tokenization preserves Unicode better for poetry domain

### DBBC Scraper Filtering Issues
- **Root cause**: Overly restrictive filters rejected valid poems
  - Length limit: 3000 chars too low (Greta Schledorn poem was 5846 chars)
  - Line count logic: Required 2+ lines, rejected long single paragraphs
  - HTML parsing: Lost line breaks from `<br>` tags
- **Fix implemented**: Length → 8000 chars, allow long paragraphs, preserve HTML breaks
- **Validation**: Test poem now passes filtering logic

### Zipf's Law Vocabulary Selection
- **Discovery**: Goodness-of-fit is non-monotonic across vocabulary sizes
- **Implication**: Multiple optimal regions exist (e.g., ~34 and ~570 both good)
- **Solution**: Built multi-region analysis to find all good vocabulary sizes
- **Trade-off**: Smaller vocab = faster training, larger vocab = better coverage

## Educational Framework Built
1. **`glove_preprocessing_tutorial.ipynb`** - 12 hands-on exercises linking theory to practice
2. **`zipf_vocabulary_analysis.py`** - Statistical toolkit for principled vocab selection  
3. **Dataset consolidation** - 277 poems merged with harmonized metadata

## Architecture Implications for RNN Autoencoder
- **Effective dimensionality analysis** (exercises 7-8) will guide bottleneck size
- **Poetry-specific tokenization** needed to preserve semantic Unicode elements
- **Vocabulary size selection** should balance complexity vs coverage using Zipf analysis
- **Sequence length analysis** needed for curriculum learning strategy

## Next Session Priority
Complete exercises 4-8 in Jupyter notebook:
- Vocabulary construction (exercise 4)
- Co-occurrence matrix computation (exercise 5-6) 
- PCA effective dimensionality analysis (exercise 7-8)
- Then begin RNN autoencoder architecture design

## Technical Debt/Issues
- Geckodriver compatibility warning (doesn't block functionality)
- Haven't downloaded pre-trained GLoVe embeddings yet
- Need to test vocabulary alignment with GLoVe coverage

## Files Created This Session
- `glove_preprocessing_tutorial.ipynb` - educational framework
- `zipf_vocabulary_analysis.py` - vocabulary selection tools
- `merge_poetry_datasets.py` - dataset consolidation
- Updated `dreamboybookclubscraper.py` - fixed filtering logic

Ready for core ML implementation!