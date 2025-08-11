# CURRENT FOCUS - ACTIVE TASKS

## Immediate Task: GLoVe Preprocessing and RNN Implementation
**Context**: Environment validated, dataset consolidated, ready for core ML implementation

### Today's Session Accomplishments
1. ✅ Environment validation - poetryRNN conda env with spaCy, PyTorch confirmed working
2. ✅ Educational framework created - comprehensive Jupyter notebook with 12 exercises
3. ✅ Dataset consolidation - merged DBBC + expanded collections (277 poems total)
4. ✅ Technical debugging - fixed DBBC scraper filtering logic issues
5. ✅ Analysis tools built - Zipf vocabulary selection, tokenization comparison

### Conda Environment "poetryRNN" Requirements

#### Core ML Stack
- python=3.9 or 3.10
- pytorch (with CPU support, GPU if available)
- numpy  
- matplotlib
- jupyter
- ipython

#### Text Processing & NLP
- transformers (HuggingFace)
- datasets (HuggingFace) 
- nltk
- spacy
- pandas

#### Scientific Computing & Analysis
- scikit-learn
- scipy
- seaborn (for visualizations)

#### Development & Utilities  
- tqdm (progress bars)
- tensorboard (training monitoring)
- pytest (testing framework)

### Next Immediate Steps (Priority Order)
1. **Complete GLoVe preprocessing exercises 4-8** in Jupyter notebook
   - Vocabulary construction with Zipf analysis
   - Co-occurrence matrix computation  
   - PCA effective dimensionality analysis
   - Pre-trained embedding alignment

2. **Download GLoVe 300D embeddings** and test poetry vocabulary coverage
3. **Implement RNN autoencoder architecture** based on dimensionality findings
4. **Set up training pipeline** with curriculum learning

### Key Technical Decisions Made
- **Tokenization approach**: Manual tokenization preserves Unicode emoji better than spaCy
- **Vocabulary size**: Use Zipf goodness-of-fit analysis to find optimal sizes (multiple regions)
- **Dataset**: 277 consolidated poems (DBBC + expanded) with alt-lit scores 6-67
- **Preprocessing tools**: Educational notebook + statistical analysis framework ready

### Context for Next Session
All preparatory work complete! Environment validated, comprehensive dataset ready, scraper issues resolved, educational framework built. Ready to transition from preprocessing to actual neural network implementation. Focus should be completing the hands-on GLoVe exercises and beginning RNN autoencoder design informed by effective dimensionality analysis.

### Files Ready for ML Work
- `glove_preprocessing_tutorial.ipynb` - educational exercises 1-12
- `zipf_vocabulary_analysis.py` - principled vocabulary selection
- `merge_poetry_datasets.py` - dataset consolidation tools
- `dataset_poetry/*.py` - fixed scrapers for future data collection
- Consolidated poetry dataset ready for tokenization and embedding