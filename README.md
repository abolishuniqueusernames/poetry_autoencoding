# Neural Network Autoencoder Project

Educational RNN autoencoder implementation for text dimensionality reduction using contemporary poetry.

## Project Structure

```
/
├── CLAUDE.md                     # Main collaboration instructions & memories
├── READFIRST.txt                 # Project overview and goals  
├── progress_log.txt              # Detailed progress tracking
├── your_collaborator.txt         # Andy's collaboration preferences
├── environment.yml               # Conda environment specification
│
├── 📁 analysis/                  # Data analysis & preprocessing
│   ├── glove_preprocessing_tutorial.ipynb  # GLoVe preprocessing exercises
│   └── zipf_analysis.png        # Vocabulary analysis visualization
│
├── 📁 scripts/                   # Utility scripts
│   ├── merge_poetry_datasets.py  # Dataset consolidation tools
│   ├── tabula_rasa.py           # Clean slate testing script  
│   ├── validate_environment.py  # Environment validation
│   └── zipf_vocabulary_analysis.py  # Vocabulary selection tools
│
├── 📁 dataset_poetry/           # Data collection & web scraping
│   ├── improved_dbbc_scraper.py # Enhanced DBBC scraper (67% success rate)
│   ├── scraper.py              # General poetry scraper
│   ├── dreamboybookclubscraper.py  # Original DBBC scraper
│   ├── manualscraper.py        # Manual collection tools
│   └── urlscraper.py           # URL-based scraping
│
├── 📁 docs/                     # Documentation & examples
│   ├── SETUP_CHECKLIST.md      # Environment setup checklist
│   ├── example_collaboration_instructions.txt
│   └── example_collaboration_instructions2.md
│
├── 📁 collaboration_memory/     # Project memory & decision tracking  
│   ├── collaboration_distilled.md  # Key decisions & current state
│   ├── current_focus.md         # Active tasks & priorities
│   ├── implementation_notes.md  # Technical architecture decisions
│   ├── training_results.md      # Model performance tracking
│   └── claude_notes/           # Session-specific notes
│       ├── 2025-08-11-glove-preprocessing-session.md
│       └── 2025-08-11-enhanced-dataset-session.md
│
├── 📁 GLoVe preprocessing/      # Theoretical documentation
│   ├── GloVe_documentation.tex  # Mathematical exposition
│   ├── GloVe_implementation_dictionary.tex
│   ├── GloVe_overview.tex
│   └── Style.tex
│
└── 📁 assets/                   # Images, plots, other assets
```

## Current Status

**Phase**: Step 2 - GLoVe Embeddings & Text Processing  
**Environment**: `poetryRNN` conda environment operational ✅  
**Dataset**: 20 premium alt-lit poems (avg score 23.6) ✅  
**Next**: Complete GLoVe preprocessing exercises 4-8

## Key Achievements

### Recent Breakthrough: Enhanced Dataset Quality
- **Web-scraper-debugger agent** improved DBBC scraper success rate **6.7×** (10% → 67%)
- **Premium collection**: 20 high-quality alt-lit poems with authentic aesthetic characteristics
- **Technical improvements**: Selenium → Requests+BeautifulSoup, Unicode preservation, content detection

### Theoretical Foundation Complete
- **RNN mathematical framework**: Rigorous formulation with universal approximation proofs
- **Dimensionality reduction**: Theoretical necessity proven for practical RNN training  
- **Sample complexity**: Joint input-output reduction O(ε^-600) → O(ε^-35)

### Environment Ready
- **poetryRNN conda environment**: PyTorch, spaCy, HuggingFace tools validated
- **Hardware**: Lenovo ThinkPad E14 Gen 3 (16GB RAM) operational
- **Educational framework**: Jupyter notebook with 12 GLoVe preprocessing exercises

## Quick Start

### Run Enhanced Dataset Collection
```bash
cd dataset_poetry/
python3 improved_dbbc_scraper.py --limit=30
```

### Validate Environment  
```bash
python3 scripts/validate_environment.py
```

### GLoVe Preprocessing Tutorial
```bash
jupyter notebook analysis/glove_preprocessing_tutorial.ipynb
```

### Clean Testing Environment
```bash
python3 scripts/tabula_rasa.py --dry-run  # Preview
python3 scripts/tabula_rasa.py            # Execute with confirmations
```

## Memory System

This project uses a comprehensive memory system to maintain context across sessions:

- **`progress_log.txt`**: Chronological record of all major progress
- **`collaboration_memory/`**: Structured memory files tracking decisions and current state
- **`claude_notes/`**: Detailed session-specific technical notes
- **`CLAUDE.md`**: Main collaboration instructions and project context

## Architecture Overview

**Goal**: Learn neural networks through building RNN autoencoder for text dimensionality reduction

**Approach**: Theory-driven implementation with strong mathematical foundation
- **Encoder RNN**: Poetry sequences → compressed representation (10-20D)  
- **Decoder RNN**: Compressed → reconstructed sequence
- **Training**: Curriculum learning with high-quality alt-lit examples

**Dataset**: Contemporary alt-lit poetry from Dream Boy Book Club and related sources, optimized for neural network training with preserved Unicode aesthetic characteristics.

---

*This is an educational neural networks project focused on learning through hands-on implementation with strong theoretical foundations.*