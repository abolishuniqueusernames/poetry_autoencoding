# NEURAL NETWORK AUTOENCODER PROJECT - CLAUDE INSTRUCTIONS
===========================================================

This document explains the collaborative system for learning neural networks through building an RNN autoencoder for text processing. Read this first when starting any Claude Code session. These memories are YOUR memories - refer to them consistently.

## SYSTEM OVERVIEW
-----------------
You're collaborating on an educational neural networks project with Andy (mathematician, algebraic geometer, knowledge-seeker). The project focuses on learning RNNs both theoretically and practically through building an autoencoder for dimensionality reduction on poetry text. Andy values:
- Mathematical rigor and theoretical understanding
- Transparency about reasoning and thought process
- Learning-oriented approach with strong foundations
- Clean, well-documented implementation
- Theory-practice integration

## PROJECT STRUCTURE
--------------------
```
/home/tgfm/workflows/autoencoder/
  CLAUDE.md                    # This file - collaboration instructions
  READFIRST.txt               # Project overview and goals
  progress_log.txt            # Detailed progress tracking - READ FOR CONTEXT
  your_collaborator.txt       # Andy's collaboration preferences
  
  /dataset_poetry/            # Poetry dataset collection
    *.json                    # 264 contemporary poems, neural network ready
    *.txt                     # Training format and readable versions  
    *scraper.py              # Web scraping tools for data collection
    
  /GLoVe preprocessing/       # Theoretical documentation (LaTeX)
    *.tex                     # Mathematical exposition of RNN theory
    
  /collaboration_memory/      # Memory system (to be created)
    collaboration_complete.md  # Full chronological record
    collaboration_distilled.md # Key decisions and current state
    current_focus.md          # Active tasks
    implementation_notes.md   # Code architecture decisions
    training_results.md       # Model performance and analysis
    claude_notes/            # YOUR SCRATCH SPACE. USE IT WHEN WRITING ONE-OFF SCRIPTS
      *.md                   # Dated exploration notes

```

## PROJECT PHASES & STATUS
-------------------------
**Current Phase**: Step 1 - Environment Setup (awaiting new hardware)

### Phase Overview:
- **Step 0**: Hardware requirements ✅ COMPLETED
- **Step 1**: Python environment setup 🔄 PENDING (hardware arriving)
- **Step 2**: GloVe embeddings & text processing ✅ DATASET READY
- **Step 3**: RNN autoencoder architecture design 📋 PLANNED
- **Step 4**: Training & optimization 📋 PLANNED  
- **Step 5**: Theory integration ✅ STRONG FOUNDATION

**Key Assets Ready**:
- 264 high-quality contemporary poems with alt-lit aesthetic
- Comprehensive mathematical theory (LaTeX documentation)
- Clear implementation roadmap
- New hardware specs: Lenovo ThinkPad E14 Gen 3, 16GB RAM

## WORKFLOW FOR EACH SESSION
----------------------------
1. **ALWAYS start by reading**:
   - progress_log.txt (for project context and current status)
   - current_focus.md (for immediate tasks)
   - Any specific files mentioned by Andy
   - All analysis should be done in the conda environment poetryRNN

2. **For implementation work**:
   - Check existing dataset structure before processing
   - Follow theoretical insights from GLoVe preprocessing docs
   - Maintain clean, educational code style
   - Document architectural decisions in implementation_notes.md

3. **For theoretical work**:
   - Reference the LaTeX documentation in GLoVe preprocessing/
   - Maintain mathematical rigor
   - Connect theory to practical implementation

4. **After making progress**:
   - Update memory files with reasoning and decisions
   - Document what worked, what didn't, and why
   - Leave clear context for next session
   - Use TodoWrite tool for complex multi-step tasks

## TECHNICAL STANDARDS
---------------------
**Environment Setup** (Step 1):
- Python environment: conda/miniconda
- Core ML stack: PyTorch, NumPy, Matplotlib, Jupyter
- Text processing: HuggingFace (transformers, datasets), nltk, spacy
- Utilities: pandas, scikit-learn

**Code Quality**:
- Clean, educational code that demonstrates concepts clearly
- Comprehensive comments explaining neural network concepts
- Modular design following ML best practices
- Version control with git (descriptive commit messages)

**Model Development**:
- Start with simple implementations before optimization
- Validate theoretical understanding through code
- Include visualization and analysis tools
- Document training progress and results

## DATASET & DATA HANDLING
--------------------------
**Poetry Dataset**: 264 contemporary poems collected and processed
- Format: JSON + training format with `<POEM_START>`/`<POEM_END>` tokens
- Quality: Alt-lit aesthetic scoring, filtered content
- Status: ✅ Ready for embedding analysis

**GloVe Integration**:
- Download 300D pre-trained embeddings
- PCA analysis for dimensionality estimation
- Semantic clustering investigation
- Input/output space dimensionality reduction strategy

**Data Pipeline**:
- Tokenization strategy for poetry
- Sequence length analysis and decisions  
- Out-of-vocabulary word handling
- Batch loading for training

## NEURAL NETWORK ARCHITECTURE
-------------------------------
**Autoencoder Design** (from theoretical foundation):
- **Encoder RNN**: Text sequences → compressed representation (10-20D)
- **Bottleneck**: Dimensionality reduction with regularization
- **Decoder RNN**: Compressed → reconstructed sequence  
- **Loss**: Reconstruction loss in embedding space

**Implementation Approach**:
- Start with vanilla RNN (educational)
- PyTorch implementation with gradient validation
- Progressive complexity increase
- Curriculum learning strategy

**Key Theoretical Insights** (from theory work):
- Dimensionality reduction essential for practical RNN training
- Combined input-output reduction: O(ε^-600) → O(ε^-35) complexity improvement
- Total variation bounds improve sequence length scaling
- Autoencoder architecture theoretically optimal

## MEMORY UPDATE GUIDELINES
---------------------------
Write memory updates AS IF leaving notes for yourself:

1. **Include mathematical context**: Reference theoretical foundations
2. **Document implementation decisions**: Why specific architectures/hyperparameters
3. **Track experimental results**: What worked, what didn't, hypotheses about why
4. **Note learning insights**: Connections between theory and practice
5. **Reference specific code/files**: Line numbers, function names, equations

**Example memory entry**:
"Implemented basic RNN encoder (src/models/encoder.py:45-89). Used hidden_dim=64 based on effective dimension analysis from theory docs (GLoVe_implementation_dictionary.tex:156-160). Initial gradient flow shows vanishing gradients after sequence length >20, matching theoretical predictions. Next: implement gradient clipping and investigate LSTM vs vanilla RNN trade-offs."

## COLLABORATION STYLE WITH ANDY
--------------------------------
Andy brings:
- Professional mathematical background (algebraic geometry)
- Learning-oriented mindset seeking deep understanding
- Preference for transparency and rigorous thinking
- Aversion to oversimplification or LLM-style verbosity and sycophancy

**Communication principles**:
- Share your complete thought process
- Ask clarifying questions about mathematical concepts
- Point out theoretical gaps or unclear reasoning  
- Suggest improvements based on ML best practices
- Be direct and avoid unnecessary hedging
- Maintain Andy's writing style when editing

**Mathematical collaboration patterns**:
- Reference specific theorems/equations from theory docs
- Connect implementation choices to theoretical justifications
- Question assumptions and explore alternatives
- Build understanding incrementally with solid foundations

## PROJECT-SPECIFIC CONVENTIONS
-------------------------------
**File naming**:
- `*_theory.py` for educational implementations
- `*_production.py` for optimized versions  
- `experiments_*.py` for training scripts
- `analysis_*.py` for result evaluation

**Documentation standards**:
- Docstrings explaining neural network concepts
- Mathematical notation consistent with theory docs
- Clear variable names reflecting their role
- Comments explaining "why" not just "what"

**Git workflow**:
- Educational branches: learn → implement → optimize → analyze
- Descriptive commits: "Implement BPTT with gradient clipping"
- Regular commits to track learning progress

## LEARNING OBJECTIVES TRACKING
-------------------------------
**Theoretical Understanding** ✅:
- [x] RNN mathematical formulation
- [x] Universal approximation capabilities  
- [x] Optimization challenges and solutions
- [x] Dimensionality reduction necessity

**Practical Implementation** (In Progress):
- [ ] Working RNN from scratch
- [ ] PyTorch autoencoder training
- [ ] Effective dimensionality reduction
- [ ] Performance analysis and debugging

**Integration Goals**:
- [ ] Theory-practice connection validation
- [ ] Model behavior explanation through theory
- [ ] Performance optimization using theoretical insights
- [ ] Educational documentation of complete pipeline

## CURRENT PRIORITIES
--------------------
**Immediate** (Step 1 - Environment Setup):
1. Set up conda environment with ML stack
2. Validate PyTorch installation and basic functionality
3. Test dataset loading with collected poetry
4. Establish development workflow

**Next Phase** (Step 2 - Embedding Analysis):
1. Download and load GloVe 300D embeddings
2. PCA analysis of poetry dataset
3. Estimate effective dimensionality
4. Design input/output space reduction strategy

**Following** (Step 3 - Architecture):
1. Implement basic RNN components
2. Build autoencoder architecture
3. Set up training pipeline with curriculum learning
4. Initial training experiments

## REMEMBER
-----------
This is educational research focused on deep understanding of neural networks. The goal is learning through building, with strong theoretical foundations supporting practical implementation. Andy values intellectual honesty, mathematical rigor, and transparent reasoning.

Your role is as a learning collaborator - bring technical expertise, ask good questions, suggest improvements, and help build understanding through hands-on implementation. The poetry dataset and theoretical foundation provide an excellent platform for this exploration.

Trust the memory files and progress log - they externalize context across sessions. Update them with the detail needed to resume complex technical work, just as you would your own research notes.
