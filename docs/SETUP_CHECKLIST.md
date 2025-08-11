# Hardware Setup Checklist - poetryRNN Project

## Pre-Setup (Ready ✅)
- [x] Hardware ordered: Lenovo ThinkPad E14 Gen 3 (16GB RAM, AMD Ryzen 5 5500U)
- [x] Dataset collected: 264 contemporary poems in neural network format
- [x] Theory foundation: Comprehensive mathematical documentation 
- [x] Environment specification: `environment.yml` created
- [x] Validation tools: `validate_environment.py` ready
- [x] Collaboration system: Memory files and CLAUDE.md configured

## Day 1: Initial Setup (Hardware Arrival)

### System Setup
- [ ] **Unbox and power on new hardware**
- [ ] **Complete OS setup/configuration** (if needed)
- [ ] **Install essential development tools**:
  - [ ] Git (`sudo apt install git` or equivalent)
  - [ ] Curl/wget for downloads
  - [ ] Text editor (VS Code, vim, etc.)

### Conda Installation  
- [ ] **Download Miniconda/Anaconda**:
  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  ```
- [ ] **Initialize conda** and restart terminal
- [ ] **Verify conda installation**: `conda --version`

### Environment Setup
- [ ] **Clone/transfer project files** to new machine
- [ ] **Navigate to project directory**: `cd /path/to/autoencoder`
- [ ] **Create conda environment**:
  ```bash
  conda env create -f environment.yml
  ```
- [ ] **Activate environment**: `conda activate poetryRNN`
- [ ] **Run validation script**: `python validate_environment.py`

### Dataset Verification
- [ ] **Verify dataset transfer**: Check `dataset_poetry/` directory
- [ ] **Test dataset loading**: Quick Python script to load JSON files
- [ ] **Check file permissions** and accessibility

## Day 1: Validation & Testing

### PyTorch Verification
- [ ] **Test basic PyTorch functionality**:
  ```python
  import torch
  x = torch.randn(5, 3)
  print(f"PyTorch working: {x.shape}")
  ```
- [ ] **Check GPU availability** (if applicable):
  ```python
  print(f"CUDA available: {torch.cuda.is_available()}")
  ```
- [ ] **Test RNN module**:
  ```python
  rnn = torch.nn.RNN(10, 20)
  print("RNN module loaded successfully")
  ```

### NLP Stack Verification  
- [ ] **Test HuggingFace imports**:
  ```python
  from transformers import AutoTokenizer
  print("HuggingFace working")
  ```
- [ ] **Test dataset loading**:
  ```python
  import json
  with open('dataset_poetry/expanded_contemporary_poetry.json') as f:
      data = json.load(f)
  print(f"Loaded {len(data)} poems")
  ```

### Jupyter Setup
- [ ] **Launch Jupyter Lab**: `jupyter lab`
- [ ] **Test notebook functionality** with basic PyTorch operations
- [ ] **Verify kernel has poetryRNN environment** 

## Day 2: Project Initialization

### Directory Structure
- [ ] **Create code directories**:
  ```bash
  mkdir -p src/{models,data,training,analysis,utils}
  mkdir -p notebooks experiments results
  ```
- [ ] **Set up git repository** (if not already done):
  ```bash
  git init
  git add .
  git commit -m "Initial project setup"
  ```

### Initial Development
- [ ] **Create basic data loading script** (`src/data/loader.py`)
- [ ] **Test poetry dataset loading** with first few poems
- [ ] **Verify memory system**: Update `collaboration_memory/current_focus.md`

### GloVe Embeddings Setup
- [ ] **Download GloVe embeddings** (300D):
  ```bash
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip
  unzip glove.840B.300d.zip
  ```
- [ ] **Test embedding loading**: Basic script to load and query embeddings
- [ ] **Integrate with dataset**: Test embedding lookup for poetry words

## Troubleshooting Checklist

### If Environment Creation Fails
- [ ] **Check conda channels**: `conda config --show channels`
- [ ] **Update conda**: `conda update conda`
- [ ] **Try mamba instead**: `conda install mamba`, then `mamba env create -f environment.yml`
- [ ] **Manual package installation** for problematic packages

### If PyTorch Issues
- [ ] **Check PyTorch installation**: Visit pytorch.org for platform-specific instructions
- [ ] **CPU vs GPU version**: Ensure correct PyTorch variant for hardware
- [ ] **Dependency conflicts**: `conda list` to check for conflicting packages

### If Dataset Issues  
- [ ] **Check file permissions**: `ls -la dataset_poetry/`
- [ ] **Verify file integrity**: Check file sizes match expected values
- [ ] **Test JSON parsing**: Manually open and parse one file

## Success Criteria

### Environment Ready ✅
- [ ] All validation checks pass (`python validate_environment.py`)
- [ ] PyTorch tensor operations work
- [ ] Dataset loads without errors
- [ ] Jupyter Lab functional with correct kernel

### Development Ready ✅  
- [ ] Basic data loading pipeline working
- [ ] GloVe embeddings accessible
- [ ] Git repository initialized
- [ ] Memory system updated with progress

### Ready for Step 2 ✅
- [ ] Can load and embed poetry text
- [ ] Basic PyTorch RNN modules functional
- [ ] Collaboration memory updated
- [ ] Ready to begin embedding analysis and PCA

## Post-Setup Actions

### Documentation
- [ ] **Update progress_log.txt** with setup completion
- [ ] **Update collaboration_memory/current_focus.md** with next steps
- [ ] **Document any setup modifications** in implementation_notes.md

### Next Phase Preparation
- [ ] **Review Step 2 requirements**: GloVe embedding analysis
- [ ] **Plan PCA experiments**: Effective dimensionality estimation  
- [ ] **Prepare for architecture design**: Review theoretical foundation

---

**Estimated Setup Time**: 2-4 hours for complete environment setup and validation  
**Key Success Metric**: `python validate_environment.py` passes all checks  
**Next Phase**: Step 2 - GloVe embedding analysis and effective dimension estimation