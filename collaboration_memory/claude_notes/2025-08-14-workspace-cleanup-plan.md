# Workspace Cleanup Plan - August 14, 2025

## Current Situation
Training is running with neural-network-mentor fixes. Good time to clean up accumulated files.

## Files to KEEP (Essential)
### Core Implementation
- poetry_rnn/ (entire package)
- dataset_poetry/ (main dataset)
- embeddings/ (GloVe embeddings)
- train_scaled_architecture.py (current training script)
- compare_architectures.py (evaluation script)
- best_model.pth (baseline model)

### Documentation
- CLAUDE.md (project instructions)
- collaboration_memory/ (all memory files)
- README.md
- environment.yml, requirements.txt

### Current Training Results
- scaled_model_*.pth (when training completes)
- training_logs_scaled/ (current training logs)

## Files to DELETE (Outdated/Redundant)
### Old Training Scripts (Superseded)
- train_simple_autoencoder.py (replaced by train_scaled_architecture.py)
- train_fixed_architecture.py (replaced by train_scaled_architecture.py)  
- train_architecture_comparison.py (redundant)
- diagnostic_test.py (one-time use)

### Old Model Checkpoints (Baseline training complete)
- checkpoint_epoch_*.pth (5, 10, 15, 20, 25, 30)
- fixed_architecture_*.pth (intermediate experiments)

### Old Evaluation Scripts (Redundant)
- evaluate_trained_model.py (replaced by compare_architectures.py)
- minimal_evaluation.py (replaced by compare_architectures.py)  
- simple_evaluation.py (replaced by compare_architectures.py)
- simple_model_evaluation.py (replaced by compare_architectures.py)
- quick_eval.py (replaced by compare_architectures.py)

### Old Progress Images (Training complete)
- training_progress_epoch_*.png (5, 10, 15, 20, 25, 30)
- final_training_progress.png (superseded)

### Duplicate Preprocessed Artifacts
- preprocessed_artifacts/ (many timestamped duplicates, keep only latest)
- GloVe_preprocessing/preprocessed_artifacts/ (duplicates)

### Old Documentation Files
- claude_continuity_note.md (integrated into memory system)
- TODO.md (superseded by TodoWrite system)
- progress_log.txt (superseded by memory system)
- your_collaborator.txt (integrated into CLAUDE.md)

### Development Artifacts
- dissertation.zip (external reference material)
- GRIETZER-DISSERTATION-*.pdf (external reference material)  
- scripts/ (mostly one-time utilities)
- tests/ (can be regenerated if needed)
- docs/ (old documentation)
- examples/ (can be regenerated)

## Cleanup Strategy
1. Create archive directory for removed files (in case needed later)
2. Remove files systematically  
3. Keep workspace focused on current training and evaluation

## Expected Result
- Clean workspace focused on current scaled architecture training
- Keep only essential files for ongoing work
- Maintain all memory and core implementation files