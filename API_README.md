# Poetry RNN Autoencoder - High-Level API

Transform complex neural network training into simple one-line functions with the new Poetry RNN API.

## 🚀 Quick Start

```python
from poetry_rnn.api import poetry_autoencoder

# One line to train a complete autoencoder
model = poetry_autoencoder("dataset_poetry/poems.json")

# Generate poetry with the trained model
poem = model.generate("In the quiet morning", length=100)
print(poem)
```

## 🎯 Key Features

- **One-line training**: `poetry_autoencoder("poems.json")` handles everything
- **Intelligent defaults**: Theory-driven configurations optimized for poetry
- **Auto-detection**: Finds data formats, embeddings, optimal hardware automatically
- **Progressive complexity**: Simple → advanced configuration as needed
- **Comprehensive validation**: Helpful error messages and warnings
- **Lazy initialization**: Fast feedback, expensive operations only when needed
- **Production-ready**: Built-in monitoring, checkpointing, and evaluation
- **Backward compatible**: Works alongside existing low-level API

## 📊 From Complex to Simple

**Before (50+ lines):**
```python
from poetry_rnn import PoetryPreprocessor, AutoencoderDataset, Config
from poetry_rnn.models import RNNAutoencoder
from poetry_rnn.training import RNNAutoencoderTrainer, CurriculumScheduler
from poetry_rnn.embeddings import GLoVeEmbeddingManager

# Configure preprocessing
config = Config()
config.embedding.embedding_path = "embeddings/glove.6B.300d.txt"
config.chunking.window_size = 50
config.chunking.overlap = 10

# Create preprocessor and process data
preprocessor = PoetryPreprocessor(config=config)
results = preprocessor.process_poems("poems.json")

# Create datasets
datasets = create_poetry_datasets(
    sequences=results['sequences'],
    embedding_sequences=results['embedding_sequences'],
    attention_masks=results['attention_masks'],
    metadata=results['metadata'],
    vocabulary=results['vocabulary'],
    split_ratios=(0.8, 0.15, 0.05)
)

# Create data loaders
dataloaders = create_poetry_dataloaders(
    datasets, batch_size=16, num_workers=4
)

# Build model
model = RNNAutoencoder(
    vocab_size=len(results['vocabulary']),
    embedding_dim=300,
    hidden_size=512,
    bottleneck_size=64,
    rnn_type='lstm'
)

# Setup trainer
trainer_config = {
    'learning_rate': 2.5e-4,
    'num_epochs': 30,
    'gradient_clip': 1.0,
    'curriculum_learning': True
}

trainer = RNNAutoencoderTrainer(
    model=model,
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    config=trainer_config
)

# Train model
results = trainer.train()
```

**After (1 line):**
```python
from poetry_rnn.api import poetry_autoencoder

model = poetry_autoencoder("poems.json")
```

## 🎛️ Configuration Levels

### Level 1: Minimal (Everything Automatic)
```python
model = poetry_autoencoder("poems.json")
```

### Level 2: Basic Customization
```python
model = poetry_autoencoder(
    "poems.json",
    design={"hidden_size": 512, "bottleneck_size": 64},
    training={"epochs": 50},
    output_dir="./my_experiment"
)
```

### Level 3: Advanced Configuration
```python
model = poetry_autoencoder(
    data_path="poems.json",
    design={
        "hidden_size": 768,
        "bottleneck_size": 96,
        "rnn_type": "lstm",
        "num_layers": 2,
        "dropout": 0.15,
        "attention": True
    },
    training={
        "epochs": 100,
        "curriculum_phases": 6,
        "learning_rate": 1e-4,
        "batch_size": 8,
        "scheduler": "plateau"
    },
    data={
        "max_sequence_length": 75,
        "chunking_method": "sliding",
        "split_ratios": (0.7, 0.2, 0.1)
    }
)
```

### Level 4: Expert Control
```python
from poetry_rnn.api import RNN, design_autoencoder, curriculum_learning, fetch_data

architecture = design_autoencoder(
    hidden_size=1024,
    bottleneck_size=20,  # Theory-optimal compression
    rnn_type='lstm',
    bidirectional=True,
    attention=True,
    residual=True
)

training = curriculum_learning(
    phases=8,
    epochs=150,
    decay_type='exponential',
    initial_teacher_forcing=0.95,
    final_teacher_forcing=0.05
)

data = fetch_data(
    "poems.json",
    max_length=150,
    embedding_type='glove',
    chunking_method='sliding'
)

model = RNN(architecture, training, data)
results = model.train("./expert_experiment")
```

## 🏗️ Architecture Presets

Use proven configurations optimized for different use cases:

```python
from poetry_rnn.api import preset_architecture

# Available presets
presets = {
    'tiny':     # 128→16,  334K params  - Testing & development
    'small':    # 256→32,  951K params  - Limited resources
    'medium':   # 512→64,  3M params    - Balanced performance (default)
    'large':    # 768→96,  12M params   - High quality results
    'xlarge':   # 1024→128, 21M params  - Research & experiments
    'research': # 512→20,  2.9M params  - Theory-optimal compression
}

# Usage
model = poetry_autoencoder(
    "poems.json",
    design=preset_architecture('large'),
    output_dir="./large_model"
)
```

## 🧪 Experimentation

### Compare Multiple Architectures
```python
from poetry_rnn.api import poetry_autoencoder, preset_architecture

architectures = ['small', 'medium', 'large']
results = {}

for arch in architectures:
    model = poetry_autoencoder(
        "poems.json",
        design=preset_architecture(arch),
        training={"epochs": 20},
        output_dir=f"./experiment_{arch}"
    )
    
    results[arch] = {
        'loss': model._training_results['best_loss'],
        'params': model.arch_config.estimate_parameters(),
        'compression': model.arch_config.get_compression_ratio()
    }

# Compare results
for name, result in results.items():
    print(f"{name}: loss={result['loss']:.4f}, "
          f"params={result['params']:,}, "
          f"compression={result['compression']:.1f}x")
```

### Production Workflow
```python
from poetry_rnn.api import poetry_autoencoder, production_training, suggest_architecture_for_data

# Analyze data and get recommendation
suggestion = suggest_architecture_for_data("poems.json")
print(f"Recommended: {suggestion['preset']} - {suggestion['reason']}")

# Train production model
model = poetry_autoencoder(
    "poems.json",
    design=preset_architecture(suggestion['preset']),
    training=production_training(epochs=100),  # Conservative, high-quality training
    output_dir="./production_model"
)

# Evaluate and save
metrics = model.evaluate()
model_path = model.save("production_poetry_model.pth")

print(f"Production model ready: {model_path}")
print(f"Test loss: {metrics['test_loss']:.4f}")
```

## 🔧 System Optimization

### Automatic Resource Detection
```python
from poetry_rnn.api.utils import print_system_info, get_optimal_batch_size

# Check system capabilities
print_system_info()

# Get optimal batch size for your hardware
batch_size = get_optimal_batch_size(
    model_size='medium',
    sequence_length=50,
    available_memory_gb=8.0
)

model = poetry_autoencoder(
    "poems.json",
    training={"batch_size": batch_size}
)
```

### Memory Estimation
```python
from poetry_rnn.api.utils import estimate_memory_requirements

memory = estimate_memory_requirements(
    batch_size=16,
    sequence_length=50,
    hidden_size=512,
    embedding_dim=300
)

print(f"Estimated memory: {memory['total_gb']:.1f} GB")
print(f"Recommended batch size: {memory['recommended_batch_size']}")
```

## 📈 Model Usage

### Generation
```python
# Load trained model
model = poetry_autoencoder("poems.json")

# Generate with different settings
poem1 = model.generate("The morning light", length=50, temperature=0.7)
poem2 = model.generate("Dreams of tomorrow", length=100, temperature=1.2)

print("Conservative generation:", poem1)
print("Creative generation:", poem2)
```

### Evaluation
```python
# Comprehensive evaluation
metrics = model.evaluate()

print(f"Test loss: {metrics['test_loss']:.4f}")
print(f"Reconstruction accuracy: {metrics['reconstruction_accuracy']:.2f}")
print(f"Compression ratio: {metrics['compression_ratio']:.1f}x")
```

### Save and Load
```python
# Save trained model
model_path = model.save("my_poetry_model.pth")

# Load later (with same configuration)
loaded_model = RNN(architecture, training, data).load(model_path)

# Generate with loaded model
poem = loaded_model.generate("Once upon a time", length=75)
```

## 🎓 Theory Integration

The API incorporates theoretical insights from the project's mathematical analysis:

- **Optimal compression**: Default bottleneck sizes based on effective dimensionality analysis
- **Curriculum learning**: Theory-driven teacher forcing schedules
- **Architecture presets**: Configurations validated through complexity analysis
- **Memory optimization**: Based on O(ε^-600) → O(ε^-35) complexity reduction insights

### Research Configuration
```python
# Use the theory-optimal configuration
model = poetry_autoencoder(
    "poems.json",
    design=preset_architecture('research'),  # 512→20D compression
    training=curriculum_learning(phases=6, decay_type='exponential'),
    output_dir="./theory_experiment"
)
```

## 🔄 Migration Guide

### From Low-Level API
```python
# Old way
preprocessor = PoetryPreprocessor(config)
results = preprocessor.process_poems("poems.json")
# ... 40+ more lines ...

# New way
model = poetry_autoencoder("poems.json")
```

### Gradual Migration
```python
# Keep using low-level components with new API
from poetry_rnn.api import RNN
from poetry_rnn import PoetryPreprocessor  # Still available

# Mix approaches as needed
preprocessor = PoetryPreprocessor()  # Custom preprocessing
results = preprocessor.process_poems("poems.json")

# Use API for model and training
model = poetry_autoencoder(
    "poems.json",  # Or pass preprocessed results
    design=preset_architecture('medium'),
    training=curriculum_learning(epochs=30)
)
```

## 🚨 Error Handling

The API provides comprehensive validation and helpful error messages:

```python
# Invalid configuration caught early
try:
    model = poetry_autoencoder(
        "poems.json",
        design={"hidden_size": 64, "bottleneck_size": 128}  # Invalid: bottleneck > hidden
    )
except ValueError as e:
    print(f"Configuration error: {e}")
    # Suggests valid configurations
```

## 📚 Complete API Reference

### Main Functions
- `poetry_autoencoder()`: One-line training function
- `RNN`: High-level model class

### Configuration
- `ArchitectureConfig`: Model architecture settings
- `TrainingConfig`: Training and curriculum learning
- `DataConfig`: Data processing configuration

### Factory Functions
- `design_autoencoder()`: Create architecture configs
- `curriculum_learning()`: Create training configs
- `fetch_data()`: Create data configs
- `preset_architecture()`: Load preset configurations
- `quick_training()`: Fast training for experiments
- `production_training()`: High-quality training

### Utilities
- `find_glove_embeddings()`: Auto-locate embeddings
- `auto_detect_device()`: Optimal hardware detection
- `print_system_info()`: System diagnostics
- `get_optimal_batch_size()`: Memory optimization
- `suggest_architecture_for_data()`: Data-driven recommendations

## 🎯 Next Steps

1. **Quick Test**: Run `python test_api.py` to verify everything works
2. **Examples**: Check `python api_examples.py` for comprehensive usage examples
3. **First Training**: Try `poetry_autoencoder("dataset_poetry/poems.json")`
4. **Experimentation**: Use presets and compare results
5. **Production**: Use `production_training()` for final models

The Poetry RNN API transforms neural network complexity into poetry simplicity. Start with one line, scale to any complexity needed.

---
*Built on the foundation of mathematical rigor and practical experience from the Poetry RNN Collaborative Project.*