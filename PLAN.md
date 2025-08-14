# High-Level API Implementation Plan

## Goal: Simplify Usage to Single API Call

```python
# Target: From this complexity...
from poetry_rnn import PoetryPreprocessor, AutoencoderDataset, Config
from poetry_rnn.models import RNNAutoencoder
from poetry_rnn.training import CurriculumScheduler, GradientMonitor
# ... 50 lines of setup code ...

# To this simplicity:
from poetry_rnn.api import poetry_autoencoder

model = poetry_autoencoder(
    data_path="dataset_poetry/poems.json",
    design={"hidden_size": 512, "bottleneck_size": 64, "rnn_type": "lstm"},
    training={"epochs": 30, "curriculum_phases": 4},
    output_dir="./results"
)
```

## Phase 1: API Architecture Design

### Core API Module Structure
```
poetry_rnn/api/
├── __init__.py          # Main entry points
├── design.py            # Architecture configuration
├── training.py          # Training configuration  
├── data.py              # Data handling
├── experiments.py       # Experiment management
└── utils.py             # API utilities
```

### Design Philosophy
- **Configuration over Code**: Use dictionaries/dataclasses for configuration
- **Sensible Defaults**: Work out-of-the-box with minimal configuration
- **Progressive Disclosure**: Simple API with advanced options available
- **Immutable Configs**: Prevent accidental mutations during training
- **Type Safety**: Use type hints and validation throughout

## Phase 2: Configuration System

### Architecture Configuration
```python
@dataclass
class ArchitectureConfig:
    # Core dimensions
    input_size: int = 300          # GloVe embedding dimension
    hidden_size: int = 512         # RNN hidden dimension  
    bottleneck_size: int = 64      # Compression dimension
    
    # Model type
    rnn_type: str = 'lstm'         # 'vanilla', 'lstm', 'gru'
    num_layers: int = 1            # RNN layers
    
    # Regularization
    dropout: float = 0.1           # Dropout probability
    use_batch_norm: bool = True    # Batch normalization
    
    # Advanced options
    bidirectional: bool = False    # Bidirectional RNN
    attention: bool = False        # Attention mechanism
    residual: bool = False         # Residual connections
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.bottleneck_size >= self.hidden_size:
            raise ValueError("Bottleneck must be smaller than hidden size")
        # ... more validation
```

### Training Configuration
```python
@dataclass 
class TrainingConfig:
    # Basic training
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 2.5e-4
    
    # Curriculum learning
    curriculum_phases: int = 4
    phase_epochs: List[int] = field(default_factory=lambda: [8, 10, 12, 15])
    teacher_forcing_schedule: List[float] = field(default_factory=lambda: [0.9, 0.7, 0.3, 0.1])
    
    # Optimization
    optimizer: str = 'adam'
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    scheduler: str = 'plateau'     # 'plateau', 'cosine', 'exponential'
    
    # Monitoring
    save_every: int = 5
    log_every: int = 20
    early_stopping_patience: int = 10
    
    # Hardware
    device: str = 'auto'           # 'auto', 'cpu', 'cuda', 'cuda:0'
    num_workers: int = 4           # DataLoader workers
    pin_memory: bool = True
```

### Data Configuration
```python
@dataclass
class DataConfig:
    # Dataset
    data_path: str                 # Path to poetry dataset
    split_ratios: Tuple[float, float, float] = (0.8, 0.15, 0.05)  # train/val/test
    
    # Preprocessing  
    max_sequence_length: int = 50
    embedding_type: str = 'glove'  # 'glove', 'word2vec', 'custom'
    embedding_dim: int = 300
    embedding_path: Optional[str] = None
    
    # Chunking strategy
    chunking_method: str = 'sliding'  # 'sliding', 'truncate', 'pad'
    window_size: int = 50
    overlap: int = 10
    
    # Caching
    cache_preprocessed: bool = True
    cache_dir: str = 'preprocessed_artifacts'
```

## Phase 3: Factory Functions

### Architecture Factory
```python
def design_autoencoder(
    hidden_size: int = 512,
    bottleneck_size: int = 64, 
    input_size: int = 300,
    rnn_type: str = 'lstm',
    **kwargs
) -> ArchitectureConfig:
    """Create architecture configuration with validation."""
    config = ArchitectureConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        bottleneck_size=bottleneck_size,
        rnn_type=rnn_type,
        **kwargs
    )
    config.validate()
    return config

def preset_architecture(preset: str) -> ArchitectureConfig:
    """Load preset architectures."""
    presets = {
        'small': ArchitectureConfig(hidden_size=128, bottleneck_size=16),
        'medium': ArchitectureConfig(hidden_size=256, bottleneck_size=32), 
        'large': ArchitectureConfig(hidden_size=512, bottleneck_size=64),
        'xlarge': ArchitectureConfig(hidden_size=1024, bottleneck_size=128)
    }
    return presets[preset]
```

### Training Factory
```python
def curriculum_learning(
    phases: int = 4,
    epochs: List[int] = None,
    decay_type: str = 'linear',
    **kwargs
) -> TrainingConfig:
    """Create curriculum learning configuration."""
    if epochs is None:
        # Auto-generate epoch distribution
        total_epochs = kwargs.get('total_epochs', 30)
        epochs = distribute_epochs(total_epochs, phases)
    
    # Auto-generate teacher forcing schedule
    tf_schedule = generate_tf_schedule(phases, decay_type)
    
    return TrainingConfig(
        curriculum_phases=phases,
        phase_epochs=epochs,
        teacher_forcing_schedule=tf_schedule,
        **kwargs
    )
```

### Data Factory
```python
def fetch_data(
    data_path: str,
    embedding_type: str = 'glove',
    max_length: int = 50,
    **kwargs
) -> DataConfig:
    """Create data configuration with auto-detection."""
    # Auto-detect data format
    data_format = detect_data_format(data_path)
    
    # Auto-locate embeddings
    if embedding_type == 'glove' and 'embedding_path' not in kwargs:
        kwargs['embedding_path'] = find_glove_embeddings()
    
    return DataConfig(
        data_path=data_path,
        embedding_type=embedding_type,
        max_sequence_length=max_length,
        **kwargs
    )
```

## Phase 4: Main API Classes

### High-Level RNN Class
```python
class RNN:
    """High-level autoencoder interface."""
    
    def __init__(
        self,
        architecture: ArchitectureConfig,
        training: TrainingConfig,
        data: DataConfig
    ):
        self.arch_config = architecture
        self.train_config = training
        self.data_config = data
        
        # Initialize components lazily
        self._model = None
        self._trainer = None
        self._data_loaders = None
        
    def train(self, output_dir: str = "./results") -> Dict[str, Any]:
        """Train the autoencoder and return results."""
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Build components
        self._build_model()
        self._prepare_data()
        self._setup_training()
        
        # Train model
        results = self._trainer.train()
        
        # Save everything
        self._save_results(results)
        
        return results
    
    def load(self, model_path: str) -> 'RNN':
        """Load trained model."""
        # Implementation...
        return self
        
    def generate(self, seed_text: str = None, length: int = 50) -> str:
        """Generate poetry from trained model."""
        # Implementation...
        pass
```

### Convenience Function
```python
def poetry_autoencoder(
    data_path: str,
    design: Dict[str, Any] = None,
    training: Dict[str, Any] = None,
    data: Dict[str, Any] = None,
    output_dir: str = "./results"
) -> RNN:
    """One-line poetry autoencoder training."""
    
    # Create configurations with defaults
    arch_config = design_autoencoder(**(design or {}))
    train_config = curriculum_learning(**(training or {}))  
    data_config = fetch_data(data_path, **(data or {}))
    
    # Create and train model
    model = RNN(arch_config, train_config, data_config)
    model.train(output_dir)
    
    return model
```

## Phase 5: Implementation Strategy

### Step 1: Configuration System (Week 1)
- [ ] Create config dataclasses with validation
- [ ] Implement factory functions  
- [ ] Add preset configurations
- [ ] Write comprehensive tests

### Step 2: API Wrapper (Week 2)  
- [ ] Implement RNN class with lazy initialization
- [ ] Create component builders (model, data, trainer)
- [ ] Add result saving and loading
- [ ] Integration with existing codebase

### Step 3: Convenience Functions (Week 3)
- [ ] Implement poetry_autoencoder() main function
- [ ] Add auto-detection utilities (data format, embeddings)
- [ ] Create sensible defaults system
- [ ] Documentation and examples

### Step 4: Advanced Features (Week 4)
- [ ] Experiment management (multiple runs, hyperparameter sweeps)
- [ ] Model comparison utilities
- [ ] Generation and evaluation tools
- [ ] Performance monitoring

## Phase 6: Backward Compatibility

### Migration Strategy
- Keep existing low-level API intact
- Add deprecation warnings gradually
- Provide migration guide
- Support both APIs during transition

### Testing Strategy
- Unit tests for all configurations
- Integration tests with real data
- Performance regression tests
- API contract tests

## Usage Examples

### Basic Usage
```python
from poetry_rnn.api import poetry_autoencoder

# Minimal usage - everything auto-configured
model = poetry_autoencoder("my_poems.json")
```

### Advanced Usage
```python
from poetry_rnn.api import design_autoencoder, curriculum_learning, fetch_data, RNN

# Custom configuration
arch = design_autoencoder(hidden_size=1024, bottleneck_size=128, rnn_type='lstm')
curriculum = curriculum_learning(phases=6, decay_type='exponential')
data = fetch_data("poems.json", max_length=100, chunking_method='sliding')

model = RNN(arch, curriculum, data)
results = model.train("./experiment_1")

# Generate poetry
poem = model.generate(seed_text="In the beginning", length=100)
```

### Experiment Management
```python
from poetry_rnn.api import run_experiments

# Hyperparameter sweep
experiments = run_experiments(
    data_path="poems.json",
    architectures=[
        {"hidden_size": 256, "bottleneck_size": 32},
        {"hidden_size": 512, "bottleneck_size": 64}, 
        {"hidden_size": 1024, "bottleneck_size": 128}
    ],
    output_dir="./experiments"
)
```

This plan maintains the power and flexibility of the current system while dramatically simplifying the user experience. The factory pattern makes it easy to extend with new features while keeping the API clean and intuitive.