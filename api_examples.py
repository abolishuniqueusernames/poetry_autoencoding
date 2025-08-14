#!/usr/bin/env python3
"""
Comprehensive examples for the Poetry RNN API.

This script demonstrates all the ways to use the new high-level API,
from simple one-line training to advanced configuration and experimentation.

Run this script in the poetry autoencoding project directory.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def example_1_minimal_usage():
    """Example 1: Minimal usage - everything auto-configured."""
    print("=" * 60)
    print("EXAMPLE 1: MINIMAL USAGE")
    print("=" * 60)
    
    from poetry_rnn.api import poetry_autoencoder
    
    # Check if we have actual poetry data
    poetry_files = list(Path("dataset_poetry").glob("*.json")) if Path("dataset_poetry").exists() else []
    
    if poetry_files:
        data_file = str(poetry_files[0])
        print(f"Using real data: {data_file}")
        
        # This would actually train the model
        print("To train with minimal configuration:")
        print(f'model = poetry_autoencoder("{data_file}")')
        print()
        print("Note: Training disabled in this example to avoid long runtime.")
        print("To actually train, uncomment the line below and run:")
        print("# model = poetry_autoencoder(data_file)")
    else:
        print("No poetry data found in dataset_poetry/")
        print("Example of minimal usage (with real data):")
        print('model = poetry_autoencoder("dataset_poetry/poems.json")')
    
    print("\nThis single line:")
    print("1. Auto-detects data format")
    print("2. Finds GLoVe embeddings automatically")
    print("3. Uses optimal architecture for detected data")
    print("4. Implements curriculum learning")
    print("5. Trains model with monitoring")
    print("6. Saves results to ./results")


def example_2_custom_configuration():
    """Example 2: Custom configuration with explicit parameters."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: CUSTOM CONFIGURATION")
    print("=" * 60)
    
    from poetry_rnn.api import poetry_autoencoder
    
    print("Custom configuration example:")
    print("""
model = poetry_autoencoder(
    data_path="dataset_poetry/poems.json",
    design={
        "hidden_size": 512, 
        "bottleneck_size": 64, 
        "rnn_type": "lstm",
        "dropout": 0.15,
        "num_layers": 2
    },
    training={
        "epochs": 50, 
        "curriculum_phases": 6,
        "learning_rate": 1e-4,
        "batch_size": 8
    },
    data={
        "max_sequence_length": 75,
        "chunking_method": "sliding",
        "split_ratios": (0.7, 0.2, 0.1)
    },
    output_dir="./experiment_1"
)
""")
    
    # Show what these configurations mean
    from poetry_rnn.api import design_autoencoder, curriculum_learning, fetch_data
    
    print("This creates:")
    arch = design_autoencoder(hidden_size=512, bottleneck_size=64, rnn_type="lstm", dropout=0.15, num_layers=2)
    print(f"- Architecture: {arch.get_compression_ratio():.1f}x compression, {arch.estimate_parameters():,} parameters")
    
    training = curriculum_learning(epochs=50, phases=6, learning_rate=1e-4)
    print(f"- Training: {len(training.phase_epochs)} phases over {sum(training.phase_epochs)} epochs")
    print(f"- Teacher forcing decay: {training.teacher_forcing_schedule[0]:.2f} → {training.teacher_forcing_schedule[-1]:.2f}")


def example_3_preset_architectures():
    """Example 3: Using preset architectures."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: PRESET ARCHITECTURES")
    print("=" * 60)
    
    from poetry_rnn.api import preset_architecture
    
    print("Available preset architectures:")
    
    presets = ['tiny', 'small', 'medium', 'large', 'xlarge', 'research']
    
    for preset_name in presets:
        try:
            preset = preset_architecture(preset_name)
            compression = preset.get_compression_ratio()
            params = preset.estimate_parameters()
            
            print(f"- {preset_name:8}: {preset.hidden_size:4}→{preset.bottleneck_size:3} "
                  f"({compression:4.1f}x compression, {params:8,} params)")
        except Exception as e:
            print(f"- {preset_name:8}: Error - {e}")
    
    print(f"\nUsage example:")
    print("""
from poetry_rnn.api import poetry_autoencoder

# Use preset with optional overrides
model = poetry_autoencoder(
    "poems.json",
    design=preset_architecture('large', dropout=0.2),  # Large model with more dropout
    output_dir="./large_model_experiment"
)
""")


def example_4_advanced_usage():
    """Example 4: Advanced usage with manual control."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: ADVANCED MANUAL CONTROL")
    print("=" * 60)
    
    print("For maximum control, create configurations separately:")
    print("""
from poetry_rnn.api import (
    RNN, design_autoencoder, curriculum_learning, fetch_data,
    validate_configuration_compatibility
)

# Create custom configurations
architecture = design_autoencoder(
    hidden_size=768,
    bottleneck_size=96,
    rnn_type='lstm',
    num_layers=2,
    dropout=0.15,
    use_batch_norm=True,
    attention=True  # Advanced feature
)

training = curriculum_learning(
    phases=4,
    epochs=60,
    decay_type='exponential',
    learning_rate=2e-4,
    batch_size=12,
    scheduler='plateau'
)

data = fetch_data(
    "dataset_poetry/poems.json",
    max_length=100,
    chunking_method='sliding',
    cache=True
)

# Validate configurations work together
validate_configuration_compatibility(architecture, training, data)

# Create model (doesn't train immediately)
model = RNN(architecture, training, data, output_dir="./advanced_experiment")

# Train when ready
results = model.train()

# Use trained model
poem = model.generate("In the quiet morning", length=50, temperature=0.8)
print(poem)

# Evaluate performance
metrics = model.evaluate()
print(f"Test loss: {metrics['test_loss']:.4f}")

# Save for later use
model_path = model.save("my_poetry_model.pth")

# Load later
loaded_model = RNN(architecture, training, data).load(model_path)
""")


def example_5_experimentation():
    """Example 5: Running multiple experiments."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: EXPERIMENTATION FRAMEWORK")
    print("=" * 60)
    
    print("Example of running multiple experiments:")
    print("""
from poetry_rnn.api import poetry_autoencoder, preset_architecture

# Test different architectures
architectures = ['small', 'medium', 'large']
results = {}

for arch_name in architectures:
    print(f"Training {arch_name} architecture...")
    
    model = poetry_autoencoder(
        "dataset_poetry/poems.json",
        design=preset_architecture(arch_name),
        training={"epochs": 20},  # Shorter for comparison
        output_dir=f"./experiment_{arch_name}"
    )
    
    results[arch_name] = {
        'final_loss': model._training_results['final_loss'],
        'parameters': model.arch_config.estimate_parameters(),
        'compression': model.arch_config.get_compression_ratio()
    }

# Compare results
for name, result in results.items():
    print(f"{name}: loss={result['final_loss']:.4f}, "
          f"params={result['parameters']:,}, "
          f"compression={result['compression']:.1f}x")
""")


def example_6_system_diagnostics():
    """Example 6: System diagnostics and optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: SYSTEM DIAGNOSTICS")
    print("=" * 60)
    
    from poetry_rnn.api.utils import (
        print_system_info, get_optimal_batch_size, 
        estimate_memory_requirements, suggest_architecture_for_data
    )
    
    print("System diagnostics:")
    print_system_info()
    
    print(f"\nMemory optimization:")
    
    # Get optimal batch size for different configurations
    for model_size in ['small', 'medium', 'large']:
        batch_size = get_optimal_batch_size(model_size, sequence_length=50, available_memory_gb=8.0)
        print(f"- {model_size} model: recommended batch size = {batch_size}")
    
    print(f"\nMemory estimation for medium model:")
    memory = estimate_memory_requirements(
        batch_size=16, sequence_length=50, hidden_size=512, embedding_dim=300
    )
    print(f"- Estimated memory usage: {memory['total_gb']:.1f} GB")
    print(f"- Parameters: {memory['parameters_mb']:.1f} MB")
    print(f"- Activations: {memory['activations_mb']:.1f} MB")
    
    # Data-based architecture suggestion
    poetry_files = list(Path("dataset_poetry").glob("*.json")) if Path("dataset_poetry").exists() else []
    if poetry_files:
        suggestion = suggest_architecture_for_data(str(poetry_files[0]))
        print(f"\nData-based architecture suggestion:")
        print(f"- Recommended preset: {suggestion['preset']}")
        print(f"- Reason: {suggestion['reason']}")
        if 'data_stats' in suggestion:
            stats = suggestion['data_stats']
            print(f"- Dataset: {stats['num_poems']} poems, ~{stats['avg_words_per_poem']:.0f} words/poem")


def example_7_production_workflow():
    """Example 7: Production workflow with best practices."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: PRODUCTION WORKFLOW")
    print("=" * 60)
    
    print("Production-ready workflow:")
    print("""
from poetry_rnn.api import poetry_autoencoder, production_training, suggest_architecture_for_data
import json
from pathlib import Path

# Step 1: Analyze data and get architecture recommendation
data_path = "dataset_poetry/poems.json"
suggestion = suggest_architecture_for_data(data_path)

print(f"Recommended architecture: {suggestion['preset']}")
print(f"Reason: {suggestion['reason']}")

# Step 2: Use production training configuration
model = poetry_autoencoder(
    data_path=data_path,
    design=preset_architecture(suggestion['preset']),
    training=production_training(epochs=100),  # Long, careful training
    output_dir="./production_model",
    verbose=True
)

# Step 3: Comprehensive evaluation
test_metrics = model.evaluate()

# Step 4: Save model with metadata
model_info = {
    'architecture': suggestion['preset'],
    'training_epochs': 100,
    'final_metrics': test_metrics,
    'data_stats': suggestion.get('data_stats', {}),
    'model_path': model.save("production_poetry_model.pth")
}

# Save experiment info
with open("production_experiment.json", "w") as f:
    json.dump(model_info, f, indent=2, default=str)

print("Production model ready!")
print(f"Model saved: {model_info['model_path']}")
print(f"Test loss: {test_metrics['test_loss']:.4f}")

# Step 5: Generate sample outputs for quality assessment
sample_seeds = ["In the beginning", "The morning light", "Dreams of tomorrow"]
for seed in sample_seeds:
    poem = model.generate(seed, length=75, temperature=0.7)
    print(f"\\n--- Generated from '{seed}' ---")
    print(poem)
""")


def main():
    """Run all examples."""
    print("🎭 POETRY RNN API EXAMPLES")
    print("Complete guide to using the high-level API")
    
    examples = [
        example_1_minimal_usage,
        example_2_custom_configuration,
        example_3_preset_architectures,
        example_4_advanced_usage,
        example_5_experimentation,
        example_6_system_diagnostics,
        example_7_production_workflow
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Example {example.__name__} failed: {e}")
    
    print("\n" + "=" * 60)
    print("KEY ADVANTAGES OF THE NEW API")
    print("=" * 60)
    print("1. 🚀 One-line training: poetry_autoencoder('poems.json')")
    print("2. 🎛️  Progressive complexity: simple → advanced configuration")
    print("3. 🧠 Intelligent defaults based on theoretical analysis")
    print("4. 🔍 Auto-detection of data formats and embeddings")
    print("5. ✅ Comprehensive validation and helpful error messages")
    print("6. 🏗️  Lazy initialization for fast feedback")
    print("7. 📊 Built-in experimentation and comparison tools")
    print("8. 🔧 Production-ready with monitoring and checkpointing")
    print("9. 🎯 Theory-driven presets optimized for poetry")
    print("10. 🔄 Backward compatible with existing low-level API")
    
    print(f"\n🎉 The API transforms this complex workflow:")
    print("   50+ lines of setup code")
    print("   Into this simple interface:")
    print("   model = poetry_autoencoder('poems.json')")
    
    print("\n📚 Ready to use! Check the examples above and start training.")


if __name__ == "__main__":
    main()