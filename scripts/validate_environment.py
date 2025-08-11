#!/usr/bin/env python3
"""
Environment Validation Script for poetryRNN Project
===================================================

This script validates that all required packages are installed and functioning
correctly in the poetryRNN conda environment. Run this after environment setup
to ensure everything is ready for neural network development.

Usage: python validate_environment.py
"""

import sys
import importlib
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version is appropriate for the project."""
    print("=== Python Version Check ===")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 9:
        print("❌ ERROR: Python 3.9+ required")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_core_packages():
    """Check core ML packages are installed and importable."""
    print("\n=== Core ML Stack Check ===")
    
    core_packages = {
        'torch': 'PyTorch',
        'torchvision': 'PyTorch Vision',
        'numpy': 'NumPy',
        'scipy': 'SciPy', 
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'pandas': 'Pandas'
    }
    
    all_good = True
    for package, name in core_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: NOT FOUND")
            all_good = False
    
    return all_good


def check_nlp_packages():
    """Check NLP and text processing packages."""
    print("\n=== NLP Stack Check ===")
    
    nlp_packages = {
        'transformers': 'HuggingFace Transformers',
        'datasets': 'HuggingFace Datasets', 
        'tokenizers': 'HuggingFace Tokenizers',
        'nltk': 'NLTK',
        'spacy': 'spaCy',
        'gensim': 'Gensim'
    }
    
    all_good = True
    for package, name in nlp_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: NOT FOUND") 
            all_good = False
    
    return all_good


def check_pytorch_functionality():
    """Test basic PyTorch operations."""
    print("\n=== PyTorch Functionality Check ===")
    
    try:
        import torch
        
        # Test basic tensor operations
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        z = torch.mm(x, y)
        print(f"✅ Basic tensor operations: {z.shape}")
        
        # Test autograd
        x.requires_grad_(True)
        loss = (z ** 2).sum()
        loss.backward()
        print(f"✅ Autograd functionality: gradient shape {x.grad.shape}")
        
        # Check for GPU availability
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device}")
        else:
            print("ℹ️  CUDA not available (CPU only)")
        
        # Test RNN module
        rnn = torch.nn.RNN(input_size=10, hidden_size=20, batch_first=True)
        test_input = torch.randn(5, 3, 10)  # (batch, seq, features)
        output, hidden = rnn(test_input)
        print(f"✅ RNN module: output shape {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch functionality error: {e}")
        return False


def check_jupyter_setup():
    """Check Jupyter installation."""
    print("\n=== Jupyter Environment Check ===")
    
    try:
        import jupyter
        import jupyterlab  
        import IPython
        
        print("✅ Jupyter packages installed")
        
        # Check if jupyter lab is in PATH
        result = subprocess.run(['jupyter', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Jupyter commands available")
            print(result.stdout.strip())
        else:
            print("⚠️  Jupyter command not in PATH")
        
        return True
        
    except Exception as e:
        print(f"❌ Jupyter check failed: {e}")
        return False


def check_dataset_access():
    """Check if poetry dataset is accessible."""
    print("\n=== Dataset Access Check ===")
    
    dataset_dir = Path("dataset_poetry")
    if not dataset_dir.exists():
        print("❌ dataset_poetry/ directory not found")
        return False
    
    # Check for key dataset files
    required_files = [
        "expanded_contemporary_poetry.json",
        "expanded_contemporary_poetry_training.txt",
        "dbbc_poetry_collection.json"
    ]
    
    all_good = True
    for filename in required_files:
        filepath = dataset_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024  # KB
            print(f"✅ {filename}: {size:.1f} KB")
        else:
            print(f"❌ {filename}: NOT FOUND")
            all_good = False
    
    return all_good


def check_development_tools():
    """Check development and utility packages."""
    print("\n=== Development Tools Check ===")
    
    dev_packages = {
        'tqdm': 'Progress bars',
        'tensorboard': 'TensorBoard',
        'pytest': 'Testing framework', 
        'black': 'Code formatter',
        'sentence_transformers': 'Sentence embeddings',
        'sacrebleu': 'BLEU scores'
    }
    
    all_good = True
    for package, name in dev_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name}: NOT FOUND (optional)")
    
    return all_good


def main():
    """Run all validation checks."""
    print("🧪 poetryRNN Environment Validation")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Core ML Packages", check_core_packages), 
        ("NLP Packages", check_nlp_packages),
        ("PyTorch Functionality", check_pytorch_functionality),
        ("Jupyter Environment", check_jupyter_setup),
        ("Dataset Access", check_dataset_access),
        ("Development Tools", check_development_tools)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}: EXCEPTION - {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("🏁 VALIDATION SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL" 
        print(f"{name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Environment validation successful!")
        print("Ready to begin neural network development.")
        return True
    else:
        print(f"\n⚠️  {total - passed} checks failed.")
        print("Please address the issues above before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)