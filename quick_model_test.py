#!/usr/bin/env python3
"""
Quick Model Test - Test model loading without full preprocessing
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')

from poetry_rnn.models import RNNAutoencoder

def test_model_loading(model_path: str):
    """Test loading the trained model."""
    print(f"🎭 Quick Model Test")
    print("=" * 50)
    print(f"📂 Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Load checkpoint with fixed weights_only parameter
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ Checkpoint loaded successfully!")
        
        # Show checkpoint contents
        print(f"\n📊 Checkpoint Contents:")
        for key, value in checkpoint.items():
            if isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} keys")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: tensor {value.shape}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
        
        # Extract model configuration
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
            print(f"\n🏗️  Model Configuration:")
            for key, value in model_config.items():
                print(f"  {key}: {value}")
        else:
            print("⚠️  No config found in checkpoint, using default")
            model_config = {
                'input_dim': 300,
                'hidden_dim': 512,
                'bottleneck_dim': 128,
                'rnn_type': 'LSTM',
                'num_layers': 2,
                'dropout': 0.2,
                'use_attention': True,
                'attention_heads': 4,
                'use_positional_encoding': True,
                'use_optimized_attention': True
            }
        
        # Create model
        model = RNNAutoencoder(
            input_size=model_config.get('input_dim', 300),
            hidden_size=model_config.get('hidden_dim', 512),
            bottleneck_dim=model_config.get('bottleneck_dim', 128),
            rnn_type=model_config.get('rnn_type', 'LSTM'),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.2),
            use_attention=model_config.get('use_attention', True),
            attention_heads=model_config.get('attention_heads', 4),
            use_positional_encoding=model_config.get('use_positional_encoding', True),
            use_optimized_attention=model_config.get('use_optimized_attention', True)
        )
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n✅ Model loaded and ready!")
        print(f"   Architecture: {model_config.get('input_dim', 300)}D → {model_config.get('hidden_dim', 512)}D → {model_config.get('bottleneck_dim', 128)}D")
        print(f"   RNN Type: {model_config.get('rnn_type', 'LSTM')}")
        print(f"   Attention Heads: {model_config.get('attention_heads', 4)}")
        print(f"   Total Parameters: {total_params:,}")
        
        # Show training performance if available
        if 'final_cosine_similarity' in checkpoint:
            print(f"   Final Training Performance: {checkpoint['final_cosine_similarity']:.4f} cosine similarity")
        elif 'best_cosine_similarity' in checkpoint:
            print(f"   Best Training Performance: {checkpoint['best_cosine_similarity']:.4f} cosine similarity")
        
        # Test forward pass with dummy data
        print(f"\n🧪 Testing forward pass...")
        batch_size, seq_len = 2, 20
        test_batch = {
            'input_sequences': torch.randn(batch_size, seq_len, 300),
            'attention_mask': torch.ones(batch_size, seq_len).bool()
        }
        
        with torch.no_grad():
            output = model(test_batch)
        
        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {test_batch['input_sequences'].shape}")
        print(f"   Output shape: {output['reconstructed'].shape}")
        print(f"   Bottleneck shape: {output['bottleneck'].shape}")
        
        # Compute cosine similarity for test (handle different sequence lengths)
        import torch.nn.functional as F
        input_seq = test_batch['input_sequences']  # [2, 20, 300]
        output_seq = output['reconstructed']       # [2, 50, 300] - model outputs different length
        
        # Take only the first 20 tokens to match input
        output_trimmed = output_seq[:, :input_seq.size(1), :]  # [2, 20, 300]
        
        cosine_sim = F.cosine_similarity(
            input_seq.reshape(-1, 300),
            output_trimmed.reshape(-1, 300),
            dim=1
        ).mean()
        print(f"   Test Cosine Similarity: {cosine_sim.item():.4f}")
        print(f"   Note: Output length {output_seq.size(1)} vs input length {input_seq.size(1)} (model uses max sequence length)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "attention_optimized_model.pth"
    success = test_model_loading(model_path)
    exit(0 if success else 1)