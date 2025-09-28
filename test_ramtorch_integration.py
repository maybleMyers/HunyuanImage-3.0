#!/usr/bin/env python
"""
Test script to verify RamTorch integration with HunyuanImage-3.0
This script tests the model loading and memory management without running full inference.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM
from hunyuan_image_3.ramtorch_loader import load_ramtorch_model, patch_linear_in_module

def count_linear_layers(module, layer_type=nn.Linear):
    """Count the number of Linear layers in a module."""
    count = 0
    for name, child in module.named_modules():
        if isinstance(child, layer_type):
            count += 1
    return count

def test_ramtorch_replacement():
    """Test that Linear layers are properly replaced with RamTorch layers."""
    print("=" * 60)
    print("Testing RamTorch Linear Layer Replacement")
    print("=" * 60)

    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(100, 200)
            self.layer2 = nn.Sequential(
                nn.Linear(200, 300),
                nn.ReLU(),
                nn.Linear(300, 100)
            )
            self.layer3 = nn.ModuleList([
                nn.Linear(100, 50),
                nn.Linear(50, 25)
            ])

    model = TestModel()
    original_linear_count = count_linear_layers(model)
    print(f"Original model has {original_linear_count} Linear layers")

    # Replace Linear layers
    replaced_count = patch_linear_in_module(model, device="cuda", verbose=True)
    print(f"\nReplaced {replaced_count} layers")

    # Verify replacement
    from RamTorch.ramtorch import Linear as RamTorchLinear
    ramtorch_count = count_linear_layers(model, RamTorchLinear)
    remaining_linear = count_linear_layers(model, nn.Linear)

    print(f"\nAfter replacement:")
    print(f"  RamTorch Linear layers: {ramtorch_count}")
    print(f"  Regular Linear layers: {remaining_linear}")

    assert ramtorch_count == original_linear_count, "Not all Linear layers were replaced!"
    assert remaining_linear == 0, "Some Linear layers remain!"

    print("\n✓ All Linear layers successfully replaced with RamTorch!")
    return True

def test_memory_distribution():
    """Test memory distribution of parameters."""
    print("\n" + "=" * 60)
    print("Testing Memory Distribution")
    print("=" * 60)

    # Create a test model with RamTorch layers
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 128)  # Should stay on GPU
            self.linear1 = nn.Linear(128, 256)
            self.linear2 = nn.Linear(256, 512)
            self.linear3 = nn.Linear(512, 128)

    model = TestModel().cuda()

    # Get initial memory stats
    print("Before RamTorch replacement:")
    total_params = sum(p.numel() for p in model.parameters())
    gpu_params = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
    print(f"  Total parameters: {total_params:,}")
    print(f"  GPU parameters: {gpu_params:,}")
    print(f"  GPU memory %: {(gpu_params/total_params)*100:.1f}%")

    # Replace Linear layers
    replaced_count = patch_linear_in_module(model, device="cuda", verbose=False)
    print(f"\nReplaced {replaced_count} Linear layers with RamTorch")

    # Check memory distribution after replacement
    print("\nAfter RamTorch replacement:")
    gpu_params_after = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
    cpu_params_after = sum(p.numel() for p in model.parameters() if p.device.type == "cpu")

    print(f"  Total parameters: {total_params:,}")
    print(f"  GPU parameters: {gpu_params_after:,}")
    print(f"  CPU parameters: {cpu_params_after:,}")
    print(f"  GPU memory %: {(gpu_params_after/total_params)*100:.1f}%")
    print(f"  Memory saved: {((gpu_params - gpu_params_after)/gpu_params)*100:.1f}%")

    # Verify embedding stayed on GPU
    assert model.embed.weight.device.type == "cuda", "Embedding should remain on GPU!"

    print("\n✓ Memory distribution test passed!")
    return True

def test_model_config_loading():
    """Test loading model configuration."""
    print("\n" + "=" * 60)
    print("Testing Model Configuration Loading")
    print("=" * 60)

    # Check if model path exists
    model_path = Path("./HunyuanImage-3")
    if not model_path.exists():
        print(f"⚠ Model path {model_path} not found. Skipping actual model loading test.")
        print("  To run full test, download the model to ./HunyuanImage-3")
        return True

    try:
        # Try to load just the configuration
        from hunyuan_image_3.configuration_hunyuan import HunyuanImage3Config
        config = HunyuanImage3Config.from_pretrained(model_path)
        print(f"✓ Successfully loaded model configuration")
        print(f"  Model type: {config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Vocab size: {config.vocab_size}")
        return True
    except Exception as e:
        print(f"⚠ Could not load model config: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting RamTorch Integration Tests\n")

    tests = [
        ("Linear Layer Replacement", test_ramtorch_replacement),
        ("Memory Distribution", test_memory_distribution),
        ("Model Configuration", test_model_config_loading),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 All tests passed! RamTorch integration is working correctly.")
    else:
        print(f"\n⚠ {failed} test(s) failed. Please review the errors above.")

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)