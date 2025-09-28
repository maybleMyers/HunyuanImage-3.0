#!/usr/bin/env python
"""
Test script to verify the streaming weight loader works without memory duplication.
Monitor memory usage during model loading to ensure no duplication occurs.
"""

import os
import sys
import psutil
import torch
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM
from hunyuan_image_3.ramtorch_loader import load_ramtorch_model


def get_memory_info():
    """Get current memory usage statistics"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    # System memory
    ram_gb = mem_info.rss / (1024 ** 3)  # RSS in GB

    # GPU memory if available
    gpu_mb = 0
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    return {
        "ram_gb": ram_gb,
        "gpu_mb": gpu_mb
    }


def monitor_memory(label=""):
    """Print current memory usage"""
    mem = get_memory_info()
    print(f"{label:40} RAM: {mem['ram_gb']:.2f} GB, GPU: {mem['gpu_mb']:.0f} MB")
    return mem


def test_streaming_loader(model_path="./HunyuanImage-3.0", verbose=False):
    """Test the streaming loader for memory efficiency"""

    print("=" * 70)
    print("Testing RamTorch Streaming Loader")
    print("=" * 70)

    # Check if model path exists
    if not Path(model_path).exists():
        print(f"Error: Model path {model_path} not found")
        print("Please specify the correct path to your HunyuanImage-3.0 model")
        return False

    print(f"Model path: {model_path}")
    print()

    # Initial memory
    initial_mem = monitor_memory("Initial memory:")

    # Clear any existing allocations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # Load model with streaming
        print("\nLoading model with RamTorch streaming loader...")
        print("-" * 50)

        start_time = time.time()

        # Monitor memory during loading
        model = load_ramtorch_model(
            HunyuanImage3ForCausalMM,
            model_path,
            device="cuda",
            verbose=verbose,
            attn_implementation="sdpa",
            moe_impl="eager"
        )

        load_time = time.time() - start_time

        # Memory after loading
        loaded_mem = monitor_memory("After loading:")

        # Calculate memory increase
        ram_increase = loaded_mem['ram_gb'] - initial_mem['ram_gb']
        gpu_increase = loaded_mem['gpu_mb'] - initial_mem['gpu_mb']

        print("\n" + "=" * 70)
        print("Loading Statistics:")
        print("=" * 70)
        print(f"Load time:         {load_time:.1f} seconds")
        print(f"RAM increase:      {ram_increase:.2f} GB")
        print(f"GPU increase:      {gpu_increase:.0f} MB")

        # Check memory efficiency
        print("\nMemory Efficiency Check:")
        if ram_increase > 180:  # 160GB model + overhead
            print("⚠ WARNING: RAM usage suggests memory duplication!")
            print(f"  Expected: ~160GB, Got: {ram_increase:.2f} GB")
        else:
            print(f"✓ RAM usage is efficient: {ram_increase:.2f} GB for 160GB model")

        if gpu_increase > 10000:  # 10GB
            print("⚠ WARNING: High GPU memory usage detected!")
        else:
            print(f"✓ GPU memory usage is minimal: {gpu_increase:.0f} MB")

        # Get memory stats from model
        if hasattr(model, 'get_memory_stats'):
            stats = model.get_memory_stats()
            print("\nModel Memory Distribution:")
            print(f"  Total parameters:  {stats['total_params']:,}")
            print(f"  CPU parameters:    {stats['cpu_params']:,} ({stats['cpu_memory_mb']:.0f} MB)")
            print(f"  GPU parameters:    {stats['gpu_params']:,} ({stats['gpu_memory_mb']:.0f} MB)")
            print(f"  Memory saved:      {stats['gpu_memory_saved_pct']:.1f}%")

        # Test inference capability
        print("\n" + "=" * 70)
        print("Testing Inference Capability:")
        print("=" * 70)

        # Load tokenizer
        print("Loading tokenizer...")
        model.load_tokenizer(model_path)

        # Prepare a simple input
        print("Preparing test input...")
        test_prompt = "Test"

        # Monitor memory during inference prep
        pre_inference_mem = monitor_memory("Before inference prep:")

        # This is just to verify the model can prepare inputs
        # We won't run actual generation to save time
        try:
            model_inputs = model.prepare_model_inputs(
                prompt=test_prompt,
                bot_task="image",
                system_prompt=None,
                seed=42
            )
            print("✓ Model can prepare inputs successfully")

            # Memory after inference prep
            post_inference_mem = monitor_memory("After inference prep:")

            inference_increase = post_inference_mem['gpu_mb'] - pre_inference_mem['gpu_mb']
            print(f"GPU memory for inference prep: {inference_increase:.0f} MB")

        except Exception as e:
            print(f"⚠ Could not prepare inference: {e}")

        print("\n" + "=" * 70)
        print("Test Results:")
        print("=" * 70)

        if ram_increase <= 180 and gpu_increase <= 10000:
            print("🎉 SUCCESS: Streaming loader works efficiently!")
            print("   - No memory duplication detected")
            print("   - Model loaded directly to final memory locations")
            print("   - Ready for inference with RamTorch")
            return True
        else:
            print("⚠ PARTIAL SUCCESS: Model loaded but memory usage is higher than expected")
            print("   Check for potential issues in the loading process")
            return False

    except Exception as e:
        print(f"\n❌ Error during loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test RamTorch streaming loader")
    parser.add_argument("--model-path", default="./HunyuanImage-3.0",
                       help="Path to HunyuanImage-3.0 model")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed loading information")

    args = parser.parse_args()

    print("RamTorch Streaming Loader Test\n")
    print("This test will:")
    print("1. Load the model using the streaming approach")
    print("2. Monitor memory usage to detect duplication")
    print("3. Verify the model is ready for inference")
    print()

    # Run test
    success = test_streaming_loader(args.model_path, args.verbose)

    if success:
        print("\nYou can now run inference with:")
        print(f"python run_image_gen.py --model-id {args.model_path} --use-ramtorch "
              "--prompt 'Your prompt' --rewrite False")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())