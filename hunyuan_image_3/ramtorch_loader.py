# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
RamTorch Integration for HunyuanImage-3.0
Enables memory-efficient inference by replacing nn.Linear layers with RamTorch Linear layers
that keep weights in CPU memory and transfer them to GPU on-demand.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union
from transformers import PretrainedConfig
from safetensors import safe_open
from safetensors.torch import load_file

# Add RamTorch to path
ramtorch_path = Path(__file__).parent.parent / "RamTorch"
if ramtorch_path.exists():
    sys.path.insert(0, str(ramtorch_path))

try:
    from ramtorch import Linear as RamTorchLinear
    # The actual class might be CPUBouncingLinear
    from ramtorch.modules.linear import CPUBouncingLinear
except ImportError:
    raise ImportError(
        "RamTorch not found. Please ensure RamTorch is in the RamTorch directory "
        "or install it via: pip install ramtorch"
    )


def monkey_patch_linear(device="cuda", verbose=False):
    """
    Temporarily replace nn.Linear with RamTorch Linear during model construction.

    This ensures Linear layers are created as RamTorch Linear from the start,
    avoiding duplicate memory usage during model loading.

    Args:
        device: Target device for computation
        verbose: Print creation information

    Returns:
        Original nn.Linear class (for restoration)
    """
    original_linear = nn.Linear

    class RamTorchLinearWrapper(nn.Module):
        """Wrapper that creates RamTorch Linear with nn.Linear interface"""
        def __new__(cls, in_features, out_features, bias=True, device=None, dtype=None):
            # Create RamTorch Linear instead of regular Linear
            if verbose:
                print(f"  Creating RamTorch Linear({in_features}, {out_features}, bias={bias})")

            # RamTorch expects device as computation target (cuda)
            # Weights will be on CPU, computation on specified device
            return RamTorchLinear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                dtype=dtype if dtype is not None else torch.float32,
                device=device if device is not None else "cuda"
            )

    # Replace nn.Linear globally
    nn.Linear = RamTorchLinearWrapper

    return original_linear


def patch_linear_in_module(module: nn.Module, device: str = "cuda", verbose: bool = False) -> int:
    """
    Recursively replace all nn.Linear layers in a module with RamTorch Linear layers.

    Args:
        module: PyTorch module to patch
        device: Target device for computation (default: "cuda")
        verbose: Print replacement information

    Returns:
        Number of layers replaced
    """
    replaced_count = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            # Create RamTorch Linear with same configuration
            ramtorch_linear = RamTorchLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=(child.bias is not None),
                dtype=child.weight.dtype if hasattr(child, 'weight') else torch.float32,
                device=device
            )

            # Replace the module
            setattr(module, name, ramtorch_linear)
            replaced_count += 1

            if verbose:
                param_count = child.in_features * child.out_features
                if child.bias is not None:
                    param_count += child.out_features
                print(f"  Replaced {name}: nn.Linear({child.in_features}, {child.out_features}) "
                      f"-> RamTorch.Linear [~{param_count:,} params]")
        else:
            # Recursively patch child modules
            replaced_count += patch_linear_in_module(child, device, verbose)

    return replaced_count


def load_weights_streaming(model, model_path: Union[str, Path], verbose: bool = False) -> set:
    """
    Load weights one-by-one directly into model parameters to avoid memory duplication.

    This function loads weights from disk and assigns them directly to model parameters
    without creating intermediate copies, minimizing memory usage.

    Args:
        model: PyTorch model with parameters to fill
        model_path: Path to model directory containing weight files
        verbose: Print detailed loading information

    Returns:
        Set of successfully loaded parameter keys
    """
    import gc
    model_path = Path(model_path)

    # Find weight files (safetensors preferred, then .bin)
    weight_files = list(model_path.glob("*.safetensors"))
    if not weight_files:
        bin_files = list(model_path.glob("*.bin"))
        if bin_files:
            weight_files = bin_files
        else:
            raise FileNotFoundError(f"No weight files found in {model_path}")

    loaded_keys = set()
    total_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"Loading weights from {len(weight_files)} file(s)")
        print(f"Model has {total_params:,} total parameters")

    for weight_file in weight_files:
        if verbose:
            print(f"\nLoading {weight_file.name}...")

        if weight_file.suffix == ".safetensors":
            # Use safetensors for efficient loading
            with safe_open(weight_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    try:
                        # Get tensor without loading all at once
                        tensor = f.get_tensor(key)

                        # Navigate to the parameter in the model
                        if _set_module_parameter(model, key, tensor, verbose):
                            loaded_keys.add(key)

                        # Explicitly delete tensor to free memory
                        del tensor

                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Could not load {key}: {e}")
        else:
            # Handle .bin files
            checkpoint = torch.load(weight_file, map_location="cpu")
            if isinstance(checkpoint, dict):
                # Handle different checkpoint formats
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]

                for key, tensor in checkpoint.items():
                    try:
                        if _set_module_parameter(model, key, tensor, verbose):
                            loaded_keys.add(key)
                        del tensor
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Could not load {key}: {e}")

            # Clear the checkpoint from memory
            del checkpoint

        # Force garbage collection after each file
        gc.collect()

        if verbose:
            loaded_percent = (len(loaded_keys) / len(list(model.state_dict().keys()))) * 100
            print(f"  Progress: {len(loaded_keys)} parameters loaded ({loaded_percent:.1f}%)")

    # Final garbage collection
    gc.collect()

    return loaded_keys


def _set_module_parameter(model, key: str, tensor: torch.Tensor, verbose: bool = False) -> bool:
    """
    Helper function to set a specific parameter in the model.

    Args:
        model: PyTorch model
        key: Parameter key (e.g., "layers.0.weight")
        tensor: Tensor to assign
        verbose: Print debug information

    Returns:
        True if successfully set, False otherwise
    """
    try:
        # Split the key into parts
        keys = key.split('.')

        # Navigate to the parent module
        obj = model
        for k in keys[:-1]:
            obj = getattr(obj, k)

        # Get the parameter name
        param_name = keys[-1]

        # Set the parameter
        if hasattr(obj, param_name):
            param = getattr(obj, param_name)

            with torch.no_grad():
                if isinstance(param, nn.Parameter):
                    # Check if shapes match
                    if param.shape != tensor.shape:
                        if verbose:
                            print(f"  Shape mismatch for {key}: expected {param.shape}, got {tensor.shape}")
                        return False

                    # For CPU parameters (RamTorch), pin memory for faster transfers
                    if param.device.type == "cpu":
                        if tensor.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                            param.data = tensor.pin_memory()
                        else:
                            param.data = tensor
                    else:
                        # For GPU parameters (embeddings, etc.), move to device
                        param.data = tensor.to(param.device)

                    return True
                elif isinstance(obj, nn.Module) and param_name == "weight" or param_name == "bias":
                    # Handle buffer or other attributes
                    setattr(obj, param_name, tensor)
                    return True

        return False

    except Exception as e:
        if verbose:
            print(f"  Error setting {key}: {e}")
        return False


def create_model_with_ramtorch(model_class, config: PretrainedConfig, device: str = "cuda", verbose: bool = False, **kwargs):
    """
    Create a model instance with all Linear layers created as RamTorch Linear from the start.

    This uses monkey-patching to replace nn.Linear during model construction,
    avoiding duplicate memory usage.

    Args:
        model_class: Model class to instantiate
        config: Model configuration
        device: Target device for computation
        verbose: Print detailed information
        **kwargs: Additional model initialization arguments

    Returns:
        Model with RamTorch Linear layers
    """
    print("Creating model with RamTorch Linear layers...")

    # Update config with additional kwargs
    if 'attn_implementation' in kwargs:
        config._attn_implementation = kwargs['attn_implementation']
    if 'moe_impl' in kwargs:
        config.moe_impl = kwargs['moe_impl']

    # Monkey-patch nn.Linear BEFORE model creation
    original_linear = monkey_patch_linear(device=device, verbose=verbose)

    # Also patch the linear function in hunyuan module if it exists
    hunyuan_module = sys.modules.get('hunyuan_image_3.hunyuan')
    original_hunyuan_linear = None
    if hunyuan_module and hasattr(hunyuan_module, 'linear'):
        original_hunyuan_linear = hunyuan_module.linear
        hunyuan_module.linear = lambda *args, **kwargs: nn.Linear(*args, **kwargs)

    try:
        # Create model - all nn.Linear calls will create RamTorch Linear
        model = model_class(config)

        # Count RamTorch layers created (check both possible class names)
        ramtorch_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (RamTorchLinear, CPUBouncingLinear)):
                ramtorch_count += 1

        print(f"Created model with {ramtorch_count} RamTorch Linear layers")

    finally:
        # Restore original nn.Linear
        nn.Linear = original_linear

        # Restore hunyuan linear function if it was patched
        if original_hunyuan_linear is not None:
            hunyuan_module.linear = original_hunyuan_linear

    return model


def load_ramtorch_model(model_class, model_path: Union[str, Path], device: str = "cuda",
                        verbose: bool = False, **kwargs):
    """
    Load a model with RamTorch Linear layers using streaming to avoid memory duplication.

    This function creates a model with RamTorch Linear layers and loads weights
    one at a time directly into model parameters, preventing memory duplication.

    Args:
        model_class: Model class to load
        model_path: Path to model directory
        device: Target device for computation
        verbose: Print detailed information
        **kwargs: Additional arguments for model initialization

    Returns:
        Model loaded with RamTorch Linear layers
    """
    import gc
    model_path = Path(model_path)

    # Load configuration
    print("Loading model configuration...")
    config = model_class.config_class.from_pretrained(model_path)

    # Create model with RamTorch layers
    print("Creating model with RamTorch Linear layers...")
    model = create_model_with_ramtorch(model_class, config, device=device, verbose=verbose, **kwargs)

    # Clear any existing GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Load weights using streaming approach
    print("Loading model weights (streaming mode to minimize memory)...")
    loaded_keys = load_weights_streaming(model, model_path, verbose=verbose)

    # Check for missing parameters
    model_params = set(model.state_dict().keys())
    missing_keys = model_params - loaded_keys

    if missing_keys:
        # Some keys might be buffers or have different names
        actual_missing = []
        for key in missing_keys:
            try:
                # Check if parameter exists and has been initialized
                param = model.state_dict()[key]
                if torch.all(param == 0) or torch.all(torch.isnan(param)):
                    actual_missing.append(key)
            except:
                actual_missing.append(key)

        if actual_missing and verbose:
            print(f"Warning: {len(actual_missing)} parameters not loaded: {actual_missing[:5]}{'...' if len(actual_missing) > 5 else ''}")

    print(f"Successfully loaded {len(loaded_keys)} parameters")
    print("Model loaded with RamTorch memory management!")

    # Add get_memory_stats method to the model
    model.get_memory_stats = lambda: get_memory_stats(model)

    # Set model to evaluation mode
    model.eval()

    return model


def get_memory_stats(model):
    """
    Get memory statistics for the model.

    Args:
        model: PyTorch model

    Returns:
        Dict with memory usage information
    """
    total_params = 0
    gpu_params = 0
    cpu_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.device.type == "cuda":
            gpu_params += num_params
        else:
            cpu_params += num_params

    # Convert to MB (assuming float32 by default, adjust if using different dtype)
    bytes_per_param = 4  # float32
    if hasattr(model, 'dtype'):
        if model.dtype == torch.float16 or model.dtype == torch.bfloat16:
            bytes_per_param = 2

    total_mb = (total_params * bytes_per_param) / (1024 * 1024)
    gpu_mb = (gpu_params * bytes_per_param) / (1024 * 1024)
    cpu_mb = (cpu_params * bytes_per_param) / (1024 * 1024)

    return {
        "total_params": total_params,
        "gpu_params": gpu_params,
        "cpu_params": cpu_params,
        "total_memory_mb": total_mb,
        "gpu_memory_mb": gpu_mb,
        "cpu_memory_mb": cpu_mb,
        "gpu_memory_saved_pct": (1 - gpu_mb / total_mb) * 100 if total_mb > 0 else 0
    }


class RamTorchModelMixin:
    """
    Mixin class to add RamTorch loading capability to existing model classes.
    """

    @classmethod
    def from_pretrained_ramtorch(cls, pretrained_model_name_or_path, device="cuda",
                                 verbose=False, **kwargs):
        """
        Load a pretrained model with RamTorch Linear layers.

        This method replaces all nn.Linear layers with RamTorch Linear layers
        during model loading to avoid GPU memory allocation.
        """
        return load_ramtorch_model(
            cls,
            pretrained_model_name_or_path,
            device=device,
            verbose=verbose,
            **kwargs
        )