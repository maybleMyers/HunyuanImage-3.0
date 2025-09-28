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
except ImportError:
    raise ImportError(
        "RamTorch not found. Please ensure RamTorch is in the RamTorch directory "
        "or install it via: pip install ramtorch"
    )


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


def load_state_dict_cpu_pinned(model_path: Union[str, Path], device_map: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    """
    Load model state dict directly to CPU with pinned memory for efficient transfer.

    Args:
        model_path: Path to model directory or checkpoint file
        device_map: Optional device map for model parallelism

    Returns:
        State dictionary with CPU-pinned tensors
    """
    model_path = Path(model_path)
    state_dict = {}

    # Find safetensors files
    safetensor_files = list(model_path.glob("*.safetensors"))

    if not safetensor_files:
        # Try loading from pytorch_model.bin
        pytorch_file = model_path / "pytorch_model.bin"
        if pytorch_file.exists():
            print(f"Loading weights from {pytorch_file}")
            checkpoint = torch.load(pytorch_file, map_location="cpu")
            state_dict = checkpoint if isinstance(checkpoint, dict) and "state_dict" not in checkpoint else checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError(f"No model weights found in {model_path}")
    else:
        # Load from safetensors files
        for file_path in safetensor_files:
            print(f"Loading weights from {file_path}")
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # Pin memory for faster CPU-GPU transfers
                    if tensor.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                        tensor = tensor.pin_memory()
                    state_dict[key] = tensor

    return state_dict


def create_model_with_ramtorch(model_class, config: PretrainedConfig, device: str = "cuda", verbose: bool = False, **kwargs):
    """
    Create a model instance with all Linear layers replaced by RamTorch Linear layers.

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

    # Create model normally first (we'll replace Linear layers before loading weights)
    model = model_class(config)

    # Replace all Linear layers with RamTorch Linear
    replaced_count = patch_linear_in_module(model, device=device, verbose=verbose)
    print(f"Replaced {replaced_count} Linear layers with RamTorch Linear layers")

    return model


def load_ramtorch_model(model_class, model_path: Union[str, Path], device: str = "cuda",
                        verbose: bool = False, **kwargs):
    """
    Load a model with RamTorch Linear layers, avoiding GPU memory allocation during loading.

    Args:
        model_class: Model class to load
        model_path: Path to model directory
        device: Target device for computation
        verbose: Print detailed information
        **kwargs: Additional arguments for model initialization

    Returns:
        Model loaded with RamTorch Linear layers
    """
    model_path = Path(model_path)

    # Load configuration
    config = model_class.config_class.from_pretrained(model_path)

    # Create model with RamTorch layers
    model = create_model_with_ramtorch(model_class, config, device=device, verbose=verbose, **kwargs)

    # Load weights directly to CPU-pinned memory
    print("Loading model weights to CPU memory...")
    state_dict = load_state_dict_cpu_pinned(model_path)

    # Custom loading to handle RamTorch Linear layers
    missing_keys = []
    unexpected_keys = []
    mismatched_keys = []

    # Get model state dict for comparison
    model_state = model.state_dict()

    for key in state_dict.keys():
        if key not in model_state:
            unexpected_keys.append(key)

    for name, param in model.named_parameters():
        if name in state_dict:
            with torch.no_grad():
                # Get the tensor from state_dict
                loaded_tensor = state_dict[name]

                # Check shape compatibility
                if param.shape != loaded_tensor.shape:
                    mismatched_keys.append(f"{name}: expected {param.shape}, got {loaded_tensor.shape}")
                    continue

                # For RamTorch Linear layers, weights should stay on CPU
                if "weight" in name or "bias" in name:
                    # Check if this parameter belongs to a RamTorch Linear layer
                    module_name = ".".join(name.split(".")[:-1])
                    try:
                        module = model
                        for part in module_name.split("."):
                            if part:
                                module = getattr(module, part)

                        if isinstance(module, RamTorchLinear):
                            # Keep on CPU for RamTorch layers
                            param.data = loaded_tensor.clone().to(dtype=param.dtype)
                            if param.data.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                                param.data = param.data.share_memory_().pin_memory()
                        else:
                            # Move to device for non-RamTorch layers
                            param.data = loaded_tensor.to(device=device, dtype=param.dtype)
                    except:
                        # Default behavior if we can't determine the module type
                        if param.device.type == "cpu":
                            param.data = loaded_tensor.clone().to(dtype=param.dtype)
                            if param.data.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                                param.data = param.data.share_memory_().pin_memory()
                        else:
                            param.data = loaded_tensor.to(device=device, dtype=param.dtype)
        else:
            missing_keys.append(name)

    # Load buffers (non-parameter tensors)
    for name, buffer in model.named_buffers():
        if name in state_dict:
            buffer.copy_(state_dict[name].to(device))

    # Report loading issues
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    if mismatched_keys:
        print(f"Warning: Shape mismatches: {mismatched_keys[:5]}{'...' if len(mismatched_keys) > 5 else ''}")

    print("Model loaded successfully with RamTorch memory management!")

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