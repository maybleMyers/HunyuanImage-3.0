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
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union
from transformers import PretrainedConfig
from transformers.generation.utils import GenerationConfig
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


def load_weights_from_disk_to_meta(model, model_path: Union[str, Path], verbose: bool = False) -> set:
    """
    Load weights from disk directly into a meta model, converting from meta to CPU/GPU.

    Args:
        model: Model with meta device parameters
        model_path: Path to model weights
        verbose: Print detailed information

    Returns:
        Set of loaded parameter keys
    """
    import gc
    model_path = Path(model_path)

    # Find weight files
    weight_files = list(model_path.glob("*.safetensors"))
    if not weight_files:
        weight_files = list(model_path.glob("*.bin"))
        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {model_path}")

    loaded_keys = set()

    for weight_file in weight_files:
        if verbose:
            print(f"Loading {weight_file.name}...")

        if weight_file.suffix == ".safetensors":
            with safe_open(weight_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    try:
                        # Get tensor from file
                        tensor = f.get_tensor(key)

                        # Set in model (converting from meta to actual device)
                        if _set_module_tensor_from_meta(model, key, tensor, verbose):
                            loaded_keys.add(key)

                        del tensor
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Could not load {key}: {e}")
        else:
            # Handle .bin files
            checkpoint = torch.load(weight_file, map_location="cpu")
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]

                for key, tensor in checkpoint.items():
                    try:
                        if _set_module_tensor_from_meta(model, key, tensor, verbose):
                            loaded_keys.add(key)
                        del tensor
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Could not load {key}: {e}")

            del checkpoint

        gc.collect()

    return loaded_keys


def _set_module_tensor_from_meta(model, key: str, tensor: torch.Tensor, verbose: bool = False) -> bool:
    """
    Set a tensor in a model that has meta device parameters.

    Args:
        model: Model with meta parameters
        key: Parameter key
        tensor: Tensor to set
        verbose: Print debug info

    Returns:
        True if successful
    """
    try:
        keys = key.split('.')
        obj = model
        for k in keys[:-1]:
            obj = getattr(obj, k)

        param_name = keys[-1]

        if hasattr(obj, param_name):
            param = getattr(obj, param_name)

            if isinstance(param, nn.Parameter):
                # Replace meta parameter with actual tensor
                # Avoid copy - convert dtype only if needed
                if tensor.dtype != param.dtype:
                    tensor = tensor.to(dtype=param.dtype)
                # Direct assignment, no pinning
                new_param = nn.Parameter(tensor)
                setattr(obj, param_name, new_param)
                return True

        return False

    except Exception as e:
        if verbose:
            print(f"  Error setting {key}: {e}")
        return False


def convert_to_ramtorch_post_load(model, device: str = "cuda", verbose: bool = False, max_gpu_gb: float = 30.0):
    """
    Convert nn.Linear layers to RamTorch AFTER weights are already loaded.
    This avoids the double allocation issue.

    For 32GB VRAM, we keep only essential layers on GPU to fit within memory constraints.

    Args:
        model: Model with loaded weights
        device: Computation device for RamTorch
        verbose: Print conversion details
        max_gpu_gb: Maximum GPU memory to use (default 30GB for safety on 32GB cards)
    """
    import re
    converted_count = 0
    skipped_count = 0
    gpu_memory_used = 0.0

    # Essential components that MUST stay on GPU (estimated sizes from analysis)
    essential_gpu_layers = {
        'wte': 1.02,  # Embeddings - 1GB
        'embed': 0.1,  # Other embeddings
        'norm': 0.01,  # Normalizations - tiny
        'ln': 0.01,    # Layer norms - tiny
    }

    # Calculate GPU memory budget
    gpu_budget = max_gpu_gb
    for key, size in essential_gpu_layers.items():
        gpu_budget -= size

    print(f"GPU Memory Budget: {max_gpu_gb:.1f} GB total, {gpu_budget:.1f} GB available after essentials")

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Debug: Show ALL Linear layers to find shared_mlp
            if 'shared_mlp' in name and 'down_proj' in name:
                print(f"FOUND SHARED_MLP.DOWN_PROJ: {name}, shape: {module.weight.shape}")

            # Extract layer index if present
            layer_idx = None
            layer_match = re.search(r'layers\.(\d+)', name)
            if layer_match:
                layer_idx = int(layer_match.group(1))

            # Check if this is an essential layer that must stay on GPU
            is_essential = any(key in name.lower() for key in essential_gpu_layers.keys())

            if is_essential:
                # Keep embeddings and normalizations on GPU
                if verbose:
                    print(f"  Keeping {name}: Linear({module.in_features}, {module.out_features}) on GPU (essential)")
                skipped_count += 1
                module.to(device)
                continue

            # For 32GB limit, we can only keep a few attention layers on GPU
            # Prioritize early layers (0-2) for initial processing
            if layer_idx is not None and layer_idx <= 2:
                if 'o_proj' in name:
                    # Keep o_proj for first few layers for stability
                    estimated_size_gb = (module.in_features * module.out_features * 2) / (1024**3)  # bfloat16
                    if gpu_memory_used + estimated_size_gb < gpu_budget:
                        if verbose:
                            print(f"  Keeping {name}: Linear({module.in_features}, {module.out_features}) on GPU (early o_proj)")
                        skipped_count += 1
                        module.to(device)
                        gpu_memory_used += estimated_size_gb
                        continue

            # Everything else goes to RamTorch, including ALL down_proj layers
            # Handle down_proj with dimension mismatch
            # Also handle shared_mlp layers which may have dimension issues
            if 'down_proj' in name or 'shared_mlp' in name:
                # These have dimension mismatches, need special handling
                ramtorch_layer = create_ramtorch_from_loaded(module, device, handle_mismatch=True, layer_name=name)
                if verbose or 'shared_mlp' in name:  # Always show shared_mlp for debugging
                    print(f"  Converted {name}: Linear({module.in_features}, {module.out_features}) -> RamTorch (with mismatch handling)")
            else:
                # Regular conversion
                ramtorch_layer = create_ramtorch_from_loaded(module, device, handle_mismatch=False, layer_name=name)
                if verbose and converted_count < 10:  # Only show first 10 to avoid spam
                    print(f"  Converted {name}: Linear({module.in_features}, {module.out_features}) -> RamTorch")

            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
            child_name = name.split('.')[-1]
            parent = model if parent_name == '' else model.get_submodule(parent_name)

            # Replace in parent
            setattr(parent, child_name, ramtorch_layer)
            converted_count += 1

    print(f"Converted {converted_count} Linear layers to RamTorch")
    print(f"Kept {skipped_count} layers on GPU (essentials + selected for performance)")
    print(f"Estimated GPU memory usage: {gpu_memory_used + sum(essential_gpu_layers.values()):.2f} GB")

    # Force garbage collection to free any temporary tensors
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_ramtorch_from_loaded(linear_module: nn.Linear, device: str = "cuda", handle_mismatch: bool = False, layer_name: str = ""):
    """
    Create a RamTorch Linear layer from an already loaded nn.Linear.
    This transfers the weights without creating duplicates.

    Args:
        linear_module: Loaded nn.Linear module
        device: Computation device
        handle_mismatch: If True, handle dimension mismatches in forward pass
        layer_name: Name of the layer for debugging and special handling

    Returns:
        RamTorch Linear layer with transferred weights
    """
    # Create a custom RamTorch layer that accepts pre-loaded weights
    class LoadedRamTorchLinear(CPUBouncingLinear):
        def __init__(self, weight, bias, device, layer_name="", is_down_proj_mismatch=False):
            # Skip the parent __init__ to avoid weight initialization
            nn.Module.__init__(self)
            self.layer_name = layer_name
            self.device = device
            self.is_down_proj_mismatch = is_down_proj_mismatch

            # For down_proj layers with SwiGLU mismatch, slice the weight
            # The weight is [out_features, 6144] but we only need [out_features, 3072]
            if is_down_proj_mismatch:
                print(f"    DEBUG in __init__: is_down_proj_mismatch={is_down_proj_mismatch}, weight.shape={weight.shape}")
                if weight.shape[1] == 6144:
                    # Only use the first 3072 columns of the weight matrix
                    print(f"    SLICING weight from {weight.shape} to [{weight.shape[0]}, 3072]")
                    weight = weight[:, :3072].contiguous()
                    print(f"    After slicing: weight.shape={weight.shape}")
                elif weight.shape[1] == 4096:
                    # For shared_mlp down_proj, slice differently
                    print(f"    SLICING weight from {weight.shape} to [{weight.shape[0]}, 3072]")
                    weight = weight[:, :3072].contiguous()
                    print(f"    After slicing: weight.shape={weight.shape}")

            # Store actual weight dimensions after potential slicing
            self.in_features = weight.shape[1]
            self.out_features = weight.shape[0]

            # Direct assignment - NO share_memory to avoid file descriptor issues, NO pinning to avoid copies
            # Use the tensors as-is if already on CPU, otherwise move them
            if weight.is_cpu:
                self.weight = nn.Parameter(weight)
            else:
                self.weight = nn.Parameter(weight.cpu())

            if bias is not None:
                if bias.is_cpu:
                    self.bias = nn.Parameter(bias)
                else:
                    self.bias = nn.Parameter(bias.cpu())
            else:
                self.bias = None

        def forward(self, x):
            """Forward pass - weight dimensions should now match after slicing."""
            # Call the original RamTorch forward method from CPUBouncingLinear
            # This handles the weight transfer from CPU to GPU
            from ramtorch.modules.linear import BouncingLinearFn
            return BouncingLinearFn.apply(x, self.weight, self.bias, self.device)

    # Pass the actual parameter data (not cloned)
    weight_data = linear_module.weight.data
    bias_data = linear_module.bias.data if linear_module.bias is not None else None

    # Check if this is a down_proj layer with the SwiGLU dimension mismatch
    # Also check for shared_mlp layers which may have similar issues
    is_down_proj_mismatch = False
    if 'down_proj' in layer_name or 'shared_mlp' in layer_name:
        # Always print for debugging
        print(f"DEBUG: Processing layer {layer_name}")
        print(f"  Weight shape: {weight_data.shape}")
        print(f"  Handle mismatch: {handle_mismatch}")

        if handle_mismatch:
            # Check if weight has the problematic shape [out_features, 6144] or [out_features, 4096]
            if weight_data.shape[1] == 6144:
                is_down_proj_mismatch = True
                print(f"  WILL FIX: Slicing weight from [{weight_data.shape[0]}, 6144] to [{weight_data.shape[0]}, 3072]")
            elif weight_data.shape[1] == 4096:
                # This might be the shared_mlp down_proj which has different dimensions
                is_down_proj_mismatch = True
                print(f"  WILL FIX: Weight has shape [{weight_data.shape[0]}, 4096] - slicing to 3072")
            else:
                print(f"  No mismatch detected for shape {weight_data.shape}")

    # Create the layer - the weights will be moved inside __init__ if needed
    layer = LoadedRamTorchLinear(weight_data, bias_data, device, layer_name=layer_name, is_down_proj_mismatch=is_down_proj_mismatch)

    # Clear the original module's weights to free memory
    del linear_module.weight
    if linear_module.bias is not None:
        del linear_module.bias

    return layer


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

                    # For CPU parameters (RamTorch), no pinning to avoid copies
                    if param.device.type == "cpu":
                        param.data = tensor  # Direct assignment, no pinning
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
    Load a model with RamTorch Linear layers without memory duplication.

    This function:
    1. Creates model on meta device (no memory allocation)
    2. Loads weights directly from disk
    3. Converts nn.Linear to RamTorch after weights are loaded

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

    # Update config with kwargs
    if 'attn_implementation' in kwargs:
        config._attn_implementation = kwargs['attn_implementation']
    if 'moe_impl' in kwargs:
        config.moe_impl = kwargs['moe_impl']

    # Step 1: Create model on meta device (no memory allocation)
    print("Creating model structure on meta device (no memory allocated)...")
    with torch.device('meta'):
        model = model_class(config)

    # Step 2: Load weights directly from disk (streaming, no duplication)
    print("Loading model weights from disk (streaming mode)...")
    loaded_keys = load_weights_from_disk_to_meta(model, model_path, verbose=verbose)

    # Step 3: Convert nn.Linear layers to RamTorch AFTER weights are loaded
    print("Converting Linear layers to RamTorch (memory-efficient mode)...")
    convert_to_ramtorch_post_load(model, device=device, verbose=verbose)

    # Clear any GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

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

    # Load generation_config.json if it exists
    generation_config_path = model_path / "generation_config.json"
    if generation_config_path.exists():
        print("Loading generation config...")
        with open(generation_config_path, 'r') as f:
            generation_config_dict = json.load(f)
        model.generation_config = GenerationConfig(**generation_config_dict)
    else:
        print("Warning: generation_config.json not found, using default generation config")
        model.generation_config = GenerationConfig()

    # Move non-RamTorch components to GPU
    print("Moving non-RamTorch components to GPU...")

    # Specifically move known components that need to be on GPU
    # These are typically embeddings, layer norms, and other non-Linear layers
    components_to_move = []

    # Find and move embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'wte'):
        model.model.wte = model.model.wte.to(device)
        components_to_move.append('model.wte (embeddings)')

    # Move vision model if it exists
    if hasattr(model, 'vision_model'):
        model.vision_model = model.vision_model.to(device)
        components_to_move.append('vision_model')

    # Move vision aligner if it exists
    if hasattr(model, 'vision_aligner'):
        model.vision_aligner = model.vision_aligner.to(device)
        components_to_move.append('vision_aligner')

    # Move VAE if it exists
    if hasattr(model, 'vae'):
        model.vae = model.vae.to(device)
        components_to_move.append('vae')

    # Move timestep embedders if they exist
    if hasattr(model, 'timestep_emb'):
        model.timestep_emb = model.timestep_emb.to(device)
        components_to_move.append('timestep_emb')
    if hasattr(model, 'time_embed'):
        model.time_embed = model.time_embed.to(device)
        components_to_move.append('time_embed')
    if hasattr(model, 'time_embed_2'):
        model.time_embed_2 = model.time_embed_2.to(device)
        components_to_move.append('time_embed_2')

    # Move patch_embed and final_layer if they exist
    if hasattr(model, 'patch_embed'):
        model.patch_embed = model.patch_embed.to(device)
        components_to_move.append('patch_embed')
    if hasattr(model, 'final_layer'):
        model.final_layer = model.final_layer.to(device)
        components_to_move.append('final_layer')

    # Move all LayerNorm and RMSNorm layers to GPU
    for name, module in model.named_modules():
        if 'norm' in module.__class__.__name__.lower() or 'Norm' in module.__class__.__name__:
            module.to(device)
            if verbose and name not in components_to_move:
                components_to_move.append(f'{name} ({module.__class__.__name__})')

    if verbose and components_to_move:
        print(f"  Moved to {device}: {', '.join(components_to_move[:5])}" +
              (f" and {len(components_to_move)-5} more" if len(components_to_move) > 5 else ""))

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