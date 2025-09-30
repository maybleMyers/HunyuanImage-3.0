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

import gc
import torch
import time
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import psutil


class OffloadManager:
    """
    Intelligent offloading manager for HunyuanImage-3.0 model layers.
    Dynamically moves layers between CPU and GPU to maximize GPU utilization
    while preventing OOM errors.
    """

    def __init__(
        self,
        model,
        strategy: str = "sequential",
        memory_threshold: float = 0.85,
        prefetch_distance: int = 2,
        min_gpu_layers: int = 2,
        max_gpu_layers: int = 8,
        verbose: bool = False
    ):
        """
        Initialize the offload manager.

        Args:
            model: The HunyuanImage3 model
            strategy: Offloading strategy ("sequential", "memory_aware", "performance")
            memory_threshold: Maximum VRAM usage percentage before offloading
            prefetch_distance: Number of layers to prefetch
            min_gpu_layers: Minimum number of layers to keep on GPU
            max_gpu_layers: Maximum number of layers to keep on GPU
            verbose: Print debug information
        """
        self.model = model
        self.strategy = strategy
        self.memory_threshold = memory_threshold
        self.prefetch_distance = prefetch_distance
        self.min_gpu_layers = min_gpu_layers
        self.max_gpu_layers = max_gpu_layers
        self.verbose = verbose

        # Track layer locations
        self.layer_devices = {}
        self.gpu_layers = OrderedDict()  # Ordered dict to track LRU

        # Performance metrics
        self.transfer_times = []
        self.gpu_utilization = []

        # Initialize layer tracking
        self._initialize_layers()

    def _initialize_layers(self):
        """Initialize layer device tracking."""
        # Track all model layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for i, layer in enumerate(self.model.model.layers):
                # Get current device
                device = next(layer.parameters()).device
                self.layer_devices[f'model.layers.{i}'] = device

                if device.type == 'cuda':
                    self.gpu_layers[f'model.layers.{i}'] = layer

        # Track other components
        components = ['vae', 'vision_model', 'vision_aligner', 'timestep_emb',
                     'patch_embed', 'time_embed', 'final_layer', 'time_embed_2',
                     'model.wte', 'model.ln_f', 'lm_head']

        for comp_name in components:
            comp = self._get_component(comp_name)
            if comp is not None:
                device = next(comp.parameters()).device if hasattr(comp, 'parameters') else torch.device('cpu')
                self.layer_devices[comp_name] = device
                if device.type == 'cuda':
                    self.gpu_layers[comp_name] = comp

    def _get_component(self, name: str):
        """Get a component from the model by name."""
        parts = name.split('.')
        obj = self.model
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj

    def _set_component_device(self, name: str, device: torch.device):
        """Move a component to a specific device."""
        comp = self._get_component(name)
        if comp is not None:
            # Check if the component is on the meta device
            try:
                is_meta = next(comp.parameters()).is_meta
            except StopIteration:  # handle modules with no parameters
                is_meta = False

            if is_meta and hasattr(comp, "_hf_hook"):
                # This module is a meta-proxy for offloaded weights.
                # Instruct the accelerate hook to load weights directly onto the target device.
                comp._hf_hook.execution_device = device
                # Manually trigger the hook to materialize the module.
                comp._hf_hook.pre_forward(comp)
                # After this, `comp` is a fully materialized module on the target `device`.

            comp = comp.to(device)
            # Update the reference in the model
            parts = name.split('.')
            if len(parts) == 1:
                setattr(self.model, name, comp)
            elif len(parts) == 2:
                parent = getattr(self.model, parts[0])
                setattr(parent, parts[1], comp)
            elif len(parts) == 3:
                parent = getattr(self.model, parts[0])
                parent = getattr(parent, parts[1])
                setattr(parent, parts[2], comp)

    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current GPU memory usage in GB and percentage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            percentage = allocated / total
            return allocated, percentage
        return 0.0, 0.0

    def should_offload(self) -> bool:
        """Check if we should offload layers based on memory usage."""
        _, mem_percentage = self.get_memory_usage()
        return mem_percentage > self.memory_threshold and len(self.gpu_layers) > self.min_gpu_layers

    def offload_layer(self, layer_name: str):
        """Offload a specific layer to CPU."""
        if layer_name in self.gpu_layers:
            start_time = time.time()

            # Move to CPU
            self._set_component_device(layer_name, torch.device('cpu'))

            # Update tracking
            del self.gpu_layers[layer_name]
            self.layer_devices[layer_name] = torch.device('cpu')

            # Clear cache
            torch.cuda.empty_cache()

            transfer_time = time.time() - start_time
            self.transfer_times.append(transfer_time)

            if self.verbose:
                mem_gb, mem_pct = self.get_memory_usage()
                print(f"Offloaded {layer_name} to CPU in {transfer_time:.3f}s. "
                      f"GPU memory: {mem_gb:.2f}GB ({mem_pct:.1%})")

    def load_layer(self, layer_name: str):
        """Load a specific layer to GPU."""
        if layer_name not in self.gpu_layers and layer_name in self.layer_devices:
            # Check if we need to make space
            while self.should_offload() and len(self.gpu_layers) > 0:
                # Offload least recently used layer
                lru_layer = next(iter(self.gpu_layers))
                self.offload_layer(lru_layer)

            start_time = time.time()

            # Move to GPU
            self._set_component_device(layer_name, torch.device('cuda'))

            # Update tracking (add to end for LRU)
            comp = self._get_component(layer_name)
            self.gpu_layers[layer_name] = comp
            self.layer_devices[layer_name] = torch.device('cuda')

            transfer_time = time.time() - start_time
            self.transfer_times.append(transfer_time)

            if self.verbose:
                mem_gb, mem_pct = self.get_memory_usage()
                print(f"Loaded {layer_name} to GPU in {transfer_time:.3f}s. "
                      f"GPU memory: {mem_gb:.2f}GB ({mem_pct:.1%})")

    def prefetch_layers(self, current_layer_idx: int):
        """Prefetch upcoming layers to GPU."""
        if self.strategy == "sequential":
            # Prefetch next N layers
            for i in range(1, self.prefetch_distance + 1):
                next_idx = current_layer_idx + i
                if next_idx < 32:  # Total 32 layers
                    layer_name = f'model.layers.{next_idx}'
                    if layer_name not in self.gpu_layers and len(self.gpu_layers) < self.max_gpu_layers:
                        self.load_layer(layer_name)

    def execute_with_offload(self, layer_idx: int, forward_fn, *args, **kwargs):
        """Execute a layer's forward pass with intelligent offloading."""
        layer_name = f'model.layers.{layer_idx}'

        # Ensure layer is on GPU
        if layer_name not in self.gpu_layers:
            self.load_layer(layer_name)

        # Mark as recently used (move to end)
        if layer_name in self.gpu_layers:
            self.gpu_layers.move_to_end(layer_name)

        # Prefetch upcoming layers
        self.prefetch_layers(layer_idx)

        # Execute forward pass
        result = forward_fn(*args, **kwargs)

        # Offload if we're past this layer and memory is tight
        if self.strategy == "sequential" and self.should_offload():
            # Keep only recent and upcoming layers
            keep_range = range(max(0, layer_idx - 1), min(32, layer_idx + self.prefetch_distance + 1))
            keep_layers = {f'model.layers.{i}' for i in keep_range}

            # Offload layers outside the range
            for layer in list(self.gpu_layers.keys()):
                if layer.startswith('model.layers.') and layer not in keep_layers:
                    self.offload_layer(layer)

        return result

    def prepare_for_vae_decode(self):
        """Prepare for VAE decode by offloading model layers."""
        if self.verbose:
            print("Preparing for VAE decode, offloading model layers...")

        # Offload all model layers except essential components
        layers_to_offload = [k for k in self.gpu_layers.keys() if k.startswith('model.layers.')]
        for layer_name in layers_to_offload:
            self.offload_layer(layer_name)

        # Force garbage collection and clear cache
        gc.collect()
        torch.cuda.empty_cache()

        if self.verbose:
            mem_gb, mem_pct = self.get_memory_usage()
            print(f"Ready for VAE decode. GPU memory: {mem_gb:.2f}GB ({mem_pct:.1%})")

    def cleanup_after_vae(self):
        """Clean up memory after VAE decode."""
        gc.collect()
        torch.cuda.empty_cache()

        if self.verbose:
            mem_gb, mem_pct = self.get_memory_usage()
            print(f"Cleaned up after VAE. GPU memory: {mem_gb:.2f}GB ({mem_pct:.1%})")

    def get_optimal_device_map(self) -> Dict[str, Any]:
        """Generate an optimal device map based on available memory."""
        # Get available GPU memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

            # Essential components that should stay on GPU
            device_map = {
                'vae': 0,
                'vision_model': 0,
                'vision_aligner': 0,
                'timestep_emb': 0,
                'patch_embed': 0,
                'time_embed': 0,
                'final_layer': 0,
                'time_embed_2': 0,
                'model.wte': 0,
                'model.ln_f': 0,
                'lm_head': 0,
            }

            # Determine how many layers can fit on GPU
            if total_memory > 24:  # High VRAM (3090, 4090, A100)
                # Keep more layers on GPU
                gpu_layers = min(16, self.max_gpu_layers)
            elif total_memory > 12:  # Medium VRAM (3080, 4070)
                gpu_layers = min(8, self.max_gpu_layers)
            else:  # Low VRAM
                gpu_layers = min(4, self.max_gpu_layers)

            # Assign layers
            for i in range(32):
                if i < gpu_layers:
                    device_map[f'model.layers.{i}'] = 0
                else:
                    device_map[f'model.layers.{i}'] = 'cpu'

            return device_map
        else:
            # CPU only
            return 'cpu'

    def print_summary(self):
        """Print performance summary."""
        if self.transfer_times:
            avg_transfer = sum(self.transfer_times) / len(self.transfer_times)
            print(f"\nOffload Manager Summary:")
            print(f"  Strategy: {self.strategy}")
            print(f"  Total transfers: {len(self.transfer_times)}")
            print(f"  Avg transfer time: {avg_transfer:.3f}s")
            print(f"  GPU layers: {len(self.gpu_layers)}/{32}")

            mem_gb, mem_pct = self.get_memory_usage()
            print(f"  Current GPU memory: {mem_gb:.2f}GB ({mem_pct:.1%})")