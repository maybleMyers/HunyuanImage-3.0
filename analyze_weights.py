#!/usr/bin/env python3
"""
Weight Analysis Script for HunyuanImage-3.0 Model
Analyzes model weights to determine optimal GPU/RamTorch allocation strategy
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from collections import defaultdict
import re

def parse_weight_name(weight_name: str) -> Dict[str, str]:
    """Parse a weight name to extract layer type and components."""
    parts = {
        'full_name': weight_name,
        'layer_type': 'unknown',
        'layer_idx': None,
        'component': None,
        'expert_idx': None,
        'is_moe': False,
        'is_norm': False,
        'is_embedding': False,
        'is_attention': False,
        'is_mlp': False,
        'is_vision': False,
        'is_vae': False,
    }

    # Check for special components
    if 'wte' in weight_name or 'embed' in weight_name.lower():
        parts['is_embedding'] = True
        parts['layer_type'] = 'embedding'
    elif 'norm' in weight_name.lower() or 'ln' in weight_name.lower():
        parts['is_norm'] = True
        parts['layer_type'] = 'normalization'
    elif 'vision' in weight_name.lower():
        parts['is_vision'] = True
        parts['layer_type'] = 'vision'
    elif 'vae' in weight_name.lower():
        parts['is_vae'] = True
        parts['layer_type'] = 'vae'

    # Check for MoE experts
    if 'experts' in weight_name or 'expert' in weight_name:
        parts['is_moe'] = True
        parts['layer_type'] = 'moe_expert'
        # Extract expert index if present
        expert_match = re.search(r'experts\.(\d+)', weight_name)
        if expert_match:
            parts['expert_idx'] = int(expert_match.group(1))

    # Check for attention components
    attention_patterns = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv_proj']
    for pattern in attention_patterns:
        if pattern in weight_name:
            parts['is_attention'] = True
            parts['layer_type'] = 'attention'
            parts['component'] = pattern
            break

    # Check for MLP components
    mlp_patterns = ['gate_proj', 'up_proj', 'down_proj', 'gate_and_up_proj', 'mlp']
    for pattern in mlp_patterns:
        if pattern in weight_name:
            parts['is_mlp'] = True
            if not parts['is_moe']:
                parts['layer_type'] = 'mlp'
            parts['component'] = pattern
            break

    # Extract layer index
    layer_match = re.search(r'layers\.(\d+)', weight_name)
    if layer_match:
        parts['layer_idx'] = int(layer_match.group(1))

    # Gate weights
    if 'gate.wg' in weight_name or 'router' in weight_name:
        parts['layer_type'] = 'moe_gate'
        parts['is_moe'] = True

    return parts

def calculate_weight_size(shape: List[int], dtype: str = 'float32') -> Dict[str, float]:
    """Calculate memory size for a weight tensor."""
    num_elements = 1
    for dim in shape:
        num_elements *= dim

    # Determine bytes per element
    bytes_per_element = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'int8': 1,
        'int32': 4,
    }.get(dtype, 4)

    total_bytes = num_elements * bytes_per_element

    return {
        'num_elements': num_elements,
        'bytes': total_bytes,
        'mb': total_bytes / (1024 * 1024),
        'gb': total_bytes / (1024 * 1024 * 1024),
    }

def analyze_weights_from_index(index_path: Path, config_path: Path) -> Dict:
    """Analyze weights from safetensors index file."""

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load index file
    with open(index_path, 'r') as f:
        index_data = json.load(f)

    # Get weight map and metadata
    weight_map = index_data.get('weight_map', {})
    metadata = index_data.get('metadata', {})

    # Analysis results
    analysis = {
        'total_weights': len(weight_map),
        'total_size_gb': 0,
        'by_type': defaultdict(lambda: {'count': 0, 'size_gb': 0, 'weights': []}),
        'by_layer': defaultdict(lambda: {'count': 0, 'size_gb': 0, 'weights': []}),
        'large_weights': [],  # Weights > 100MB
        'recommendations': {
            'keep_on_gpu': [],
            'convert_to_ramtorch': [],
            'critical_for_performance': [],
        },
        'shape_mismatches': [],
    }

    # Analyze each weight
    for weight_name, file_name in weight_map.items():
        parsed = parse_weight_name(weight_name)

        # Try to infer shape from weight name and config
        # This is an approximation since we don't have the actual tensor shapes
        shape = infer_weight_shape(weight_name, config)

        if shape:
            size_info = calculate_weight_size(shape, 'bfloat16')  # Assume bfloat16

            # Update totals
            analysis['total_size_gb'] += size_info['gb']

            # Group by type
            layer_type = parsed['layer_type']
            analysis['by_type'][layer_type]['count'] += 1
            analysis['by_type'][layer_type]['size_gb'] += size_info['gb']
            analysis['by_type'][layer_type]['weights'].append({
                'name': weight_name,
                'shape': shape,
                'size_mb': size_info['mb'],
                'component': parsed.get('component'),
                'layer_idx': parsed.get('layer_idx'),
            })

            # Group by layer index
            if parsed['layer_idx'] is not None:
                layer_key = f"layer_{parsed['layer_idx']}"
                analysis['by_layer'][layer_key]['count'] += 1
                analysis['by_layer'][layer_key]['size_gb'] += size_info['gb']
                analysis['by_layer'][layer_key]['weights'].append(weight_name)

            # Track large weights
            if size_info['mb'] > 100:
                analysis['large_weights'].append({
                    'name': weight_name,
                    'size_mb': size_info['mb'],
                    'type': layer_type,
                })

            # Make recommendations
            if parsed['is_embedding'] or parsed['is_norm']:
                analysis['recommendations']['keep_on_gpu'].append(weight_name)
                analysis['recommendations']['critical_for_performance'].append(weight_name)
            elif parsed['is_vision'] or parsed['is_vae']:
                analysis['recommendations']['keep_on_gpu'].append(weight_name)
            elif 'o_proj' in weight_name:
                # o_proj layers have known issues with RamTorch
                analysis['recommendations']['keep_on_gpu'].append(weight_name)
            elif 'down_proj' in weight_name and shape and len(shape) >= 2:
                # Check for potential shape mismatches
                if shape[1] == 4096 and config.get('intermediate_size') == 3072:
                    analysis['shape_mismatches'].append({
                        'name': weight_name,
                        'expected_input': 3072,
                        'actual_input': shape[1],
                        'recommendation': 'keep_on_gpu',
                    })
                    analysis['recommendations']['keep_on_gpu'].append(weight_name)
                else:
                    analysis['recommendations']['convert_to_ramtorch'].append(weight_name)
            elif parsed['is_moe'] and parsed['expert_idx'] is not None:
                # MoE expert weights are good candidates for RamTorch
                analysis['recommendations']['convert_to_ramtorch'].append(weight_name)
            elif parsed['is_attention'] or parsed['is_mlp']:
                # Regular attention/MLP layers can use RamTorch
                if size_info['mb'] > 10:  # Only convert if reasonably large
                    analysis['recommendations']['convert_to_ramtorch'].append(weight_name)

    return analysis

def infer_weight_shape(weight_name: str, config: Dict) -> List[int]:
    """Infer weight shape from name and config."""
    hidden_size = config.get('hidden_size', 4096)
    intermediate_size = config.get('intermediate_size', 3072)
    num_attention_heads = config.get('num_attention_heads', 32)
    attention_head_dim = config.get('attention_head_dim', 128)
    vocab_size = config.get('vocab_size', 128256)

    # Parse layer index for MoE dimensions
    layer_match = re.search(r'layers\.(\d+)', weight_name)
    layer_idx = int(layer_match.group(1)) if layer_match else 0

    # Get MoE-specific dimensions
    moe_intermediate_sizes = config.get('moe_intermediate_size', [intermediate_size] * 32)
    if layer_idx < len(moe_intermediate_sizes):
        moe_intermediate_size = moe_intermediate_sizes[layer_idx]
    else:
        moe_intermediate_size = intermediate_size

    # Embeddings
    if 'wte.weight' in weight_name:
        return [vocab_size, hidden_size]

    # Attention projections
    if 'q_proj' in weight_name or 'k_proj' in weight_name or 'v_proj' in weight_name:
        return [hidden_size, hidden_size]
    elif 'o_proj' in weight_name:
        return [hidden_size, hidden_size]

    # MLP projections
    if 'gate_and_up_proj' in weight_name:
        if 'experts' in weight_name:
            # MoE expert
            if config.get('hidden_act') == 'silu':
                return [hidden_size, moe_intermediate_size * 2]
            else:
                return [hidden_size, moe_intermediate_size]
        else:
            # Shared MLP
            num_shared = config.get('num_shared_expert', [1] * 32)
            shared_experts = num_shared[layer_idx] if layer_idx < len(num_shared) else 1
            actual_intermediate = moe_intermediate_size * shared_experts
            if config.get('hidden_act') == 'silu':
                return [hidden_size, actual_intermediate * 2]
            else:
                return [hidden_size, actual_intermediate]
    elif 'down_proj' in weight_name:
        if 'experts' in weight_name:
            # MoE expert down projection
            if config.get('hidden_act') == 'silu':
                return [moe_intermediate_size, hidden_size]  # SwiGLU uses intermediate_size (not *2)
            else:
                return [moe_intermediate_size, hidden_size]
        else:
            # Shared MLP down projection
            num_shared = config.get('num_shared_expert', [1] * 32)
            shared_experts = num_shared[layer_idx] if layer_idx < len(num_shared) else 1
            actual_intermediate = moe_intermediate_size * shared_experts
            if config.get('hidden_act') == 'silu':
                return [actual_intermediate, hidden_size]  # Note: for shared, it's not divided
            else:
                return [actual_intermediate, hidden_size]

    # Layer norms
    if 'norm' in weight_name.lower() or 'ln' in weight_name.lower():
        if 'weight' in weight_name or 'bias' in weight_name:
            return [hidden_size]

    # MoE gates
    if 'gate.wg' in weight_name:
        num_experts = config.get('num_experts', [48] * 32)
        experts = num_experts[layer_idx] if layer_idx < len(num_experts) else 48
        return [hidden_size, experts]

    # Default: can't infer
    return None

def print_analysis(analysis: Dict):
    """Print analysis results in a readable format."""
    print("=" * 80)
    print("HUNYUAN IMAGE 3.0 MODEL WEIGHT ANALYSIS")
    print("=" * 80)

    print(f"\nTotal Weights: {analysis['total_weights']:,}")
    print(f"Total Size: {analysis['total_size_gb']:.2f} GB")

    print("\n" + "=" * 80)
    print("WEIGHTS BY TYPE")
    print("=" * 80)

    # Sort by size
    sorted_types = sorted(
        analysis['by_type'].items(),
        key=lambda x: x[1]['size_gb'],
        reverse=True
    )

    for layer_type, info in sorted_types:
        print(f"\n{layer_type.upper()}:")
        print(f"  Count: {info['count']}")
        print(f"  Total Size: {info['size_gb']:.3f} GB ({info['size_gb']/analysis['total_size_gb']*100:.1f}%)")

        # Show top 5 largest weights of this type
        if info['weights']:
            sorted_weights = sorted(info['weights'], key=lambda x: x['size_mb'], reverse=True)[:5]
            print(f"  Largest weights:")
            for w in sorted_weights:
                shape_str = f"{w['shape']}" if w['shape'] else "unknown"
                print(f"    - {w['name']}: {w['size_mb']:.1f} MB {shape_str}")

    print("\n" + "=" * 80)
    print("SHAPE MISMATCHES DETECTED")
    print("=" * 80)

    if analysis['shape_mismatches']:
        for mismatch in analysis['shape_mismatches']:
            print(f"\n{mismatch['name']}:")
            print(f"  Expected input dim: {mismatch['expected_input']}")
            print(f"  Actual input dim: {mismatch['actual_input']}")
            print(f"  Recommendation: {mismatch['recommendation']}")
    else:
        print("\nNo shape mismatches detected")

    print("\n" + "=" * 80)
    print("MEMORY ALLOCATION RECOMMENDATIONS")
    print("=" * 80)

    # Calculate sizes for recommendations
    gpu_weights = set(analysis['recommendations']['keep_on_gpu'])
    ramtorch_weights = set(analysis['recommendations']['convert_to_ramtorch'])

    gpu_size = 0
    ramtorch_size = 0

    for layer_type, info in analysis['by_type'].items():
        for weight in info['weights']:
            if weight['name'] in gpu_weights:
                gpu_size += weight['size_mb'] / 1024  # Convert to GB
            elif weight['name'] in ramtorch_weights:
                ramtorch_size += weight['size_mb'] / 1024

    print(f"\nKEEP ON GPU ({len(gpu_weights)} weights, ~{gpu_size:.2f} GB):")
    print("  - All embeddings and normalization layers")
    print("  - Vision and VAE components")
    print("  - Output projection layers (o_proj)")
    print("  - Layers with shape mismatches (down_proj with incompatible dims)")

    print(f"\nCONVERT TO RAMTORCH ({len(ramtorch_weights)} weights, ~{ramtorch_size:.2f} GB):")
    print("  - MoE expert layers")
    print("  - Most attention projections (q_proj, k_proj, v_proj)")
    print("  - Compatible MLP layers")
    print("  - Gate and up projections")

    print(f"\nMEMORY SAVINGS:")
    print(f"  GPU Memory Required: ~{gpu_size:.2f} GB")
    print(f"  CPU Memory (RamTorch): ~{ramtorch_size:.2f} GB")
    print(f"  GPU Memory Saved: ~{ramtorch_size:.2f} GB ({ramtorch_size/analysis['total_size_gb']*100:.1f}%)")

    print("\n" + "=" * 80)
    print("TOP 10 LARGEST WEIGHTS")
    print("=" * 80)

    sorted_large = sorted(analysis['large_weights'], key=lambda x: x['size_mb'], reverse=True)[:10]
    for i, weight in enumerate(sorted_large, 1):
        print(f"{i:2d}. {weight['name']}: {weight['size_mb']:.1f} MB ({weight['type']})")

def main():
    """Main analysis function."""
    # Paths
    config_path = Path("model/config.json")
    index_path = Path("model/model.safetensors.index.json")

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    if not index_path.exists():
        print(f"Error: Index file not found at {index_path}")
        return

    print("Analyzing model weights...")
    analysis = analyze_weights_from_index(index_path, config_path)

    # Save detailed analysis to JSON
    output_path = Path("weight_analysis.json")
    with open(output_path, 'w') as f:
        # Convert defaultdicts to regular dicts for JSON serialization
        json_safe = {
            'total_weights': analysis['total_weights'],
            'total_size_gb': analysis['total_size_gb'],
            'by_type': dict(analysis['by_type']),
            'by_layer': dict(analysis['by_layer']),
            'large_weights': analysis['large_weights'][:20],  # Top 20 only
            'recommendations': {
                'keep_on_gpu': analysis['recommendations']['keep_on_gpu'][:100],  # Limit for readability
                'convert_to_ramtorch': analysis['recommendations']['convert_to_ramtorch'][:100],
            },
            'shape_mismatches': analysis['shape_mismatches'],
        }
        json.dump(json_safe, f, indent=2)

    print(f"\nDetailed analysis saved to: {output_path}")

    # Print summary
    print_analysis(analysis)

if __name__ == "__main__":
    main()