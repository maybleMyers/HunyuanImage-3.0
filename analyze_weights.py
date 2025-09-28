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
    if isinstance(moe_intermediate_sizes, list) and layer_idx < len(moe_intermediate_sizes):
        moe_intermediate_size = moe_intermediate_sizes[layer_idx]
    elif isinstance(moe_intermediate_sizes, int):
        moe_intermediate_size = moe_intermediate_sizes
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
            if isinstance(num_shared, list):
                shared_experts = num_shared[layer_idx] if layer_idx < len(num_shared) else 1
            else:
                shared_experts = num_shared if num_shared else 1
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
            if isinstance(num_shared, list):
                shared_experts = num_shared[layer_idx] if layer_idx < len(num_shared) else 1
            else:
                shared_experts = num_shared if num_shared else 1
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
        num_experts = config.get('num_experts', 48)
        # Handle both int and list types
        if isinstance(num_experts, list):
            experts = num_experts[layer_idx] if layer_idx < len(num_experts) else 48
        else:
            experts = num_experts
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
    print("DETAILED WEIGHTS BY TYPE")
    print("=" * 80)

    # Sort by size
    sorted_types = sorted(
        analysis['by_type'].items(),
        key=lambda x: x[1]['size_gb'],
        reverse=True
    )

    # Create summary table
    print("\n{:<25} {:>10} {:>15} {:>10}".format("Type", "Count", "Size (GB)", "Percent"))
    print("-" * 60)

    for layer_type, info in sorted_types:
        print("{:<25} {:>10} {:>15.2f} {:>9.1f}%".format(
            layer_type.upper()[:25],
            info['count'],
            info['size_gb'],
            info['size_gb']/analysis['total_size_gb']*100 if analysis['total_size_gb'] > 0 else 0
        ))

    print("-" * 60)
    print("{:<25} {:>10} {:>15.2f} {:>9.1f}%".format(
        "TOTAL",
        analysis['total_weights'],
        analysis['total_size_gb'],
        100.0
    ))

    # Detailed breakdown for major types
    print("\n" + "=" * 80)
    print("COMPONENT BREAKDOWN")
    print("=" * 80)

    # Group by component type for better analysis
    component_stats = {
        'MoE Experts': {'count': 0, 'size_gb': 0, 'patterns': ['experts']},
        'Attention Q/K/V': {'count': 0, 'size_gb': 0, 'patterns': ['q_proj', 'k_proj', 'v_proj']},
        'Attention O': {'count': 0, 'size_gb': 0, 'patterns': ['o_proj']},
        'MLP Gate/Up': {'count': 0, 'size_gb': 0, 'patterns': ['gate_and_up_proj', 'gate_proj', 'up_proj']},
        'MLP Down': {'count': 0, 'size_gb': 0, 'patterns': ['down_proj']},
        'Embeddings': {'count': 0, 'size_gb': 0, 'patterns': ['wte', 'embed']},
        'Normalizations': {'count': 0, 'size_gb': 0, 'patterns': ['norm', 'ln']},
        'MoE Gates': {'count': 0, 'size_gb': 0, 'patterns': ['gate.wg']},
    }

    # Calculate component stats
    for layer_type, info in analysis['by_type'].items():
        for weight in info['weights']:
            weight_name = weight['name'].lower()
            weight_mb = weight.get('size_mb', 0)

            for comp_name, comp_info in component_stats.items():
                if any(pattern in weight_name for pattern in comp_info['patterns']):
                    comp_info['count'] += 1
                    comp_info['size_gb'] += weight_mb / 1024
                    break

    print("\n{:<20} {:>10} {:>15} {:>10}".format("Component", "Count", "Size (GB)", "Percent"))
    print("-" * 60)

    for comp_name, comp_info in sorted(component_stats.items(), key=lambda x: x[1]['size_gb'], reverse=True):
        if comp_info['count'] > 0:
            print("{:<20} {:>10} {:>15.2f} {:>9.1f}%".format(
                comp_name,
                comp_info['count'],
                comp_info['size_gb'],
                comp_info['size_gb']/analysis['total_size_gb']*100 if analysis['total_size_gb'] > 0 else 0
            ))

    print("\n" + "=" * 80)
    print("LAYER DISTRIBUTION (Top 10 Layers by Size)")
    print("=" * 80)

    # Get layer statistics
    layer_sorted = sorted(
        [(k, v) for k, v in analysis['by_layer'].items()],
        key=lambda x: x[1]['size_gb'],
        reverse=True
    )[:10]

    print("\n{:<15} {:>10} {:>15}".format("Layer", "Weights", "Size (GB)"))
    print("-" * 40)
    for layer_name, layer_info in layer_sorted:
        print("{:<15} {:>10} {:>15.2f}".format(
            layer_name,
            layer_info['count'],
            layer_info['size_gb']
        ))

    print("\n" + "=" * 80)
    print("SHAPE MISMATCHES DETECTED")
    print("=" * 80)

    if analysis['shape_mismatches']:
        print(f"\nFound {len(analysis['shape_mismatches'])} weights with shape mismatches:")
        print("\n{:<60} {:>10} {:>10}".format("Weight Name", "Expected", "Actual"))
        print("-" * 80)
        for mismatch in analysis['shape_mismatches'][:10]:  # Show first 10
            name_short = mismatch['name']
            if len(name_short) > 60:
                name_short = "..." + name_short[-57:]
            print("{:<60} {:>10} {:>10}".format(
                name_short,
                mismatch['expected_input'],
                mismatch['actual_input']
            ))
        if len(analysis['shape_mismatches']) > 10:
            print(f"... and {len(analysis['shape_mismatches']) - 10} more")
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
    gpu_breakdown = defaultdict(float)
    ramtorch_breakdown = defaultdict(float)

    for layer_type, info in analysis['by_type'].items():
        for weight in info['weights']:
            if weight['name'] in gpu_weights:
                gpu_size += weight['size_mb'] / 1024  # Convert to GB
                # Categorize GPU weights
                if 'embed' in weight['name'].lower() or 'wte' in weight['name'].lower():
                    gpu_breakdown['Embeddings'] += weight['size_mb'] / 1024
                elif 'norm' in weight['name'].lower():
                    gpu_breakdown['Normalizations'] += weight['size_mb'] / 1024
                elif 'o_proj' in weight['name']:
                    gpu_breakdown['O_proj layers'] += weight['size_mb'] / 1024
                elif 'down_proj' in weight['name']:
                    gpu_breakdown['Down_proj (mismatched)'] += weight['size_mb'] / 1024
                else:
                    gpu_breakdown['Other'] += weight['size_mb'] / 1024
            elif weight['name'] in ramtorch_weights:
                ramtorch_size += weight['size_mb'] / 1024
                # Categorize RamTorch weights
                if 'experts' in weight['name']:
                    ramtorch_breakdown['MoE Experts'] += weight['size_mb'] / 1024
                elif any(proj in weight['name'] for proj in ['q_proj', 'k_proj', 'v_proj']):
                    ramtorch_breakdown['Attention QKV'] += weight['size_mb'] / 1024
                elif 'gate_and_up_proj' in weight['name'] or 'up_proj' in weight['name']:
                    ramtorch_breakdown['Gate/Up proj'] += weight['size_mb'] / 1024
                else:
                    ramtorch_breakdown['Other'] += weight['size_mb'] / 1024

    print(f"\n{'='*40}")
    print(f"KEEP ON GPU: {len(gpu_weights)} weights, {gpu_size:.2f} GB")
    print(f"{'='*40}")
    for category, size in sorted(gpu_breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category:<25} {size:>8.2f} GB ({size/gpu_size*100:>5.1f}%)")

    print(f"\n{'='*40}")
    print(f"CONVERT TO RAMTORCH: {len(ramtorch_weights)} weights, {ramtorch_size:.2f} GB")
    print(f"{'='*40}")
    for category, size in sorted(ramtorch_breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category:<25} {size:>8.2f} GB ({size/ramtorch_size*100:>5.1f}%)")

    print(f"\n{'='*40}")
    print("MEMORY SUMMARY")
    print(f"{'='*40}")
    print(f"  Total Model Size:        {analysis['total_size_gb']:>8.2f} GB")
    print(f"  GPU Memory Required:     {gpu_size:>8.2f} GB ({gpu_size/analysis['total_size_gb']*100:.1f}%)")
    print(f"  CPU Memory (RamTorch):   {ramtorch_size:>8.2f} GB ({ramtorch_size/analysis['total_size_gb']*100:.1f}%)")
    print(f"  GPU Memory Saved:        {ramtorch_size:>8.2f} GB ({ramtorch_size/analysis['total_size_gb']*100:.1f}%)")

    # Provide guidance based on common GPU sizes
    print(f"\n{'='*40}")
    print("GPU COMPATIBILITY")
    print(f"{'='*40}")
    gpu_configs = [
        (24, "RTX 3090/4090"),
        (40, "A100-40GB"),
        (48, "RTX A6000/L40"),
        (80, "A100-80GB/H100"),
    ]

    for gpu_mem, gpu_name in gpu_configs:
        if gpu_size <= gpu_mem:
            print(f"  ✓ {gpu_name:<20} Can fit GPU weights ({gpu_size:.1f}/{gpu_mem} GB)")
        else:
            print(f"  ✗ {gpu_name:<20} Insufficient ({gpu_size:.1f}/{gpu_mem} GB)")

    print("\n" + "=" * 80)
    print("TOP 10 LARGEST WEIGHTS")
    print("=" * 80)

    print("\n{:>3} {:<50} {:>10} {:<15}".format("#", "Weight Name", "Size (MB)", "Type"))
    print("-" * 80)

    sorted_large = sorted(analysis['large_weights'], key=lambda x: x['size_mb'], reverse=True)[:10]
    for i, weight in enumerate(sorted_large, 1):
        name_short = weight['name']
        if len(name_short) > 50:
            name_short = "..." + name_short[-47:]
        print(f"{i:>3} {name_short:<50} {weight['size_mb']:>10.1f} {weight['type']:<15}")

def main():
    """Main analysis function."""
    import sys

    # Get model directory from command line or use default
    if len(sys.argv) > 1:
        model_dir = Path(sys.argv[1])
    else:
        model_dir = Path("model")

    # Paths
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        print(f"Looking in directory: {model_dir.absolute()}")
        # Try to list what files are there
        if model_dir.exists():
            print(f"Files found in {model_dir}:")
            for f in sorted(model_dir.glob("*.json"))[:10]:
                print(f"  - {f.name}")
        return

    if not index_path.exists():
        print(f"Error: Index file not found at {index_path}")
        print(f"Looking in directory: {model_dir.absolute()}")
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