#!/usr/bin/env python3
"""
Generalized Neuron Modifier

This script modifies the biases of a GPT-2 model based on the difference between two sets of activation vectors (target and neutral),
to enhance a specific property in the model outputs.
"""

import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

# Add parent directory to path to import from nanoGPT
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from model import GPT


def load_model(model_path, device):
    """Load model from checkpoint"""
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Load config and model
    from model import GPTConfig

    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)

    # Load state dict
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"Model loaded successfully")
    return model


def find_layer_vectors(data_dir):
    """Find all available layer vectors in the directory"""
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")

    mean_files = [f for f in os.listdir(data_dir) if f.endswith("_mean.pt")]
    if not mean_files:
        raise ValueError(f"No mean files found in {data_dir}")

    print(f"Found {len(mean_files)} layer vectors in {data_dir}")
    layer_files = {
        file_name.replace("_mean.pt", ""): os.path.join(data_dir, file_name)
        for file_name in mean_files
    }
    return layer_files


def calculate_difference(
    target_layer_files,
    neutral_layer_files,
):
    """Calculate activation differences between target and neutral data"""
    differences = {}
    for layer_name, target_file in tqdm(
        target_layer_files.items(), desc="Processing layers"
    ):
        if layer_name not in neutral_layer_files:
            print(f"Layer {layer_name} not found in neutral data, skipping")
            continue
        neutral_file = neutral_layer_files[layer_name]
        try:
            target_activation = torch.load(target_file, weights_only=False)
            neutral_activation = torch.load(neutral_file, weights_only=False)
            if target_activation.shape != neutral_activation.shape:
                print(
                    f"Shape mismatch for {layer_name}: {target_activation.shape} vs {neutral_activation.shape}"
                )
                continue
            difference = target_activation - neutral_activation
            differences[layer_name] = difference
        except Exception as e:
            print(f"Error processing {layer_name}: {e}")
    return differences


def map_layer_to_model_component(model, layer_name):
    """Map layer name to model component"""
    if layer_name == "final_layer":
        return {"component": model.transformer.ln_f, "bias_attr": "bias", "type": "ln"}
    parts = layer_name.split("_")
    if parts[0] != "layer":
        return None
    layer_idx = int(parts[1])
    component_type = "_".join(parts[2:])
    if component_type == "ln_1":
        return {
            "component": model.transformer.h[layer_idx].ln_1,
            "bias_attr": "bias",
            "type": "ln",
        }
    elif component_type == "ln_2":
        return {
            "component": model.transformer.h[layer_idx].ln_2,
            "bias_attr": "bias",
            "type": "ln",
        }
    elif component_type == "attn":
        return {
            "component": model.transformer.h[layer_idx].attn,
            "bias_attr": None,
            "type": "attn",
        }
    elif component_type == "attn_output_proj":
        return {
            "component": model.transformer.h[layer_idx].attn.c_proj,
            "bias_attr": "bias",
            "type": "proj",
        }
    elif component_type == "attn_qkv_proj":
        return {
            "component": model.transformer.h[layer_idx].attn.c_attn,
            "bias_attr": "bias",
            "type": "proj",
        }
    elif component_type == "mlp":
        return {
            "component": model.transformer.h[layer_idx].mlp,
            "bias_attr": None,
            "type": "mlp",
        }
    elif component_type == "mlp_fc":
        return {
            "component": model.transformer.h[layer_idx].mlp.c_fc,
            "bias_attr": "bias",
            "type": "proj",
        }
    elif component_type == "mlp_proj":
        return {
            "component": model.transformer.h[layer_idx].mlp.c_proj,
            "bias_attr": "bias",
            "type": "proj",
        }
    elif component_type == "output":
        return {
            "component": model.transformer.h[layer_idx],
            "bias_attr": None,
            "type": "block",
        }
    return None


def modify_model_biases(
    model, differences, scaling_factor, limit_to=None, component_types=None
):
    """Apply differences to model biases"""
    if component_types is None:
        component_types = ["ln", "proj"]
    modifications = {}
    for layer_name, difference in differences.items():
        mapping = map_layer_to_model_component(model, layer_name)
        if mapping is None:
            print(f"No mapping found for {layer_name}, skipping")
            continue
        if mapping["bias_attr"] is None:
            print(f"No bias attribute for {layer_name}, skipping")
            continue
        if mapping["type"] not in component_types:
            print(f"Skipping {layer_name} (type: {mapping['type']})")
            continue
        component = mapping["component"]
        bias_attr = mapping["bias_attr"]
        bias = getattr(component, bias_attr)
        if bias.shape != difference.shape:
            print(
                f"Shape mismatch for {layer_name}: bias {bias.shape} vs diff {difference.shape}"
            )
            continue
        if limit_to is not None:
            difference = torch.clamp(difference, -limit_to, limit_to)
        modified_bias = bias + (difference * scaling_factor)
        setattr(component, bias_attr, torch.nn.Parameter(modified_bias))
        modifications[layer_name] = {
            "original_norm": bias.norm().item(),
            "difference_norm": difference.norm().item(),
            "modified_norm": modified_bias.norm().item(),
            "max_diff": difference.abs().max().item(),
            "scaled_max_diff": (difference * scaling_factor).abs().max().item(),
        }
    return modifications


def save_modified_model(model, output_path, modifications, args):
    """Save the modified model and modification details"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_args = {
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
        "vocab_size": model.config.vocab_size,
        "block_size": model.config.block_size,
        "bias": model.config.bias,
        "dropout": model.config.dropout,
    }
    metadata = {
        "target_data": args.target_data,
        "neutral_data": args.neutral_data,
        "scaling_factor": args.scaling_factor,
        "modification_details": modifications,
        "modification_time": pd.Timestamp.now().isoformat(),
    }
    torch.save(
        {
            "model_args": model_args,
            "model": model.state_dict(),
            "modification_metadata": metadata,
        },
        output_path,
    )
    metadata_path = output_path.replace(".pt", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "target_data": args.target_data,
                "neutral_data": args.neutral_data,
                "scaling_factor": args.scaling_factor,
                "modification_details": {
                    k: {
                        kk: float(vv)
                        if isinstance(vv, (np.float32, np.float64))
                        else vv
                        for kk, vv in v.items()
                    }
                    for k, v in modifications.items()
                },
                "modification_time": pd.Timestamp.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"Model saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Modify GPT-2 biases using the difference between two sets of activation vectors"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save modified model",
    )
    parser.add_argument(
        "--target_data",
        type=str,
        required=True,
        help="Directory containing target activation vectors (_mean.pt files)",
    )
    parser.add_argument(
        "--neutral_data",
        type=str,
        required=True,
        help="Directory containing neutral activation vectors (_mean.pt files)",
    )
    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=0.2,
        help="Factor to scale differences (0.1-1.0 recommended)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda)",
    )
    parser.add_argument(
        "--limit_to",
        type=float,
        default=None,
        help="Limit individual neuron differences to this value",
    )
    parser.add_argument(
        "--component_types",
        type=str,
        nargs="+",
        default=["proj", "ln"],
        help="Component types to modify (proj, ln)",
    )
    args = parser.parse_args()

    # Find layer vectors
    target_layer_files = find_layer_vectors(args.target_data)
    neutral_layer_files = find_layer_vectors(args.neutral_data)
    print(
        f"Found {len(target_layer_files)} target layers and {len(neutral_layer_files)} neutral layers"
    )

    # Load model
    model = load_model(args.model_path, args.device)

    # Calculate differences
    differences = calculate_difference(target_layer_files, neutral_layer_files)
    print(f"Calculated differences for {len(differences)} layers")

    # Modify model biases
    modifications = modify_model_biases(
        model,
        differences,
        args.scaling_factor,
        args.limit_to,
        args.component_types,
    )
    print(f"Modified {len(modifications)} components")

    # Save modified model
    save_modified_model(model, args.output_path, modifications, args)

    print("Modification complete!")


if __name__ == "__main__":
    main()
