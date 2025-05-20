#!/usr/bin/env python3
"""
Generalized Neuronal Activation Averager
Processes a dataset of texts (single 'text' column) and computes the average neuron activations across all samples.
Saves the averaged activations to a specified output directory and file name.
"""

import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import tempfile

# Add parent directory to path to import from nanoGPT
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from model import GPT

# For loading datasets
try:
    from datasets import load_dataset
except ImportError:
    pass  # Not needed for this script


def load_text_dataset(parquet_file):
    """Load a Parquet file with a single 'text' column."""
    print(f"Loading texts from file: {parquet_file}")
    if not parquet_file.lower().endswith(".parquet"):
        raise ValueError("Only Parquet files are supported for this script.")
    df = pd.read_parquet(parquet_file)
    if "text" not in df.columns:
        raise ValueError("Input file must have a 'text' column.")
    print(f"Loaded {len(df)} samples.")
    return df


def create_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def load_model(model_path, device):
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    from model import GPTConfig

    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"Model loaded successfully and placed on {device}")
    return model


def register_comprehensive_hooks(model):
    hooks = []
    activations = {}
    for i in range(model.config.n_layer):
        hooks.append(
            model.transformer.h[i].ln_1.register_forward_hook(
                lambda mod, inp, out, layer_idx=i: activations.update(
                    {f"layer_{layer_idx}_ln_1": out.detach().cpu()}
                )
            )
        )
        hooks.append(
            model.transformer.h[i].attn.register_forward_hook(
                lambda mod, inp, out, layer_idx=i: activations.update(
                    {f"layer_{layer_idx}_attn": out.detach().cpu()}
                )
            )
        )
        hooks.append(
            model.transformer.h[i].attn.c_attn.register_forward_hook(
                lambda mod, inp, out, layer_idx=i: activations.update(
                    {f"layer_{layer_idx}_attn_qkv_proj": out.detach().cpu()}
                )
            )
        )
        hooks.append(
            model.transformer.h[i].attn.c_proj.register_forward_hook(
                lambda mod, inp, out, layer_idx=i: activations.update(
                    {f"layer_{layer_idx}_attn_output_proj": out.detach().cpu()}
                )
            )
        )
        hooks.append(
            model.transformer.h[i].ln_2.register_forward_hook(
                lambda mod, inp, out, layer_idx=i: activations.update(
                    {f"layer_{layer_idx}_ln_2": out.detach().cpu()}
                )
            )
        )
        hooks.append(
            model.transformer.h[i].mlp.register_forward_hook(
                lambda mod, inp, out, layer_idx=i: activations.update(
                    {f"layer_{layer_idx}_mlp": out.detach().cpu()}
                )
            )
        )
        hooks.append(
            model.transformer.h[i].mlp.c_fc.register_forward_hook(
                lambda mod, inp, out, layer_idx=i: activations.update(
                    {f"layer_{layer_idx}_mlp_fc": out.detach().cpu()}
                )
            )
        )
        hooks.append(
            model.transformer.h[i].mlp.gelu.register_forward_hook(
                lambda mod, inp, out, layer_idx=i: activations.update(
                    {f"layer_{layer_idx}_mlp_gelu": out.detach().cpu()}
                )
            )
        )
        hooks.append(
            model.transformer.h[i].mlp.c_proj.register_forward_hook(
                lambda mod, inp, out, layer_idx=i: activations.update(
                    {f"layer_{layer_idx}_mlp_proj": out.detach().cpu()}
                )
            )
        )
        hooks.append(
            model.transformer.h[i].register_forward_hook(
                lambda mod, inp, out, layer_idx=i: activations.update(
                    {f"layer_{layer_idx}_output": out.detach().cpu()}
                )
            )
        )
    hooks.append(
        model.transformer.ln_f.register_forward_hook(
            lambda mod, inp, out: activations.update(
                {"final_layer": out.detach().cpu()}
            )
        )
    )
    return hooks, activations


def get_encoder():
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode


def process_all_samples(
    model,
    samples,
    output_dir,
    output_name,
    device="cpu",
    batch_size=32,
    checkpoint_interval=100,
):
    """
    Process all samples, maintaining running means without storing all raw activations.
    Args:
        model: The GPT model
        samples: DataFrame with 'text' column
        output_dir: Output directory
        output_name: Output file name (no extension)
        device: Device to run model on
        batch_size: Number of samples to process in memory at once
        checkpoint_interval: How often to save checkpoints of running means
    """
    encode, _ = get_encoder()
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(
        output_dir, f"{output_name}_processing_checkpoint.json"
    )
    start_idx = 0
    running_means = {}
    sample_counts = {}
    total_loss = 0.0
    total_loss_count = 0
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            start_idx = checkpoint.get("last_processed_idx", 0)
            print(f"Resuming from checkpoint at sample {start_idx}")
            for layer_name in checkpoint.get("layers", []):
                mean_file = os.path.join(output_dir, f"{layer_name}_running_mean.pt")
                count_file = os.path.join(output_dir, f"{layer_name}_count.pt")
                if os.path.exists(mean_file) and os.path.exists(count_file):
                    running_means[layer_name] = torch.load(
                        mean_file, weights_only=False
                    )
                    sample_counts[layer_name] = torch.load(
                        count_file, weights_only=False
                    )
            total_loss = checkpoint.get("total_loss", 0.0)
            total_loss_count = checkpoint.get("total_loss_count", 0)
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_idx = 0
            running_means = {}
            sample_counts = {}
            total_loss = 0.0
            total_loss_count = 0
    total_samples = len(samples)
    print(f"Processing {total_samples} samples for activation averaging.")
    samples_list = samples.iloc[start_idx:].to_dict(orient="records")
    for batch_idx in tqdm(
        range(0, len(samples_list), batch_size),
        desc=f"Processing activations",
    ):
        current_batch = samples_list[batch_idx : batch_idx + batch_size]
        for sample_idx, sample in enumerate(current_batch):
            global_idx = start_idx + batch_idx + sample_idx
            if not sample.get("text"):
                continue
            tokens = encode(sample["text"])
            x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            # print(f"Text: {sample['text']} | Token count: {x.size(1)}")
            if (
                hasattr(model.config, "block_size")
                and x.size(1) > model.config.block_size
            ):
                x = x[:, -model.config.block_size :]
            hooks, activations = register_comprehensive_hooks(model)
            with torch.no_grad():
                logits, loss = model(x, x)
                if loss is not None:
                    # print(f"Loss for sample: {loss.item()}")
                    total_loss += loss.item()
                    total_loss_count += 1
            for layer_name, activation in activations.items():
                activation = activation.float()
                if activation.dim() > 2:
                    activation = activation.mean(dim=1)
                if activation.dim() == 2:
                    activation = activation.mean(dim=0)
                activation = activation.view(-1)
                if layer_name in running_means:
                    count = sample_counts[layer_name]
                    old_mean = running_means[layer_name]
                    new_count = count + 1
                    new_mean = old_mean + (activation - old_mean) / new_count
                    running_means[layer_name] = new_mean
                    sample_counts[layer_name] = new_count
                else:
                    running_means[layer_name] = activation
                    sample_counts[layer_name] = 1
            for hook in hooks:
                hook.remove()
            if device != "cpu":
                torch.cuda.empty_cache()
            if (
                global_idx + 1
            ) % checkpoint_interval == 0 or global_idx == total_samples - 1:
                for layer_name, mean in running_means.items():
                    torch.save(
                        mean,
                        os.path.join(output_dir, f"{layer_name}_running_mean.pt"),
                    )
                    torch.save(
                        sample_counts[layer_name],
                        os.path.join(output_dir, f"{layer_name}_count.pt"),
                    )
                    if global_idx == total_samples - 1:
                        torch.save(
                            mean,
                            os.path.join(output_dir, f"{layer_name}_mean.pt"),
                        )
                with open(checkpoint_file, "w") as f:
                    checkpoint = {
                        "last_processed_idx": global_idx + 1,
                        "total_samples": total_samples,
                        "layers": list(running_means.keys()),
                        "total_loss": total_loss,
                        "total_loss_count": total_loss_count,
                    }
                    json.dump(checkpoint, f)
                # Save average loss at each checkpoint
                if total_loss_count > 0:
                    avg_loss = total_loss / total_loss_count
                    with open(
                        os.path.join(output_dir, f"{output_name}_average_loss.txt"), "w"
                    ) as f_loss:
                        f_loss.write(str(avg_loss) + "\n")
                print(f"Checkpoint saved at sample {global_idx + 1}/{total_samples}")
    for layer_name in running_means.keys():
        running_mean_file = os.path.join(output_dir, f"{layer_name}_running_mean.pt")
        count_file = os.path.join(output_dir, f"{layer_name}_count.pt")
        if os.path.exists(running_mean_file):
            os.remove(running_mean_file)
        if os.path.exists(count_file):
            os.remove(count_file)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    if total_loss_count > 0:
        avg_loss = total_loss / total_loss_count
        print(f"Average loss over all samples: {avg_loss}")
        with open(
            os.path.join(output_dir, f"{output_name}_average_loss.txt"), "w"
        ) as f:
            f.write(str(avg_loss) + "\n")
    print(f"Completed processing. Averaged activations saved to {output_dir}'.")
    return running_means


def main():
    parser = argparse.ArgumentParser(
        description="Generalized neuronal activation averaging for a text dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output files",
    )
    parser.add_argument(
        "--parquet_file",
        type=str,
        default=None,
        help="Path to Parquet file with a 'text' column",
    )
    parser.add_argument(
        "--parquet_files",
        type=str,
        nargs="*",
        default=None,
        help="List of Parquet files with a 'text' column",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Prefix for output files (default: base name of parquet file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run model on (cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples to process in memory at once",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="How often to save checkpoints of running means",
    )
    args = parser.parse_args()
    create_output_dir(args.output_dir)
    # Collect all parquet files
    parquet_files = []
    if args.parquet_file:
        parquet_files.append(args.parquet_file)
    if args.parquet_files:
        parquet_files.extend(args.parquet_files)
    if not parquet_files:
        raise ValueError(
            "At least one of --parquet_file or --parquet_files must be provided."
        )
    # If more than one file, concatenate to a temp parquet file
    temp_file = None
    try:
        if len(parquet_files) == 1:
            parquet_to_use = parquet_files[0]
        else:
            samples_list = [load_text_dataset(f) for f in parquet_files]
            samples = pd.concat(samples_list, ignore_index=True)
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
            samples.to_parquet(temp.name)
            temp_file = temp.name
            temp.close()
            parquet_to_use = temp_file
        samples = load_text_dataset(parquet_to_use)
        model = load_model(args.model_path, args.device)
        output_name = args.output_name
        if output_name is None:
            if len(parquet_files) == 1:
                output_name = os.path.splitext(os.path.basename(parquet_files[0]))[0]
            else:
                output_name = "_and_".join(
                    [os.path.splitext(os.path.basename(f))[0] for f in parquet_files]
                )
        process_all_samples(
            model=model,
            samples=samples,
            output_dir=args.output_dir,
            output_name=output_name,
            device=args.device,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
        )
        print("All samples processed successfully.")
    finally:
        if temp_file is not None and os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    main()
