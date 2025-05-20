#!/usr/bin/env python3
"""
Interactive CLI for emotion-enhanced GPT models
"""

import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model import GPTConfig, GPT


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Interactive CLI for emotion-enhanced GPT models"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="out/joy_enhanced_model.pt",
        help="Path to the emotion-enhanced model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling (lower = more deterministic)",
    )
    parser.add_argument(
        "--top_k", type=int, default=200, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cpu, cuda)",
    )
    args = parser.parse_args()

    # Initialize the device & precision settings
    device = args.device
    device_type = "cuda" if "cuda" in device else "cpu"

    if device_type == "cuda":
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
    else:
        dtype = "float32"

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # Allow TF32 for faster computation if available
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load the model
    print(f"Loading model from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)

        # Try to determine model configuration
        if "model_args" in checkpoint:
            # Standard format
            model_args = checkpoint["model_args"]
        elif "config" in checkpoint and "model_args" in checkpoint["config"]:
            # Config nested in 'config'
            model_args = checkpoint["config"]["model_args"]
        else:
            print(
                "Model format not recognized. Available keys:", list(checkpoint.keys())
            )
            # Use default GPT-2 config
            model_args = {
                "n_layer": 12,  # Default for GPT-2 small
                "n_head": 12,  # Default for GPT-2 small
                "n_embd": 768,  # Default for GPT-2 small
                "vocab_size": 50257,  # GPT-2 vocabulary size
                "block_size": 1024,  # Default context size
                "bias": True,  # Use bias terms
                "dropout": 0.1,  # Default dropout
            }
            print(f"Using default GPT-2 config: {model_args}")

        # Initialize model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        # Get the state dict
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Clean up state dict if needed
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        print("Model loaded successfully")

        # Get model metadata if available
        if "emotion_modification_metadata" in checkpoint:
            emotion_metadata = checkpoint["emotion_modification_metadata"]
            print(f"Model enhanced for emotion: {emotion_metadata['target_emotion']}")
            print(f"Scaling factor: {emotion_metadata['scaling_factor']}")
            if "normalize" in emotion_metadata:
                print(
                    f"Normalization: {'enabled' if emotion_metadata['normalize'] else 'disabled'}"
                )
            print(
                f"Modified {len(emotion_metadata['modification_details'])} components"
            )
        else:
            print("No emotion enhancement metadata found")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize the tokenizer (GPT-2 encoding)
    print("Initializing GPT-2 tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    # Interactive CLI loop
    print("\n" + "=" * 50)
    print("Emotion-Enhanced GPT Interactive CLI")
    print("Type your prompt and press Enter. Type 'quit' or 'exit' to end.")
    print("Type 'settings' to view/change generation settings.")
    print("=" * 50 + "\n")

    # Store current settings
    settings = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }

    with torch.no_grad():
        while True:
            # Get user input
            user_input = input("Your prompt > ")

            # Check for exit command
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Check for settings command
            if user_input.lower() == "settings":
                print("\nCurrent settings:")
                for k, v in settings.items():
                    print(f"  {k}: {v}")

                change = input("\nChange settings? (y/n) > ")
                if change.lower() == "y":
                    try:
                        max_tokens = input(f"Max tokens [{settings['max_tokens']}] > ")
                        if max_tokens.strip():
                            settings["max_tokens"] = int(max_tokens)

                        temp = input(f"Temperature [{settings['temperature']}] > ")
                        if temp.strip():
                            settings["temperature"] = float(temp)

                        top_k = input(f"Top-k [{settings['top_k']}] > ")
                        if top_k.strip():
                            settings["top_k"] = int(top_k)

                        print("\nSettings updated:")
                        for k, v in settings.items():
                            print(f"  {k}: {v}")
                    except ValueError as e:
                        print(f"Error updating settings: {e}")

                print()  # Add a newline for spacing
                continue

            # If input is empty, skip
            if not user_input.strip():
                continue

            # Encode input
            input_ids = encode(user_input)
            x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]

            # Generate response (streaming)
            print("\nModel response:\n" + "-" * 20)
            with ctx:
                generated = x
                for _ in range(settings["max_tokens"]):
                    y = model.generate(
                        generated,
                        1,  # generate one token at a time
                        temperature=settings["temperature"],
                        top_k=settings["top_k"],
                    )
                    new_token = y[0, -1].item()
                    # Stop if endoftext token is generated
                    if new_token == enc.eot_token:  # tiktoken's endoftext
                        break
                    print(decode([new_token]), end="", flush=True)
                    generated = y
            print("\n" + "-" * 20 + "\n")


if __name__ == "__main__":
    main()
