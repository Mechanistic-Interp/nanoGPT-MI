#!/usr/bin/env python3
"""
Prepare GPT-2 model for activation analysis
"""

import os
import torch
from model import GPT

# Make the output directory
os.makedirs("out", exist_ok=True)

# Download and create GPT-2 model
print("Loading pretrained GPT-2 model...")
model = GPT.from_pretrained("gpt2")

# Create a checkpoint dictionary compatible with tweet_activations.py
checkpoint = {
    "model": model.state_dict(),
    "model_args": {
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
        "block_size": model.config.block_size,
        "bias": model.config.bias,
        "vocab_size": model.config.vocab_size,
        "dropout": 0.0,  # Set to 0 for inference
    },
}

# Save the model to a file
print("Saving model to out/gpt2.pt")
torch.save(checkpoint, "out/gpt2.pt")
print("Done!")
