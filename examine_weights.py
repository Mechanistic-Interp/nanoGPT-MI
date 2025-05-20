import torch
import argparse
from model import GPT, GPTConfig


def examine_model(model_path=None):
    # Load the model
    if model_path:
        print(f"\nLoading model from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")

        # Check what type of checkpoint format we have
        if "model_args" in checkpoint:
            model_args = checkpoint["model_args"]
            print(f"Found model_args: {model_args}")

            # Create a GPTConfig object with the correct parameters
            config = GPTConfig(
                block_size=model_args.get("block_size", 1024),
                vocab_size=model_args.get("vocab_size", 50304),
                n_layer=model_args.get("n_layer", 12),
                n_head=model_args.get("n_head", 12),
                n_embd=model_args.get("n_embd", 768),
                dropout=model_args.get("dropout", 0.0),
                bias=model_args.get("bias", True),
            )

            # Initialize model with the config
            model = GPT(config)

            # Load weights
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                # Try loading the full checkpoint
                try:
                    model.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"Error loading model weights: {e}")
                    print("Keys in checkpoint:", checkpoint.keys())
                    return

            print(f"Model loaded from checkpoint with config: {config}")
        else:
            # Might be a simple state dict
            try:
                # Create default config
                config = GPTConfig()
                model = GPT(config)
                model.load_state_dict(checkpoint)
                print(f"Model loaded with default config")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Keys in checkpoint:", checkpoint.keys())
                return
    else:
        print("\nLoading pre-trained GPT-2 model")
        model = GPT.from_pretrained("gpt2")

    # Print model structure and weights
    print("\nModel Structure:")
    print("=" * 50)
    for name, param in model.named_parameters():
        print(f"\nLayer: {name}")
        print(f"Shape: {param.shape}")
        print(f"First few values: {param.data.flatten()[:5]}")
        print(f"Mean: {param.data.mean():.4f}")
        print(f"Std: {param.data.std():.4f}")
        print("-" * 50)

    # Determine output filename
    output_file = "model_weights.pt" if model_path else "gpt2_weights.pt"

    # Save weights to a file for inspection
    torch.save(model.state_dict(), output_file)
    print(f"\nWeights saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examine model weights")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (default: load pre-trained GPT-2)",
    )
    args = parser.parse_args()

    examine_model(args.model)
