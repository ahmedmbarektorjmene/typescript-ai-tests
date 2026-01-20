"""
Export Inference-Only Model with torch.compile Optimization
"""

import os
import torch
import argparse
from models.shutka import UltraEfficientTextJEPA


def export_inference_model(
    checkpoint_path: str,
    output_path: str,
    fast_mode: bool = True,
    optimize: bool = True,
    script: bool = True,
    verbose: bool = True,
):
    """
    Export a clean inference-only model.
    """
    if verbose:
        print("=" * 60)
        print("INFERENCE MODEL EXPORT (Shutka-v2)")
        print("=" * 60)
        print("\n[*] Strategy: Clean inference-only export")

    device = torch.device("cpu")

    # Configuration for ~350M parameters
    # source_dim=512, depth=24

    config = {
        "vocab_size": 100277,
        "source_dim": 512,
        "source_depth": 24,  # Main backbone
        "predictor_dim": 512,
        "predictor_depth": 4,  # Lighter predictor
        "output_dim": 512,
        "max_source_len": 4096,
        "engram_vocab_size": 370000,
    }

    if os.path.exists(checkpoint_path):
        if verbose:
            print(f"\n[*] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load config from checkpoint if available
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            if verbose:
                print("    • Loaded config from checkpoint")
            config = checkpoint["config"]
            # Ensure engram_vocab_size is set (backward compatibility)
            if "engram_vocab_size" not in config:
                config["engram_vocab_size"] = 370000
        else:
            if verbose:
                print(
                    "    ! Config not found in checkpoint. Using default 350M config."
                )

    # Validating Config against Weights (CRITICAL FIX for T4/350M models)
    if "model_state_dict" in checkpoint:
        sd = checkpoint["model_state_dict"]
        if "token_embed.weight" in sd:
            weight_dim = sd["token_embed.weight"].shape[1]
            if config.get("source_dim") != weight_dim:
                if verbose:
                    print(
                        f"    ! Config mismatch detected: config.source_dim={config.get('source_dim')} but weights are {weight_dim}."
                    )
                    print(
                        f"    ! Overriding config to match weights: source_dim={weight_dim}"
                    )
                config["source_dim"] = weight_dim
                config["target_dim"] = weight_dim
                config["predictor_dim"] = weight_dim
                config["output_dim"] = weight_dim

    model = UltraEfficientTextJEPA(**config)

    # Load weights (CRITICAL FIX)
    if "model_state_dict" in checkpoint:
        if verbose:
            print("    • Loading model weights...")
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        except Exception as e:
            print(f"    ! Warning loading weights: {e}")

    model.eval()

    # Optimization: Precompute mHC and strip training layers
    if hasattr(model, "precompute_mhc"):
        if verbose:
            print("    • Precomputing mHC matrices...")
        model.precompute_mhc()

    if hasattr(model, "strip_for_inference"):
        if verbose:
            print("    • Stripping training-only layers (Predictor/Target)...")
        model.strip_for_inference()

    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False

    if verbose:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n[*] Optimized Model Parameter Count: {param_count / 1e6:.1f}M")

    # Scripting Support
    if script:
        if verbose:
            print("\n[*] Attempting TorchScript Export...")
        try:
            # Create dummy inputs for tracing
            dummy_input = torch.randint(0, 1000, (1, 128), dtype=torch.long)
            dummy_hashes = torch.randint(
                0, config["engram_vocab_size"], (1, 128, 6), dtype=torch.long
            )

            # Trace the model
            traced_model = torch.jit.trace(model, (dummy_input, None, dummy_hashes))

            script_path = output_path.replace(".pt", "_script.pt")
            torch.jit.save(traced_model, script_path)
            if verbose:
                print(f"    • TorchScript model saved: {script_path} [OK]")
        except Exception as e:
            print(f"    ! TorchScript export failed: {e}")

    # Standard Save with Ternary Weights
    # BitNet 1.58-bit: weights are {-1, 0, 1}
    if verbose:
        print("\n[*] Saving model in native Ternary (1.58-bit) format...")

    export_data = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "inference_only": True,
        "optimized": optimize,
        "quantization": "bitnet_1.58",
    }

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    torch.save(export_data, output_path)

    if verbose:
        print(f"\n[OK] Export complete: {output_path}")
        print(
            "    💡 Note: Model is saved in Ternary format. Load with UltraEfficientTextJEPA."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="models/shutka.pt")
    parser.add_argument("--no-script", action="store_true")
    args = parser.parse_args()

    export_inference_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        script=not args.no_script,
    )
