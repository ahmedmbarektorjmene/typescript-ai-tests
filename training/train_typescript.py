"""
Train Shutka (VL-JEPA) on TypeScript/JavaScript Code - T4 GPU Optimized

Optimizations for Kaggle T4 GPU:
1. Reduced model size (350M params)
2. Gradient accumulation for larger effective batch size
3. Mixed precision (BF16/FP16)
4. Gradient checkpointing to save memory
5. Optimized batch sizes for T4 (16GB VRAM)
6. Fast data loading with prefetching

Usage:
    python train_typescript.py --epochs 3 --batch_size 4
"""

import argparse
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.shutka import UltraEfficientTextJEPA
from training.typescript_loader import create_typescript_dataloader
from training.trainer import Trainer
from config import TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Shutka on TypeScript/JavaScript"
    )

    # Data
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=10000,
        help="Maximum training buffer size to stream",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="typescript,javascript",
        help="Comma-separated languages to include",
    )

    # Model (Shutka-v2 350M Spec)
    parser.add_argument(
        "--source_dim", type=int, default=512, help="Model dimension (512 for 350M)"
    )
    parser.add_argument("--target_dim", type=int, default=512)
    parser.add_argument("--predictor_dim", type=int, default=512)
    parser.add_argument(
        "--source_depth", type=int, default=24, help="Encoder layers (24 for 350M)"
    )
    parser.add_argument("--target_depth", type=int, default=4, help="Target layers")
    parser.add_argument(
        "--predictor_depth", type=int, default=4, help="Predictor layers"
    )
    parser.add_argument(
        "--max_source_len",
        type=int,
        default=1024,
        help="Max source context (1024 for T4)",
    )
    parser.add_argument(
        "--max_target_len", type=int, default=256, help="Max target length (256 for T4)"
    )

    # Training (T4-optimized)
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per GPU (4 for T4)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Accumulate gradients (effective batch = 4*4=16)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-5,
        help="Learning rate (lower for stability)",
    )
    parser.add_argument(
        "--use_rag", action="store_true", default=False, help="Disable RAG for speed"
    )

    # Optimization (T4-specific)
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable torch.compile (can be slow on T4)",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision (fp16 for T4)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing (saves memory)",
    )

    # Checkpointing (faster saves)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--save_every", type=int, default=1000, help="Save every N steps"
    )
    parser.add_argument(
        "--eval_every", type=int, default=500, help="Eval every N steps"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of data loader workers"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate batch size for InfoNCE loss
    if args.batch_size < 2:
        print(
            "⚠️  Warning: InfoNCE loss requires batch_size >= 2 for contrastive learning."
        )
        print("   Setting batch_size to 2 (minimum).")
        args.batch_size = 2

    print("=" * 60)
    print("SHUTKA Training - T4 GPU Optimized")
    print("=" * 60)
    print(
        f"Model: {args.source_dim}d, {args.source_depth}+{args.target_depth}+{args.predictor_depth} layers"
    )
    print(
        f"Batch: {args.batch_size} × {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps} effective"
    )
    print(f"Precision: {args.mixed_precision}")
    print(f"Workers: {args.num_workers}")

    # Parse languages
    languages = [lang.strip() for lang in args.languages.split(",")]
    print(f"\nLanguages: {languages}")
    print(f"Buffer Size: {args.buffer_size}")

    # Create dataloaders (streams from The Stack - only TS/JS!)
    print("\n" + "-" * 40)
    print("Creating dataloaders...")
    print("-" * 40)

    train_loader, val_loader = create_typescript_dataloader(
        batch_size=args.batch_size,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        buffer_size=args.buffer_size,
        num_workers=args.num_workers,
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model (T4-optimized: 350M params)
    print("\n" + "-" * 40)
    print("Initializing model...")
    print("-" * 40)

    model = UltraEfficientTextJEPA(
        vocab_size=100277,
        source_dim=args.source_dim,
        source_depth=args.source_depth,
        target_dim=args.target_dim,
        target_depth=args.target_depth,
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        output_dim=args.source_dim,  # Matches source dim
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        use_rag=args.use_rag,
        use_enhanced_encoder=True,
        use_titans=True,
        use_miras=False,  # Disable for speed
        use_hoprag=False,  # Disable for speed
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Create config
    config = TrainingConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.checkpoint_dir = args.checkpoint_dir
    config.save_every = args.save_every
    config.eval_every = args.eval_every
    config.max_source_len = args.max_source_len
    config.max_target_len = args.max_target_len
    config.gradient_checkpointing = args.gradient_checkpointing
    config.use_mixed_precision = args.mixed_precision != "no"

    # Create trainer with T4 optimizations
    print("\n" + "-" * 40)
    print("Initializing trainer...")
    print("-" * 40)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        use_compile=args.compile,
        use_mixed_precision=(args.mixed_precision != "no"),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving checkpoint...")
        trainer.save_checkpoint("interrupted_checkpoint.pt")
        print("Checkpoint saved.")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model saved to: {args.checkpoint_dir}/best_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
