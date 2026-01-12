"""
Train Shutka (VL-JEPA) on TypeScript/JavaScript Code

This script:
1. Streams TypeScript/JavaScript code from The Stack (HuggingFace)
2. Only downloads TypeScript and JavaScript - NOT the entire dataset!
3. Uses modern GPU optimizations (torch.compile, BF16, TF32)
4. Falls back gracefully to CPU if no GPU available

Usage:
    python train_typescript.py --max_samples 10000 --epochs 10
"""
import argparse
import os
import sys
import torch

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.shutka import UltraEfficientTextJEPA
from training.typescript_loader import create_typescript_dataloader
from training.trainer import Trainer
from config import TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train Shutka on TypeScript/JavaScript")
    
    # Data
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum training samples to stream")
    parser.add_argument("--languages", type=str, default="typescript,javascript",
                        help="Comma-separated languages to include")
    
    # Model
    parser.add_argument("--source_dim", type=int, default=768)
    parser.add_argument("--target_dim", type=int, default=768)
    parser.add_argument("--predictor_dim", type=int, default=768)
    parser.add_argument("--max_source_len", type=int, default=2048,
                        help="Max source context length")
    parser.add_argument("--max_target_len", type=int, default=512,
                        help="Max target code length")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--use_rag", action="store_true", default=True,
                        help="Enable FAISS RAG memory")
    
    # Optimization
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--no_mixed_precision", action="store_true",
                        help="Disable mixed precision training")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=250)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("SHUTKA (VL-JEPA) TypeScript/JavaScript Training")
    print("="*60)
    
    # Parse languages
    languages = [lang.strip() for lang in args.languages.split(",")]
    print(f"\nLanguages: {languages}")
    print(f"Max samples: {args.max_samples}")
    
    # Create dataloaders (streams from The Stack - only TS/JS!)
    print("\n" + "-"*40)
    print("Creating TypeScript/JavaScript dataloaders...")
    print("-"*40)
    
    train_loader, val_loader = create_typescript_dataloader(
        batch_size=args.batch_size,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        buffer_size=args.max_samples,
        languages=languages
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "-"*40)
    print("Initializing Shutka model...")
    print("-"*40)
    
    model = UltraEfficientTextJEPA(
        vocab_size=100277,  # cl100k_base vocab size (GPT-4 tokenizer)
        source_dim=args.source_dim,
        target_dim=args.target_dim,
        predictor_dim=args.predictor_dim,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        use_rag=args.use_rag
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
    
    # Create trainer with GPU optimizations
    print("\n" + "-"*40)
    print("Initializing trainer with optimizations...")
    print("-"*40)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        use_compile=not args.no_compile,
        use_mixed_precision=not args.no_mixed_precision
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train!
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving checkpoint...")
        trainer.save_checkpoint("interrupted_checkpoint.pt")
        print("Checkpoint saved.")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best model saved to: {args.checkpoint_dir}/best_model.pt")
    print("="*60)


if __name__ == "__main__":
    main()
