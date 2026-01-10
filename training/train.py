"""
Main training script for all three models
"""
import argparse
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TrainingConfig
from tokenizer.tokenizer import SimpleByteTokenizer, BytePairTokenizer
from models.mamba2 import Mamba2Model
from models.rwkv_x import RWKVXModel
from models.xlstm import XLSTMModel
from training.trainer import Trainer
from training.data_loader import create_dataloader


def get_model(model_name: str, vocab_size: int, config: TrainingConfig):
    """Get model instance by name"""
    if model_name == 'mamba2':
        return Mamba2Model(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            d_state=config.mamba2_d_state,
            d_conv=config.mamba2_d_conv,
            expand=config.mamba2_expand,
        )

    elif model_name == 'rwkv_x':
        return RWKVXModel(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            attn_size=config.rwkv_x_attn_size,
            sparse_topk=config.rwkv_x_sparse_topk,
            max_seq_len=config.max_seq_len
        )
    elif model_name == 'xlstm':
        return XLSTMModel(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            head_dim=config.xlstm_head_dim,
            use_mlstm=config.xlstm_use_mlstm,
            max_seq_len=config.max_seq_len
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='Train sequence models')
    parser.add_argument('--model', type=str, required=True, choices=['mamba2', 'rwkv_x', 'xlstm'],
                        help='Model architecture to train')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing training data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Number of layers')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--tokenizer', type=str, default='byte', choices=['byte', 'bpe'],
                        help='Tokenizer type')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (auto=detect GPU, cpu=force CPU, cuda=force GPU)')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.data_dir = args.data_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.d_model = args.d_model
    config.n_layers = args.n_layers
    config.max_seq_len = args.max_seq_len
    
    # Create tokenizer
    if args.tokenizer == 'byte':
        tokenizer = SimpleByteTokenizer()
    else:
        tokenizer = BytePairTokenizer()
        # Note: In practice, you'd train the BPE tokenizer first
        # For now, we'll use it with base vocabulary
    
    config.vocab_size = tokenizer.vocab_size
    
    print(f"Training {args.model} model")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Model dim: {config.d_model}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Create dataloaders (num_workers will be auto-detected)
    print("\nLoading data...")
    train_loader, val_loader = create_dataloader(
        data_dir=config.data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_seq_len,
        train_split=config.train_split,
        num_workers=None  # Auto-detect based on GPU availability
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = get_model(args.model, config.vocab_size, config)
    
    # Determine device
    if args.device == 'auto':
        device = None  # Let trainer auto-detect
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    print(f"\nTraining completed! Best model saved to {config.checkpoint_dir}/best_model.pt")


if __name__ == '__main__':
    main()
