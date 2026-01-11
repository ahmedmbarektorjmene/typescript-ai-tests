"""
Configuration file for training and evaluation
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """Shared training configuration for all models"""
    # Model parameters
    vocab_size: int = 50257  # Will be set by tokenizer
    d_model: int = 512
    n_layers: int = 6
    max_seq_len: int = 512
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 8  # CPU-friendly batch size
    num_epochs: int = 10
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    
    # Data parameters
    data_dir: str = "data"
    train_split: float = 0.9
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000  # Save checkpoint every N steps
    eval_every: int = 500   # Evaluate every N steps
    
    # Device optimization
    num_workers: int = 0  # Set to 2-4 for GPU training
    pin_memory: bool = True  # Will be auto-set based on device
    gradient_checkpointing: bool = True
    
    # Model-specific parameters
    mamba2_d_state: int = 16
    mamba2_d_conv: int = 4
    mamba2_expand: int = 2
    
    # Enhanced Mamba2 parameters
    enhanced_mamba2_n_scales: int = 3
    enhanced_mamba2_n_symbols: int = 32
    enhanced_mamba2_symbol_dim: int = 64
    enhanced_mamba2_use_rule_aux_loss: bool = True
    
    rwkv_x_attn_size: int = 64
    rwkv_x_sparse_topk: int = 32
    
    xlstm_head_dim: int = 64
    xlstm_use_mlstm: bool = True  # Use matrix memory variant

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    checkpoint_path: str = ""
    test_suite_dir: str = "evaluation/test_suites"
    results_dir: str = "results"
    max_gen_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
