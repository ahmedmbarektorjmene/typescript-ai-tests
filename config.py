"""
Configuration file for Shutka (VL-JEPA) training and evaluation
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """Training configuration for Shutka (VL-JEPA with Flash Linear Attention)"""
    # Flash Linear Attention Architecture parameters
    vocab_size: int = 50257   # Tiktoken (GPT-2) vocabulary size
    source_dim: int = 768     # Standard Base dimension
    source_depth: int = 12    # Deeper encoder for better semantics
    target_dim: int = 768     # Match source dim
    target_depth: int = 6     # Efficient target encoder
    predictor_dim: int = 768  # Match dimensions
    predictor_depth: int = 6  # Efficient predictor
    output_dim: int = 1536    # Projection dimension
    temperature: float = 0.07 # InfoNCE temperature

    # Sequence & Tiling (Flash Linear Attention)
    max_source_len: int = 4096   # Supported Context (can go higher with O(N))
    max_target_len: int = 512    # Target prediction length
    chunk_size: int = 128        # Chunk size for Flash Linear Attention
    
    # Training parameters
    learning_rate: float = 3e-4  # Slightly higher for BPE
    batch_size: int = 8          # Adjusted for larger dim
    num_epochs: int = 10
    optimizer: str = "galore"    # Keep GaLore for memory efficiency
    weight_decay: float = 0.01

    # Data parameters
    data_dir: str = "data"
    train_split: float = 0.9
    tokenizer: str = "gpt2"      # Use tiktoken gpt2/cl100k_base

    # RAG parameters (Huge Memory)
    use_rag: bool = True
    rag_index_type: str = "IVF4096,Flat" # Efficient clustering for huge index
    rag_storage_mode: str = "mmap"       # Load from disk, not RAM

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000
    eval_every: int = 500

    # Device optimization
    num_workers: int = 4
    pin_memory: bool = True
    gradient_checkpointing: bool = True

@dataclass
class EvaluationConfig:
    """Evaluation configuration for Shutka"""
    checkpoint_path: str = ""
    test_suite_dir: str = "evaluation/test_suites"
    results_dir: str = "results"
    max_gen_length: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    use_rag: bool = True
