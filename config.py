"""
Configuration file for Shutka (VL-JEPA) training and evaluation
"""
import os
import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple


def validate_config(config: 'TrainingConfig') -> Tuple[bool, List[str]]:
    """
    Validate configuration for hardware compatibility and logical consistency.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Hardware capability checks
    cuda_available = torch.cuda.is_available()
    
    # Check dimension compatibility
    if config.source_dim % 8 != 0:
        errors.append(f"source_dim ({config.source_dim}) must be divisible by 8 for efficient computation")
    
    if config.target_dim % 8 != 0:
        errors.append(f"target_dim ({config.target_dim}) must be divisible by 8 for efficient computation")
    
    if config.predictor_dim % 8 != 0:
        errors.append(f"predictor_dim ({config.predictor_dim}) must be divisible by 8 for efficient computation")
    
    # Check sequence length limits
    if config.max_source_len < 128:
        errors.append(f"max_source_len ({config.max_source_len}) should be at least 128 tokens")
    
    if config.max_target_len < 32:
        errors.append(f"max_target_len ({config.max_target_len}) should be at least 32 tokens")
    
    # Check batch size
    if config.batch_size < 1:
        errors.append(f"batch_size ({config.batch_size}) must be at least 1")
    
    if config.batch_size > 128:
        errors.append(f"batch_size ({config.batch_size}) is very large and may cause OOM errors")
    
    # Check learning rate
    if config.learning_rate <= 0 or config.learning_rate > 1.0:
        errors.append(f"learning_rate ({config.learning_rate}) should be between 0 and 1.0")
    
    # Check Titans Memory configuration
    if config.use_titans:
        if config.titans_capacity < 100:
            errors.append(f"titans_capacity ({config.titans_capacity}) should be at least 100")
        
        if config.titans_depth < 1 or config.titans_depth > 10:
            errors.append(f"titans_depth ({config.titans_depth}) should be between 1 and 10")
        
        if config.titans_surprise_threshold < 0 or config.titans_surprise_threshold > 1:
            errors.append(f"titans_surprise_threshold ({config.titans_surprise_threshold}) should be between 0 and 1")
    
    # Check MIRAS configuration
    if config.use_miras and not config.use_titans:
        errors.append("MIRAS requires Titans Memory to be enabled (use_titans=True)")
    
    if config.use_miras:
        if config.miras_confidence_threshold < 0 or config.miras_confidence_threshold > 1:
            errors.append(f"miras_confidence_threshold ({config.miras_confidence_threshold}) should be between 0 and 1")
    
    # Check HopRAG configuration
    if config.use_hoprag and not config.use_miras:
        errors.append("HopRAG requires MIRAS to be enabled (use_miras=True)")
    
    if config.use_hoprag:
        if config.hoprag_max_hops < 1 or config.hoprag_max_hops > 10:
            errors.append(f"hoprag_max_hops ({config.hoprag_max_hops}) should be between 1 and 10")
    
    # Check gradient checkpointing compatibility
    if config.gradient_checkpointing and not config.use_enhanced_encoder:
        errors.append("Gradient checkpointing requires enhanced encoder (use_enhanced_encoder=True)")
    
    # Check directory existence
    if not os.path.exists(config.data_dir):
        errors.append(f"data_dir '{config.data_dir}' does not exist")
    
    # Hardware-specific warnings (not errors)
    if not cuda_available and config.batch_size > 4:
        errors.append("WARNING: Running on CPU with batch_size > 4 may be very slow")
    
    if cuda_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        estimated_memory = (config.source_dim * config.source_depth * config.batch_size) / 1e6
        
        if estimated_memory > gpu_memory * 0.8:
            errors.append(f"WARNING: Estimated memory usage ({estimated_memory:.1f}GB) may exceed GPU memory ({gpu_memory:.1f}GB)")
    
    return len(errors) == 0, errors


def check_hardware_capabilities() -> dict:
    """
    Check hardware capabilities and return a summary.
    
    Returns:
        Dictionary with hardware information
    """
    capabilities = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': None,
        'device_memory_gb': None,
        'bf16_supported': False,
        'tf32_supported': False,
    }
    
    if torch.cuda.is_available():
        capabilities['device_name'] = torch.cuda.get_device_name(0)
        capabilities['device_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        capabilities['bf16_supported'] = torch.cuda.is_bf16_supported()
        
        # Check for TF32 support (Ampere and newer)
        compute_capability = torch.cuda.get_device_capability(0)
        capabilities['tf32_supported'] = compute_capability[0] >= 8
        capabilities['compute_capability'] = f"{compute_capability[0]}.{compute_capability[1]}"
    
    return capabilities

@dataclass
class TrainingConfig:
    """Training configuration for Shutka (JEPA with lightning Attention 2)"""
    # lightning Attention 2 Architecture parameters
    vocab_size: int = 100277  # Modern OpenAI (cl100k_base) vocabulary size
    source_dim: int = 768     # Standard Base dimension
    source_depth: int = 12    # Deeper encoder for better semantics
    target_dim: int = 768     # Match source dim
    target_depth: int = 6     # Efficient target encoder
    predictor_dim: int = 768  # Match dimensions
    predictor_depth: int = 6  # Efficient predictor
    output_dim: int = 1536    # Projection dimension
    temperature: float = 0.07 # InfoNCE temperature

    # Sequence & Tiling (lightning Attention 2)
    max_source_len: int = 4096   # Supported Context (can go higher with O(N))
    max_target_len: int = 512    # Target prediction length
    chunk_size: int = 128        # Chunk size for lightning Attention 2
    
    # Training parameters
    learning_rate: float = 3e-4  # Slightly higher for BPE
    batch_size: int = 8          # Adjusted for larger dim
    num_epochs: int = 10
    optimizer: str = "galore"    # Options: "galore", "adamw", "muon"
    weight_decay: float = 0.01

    # Data parameters
    data_dir: str = "data"
    train_split: float = 0.9

    # RAG parameters (Huge Memory)
    use_rag: bool = True
    rag_index_type: str = "IVF4096,Flat" # Efficient clustering for huge index
    rag_storage_mode: str = "mmap"       # Load from disk, not RAM
    
    # Enhanced Architecture (CPU-Optimized)
    use_enhanced_encoder: bool = True    # Enable CPU-optimized architecture
    use_titans: bool = True              # Titans Memory (test-time learning)
    use_miras: bool = True               # MIRAS three-tier retrieval
    use_hoprag: bool = True              # HopRAG multi-hop reasoning
    bing_api_key: Optional[str] = None   # Bing Search API key (optional)
    
    # Titans Memory Configuration
    titans_capacity: int = 10000         # Memory capacity
    titans_depth: int = 3                # Memory MLP depth
    titans_surprise_threshold: float = 0.5  # Update threshold
    
    # MIRAS Configuration
    miras_confidence_threshold: float = 0.7  # Retrieval confidence threshold
    
    # HopRAG Configuration
    hoprag_max_hops: int = 3             # Maximum reasoning hops
    hoprag_adaptive_threshold: bool = True  # Use adaptive sufficiency

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000
    eval_every: int = 500

    # Device optimization
    num_workers: int = 4
    pin_memory: bool = True
    gradient_checkpointing: bool = True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this configuration"""
        return validate_config(self)
    
    def print_validation_report(self):
        """Print a validation report"""
        is_valid, errors = self.validate()
        
        print("\n" + "=" * 60)
        print("CONFIGURATION VALIDATION REPORT")
        print("=" * 60)
        
        if is_valid:
            print("✅ Configuration is valid!")
        else:
            print("❌ Configuration has issues:")
            for i, error in enumerate(errors, 1):
                if error.startswith("WARNING"):
                    print(f"  ⚠️  {error}")
                else:
                    print(f"  {i}. {error}")
        
        print("\nHardware Capabilities:")
        caps = check_hardware_capabilities()
        print(f"  CUDA Available: {caps['cuda_available']}")
        if caps['cuda_available']:
            print(f"  Device: {caps['device_name']}")
            print(f"  Memory: {caps['device_memory_gb']:.1f}GB")
            print(f"  Compute Capability: {caps.get('compute_capability', 'N/A')}")
            print(f"  BF16 Support: {caps['bf16_supported']}")
            print(f"  TF32 Support: {caps['tf32_supported']}")
        
        print("=" * 60 + "\n")
        
        return is_valid

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
