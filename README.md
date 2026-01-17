# Shutka V2: Ultra-Efficient Text-JEPA Model

Shutka is an **Ultra-Efficient Text-JEPA** (Joint Embedding Predictive Architecture) model designed for high-performance text understanding and generation on local hardware, including low-end GPUs and CPUs.

## 🚀 Key Features

### Core Architecture
- **Text-JEPA Paradigm**: Non-autoregressive representation learning with predictive embeddings
- **BitNet 1.58 Quantization**: Ternary weights {-1, 0, 1} for 8-16x memory reduction and 2-3x CPU speedup
- **Lightning Attention 2**: O(N) complexity tiled linear attention with intra/inter block processing
- **Modern Components**: RMSNorm, SwiGLU activation, RoPE positional embeddings
- **FAISS RAG**: Dynamic memory bank with mutable knowledge injection

### Enhanced Architecture (CPU-Optimized)
- **Titans Memory**: Test-time learnable memory with surprise-based updates
  - 10,000 memory slots with MLP-based storage
  - Surprise threshold: 0.5 for efficient updates
  - Persistent save/load for long-term learning
- **MIRAS**: Three-tier retrieval system (Titans → FAISS → Bing Search)
  - Adaptive confidence thresholds
  - Project-specific context isolation
- **HopRAG**: Multi-hop reasoning with adaptive sufficiency learning
  - Up to 3 reasoning hops
  - Neural network learns optimal stopping threshold (0.6-0.9 range)
- **Unified Hybrid Attention**: Combines DeBERTa, TransMLA, ELFATT, and Lightning Attention 2
  - Content/position separation (DeBERTa)
  - KV cache compression (TransMLA)
  - Kernel approximation (ELFATT)
  - Tiled processing (Lightning Attention 2)

### Dynamic Optimization
- **EAU (Evaluator Adjuster Unit)**: Dynamic complexity assessment
- **Gated Residuals**: Learned information flow control
- **Block Skipping**: Skip computation on simple inputs (CPU efficiency)
- **mHC (Manifold-Constrained Hyper-Connections)**: Improved gradient flow
- **Memory as Layer (MAL)**: Memory-augmented MLP replacement

### Training Optimizations
- **Gradient Checkpointing**: 30-50% memory reduction during training
- **BitNet 1.58 Quantization**: 8-16x memory reduction, 2-3x CPU speedup
  - Ternary weights {-1, 0, 1}
  - 8-bit activation quantization
  - Straight-through estimator for gradients
- **GaLore Optimizer**: Low-rank gradient projection for memory efficiency
  - Rank: 128, Update frequency: 200 steps
  - CANS orthogonalization for fast convergence
- **Mixed Precision**: Automatic FP16/BF16 with dynamic loss scaling
  - BF16 on Ampere+ GPUs (RTX 30xx/40xx)
  - FP16 on older GPUs (GTX 1050, Tesla T4)
- **torch.compile**: 2-3x inference speedup with PyTorch 2.0+

## 📂 Project Structure

```bash
.
├── models/
│   └── shutka.py             # UltraEfficientTextJEPA model with all enhancements
├── training/
│   ├── trainer.py            # GPU/CPU optimized trainer with mixed precision
│   ├── train_typescript.py   # TypeScript/JavaScript training script
│   ├── train_real_data.py    # Instruction-following training script
│   ├── typescript_loader.py  # TypeScript semantic data loader
│   └── real_instruction_loader.py # Multi-source instruction streamer
├── evaluation/
│   ├── eval.py               # Main evaluation entry point
│   ├── evaluator.py          # Comprehensive evaluation metrics
│   ├── test_syntax.py        # Syntax verification tests
│   ├── test_programming.py   # Programming logic tests
│   ├── test_algorithmic.py   # Complex algorithmic tests
│   └── test_suites/          # JSON test definitions
├── config.py                 # Unified configuration with validation
├── api_server.py             # OpenAI-compatible API server
├── export_model.py           # Export optimized inference models
└── README.md                 # This file
```

## 🛠️ Dynamic Memory Management

Shutka features a mutable FAISS memory bank and Titans Memory system:

```python
from models.shutka import UltraEfficientTextJEPA

model = UltraEfficientTextJEPA(use_enhanced_encoder=True)

# FAISS Memory Bank
bank = model.predictor.memory_bank
ids = bank.add_memory(embeddings, ["New knowledge..."])
bank.delete_memory(ids)
bank.update_memory(old_id, new_embeddings, "Updated info...")

# Titans Memory (test-time learning)
model.save_titans_memory("titans_state.pt")
model.load_titans_memory("titans_state.pt")
model.forget_titans_memory(n=100)  # Forget least accessed
```

## 🚀 Getting Started

### 1. Installation

**For CPU:**
```bash
pip install faiss-cpu datasets tiktoken torch numpy tqdm hypercorn h2 huggingface_hub
```

**For GPU:**
```bash
pip install faiss-gpu datasets tiktoken torch numpy tqdm bitsandbytes fastapi hypercorn h2 huggingface_hub
```

### 2. OpenAI-Compatible API Server

Start the local API server:

```bash
python api_server.py
```

**API Endpoints:**
- `POST /v1/chat/completions` - Chat completions with streaming
- `POST /v1/embeddings` - Generate embeddings
- `POST /v1/memory` - Add to FAISS memory bank
- `POST /v1/titans/save` - Save Titans Memory state
- `POST /v1/titans/load` - Load Titans Memory state
- `POST /v1/titans/forget?n=100` - Forget least accessed memories

**Project-Specific FAISS:**
- `POST /v1/faiss/create_project?project_name=myproject`
- `POST /v1/faiss/add_to_project?project_name=myproject`
- `GET /v1/faiss/list_projects`
- `DELETE /v1/faiss/delete_project?project_name=myproject`

### 3. Configuration for Continue vscode extension

```yaml
```

### 4. Training

**Phase 1: TypeScript Syntax Learning**
```bash
python training/train_typescript.py --epochs 3 --batch_size 4
```

**Phase 2: Instruction Following**
```bash
python training/train_real_data.py --resume checkpoints/best_model.pt --epochs 10
```

### 5. Evaluation

```bash
python evaluation/eval.py --checkpoint checkpoints/best_model.pt
```

### 6. Export Optimized Model

```bash
python export_model.py --checkpoint checkpoints/best_model.pt --output models/shutka.pt
```

## ⚙️ Configuration

Enable all features in `config.py`:

```python
config = TrainingConfig(
    # Enhanced Architecture
    use_enhanced_encoder=True,    # CPU-optimized architecture
    use_titans=True,              # Titans Memory
    use_miras=True,               # MIRAS retrieval
    use_hoprag=True,              # HopRAG multi-hop reasoning
    bing_api_key=None,            # Optional Bing Search
    
    # Memory Configuration
    titans_capacity=10000,        # Memory slots
    titans_depth=3,               # MLP depth
    titans_surprise_threshold=0.5,
    miras_confidence_threshold=0.7,
    hoprag_max_hops=3,
    
    # Training Optimizations
    gradient_checkpointing=True,
    batch_size=8,
    learning_rate=3e-4,
    optimizer="galore",           # GaLore optimizer
)
```

## 🔧 Hardware Compatibility

**Minimum Requirements:**
- CPU: Any modern x64 processor
- RAM: 4GB (CPU mode), 8GB recommended
- GPU: Optional, GTX 1050+ or any CUDA-capable GPU


## 📊 Performance Benchmarks

**Model Sizes:**
- **Fast Mode**: ~350M parameters (320d, 6+3+3 layers)
- **Standard Mode**: ~1B parameters (768d, 12+6+6 layers)
- **Large Mode**: ~3B parameters (1024d, 24+12+12 layers)

**Key Optimizations:**
- BitNet 1.58: 8-16x memory reduction
- Lightning Attention 2: O(N) complexity
- torch.compile: 2-3x inference speedup
- Mixed precision: 2x training speedup
- Gradient checkpointing: 50% memory reduction

## 🧪 Backward Compatibility

Load old checkpoints with new architecture:

```python
model = UltraEfficientTextJEPA(use_enhanced_encoder=True)
checkpoint = torch.load("old_checkpoint.pt")
model.load_state_dict_with_compatibility(checkpoint['model_state_dict'])
```

Old parameters are automatically mapped. New components are randomly initialized.

## 📄 Research Papers

This model is based on the following research papers:

### Core Architecture
- [ELFATT: Efficient Linear Fast Attention for Vision Transformers](https://arxiv.org/html/2501.06098v4)
- [DeBERTa: Decoding-Enhanced BERT with Disentangled Attention](https://arxiv.org/pdf/2006.03654)
- [Dynamic Context Adaptation and Information Flow Control in Transformers: Introducing the Evaluator Adjuster Unit and Gated Residual Connections](https://arxiv.org/html/2405.13407v1)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)

### Memory and Retrieval Systems
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/html/2501.00663v1)
- [Titans + MIRAS: Helping AI have long-term memory](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
- [HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval-Augmented Generation](https://arxiv.org/html/2502.12442v2)

### Attention Mechanisms
- [Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths](https://arxiv.org/html/2506.13585v1)
- [TransMLA: Multi-Head Latent Attention Is All You Need](https://arxiv.org/html/2502.07864v5)

### Training Optimizations
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/html/2403.03507v2)
- [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/html/2412.09871v1)
- [Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials](https://arxiv.org/html/2506.10935v1)

### Model Architecture Enhancements
- [GM-Skip: Metric-Guided Transformer Block Skipping for Efficient Vision-Language Models](https://arxiv.org/html/2508.18227v1)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/html/2402.03300v3) (for mHC - Manifold-Constrained Hyper-Connections)

### Foundation
- [Text-JEPA: Joint Embedding Predictive Architecture](https://arxiv.org/abs/2301.08727)
- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453)
