# Shutka V2: Ultra-Efficient TypeScript Coding Agent

Shutka is a state-of-the-art **VL-JEPA** (Vision-Language Joint Embedding Predictive Architecture) model specifically optimized for low-latency, high-precision TypeScript code assistance on local hardware.

## 🚀 Key Innovations (V2)

Shutka V2 incorporates cutting-edge architectural advancements to provide a premium coding experience:

- **VL-JEPA Paradigm**: Non-autoregressive representation learning using semantic patches.
- **Efficient Tokenizer (cl100k_base)**: Upgraded to 100k vocabulary for superior code compression and GPT-4 compatibility.
- **RMSNorm & SwiGLU**: Modern, hardware-efficient architecture components.
- **HTTP/2 Support**: Optimized API server for ultra-low latency streaming.
- **Dynamic FAISS Memory (RAG)**: Mutable memory bank with live API-based context injection.

## 📂 Project Structure

```bash
.
├── models/
│   └── shutka.py             # Shutka V2 (RMSNorm, RoPE, SwiGLU, BitLinear)
├── training/
│   ├── trainer.py            # Phase-aware trainer with GPU optimizations
│   ├── train_typescript.py   # PHASE 1: Syntax & Structure training script
│   ├── train_real_data.py    # PHASE 2: Instruction-following training script
│   ├── typescript_loader.py  # Rich semantic extractor (classes, types, etc.)
│   └── real_instruction_loader.py # Multi-source instruction streamer
├── evaluation/
│   ├── eval.py               # Main evaluation entry point
│   ├── evaluator.py          # Representation & Retrieval metrics
│   ├── test_syntax.py        # Bun-powered TS syntax verification
│   ├── test_programming.py   # Functional logic verification
│   ├── test_algorithmic.py   # Complex algorithmic tests
│   └── test_suites/          # JSON definitions for all tests
├── config.py                 # Unified hyperparameter management
├── evaluate_shutka.py        # Wrapper for evaluation runs
├── KAGGLE_GUIDE.md           # Cloud training blueprints
└── README.md                 # Project documentation
```

## 🛠️ Dynamic Memory Management

Shutka V2 features a mutable FAISS memory bank. You can manage the model's knowledge without retraining:

```python
from models.shutka import UltraEfficientTextJEPA

model = UltraEfficientTextJEPA()
bank = model.predictor.memory_bank

# 1. Add new knowledge
ids = bank.add_memory(new_embeddings, ["Updated API documentation..."])

# 2. Delete stale info
bank.delete_memory(ids)

# 3. Update existing entry
bank.update_memory(old_id, new_embeddings, "New implementation...")
```

## 🚀 Getting Started

### 1. Installation

## for CPU

```bash
# Optimized for Bun & Python 3.10+
pip install faiss-cpu datasets tiktoken torch numpy tqdm hypercorn h2
```

#### for GPU

```bash
# Optimized for Bun & Python 3.10+
pip install faiss-gpu datasets tiktoken torch numpy tqdm bitsandbytes fastapi hypercorn h2
```

### 2. OpenAI-Compatible API

Expose Shutka as a local server for **Cursor**, **VS Code**, or **Continue**:

```bash
python api_server.py
```

#### Recommended Configuration (continue)

configure continue vscode extension settings:

```yaml
name: Local Config
version: 1.0.0
schema: v1

models:
  - name: Shutka Local
    provider: openai
    model: shutka-v2
    apiBase: http://localhost:8000/v1
    apiKey: anything
```

### 4. Dynamic Context Injection

You can "teach" Shutka new information on the fly using the specialized memory endpoint:

```bash
curl -X POST http://localhost:8000/v1/memory \
     -H "Content-Type: application/json" \
     -d '{"text": "The new project structure uses Bun for execution..."}'
```

### 3. Recommended Training Sequence

To turn Shutka into a premium coding agent, we recommend a two-phase training approach:

#### Phase 1: Syntax (The Grammar)

Learn the syntax and structural patterns of TypeScript.

```bash
python training/train_typescript.py --max_samples 50000 --epochs 5
```

#### Phase 2: Instruction Following (The Agent)

Train the model to map natural language to code using real-world data.

```bash
python training/train_real_data.py --resume checkpoints/best_model.pt --epochs 10
```

### 4. Evaluation (Bun Optimized)

```bash
# Verify syntax and programming logic
python evaluation/eval.py --checkpoint checkpoints/best_model.pt
```

## 📊 Performance

Shutka V2 is designed to run on a **GTX 1050 (4GB)** or even purely on **CPUs** while maintaining high accuracy, achieving a ~2.0GB memory footprint in full training mode and <1GB during inference.

## 📄 References

- [VL-JEPA (Joint Embedding Predictive Architecture)](https://arxiv.org/abs/2512.10942)
- [RoPE (Rotary Positional Embeddings)](https://arxiv.org/abs/2104.09864)
- [Llama-3 Architecture (RMSNorm & SwiGLU)](https://ai.meta.com/blog/meta-llama-3/)
