# Shutka (VL-JEPA) Training and Evaluation Framework

This project implements and evaluates **Shutka**, an Ultra-Efficient VL-JEPA (Vision-Language Joint Embedding Predictive Architecture) model.

**VL-JEPA** is a paradigm shift from traditional token-based language models to patch-based representation learning. Instead of predicting next tokens, VL-JEPA predicts target patch representations from source patches, creating rich joint embeddings.

**Shutka** combines:
- **BitLinear 1.58b**: Ternary weight quantization (8-16x memory reduction)
- **Linear Attention**: O(n) complexity instead of O(n²)
- **FAISS RAG**: External memory for knowledge without model bloat
- **VL-JEPA Architecture**: Patch-based representation learning

The model is optimized for low-end hardware (GTX 1050 / CPU) with ~500MB memory footprint and 2-5x faster inference than traditional transformers.

## Project Structure

```
.
├── models/              # Model implementations
│   └── shutka.py       # Shutka (VL-JEPA) architecture
├── training/           # Training scripts
│   ├── train.py        # Main training loop
│   ├── trainer.py      # Trainer class
│   └── data_loader.py  # VL-JEPA patch-based data loading
├── evaluation/         # Evaluation scripts
│   ├── evaluator.py    # VL-JEPA representation quality evaluation
│   └── test_suites/    # Test suites (if needed)
├── evaluate_shutka.py  # Shutka evaluation script
├── data/               # Data directory (text files for patch extraction)
├── checkpoints/        # Model checkpoints
└── results/            # Evaluation results

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up training data (optional - creates sample data if directory is empty):
```bash
python setup_data.py --data_dir data/ --num_files 50
```

## Usage

### Training

Train Shutka on text data (VL-JEPA works with patches extracted from text):

```bash
python training/train.py --data_dir data/ --epochs 10 --batch_size 8
```

Training options:
- `--data_dir`: Directory containing text files for patch extraction
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (default: 8, CPU-friendly)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--source_dim`: Source encoder dimension (default: 768)
- `--target_dim`: Target encoder dimension (default: 768)
- `--predictor_dim`: Predictor dimension (default: 768)
- `--max_source_len`: Maximum source sequence length (default: 16384)
- `--max_target_len`: Maximum target sequence length (default: 512)
- `--use_rag`: Enable FAISS retrieval-augmented generation (default: True)
- `--resume`: Resume from checkpoint path

### Evaluation

Evaluate Shutka model representation quality and retrieval capability:

```bash
python evaluate_shutka.py --checkpoint checkpoints/best_model.pt
```

Evaluation options:
- `--checkpoint`: Path to Shutka checkpoint
- `--results_dir`: Directory to save results (default: `results`)

VL-JEPA evaluation measures:
- **Representation Quality**: How well patches cluster by semantic similarity
- **Retrieval Accuracy**: Effectiveness of FAISS-based memory retrieval
- **Composite Score**: Overall VL-JEPA performance metric

## VL-JEPA Evaluation

Shutka evaluation focuses on representation learning quality rather than text generation:

1. **Representation Quality**: Measures how well the model clusters semantically similar patches
2. **Retrieval Capability**: Tests FAISS-based memory retrieval effectiveness
3. **Composite Score**: Overall VL-JEPA performance metric

Results are saved as JSON files in the `results/` directory with detailed metrics for representation quality and retrieval accuracy.

## Key Innovations

- **VL-JEPA Paradigm**: Learns from patches rather than token prediction
- **BitLinear 1.58b**: Extreme quantization for memory efficiency
- **Linear Attention**: O(n) complexity enables long sequences
- **FAISS RAG**: External knowledge without model parameters
- **CPU/GPU Optimized**: Runs on GTX 1050 or modern CPUs

## References

- [VL-JEPA: Vision-Language Joint Embedding Predictive Architecture](https://arxiv.org/abs/2512.10942)
- [BitLinear: 1.58-bit Quantization for Efficient Language Models](https://arxiv.org/abs/2402.17764)
- [Linear Attention: Efficient Attention with O(n) Complexity](https://arxiv.org/abs/2006.04768)
- [FAISS: Efficient Similarity Search](https://github.com/facebookresearch/faiss)
