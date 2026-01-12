# Kaggle Training Guide for Shutka (VL-JEPA)

This guide shows how to set up and run the training on a Kaggle Notebook (Single or Dual T4 GPU).

## 1. Setup Environment

In a Kaggle cell:

```python
!pip install faiss-cpu datasets tiktoken torch numpy tqdm bitsandbytes huggingface_hub

```

## 2. Directory Structure

```bash
/kaggle/working/
  ├── models/
  │   └── shutka.py             # Core Architecture
  ├── training/
  │   ├── __init__.py
  │   ├── trainer.py            # Phase-aware trainer
  │   ├── train_typescript.py   # PHASE 1 Script
  │   ├── train_real_data.py    # PHASE 2 Script
  │   ├── typescript_loader.py  # Data loader
  │   └── real_instruction_loader.py
  ├── evaluation/
  │   ├── __init__.py
  │   ├── eval.py               # Main Eval script
  │   ├── evaluator.py
  │   ├── test_syntax.py
  │   ├── test_programming.py
  │   └── test_algorithmic.py
  ├── config.py                 # Unified Config
  ├── evaluate_shutka.py        # Wrapper script
  ├── KAGGLE_GUIDE.md
  └── README.md
```

## 3. Launch Phase 1: TypeScript Syntax

Learning the grammar of code (interfaces, types, classes).

```python
!python training/train_typescript.py \
    --max_samples 50000 \
    --batch_size 16 \
    --epochs 10 \
    --checkpoint_dir /kaggle/working/checkpoints_syntax
```

## 4. Launch Phase 2: Instruction Following

Training the "Intent" (Natural Language -> Code).

```python
!python training/train_real_data.py \
    --resume /kaggle/working/checkpoints_syntax/best_model.pt \
    --code_samples 10000 \
    --instruction_samples 10000 \
    --batch_size 8 \
    --epochs 20 \
    --checkpoint_dir /kaggle/working/checkpoints_agent
```

## 6. Dynamic Memory Hygiene

Shutka V2 uses a mutable FAISS bank. Proper memory management ensures your coding agent remains accurate.

### When to Update the Bank

- **Library Updates**: If a major TS library (e.g., React, Zod) changes syntax, delete old snippets and add new ones.
- **Refactoring**: When local codebases change significantly, remove stale latent representations.
- **Improved Encoder**: If you train a new checkpoint with a different architecture, you **must** re-encode and re-add all snippets (latent alignment).

### Memory Management Workflow

```python
from models.shutka import UltraEfficientTextJEPA

model = UltraEfficientTextJEPA()
bank = model.predictor.memory_bank

# Add (Appends to FAISS + Metadata)
ids = bank.add_memory(embeddings, ["interface User { id: string }"])

# Delete (Remove by ID)
bank.delete_memory(ids)

# Update (Delete + Add workflow)
bank.update_memory(old_id, new_embeddings, ["interface User { id: number }"])
```

## 7. Persistence and Scale

- **MMAP mode**: Shutka uses `IO_FLAG_MMAP`, meaning FAISS indices are mapped to disk. You don't need 32GB of RAM to search a 32GB index.
- **Auto-Save**: The `bank.auto_save` feature automatically flushes changes to `/kaggle/working/memory_bank` after every add/delete.

### Tip: Kaggle Output Persistence

Remember, `/kaggle/working` is wiped after 12 hours of inactivity. Always download your `shutka_v2_bundle.zip` at the end of your session.

## 8. Development: OpenAI API Mode

To use your trained Kaggle model in your local IDE, download the checkpoint and run:

```bash
python api_server.py
```

This starts a FastAPI server compatible with OpenAI's Chat, Completions, and Embeddings specs. Point **Cursor** or **VS Code** to `http://localhost:8000/v1` with model `shutka-v2`.
