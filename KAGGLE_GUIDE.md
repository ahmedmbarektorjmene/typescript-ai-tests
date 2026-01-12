# Kaggle Training Guide for Shutka (VL-JEPA)

This guide shows how to set up and run the training on a Kaggle Notebook (Single or Dual T4 GPU).

## 1. Setup Environment

In a Kaggle cell:

```python
!pip install faiss-gpu datasets tiktoken torch numpy tqdm
```

## 2. Directory Structure

```bash
/kaggle/working/
  ├── models/
  │   └── shutka.py
  ├── training/
  │   ├── trainer.py
  │   ├── typescript_loader.py
  │   └── real_instruction_loader.py
  ├── config.py
  ├── train_typescript.py
  └── train_real_data.py
```

## 3. Launch Phase 1: TypeScript Syntax

Tearning the grammar of code (interfaces, types, classes).

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

## 5. Persistence

Kaggle only saves files in `/kaggle/working`.
The FAISS memory bank will be saved to `memory_bank/` in that directory.

### Tip: Download results

```python
import os
import shutil
shutil.make_archive('output', 'zip', '/kaggle/working')
from IPython.display import FileLink
FileLink('output.zip')
```
