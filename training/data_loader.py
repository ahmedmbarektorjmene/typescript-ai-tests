"""
Data loading utilities for code training
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import glob
from tokenizer.tokenizer import SimpleByteTokenizer, BytePairTokenizer


class CodeDataset(Dataset):
    """
    Dataset for loading code files
    """
    def __init__(self, file_paths: List[str], tokenizer, max_length: int = 512):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        
        # Load and tokenize all files
        self.data = []
        total_chunks = 0
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if not content.strip():
                        continue
                    tokens = tokenizer.encode(content)
                    # Split into chunks of max_length
                    # Include chunks even if they're shorter (we'll pad them)
                    for i in range(0, len(tokens), max_length):
                        chunk = tokens[i:i + max_length]
                        if len(chunk) > 1:  # Need at least 2 tokens for input/labels
                            # Pad to max_length if needed
                            if len(chunk) < max_length:
                                chunk = chunk + [self.pad_token_id] * (max_length - len(chunk))
                            self.data.append(chunk)
                            total_chunks += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        # If still empty, create at least one dummy sample
        if len(self.data) == 0:
            print("Warning: No valid data chunks found. Creating dummy sample.")
            dummy_tokens = tokenizer.encode("function test() { return 42; }")
            # Pad to max_length
            if len(dummy_tokens) < max_length:
                dummy_tokens = dummy_tokens + [self.pad_token_id] * (max_length - len(dummy_tokens))
            else:
                dummy_tokens = dummy_tokens[:max_length]
            self.data = [dummy_tokens] * 10  # Create 10 copies for training
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        # Ensure we have at least 2 tokens
        if len(tokens) < 2:
            tokens = tokens + [self.pad_token_id] * (2 - len(tokens))
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def collect_code_files(data_dir: str, extensions: List[str] = ['.ts', '.js', '.tsx', '.jsx', '.txt']) -> List[str]:
    """
    Collect all code files from a directory
    """
    file_paths = []
    for ext in extensions:
        pattern = os.path.join(data_dir, '**', f'*{ext}')
        file_paths.extend(glob.glob(pattern, recursive=True))
    return file_paths


def create_dataloader(
    data_dir: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    train_split: float = 0.9,
    num_workers: Optional[int] = None
):
    """
    Create train and validation dataloaders
    """
    # Collect code files
    file_paths = collect_code_files(data_dir)
    
    if len(file_paths) == 0:
        # Create dummy data if no files found
        print("Warning: No code files found. Creating dummy dataset.")
        dummy_data = ["function test() { return 42; }"] * 100
        file_paths = []
        for i, content in enumerate(dummy_data):
            dummy_path = os.path.join(data_dir, f"dummy_{i}.ts")
            os.makedirs(data_dir, exist_ok=True)
            with open(dummy_path, 'w') as f:
                f.write(content)
            file_paths.append(dummy_path)
    
    # Split into train/val
    split_idx = int(len(file_paths) * train_split)
    train_files = file_paths[:split_idx]
    val_files = file_paths[split_idx:]
    
    # Create datasets
    print(f"  Creating datasets from {len(train_files)} train files and {len(val_files)} val files...")
    train_dataset = CodeDataset(train_files, tokenizer, max_length)
    val_dataset = CodeDataset(val_files, tokenizer, max_length)
    
    print(f"  Train dataset size: {len(train_dataset)} samples")
    print(f"  Val dataset size: {len(val_dataset)} samples")
    
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty! Please check your data files or use setup_data.py to create sample data.")
    if len(val_dataset) == 0:
        print("Warning: Validation dataset is empty. Using train dataset for validation.")
        val_dataset = train_dataset
    
    # Auto-detect optimal num_workers and pin_memory for GPU
    import torch
    if num_workers is None:
        if torch.cuda.is_available():
            num_workers = 2  # Good for GPU
            pin_memory = True
        else:
            num_workers = 0  # CPU training
            pin_memory = False
    else:
        pin_memory = torch.cuda.is_available() and num_workers > 0
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
