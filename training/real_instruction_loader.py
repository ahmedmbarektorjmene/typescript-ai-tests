"""
REAL Instruction-Code Dataset - Uses PUBLICLY ACCESSIBLE datasets

No authentication required!
Streaming - minimal memory usage.
"""
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Optional, Iterator, Dict, Tuple
import re

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("ERROR: Install datasets: pip install datasets")


class RealInstructionDataset(IterableDataset):
    """
    Stream instruction-code pairs from PUBLIC datasets
    No authentication needed!
    """
    
    def __init__(
        self,
        max_source_len: int = 512,
        max_target_len: int = 1024,
        max_samples: int = 5000,
        tokenizer_name: str = "cl100k_base"
    ):
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.max_samples = max_samples
        
        if TIKTOKEN_AVAILABLE:
            try:
                self.enc = tiktoken.get_encoding(tokenizer_name)
            except:
                self.enc = tiktoken.get_encoding("gpt2")
        else:
            self.enc = None
    
    def _encode(self, text: str) -> List[int]:
        if self.enc:
            return self.enc.encode(text)
        return [ord(c) % 50000 for c in text]
    
    def _truncate(self, tokens: List[int], max_len: int) -> List[int]:
        return tokens[:max_len]
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if not DATASETS_AVAILABLE:
            return
        
        sample_count = 0
        
        # PUBLIC datasets that work without authentication
        print("Streaming from CodeSearchNet (public)...")
        
        try:
            # CodeSearchNet - fully public, has docstrings + code
            for lang in ["javascript"]:
                dataset = load_dataset(
                    "code_search_net",
                    lang,
                    split="train",
                    streaming=True
                )
                
                for sample in dataset:
                    if sample_count >= self.max_samples:
                        return
                    
                    # Docstring = instruction, Code = target
                    doc = sample.get("func_documentation_string", "")
                    code = sample.get("func_code_string", "")
                    
                    if not doc or not code or len(doc) < 10 or len(code) < 30:
                        continue
                    
                    source_tokens = self._truncate(self._encode(doc), self.max_source_len)
                    target_tokens = self._truncate(self._encode(code), self.max_target_len)
                    
                    if len(source_tokens) < 5 or len(target_tokens) < 10:
                        continue
                    
                    yield {
                        'source_patches': torch.tensor(source_tokens, dtype=torch.long),
                        'target_patches': torch.tensor(target_tokens, dtype=torch.long)
                    }
                    
                    sample_count += 1
                    if sample_count % 500 == 0:
                        print(f"  Loaded {sample_count} instruction-code pairs...")
                        
        except Exception as e:
            print(f"  CodeSearchNet error: {e}")
        
        print(f"Total instruction samples: {sample_count}")


class CombinedRealDataset(IterableDataset):
    """Combined streaming dataset - code + instructions"""
    
    def __init__(
        self,
        max_source_len: int = 1024,
        max_target_len: int = 1024,
        max_code_samples: int = 8000,
        max_instruction_samples: int = 2000,
        languages: List[str] = ["typescript", "javascript"]
    ):
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.max_code_samples = max_code_samples
        self.max_instruction_samples = max_instruction_samples
        self.languages = languages
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        from training.typescript_loader import TypeScriptStreamingDataset
        
        sample_count = 0
        
        # Stream code patterns
        print("\n=== Streaming Code Patterns ===")
        code_ds = TypeScriptStreamingDataset(
            languages=self.languages,
            max_source_len=self.max_source_len,
            max_target_len=self.max_target_len,
            max_samples=self.max_code_samples
        )
        
        for sample in code_ds:
            yield sample
            sample_count += 1
        
        print(f"Code samples: {sample_count}")
        
        # Stream instruction pairs
        print("\n=== Streaming Instructions ===")
        instruction_count = 0
        instruction_ds = RealInstructionDataset(
            max_source_len=self.max_source_len,
            max_target_len=self.max_target_len,
            max_samples=self.max_instruction_samples
        )
        
        for sample in instruction_ds:
            yield sample
            instruction_count += 1
        
        print(f"Instruction samples: {instruction_count}")
        print(f"Total: {sample_count + instruction_count}")


def create_real_dataloader(
    batch_size: int = 8,
    max_source_len: int = 1024,
    max_target_len: int = 1024,
    code_samples: int = 5000,
    instruction_samples: int = 2000,
    num_workers: int = 0
) -> DataLoader:
    """Create streaming dataloader with real public data"""
    
    dataset = CombinedRealDataset(
        max_source_len=max_source_len,
        max_target_len=max_target_len,
        max_code_samples=code_samples,
        max_instruction_samples=instruction_samples
    )
    
    def collate_fn(batch):
        source_seqs = [item['source_patches'] for item in batch]
        target_seqs = [item['target_patches'] for item in batch]
        
        source_batch = torch.nn.utils.rnn.pad_sequence(source_seqs, batch_first=True, padding_value=0)
        target_batch = torch.nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=0)
        
        if source_batch.shape[1] > max_source_len:
            source_batch = source_batch[:, :max_source_len]
        if target_batch.shape[1] > max_target_len:
            target_batch = target_batch[:, :max_target_len]
        
        return {
            'source_patches': source_batch,
            'target_patches': target_batch
        }
    
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)


if __name__ == "__main__":
    print("Testing instruction dataset...")
    
    ds = RealInstructionDataset(max_samples=10)
    count = 0
    for sample in ds:
        count += 1
        print(f"Sample {count}: source={sample['source_patches'].shape}")
    
    print(f"Total: {count}")
