"""
JS/TS Code-Only Streaming Loader (2025)
Filters: JavaScript, TypeScript
Removes: Docstrings, Comments, Instructions
Strategy: Round-Robin (Exhausts all datasets)
"""

import torch
import re
from torch.utils.data import IterableDataset, DataLoader
from typing import Iterator, Dict, Tuple
from datasets import load_dataset

N_MIN = 3
N_MAX = 8
HASH_TABLE_SIZE = 500000

def rolling_poly_hash(bytes_tensor: torch.Tensor, n: int, prime: int = 1000003) -> torch.Tensor:
    """
    Implements RollPolyHash from Equation 4 of the paper.
    Computes a hash for every window of size n.
    """
    length = bytes_tensor.size(0)
    hashes = torch.zeros(length, dtype=torch.long)
    if length < n:
        return hashes
    
    current_hash = 0
    # Initial window
    for i in range(n):
        current_hash = (current_hash * 256 + bytes_tensor[i].item()) % HASH_TABLE_SIZE
    
    hashes[n-1] = current_hash
    
    # Rolling step
    power = pow(256, n-1, HASH_TABLE_SIZE)
    for i in range(n, length):
        # Remove leading byte, add trailing byte
        current_hash = (current_hash - bytes_tensor[i-n].item() * power) % HASH_TABLE_SIZE
        current_hash = (current_hash * 256 + bytes_tensor[i].item()) % HASH_TABLE_SIZE
        hashes[i] = current_hash
        
    return hashes

def clean_code(code: str) -> str:
    """
    Removes JSDoc, multi-line comments, and single-line comments.
    """
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    code = re.sub(r"//.*", "", code)
    lines = [line for line in code.splitlines() if line.strip()]
    return "\n".join(lines).strip()


class TypeScriptStreamingDataset(IterableDataset):
    def __init__(
        self,
        max_samples: int = 100000,
        max_seq_len: int = 1024,
        entropy_threshold: float = 0.5,
    ):
        self.max_samples = max_samples
        self.max_seq_len = max_seq_len
        self.entropy_threshold = entropy_threshold

    def _get_bytes(self, text: str) -> torch.Tensor:
        """Converts text to raw UTF-8 bytes (0-255)."""
        byte_data = text.encode("utf-8")
        return torch.tensor(list(byte_data)[: self.max_seq_len], dtype=torch.long)

    def _get_hash_ngrams(self, byte_tensor: torch.Tensor) -> torch.Tensor:
        """Section 3.2.1: Encoder Hash n-gram Embeddings."""
        length = byte_tensor.size(0)
        # Table of [SeqLen, 6] (for n=3, 4, 5, 6, 7, 8)
        all_hashes = torch.zeros((length, N_MAX - N_MIN + 1), dtype=torch.long)
        
        for idx, n in enumerate(range(N_MIN, N_MAX + 1)):
            all_hashes[:, idx] = rolling_poly_hash(byte_tensor, n)
            
        return all_hashes

    def _generate_patch_boundaries(self, byte_tensor: torch.Tensor) -> torch.Tensor:
            """
            Implements Entropy Patching (Section 2.3).
            Note: Real BLT uses a 100M parameter Byte-LM. 
            We use a space/punctuation proxy which the paper notes as a baseline (Section 2.2).
            """
            boundaries = torch.zeros_like(byte_tensor)
            if len(boundaries) > 0: boundaries[0] = 1
            
            # Triggering on "High Entropy" structural characters
            triggers = {10, 32, 46, 40, 123, 91, 59}
            for i in range(1, len(byte_tensor)):
                if byte_tensor[i].item() in triggers:
                    boundaries[i] = 1
            return boundaries

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        count = 0

        # Initialize Iterators
        iterators = []

        # 1. CodeSearchNet
        try:
            ds1 = load_dataset(
                "claudios/code_search_net", "javascript", split="train", streaming=True
            )
            iterators.append({"name": "CSN", "it": iter(ds1)})
        except Exception as e:
            print(f"Error CSN: {e}")

        # 2. StarCoder
        try:
            ds2 = load_dataset(
                "bigcode/starcoderdata",
                data_files=["javascript/*", "typescript/*"],
                split="train",
                streaming=True,
            )
            iterators.append({"name": "StarCoder", "it": iter(ds2)})
        except Exception as e:
            print(f"Error StarCoder: {e}")

        # 3. SmolLM
        try:
            ds3 = load_dataset(
                "HuggingFaceTB/smollm-corpus",
                "cosmopedia-v2",
                split="train",
                streaming=True,
            )
            iterators.append({"name": "SmolLM", "it": iter(ds3)})
        except Exception as e:
            print(f"Error SmolLM: {e}")

        # Round-Robin Loop
        while iterators and count < self.max_samples:
            to_remove = []

            for i, source in enumerate(iterators):
                try:
                    sample = next(source["it"])
                    raw_code = ""

                    # Logic specific to source schema
                    if source["name"] == "CSN":
                        raw_code = sample.get("func_code_string", "")

                    elif source["name"] == "StarCoder":
                        raw_code = sample.get("content", "")

                    elif source["name"] == "SmolLM":
                        text = sample.get("text", "")
                        if any(
                            x in text
                            for x in ["const ", "import ", "interface ", "export "]
                        ):
                            blocks = re.findall(
                                r"```(?:javascript|typescript|js|ts)?\n([\s\S]*?)```",
                                text,
                            )
                            raw_code = "\n".join(blocks) if blocks else ""

                    # Clean and Yield
                    cleaned = clean_code(raw_code)
                    byte_ids = torch.tensor(list(cleaned.encode("utf-8"))[:self.max_seq_len], dtype=torch.long)
                    if len(cleaned) > 40:
                        yield {
                                "byte_ids": byte_ids,
                                "patch_boundaries": self._generate_patch_boundaries(byte_ids),
                                "hash_ngrams": self._get_hash_ngrams(byte_ids)
                            }
                        count += 1
                        if count >= self.max_samples:
                            return

                except StopIteration:
                    to_remove.append(i)
                except Exception as e:
                    print(f"Stream Error ({source['name']}): {e}")
                    to_remove.append(i)

            # Remove exhausted streams in reverse order
            for index in sorted(to_remove, reverse=True):
                iterators.pop(index)


def create_typescript_dataloader(
    batch_size: int = 8,
    max_source_len: int = 2048,
    max_target_len: int = 512,
    buffer_size: int = 10000,
    num_workers: int = 0,
    train_split: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from a round-robin streaming dataset."""

    # Initialize streaming dataset
    stream_dataset = TypeScriptStreamingDataset(
        max_samples=buffer_size, max_seq_len=max_source_len
    )

    # Buffer samples
    full_dataset = TypeScriptBufferedDataset(stream_dataset, buffer_size=buffer_size)

    if len(full_dataset) == 0:
        raise ValueError("No samples loaded! Check your data sources.")

    # Split into train and validation
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size
    if val_size == 0:
        val_size = 1
        train_size = len(full_dataset) - 1

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # Collate function for batching
    def collate_fn(batch):
        """Custom collate to handle 3D hash n-gram tensors and convert to JEPA format."""
        byte_ids = [item["byte_ids"] for item in batch]
        boundaries = [item["patch_boundaries"] for item in batch]
        hashes = [item["hash_ngrams"] for item in batch]

        # Pad 1D sequences (Bytes and Boundaries)
        byte_batch = torch.nn.utils.rnn.pad_sequence(byte_ids, batch_first=True, padding_value=0)
        boundary_batch = torch.nn.utils.rnn.pad_sequence(boundaries, batch_first=True, padding_value=0)
        
        # Pad 2D sequences (Hashes: Batch, Seq, NGrams)
        # Manual padding for the 3rd dimension complexity
        max_len = byte_batch.shape[1]
        n_gram_counts = hashes[0].shape[1]
        padded_hashes = torch.zeros((len(batch), max_len, n_gram_counts), dtype=torch.long)
        
        for i, h in enumerate(hashes):
            seq_len = h.shape[0]
            padded_hashes[i, :seq_len, :] = h

        # Return in JEPA format expected by trainer
        return {
            "source_patches": byte_batch,  # Use byte_ids as source patches
            "target_patches": byte_batch,  # Use same byte_ids as target patches (self-supervised)
            "patch_boundaries": boundary_batch,  # Keep boundaries for model
            "hash_ngrams": padded_hashes  # Keep hash n-grams for model
        }

    # Check if CUDA is available for pin_memory optimization
    cuda_available = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=cuda_available,  # Faster GPU transfer
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=cuda_available,  # Faster GPU transfer
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
    )

    return train_loader, val_loader


class TypeScriptBufferedDataset(torch.utils.data.Dataset):
    """Buffers samples from a streaming dataset for multi-epoch training."""

    def __init__(
        self, streaming_dataset: TypeScriptStreamingDataset, buffer_size: int = 10000
    ):
        self.data = []
        for sample in streaming_dataset:
            self.data.append(sample)
            if len(self.data) >= buffer_size:
                break
        print(f"Buffered {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the original sample data for collate_fn to process
        return self.data[idx]
