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

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


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
        tokenizer_name: str = "cl100k_base",
    ):
        self.max_samples = max_samples
        self.max_seq_len = max_seq_len

        if TIKTOKEN_AVAILABLE:
            self.enc = tiktoken.get_encoding(tokenizer_name)
        else:
            self.enc = None

    def _encode(self, text: str) -> torch.Tensor:
        if self.enc:
            tokens = self.enc.encode(text, allowed_special=set())
        else:
            tokens = [ord(c) % 50000 for c in text]
        return torch.tensor(tokens[: self.max_seq_len], dtype=torch.long)

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
                    if len(cleaned) > 40:
                        yield {"input_ids": self._encode(cleaned)}
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
        source_seqs = [item["source_patches"] for item in batch]
        target_seqs = [item["target_patches"] for item in batch]

        source_batch = torch.nn.utils.rnn.pad_sequence(
            source_seqs, batch_first=True, padding_value=0
        )
        target_batch = torch.nn.utils.rnn.pad_sequence(
            target_seqs, batch_first=True, padding_value=0
        )

        if source_batch.shape[1] > max_source_len:
            source_batch = source_batch[:, :max_source_len]
        if target_batch.shape[1] > max_target_len:
            target_batch = target_batch[:, :max_target_len]

        return {"source_patches": source_batch, "target_patches": target_batch}

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


class TypeScriptBufferedDataset(torch.utils.data.Dataset):
    """Buffers samples from a streaming dataset for multi-epoch training."""

    def __init__(
        self, streaming_dataset: TypeScriptStreamingDataset, buffer_size: int = 10000
    ):
        self.data = []
        print(f"Buffering up to {buffer_size} samples from streaming dataset...")
        for sample in streaming_dataset:
            self.data.append(
                {
                    "source_patches": sample["input_ids"],
                    "target_patches": sample[
                        "input_ids"
                    ],  # syntax-only: source = target
                }
            )
            if len(self.data) >= buffer_size:
                break
        print(f"Buffered {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
