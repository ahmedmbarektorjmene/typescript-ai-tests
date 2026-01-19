"""
REAL Instruction-Code Dataset - Uses PUBLICLY ACCESSIBLE datasets

No authentication required!
Streaming - minimal memory usage.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Iterator, Dict


try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("ERROR: Install datasets: pip install datasets")


from training.typescript_loader import rolling_poly_hash, N_MIN, N_MAX


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
    ):
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.max_samples = max_samples

    def _encode(self, text: str) -> List[int]:
        # Use simple UTF-8 byte encoding
        return list(text.encode("utf-8"))

    def _truncate(self, tokens: List[int], max_len: int) -> List[int]:
        return tokens[:max_len]

    def _get_hash_ngrams(self, byte_tensor: torch.Tensor) -> torch.Tensor:
        """Generating n-gram hashes for Engram Layer compatibility"""
        length = byte_tensor.size(0)
        all_hashes = torch.zeros((length, N_MAX - N_MIN + 1), dtype=torch.long)
        for idx, n in enumerate(range(N_MIN, N_MAX + 1)):
            all_hashes[:, idx] = rolling_poly_hash(byte_tensor, n)
        return all_hashes

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
                    streaming=True,
                )

                for sample in dataset:
                    if sample_count >= self.max_samples:
                        return

                    # Docstring = instruction, Code = target
                    doc = sample.get("func_documentation_string", "")
                    code = sample.get("func_code_string", "")

                    if not doc or not code or len(doc) < 10 or len(code) < 30:
                        continue

                    source_tokens = self._truncate(
                        self._encode(doc), self.max_source_len
                    )
                    target_tokens = self._truncate(
                        self._encode(code), self.max_target_len
                    )

                    if len(source_tokens) < 5 or len(target_tokens) < 10:
                        continue

                    source_tensor = torch.tensor(source_tokens, dtype=torch.long)
                    target_tensor = torch.tensor(target_tokens, dtype=torch.long)

                    # Generate Hashes for Engram Layer (Crucial for T4/Shutka-v2)
                    source_hashes = self._get_hash_ngrams(source_tensor)

                    yield {
                        "source_patches": source_tensor,
                        "target_patches": target_tensor,
                        # Pass hashes so EngramLayer works
                        "hash_ngrams": source_hashes,
                        # Dummy boundaries for compatibility if needed, or 0s
                        "patch_boundaries": torch.zeros_like(source_tensor),
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
        languages: List[str] = ["typescript", "javascript"],
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
            max_seq_len=self.max_source_len,
            max_samples=self.max_code_samples,
        )

        for sample in code_ds:
            # Remap keys to match RealInstructionDataset schema
            # TypeScript dataset yields 'byte_ids', we need 'source_patches'/'target_patches'
            yield {
                "source_patches": sample["byte_ids"],
                "target_patches": sample[
                    "byte_ids"
                ],  # Self-supervised: target is same as source
                "hash_ngrams": sample["hash_ngrams"],
                "patch_boundaries": sample["patch_boundaries"],
            }
            sample_count += 1

        print(f"Code samples: {sample_count}")

        # Stream instruction pairs
        print("\n=== Streaming Instructions ===")
        instruction_count = 0
        instruction_ds = RealInstructionDataset(
            max_source_len=self.max_source_len,
            max_target_len=self.max_target_len,
            max_samples=self.max_instruction_samples,
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
    num_workers: int = 0,
) -> DataLoader:
    """Create streaming dataloader with real public data"""

    dataset = CombinedRealDataset(
        max_source_len=max_source_len,
        max_target_len=max_target_len,
        max_code_samples=code_samples,
        max_instruction_samples=instruction_samples,
    )

    def collate_fn(batch):
        source_seqs = [item["source_patches"] for item in batch]
        target_seqs = [item["target_patches"] for item in batch]

        # Handle hash n-grams if present (from Instruction dataset) or pass None
        hashes = [item.get("hash_ngrams", None) for item in batch]

        # If mixed batch (some have hashes, some don't - unlikely with current logic but safe)
        # We assume all have hashes now that we fixed RealInstructionDataset

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

        # Pad Hashes (Batch, Seq, NGrams)
        padded_hashes = None
        if all(h is not None for h in hashes):
            max_len = source_batch.shape[1]
            n_gram_counts = hashes[0].shape[1]
            padded_hashes = torch.zeros(
                (len(batch), max_len, n_gram_counts), dtype=torch.long
            )
            for i, h in enumerate(hashes):
                seq_len = min(h.shape[0], max_len)
                padded_hashes[i, :seq_len, :] = h[:seq_len]

        return {
            "source_patches": source_batch,
            "target_patches": target_batch,
            "hash_ngrams": padded_hashes,  # Crucial for Engram Layer
        }

    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
    )
