"""
TypeScript/JavaScript Code Training Data Loader

Uses PUBLICLY ACCESSIBLE datasets - no authentication required!
All streaming - minimal memory usage.
"""
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Optional, Iterator, Dict, Tuple
import re
import os

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Install with: pip install tiktoken")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: HuggingFace datasets not available. Install with: pip install datasets")


def extract_rich_context_typescript(code: str) -> List[Dict[str, str]]:
    """Extract functions, classes, and interfaces from TypeScript/JavaScript code"""
    items = []
    
    # 1. Extract Functions/Methods
    func_patterns = [
        (r'((?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^=]+)?\s*=>)\s*\{', 'arrow'),
        (r'((?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(?:<[^>]+>)?\s*\([^)]*\)\s*(?::\s*[^\{]+)?)\s*\{', 'function'),
        (r'((?:public|private|protected|static|async|readonly|\s)*(\w+)\s*(?:<[^>]+>)?\s*\([^)]*\)\s*(?::\s*[^\{]+)?)\s*\{', 'method'),
    ]
    
    for pattern, func_type in func_patterns:
        for match in re.finditer(pattern, code, re.MULTILINE):
            signature = match.group(1).strip()
            item_name = match.group(2) if match.lastindex >= 2 else "anonymous"
            start_pos = match.end() - 1
            
            brace_count = 1
            end_pos = start_pos + 1
            while brace_count > 0 and end_pos < len(code):
                if code[end_pos] == '{':
                    brace_count += 1
                elif code[end_pos] == '}':
                    brace_count -= 1
                end_pos += 1
            
            body = code[start_pos+1:end_pos-1].strip()
            
            if len(body) < 20:
                continue
            
            items.append({
                'name': item_name,
                'signature': signature,
                'body': body,
                'type': func_type,
                'full': f"{signature} {{\n{body}\n}}"
            })

    # 2. Extract Classes
    class_pattern = r'((?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w\s,]+)?)\s*\{'
    for match in re.finditer(class_pattern, code, re.MULTILINE):
        signature = match.group(1).strip()
        item_name = match.group(2)
        start_pos = match.end() - 1
        
        brace_count = 1
        end_pos = start_pos + 1
        while brace_count > 0 and end_pos < len(code):
            if code[end_pos] == '{':
                brace_count += 1
            elif code[end_pos] == '}':
                brace_count -= 1
            end_pos += 1
        
        body = code[start_pos+1:end_pos-1].strip()
        if len(body) > 50:
            items.append({
                'name': item_name,
                'signature': signature,
                'body': body,
                'type': 'class',
                'full': f"{signature} {{\n{body}\n}}"
            })

    # 3. Extract Interfaces/Types
    type_patterns = [
        (r'((?:export\s+)?interface\s+(\w+)(?:\s+extends\s+[\w\s,]+)?)\s*\{', 'interface'),
        (r'((?:export\s+)?type\s+(\w+)\s*(?:<[^>]+>)?\s*=\s*\{)', 'type'),
    ]
    for pattern, t_type in type_patterns:
        for match in re.finditer(pattern, code, re.MULTILINE):
            signature = match.group(1).strip()
            item_name = match.group(2)
            start_pos = match.end() - 1
            
            brace_count = 1
            end_pos = start_pos + 1
            while brace_count > 0 and end_pos < len(code):
                if code[end_pos] == '{':
                    brace_count += 1
                elif code[end_pos] == '}':
                    brace_count -= 1
                end_pos += 1
            
            body = code[start_pos+1:end_pos-1].strip()
            items.append({
                'name': item_name,
                'signature': signature,
                'body': body,
                'type': t_type,
                'full': f"{signature} {{\n{body}\n}}"
            })
    
    return items


class TypeScriptStreamingDataset(IterableDataset):
    """
    Streaming dataset using PUBLICLY ACCESSIBLE datasets
    No authentication required!
    """
    
    def __init__(
        self,
        languages: List[str] = ["typescript", "javascript"],
        max_source_len: int = 2048,
        max_target_len: int = 512,
        max_samples: Optional[int] = 10000,
        tokenizer_name: str = "cl100k_base"
    ):
        self.languages = languages
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
            try:
                return self.enc.encode(text, allowed_special=set())
            except:
                return self.enc.encode(text)
        return [ord(c) % 50000 for c in text]
    
    def _truncate(self, tokens: List[int], max_len: int) -> List[int]:
        return tokens[:max_len]
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if not DATASETS_AVAILABLE:
            yield from self._generate_dummy_samples()
            return
        
        sample_count = 0
        
        # PUBLIC DATASETS that work without authentication
        public_datasets = [
            # CodeSearchNet - publicly accessible code dataset
            ("code_search_net", {"split": "train", "streaming": True}, "func_code_string", "func_documentation_string"),
            # RedPajama code subset
            ("togethercomputer/RedPajama-Data-1T-Sample", {"split": "train", "streaming": True}, "text", None),
        ]
        
        for dataset_info in public_datasets:
            if sample_count >= self.max_samples:
                return
            
            try:
                if len(dataset_info) == 4:
                    name, kwargs, code_key, doc_key = dataset_info
                else:
                    name, kwargs = dataset_info[:2]
                    code_key, doc_key = "text", None
                
                print(f"Streaming from {name}...")
                
                # Filter for specific languages if supported
                if name == "code_search_net":
                    for lang in ["javascript"]:  # CodeSearchNet has JS
                        try:
                            dataset = load_dataset(name, lang, **kwargs)
                            for sample in dataset:
                                if sample_count >= self.max_samples:
                                    return
                                
                                code = sample.get(code_key, "")
                                doc = sample.get(doc_key, "") if doc_key else ""
                                
                                if not code or len(code) < 50:
                                    continue
                                
                                # Use docstring as source, code as target
                                if doc and len(doc) > 10:
                                    source_text = doc
                                    target_text = code
                                else:
                                    # Extract rich context signature → body
                                    rich_items = extract_rich_context_typescript(code)
                                    if not rich_items:
                                        continue
                                    item = rich_items[0]
                                    source_text = item['signature']
                                    target_text = item['body']
                                
                                source_tokens = self._truncate(self._encode(source_text), self.max_source_len)
                                target_tokens = self._truncate(self._encode(target_text), self.max_target_len)
                                
                                if len(source_tokens) < 5 or len(target_tokens) < 10:
                                    continue
                                
                                yield {
                                    'source_patches': torch.tensor(source_tokens, dtype=torch.long),
                                    'target_patches': torch.tensor(target_tokens, dtype=torch.long)
                                }
                                
                                sample_count += 1
                                if sample_count % 1000 == 0:
                                    print(f"  Loaded {sample_count} samples...")
                        except Exception as e:
                            print(f"  {name}/{lang} error: {e}")
                            continue
                else:
                    dataset = load_dataset(name, **kwargs)
                    for sample in dataset:
                        if sample_count >= self.max_samples:
                            return
                        
                        content = sample.get(code_key, sample.get("text", ""))
                        if not content:
                            continue
                        
                        # Filter for JS/TS content
                        js_indicators = ["function", "const ", "let ", "var ", "=>", "import ", "export "]
                        if not any(ind in content for ind in js_indicators):
                            continue
                        
                        rich_items = extract_rich_context_typescript(content)
                        for item in rich_items[:2]:  # Max 2 per file
                            if sample_count >= self.max_samples:
                                return
                            
                            source_tokens = self._truncate(self._encode(item['signature']), self.max_source_len)
                            target_tokens = self._truncate(self._encode(item['body']), self.max_target_len)
                            
                            if len(source_tokens) < 5 or len(target_tokens) < 10:
                                continue
                            
                            yield {
                                'source_patches': torch.tensor(source_tokens, dtype=torch.long),
                                'target_patches': torch.tensor(target_tokens, dtype=torch.long)
                            }
                            
                            sample_count += 1
                            if sample_count % 1000 == 0:
                                print(f"  Loaded {sample_count} samples...")
                                
            except Exception as e:
                print(f"  {name} failed: {e}")
                continue
        
        # If we got no samples, generate training data
        if sample_count == 0:
            print("No streaming data available. Using local code files or generating samples...")
            yield from self._load_local_code()
    
    def _load_local_code(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Load code from local data directory as fallback"""
        import glob
        
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        if os.path.exists(data_dir):
            patterns = ["**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx"]
            files = []
            for pattern in patterns:
                files.extend(glob.glob(os.path.join(data_dir, pattern), recursive=True))
            
            count = 0
            for file_path in files[:100]:  # Limit files
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    rich_items = extract_rich_context_typescript(content)
                    for item in rich_items:
                        source_tokens = self._truncate(self._encode(item['signature']), self.max_source_len)
                        target_tokens = self._truncate(self._encode(item['body']), self.max_target_len)
                        
                        if len(source_tokens) >= 5 and len(target_tokens) >= 10:
                            yield {
                                'source_patches': torch.tensor(source_tokens, dtype=torch.long),
                                'target_patches': torch.tensor(target_tokens, dtype=torch.long)
                            }
                            count += 1
                except:
                    continue
            
            if count > 0:
                print(f"Loaded {count} samples from local files")
                return
        
        # Last resort: generate synthetic training pairs
        yield from self._generate_dummy_samples()
    
    def _generate_dummy_samples(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Generate minimal training samples"""
        print("Generating minimal training samples...")
        
        samples = [
            ("function add(a: number, b: number): number {", "return a + b;"),
            ("const multiply = (x: number, y: number) => {", "return x * y;"),
            ("async function fetchData(url: string) {", "const res = await fetch(url);\nreturn res.json();"),
            ("function filterItems<T>(items: T[], predicate: (item: T) => boolean): T[] {", "return items.filter(predicate);"),
            ("const debounce = (fn: Function, delay: number) => {", "let timer: NodeJS.Timeout;\nreturn (...args: any[]) => {\nclearTimeout(timer);\ntimer = setTimeout(() => fn(...args), delay);\n};"),
        ]
        
        # Repeat samples to get enough data
        for _ in range(200):
            for source, target in samples:
                source_tokens = self._truncate(self._encode(source), self.max_source_len)
                target_tokens = self._truncate(self._encode(target), self.max_target_len)
                
                yield {
                    'source_patches': torch.tensor(source_tokens, dtype=torch.long),
                    'target_patches': torch.tensor(target_tokens, dtype=torch.long)
                }


class TypeScriptBufferedDataset(Dataset):
    """Buffered dataset for training epochs"""
    
    def __init__(
        self,
        languages: List[str] = ["typescript", "javascript"],
        max_source_len: int = 2048,
        max_target_len: int = 512,
        buffer_size: int = 10000,
        tokenizer_name: str = "cl100k_base"
    ):
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        
        print(f"Loading TypeScript/JavaScript samples...")
        
        streaming_ds = TypeScriptStreamingDataset(
            languages=languages,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
            max_samples=buffer_size,
            tokenizer_name=tokenizer_name
        )
        
        self.data = []
        for sample in streaming_ds:
            self.data.append(sample)
            if len(self.data) >= buffer_size:
                break
        
        print(f"Loaded {len(self.data)} training samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_typescript_dataloader(
    batch_size: int = 8,
    max_source_len: int = 2048,
    max_target_len: int = 512,
    buffer_size: int = 10000,
    languages: List[str] = ["typescript", "javascript"],
    num_workers: int = 0,
    train_split: float = 0.9
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    full_dataset = TypeScriptBufferedDataset(
        languages=languages,
        max_source_len=max_source_len,
        max_target_len=max_target_len,
        buffer_size=buffer_size
    )
    
    if len(full_dataset) == 0:
        raise ValueError("No samples loaded! Check your data sources.")
    
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size
    
    if val_size == 0:
        val_size = 1
        train_size = len(full_dataset) - 1
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing TypeScript data loader...")
    
    ds = TypeScriptStreamingDataset(max_samples=10)
    count = 0
    for sample in ds:
        count += 1
        print(f"Sample {count}: source={sample['source_patches'].shape}, target={sample['target_patches'].shape}")
        if count >= 5:
            break
    
    print(f"\nTotal: {count} samples")
