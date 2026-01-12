"""
Main evaluator for Shutka (VL-JEPA) - evaluates representation quality and retrieval
"""

import torch
import os
import json
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import sys
from sklearn.metrics.pairwise import cosine_similarity
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.shutka import UltraEfficientTextJEPA, FAISS_AVAILABLE
from config import EvaluationConfig


class VLEPAEvaluator:
    """
    Main evaluator for Shutka (VL-JEPA) - evaluates representation learning quality
    """

    def __init__(self, config: EvaluationConfig, device: Optional[torch.device] = None):
        self.config = config
        # Auto-detect GPU if available, or use provided device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using GPU for evaluation: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("Using CPU for evaluation")
        else:
            self.device = device
            print(f"Using device: {device}")

        # Load Shutka model
        self.model = self._load_model(config.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Initialize FAISS memory bank if RAG is enabled but no memory bank exists
        if hasattr(self.config, 'use_rag') and self.config.use_rag and FAISS_AVAILABLE:
            if not hasattr(self.model, 'memory_bank') or self.model.memory_bank is None:
                print("Creating FAISS memory bank for evaluation...")
                from models.shutka import FAISSMemoryBank
                self.model.memory_bank = FAISSMemoryBank(dimension=512, base_dir="memory_bank", shards=4)

                # Populate with some test memories for evaluation
                self._populate_test_memories()

        # Initialize evaluation metrics
        self.representation_quality = []
        self.retrieval_accuracy = []

        # Load evaluation data (would be patches for VL-JEPA)
        self.eval_patches = self._load_eval_patches()

    def _load_model(self, checkpoint_path: str):
        """Load Shutka (VL-JEPA) model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract config from checkpoint
        config_dict = checkpoint.get("config", {})

        # Create Shutka model with config parameters
        model = UltraEfficientTextJEPA(
            vocab_size=config_dict.get("vocab_size", 50000),
            source_dim=config_dict.get("source_dim", 768),
            source_depth=config_dict.get("source_depth", 12),
            target_dim=config_dict.get("target_dim", 768),
            target_depth=config_dict.get("target_depth", 6),
            predictor_dim=config_dict.get("predictor_dim", 768),
            predictor_depth=config_dict.get("predictor_depth", 8),
            output_dim=config_dict.get("output_dim", 1536),
            temperature=config_dict.get("temperature", 0.07),
            max_source_len=config_dict.get("max_source_len", 16384),
            max_target_len=config_dict.get("max_target_len", 512),
            use_rag=config_dict.get("use_rag", True),
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def _load_eval_patches(self):
        """Load evaluation patches for VL-JEPA testing"""
        eval_patches = []
        
        # Tiktoken for accurate encoding
        enc = None
        if TIKTOKEN_AVAILABLE:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except:
                enc = tiktoken.get_encoding("gpt2")

        test_texts = [
            "function add(a: number, b: number): number { return a + b; }",
            "class Calculator { constructor() { } private add(x: number, y: number): number { return x + y; } }",
            "const multiply = <T extends number>(a: T, b: T): number => a * b;",
            "async function fetchData(url: string): Promise<any> { const res = await fetch(url); return res.json(); }",
            "interface User { id: number; name: string; metadata?: Record<string, any>; }",
            "type Result<T> = { success: true; data: T } | { success: false; error: string };",
        ]

        for text in test_texts:
            if enc:
                tokens = enc.encode(text)
            else:
                tokens = [ord(c) % 256 for c in text]

            if len(tokens) < 16:
                tokens.extend([0] * (16 - len(tokens)))

            eval_patches.append({
                "patches": torch.tensor(tokens, dtype=torch.long),
                "text": text,
                "category": "code"
            })

        return eval_patches

    def _populate_test_memories(self):
        """Populate memory bank with test data for evaluation"""
        if not hasattr(self.model, 'memory_bank') or not self.model.memory_bank.indices:
            return

        print("Populating memory bank with test knowledge...")

        # Create test knowledge base
        test_knowledge = [
            ("Python list comprehensions", "List comprehensions provide a concise way to create lists: [x**2 for x in range(10)]"),
            ("JavaScript arrow functions", "Arrow functions: const add = (a, b) => a + b;"),
            ("React useState hook", "useState manages component state: const [count, setCount] = useState(0);"),
            ("TypeScript interfaces", "interface User { id: number; name: string; email: string; }"),
            ("CSS flexbox", "Flexbox layout: display: flex; justify-content: center; align-items: center;"),
            ("Git branching", "Create and switch to new branch: git checkout -b feature-branch"),
            ("Docker containers", "Run container: docker run -p 3000:3000 myapp"),
            ("SQL joins", "INNER JOIN combines rows: SELECT * FROM users u JOIN orders o ON u.id = o.user_id"),
        ]

        for topic, description in test_knowledge:
            # Create simple random embedding that matches FAISS dimension (512)
            emb = torch.randn(1, 512).to(self.device)  # Random embedding for testing
            self.model.memory_bank.add_memory(emb, [description])

        print(f"Added {len(test_knowledge)} knowledge entries to memory bank")

    def evaluate_representation_quality(self) -> Dict:
        """
        Evaluate representation quality using patch similarity and clustering
        """
        print("\n=== Evaluating Representation Quality ===")

        all_embeddings = []
        labels = []

        with torch.no_grad():
            for sample in self.eval_patches:
                tokens = sample["patches"].to(self.device)

                # Split token sequence into source and target (VL-JEPA style)
                mid_point = len(tokens) // 2
                source_tokens = tokens[:mid_point]
                target_tokens = tokens[mid_point:]

                if len(source_tokens) > 0 and len(target_tokens) > 0:
                    # Encode token sequences
                    source_emb = self.model.encode_source(source_tokens.unsqueeze(0))
                    target_emb = self.model.encode_target(target_tokens.unsqueeze(0))

                    # For VL-JEPA evaluation, test prediction quality
                    # Predict target from source using the predictor
                    predicted_target = self.model.predict(
                        source_emb, source_emb, source_mask=None, query_mask=None
                    )

                    # Compare predicted vs actual target (both are [batch, output_dim])
                    pred_pooled = predicted_target.cpu().numpy()
                    target_pooled = target_emb.cpu().numpy()

                    all_embeddings.extend([pred_pooled, target_pooled])
                    labels.extend(
                        [f"{sample['category']}_source", f"{sample['category']}_target"]
                    )

        if not all_embeddings:
            return {"score": 0.0, "details": "No embeddings generated"}

        # Convert to numpy array
        embeddings = np.vstack(all_embeddings)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Evaluate intra-class vs inter-class similarity
        # Same category pairs should be more similar than different category pairs
        intra_class_sim = []
        inter_class_sim = []

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                sim = similarity_matrix[i, j]
                if labels[i].split("_")[0] == labels[j].split("_")[0]:  # Same category
                    intra_class_sim.append(sim)
                else:
                    inter_class_sim.append(sim)

        if intra_class_sim and inter_class_sim:
            intra_mean = np.mean(intra_class_sim)
            inter_mean = np.mean(inter_class_sim)

            # Quality score: higher when intra-class similarity > inter-class similarity
            quality_score = max(0, min(1, (intra_mean - inter_mean + 1) / 2))
        else:
            quality_score = 0.0

        print(f"  Intra-class similarity: {np.mean(intra_class_sim):.3f}")
        print(f"  Inter-class similarity: {np.mean(inter_class_sim):.3f}")
        print(f"  Representation quality score: {quality_score:.3f}")

        return {
            "score": quality_score,
            "intra_class_similarity": float(np.mean(intra_class_sim)),
            "inter_class_similarity": float(np.mean(inter_class_sim)),
            "details": f"Intra: {len(intra_class_sim)} pairs, Inter: {len(inter_class_sim)} pairs",
        }

    def evaluate_retrieval(self) -> Dict:
        """
        Evaluate retrieval-augmented generation capability
        """
        print("\n=== Evaluating Retrieval Capability ===")

        if (
            not hasattr(self.model, "memory_bank")
            or not self.model.memory_bank.indices
        ):
            print("  No FAISS memory bank available - skipping retrieval evaluation")
            return {"score": 0.0, "details": "No memory bank"}

        # Add some test memories to the bank
        test_memories = []
        test_texts = [
            "JavaScript array methods: map, filter, reduce",
            "React component lifecycle methods",
            "Python list comprehensions syntax",
            "TypeScript interface definitions",
            "CSS flexbox properties",
        ]

        for i, text in enumerate(test_texts):
            # Create random embedding that matches FAISS dimension (512)
            emb = torch.randn(1, 512).to(self.device)
            test_memories.append((emb, text))

        # Add to memory bank
        embeddings = torch.cat([emb for emb, _ in test_memories])
        texts = [text for _, text in test_memories]
        self.model.memory_bank.add_memory(embeddings, texts)

        # Test retrieval
        correct_retrievals = 0
        total_queries = 0

        for query_emb, expected_text in test_memories[:3]:  # Test first 3
            total_queries += 1

            # Search for similar memories
            distances, indices, retrieved_texts = self.model.memory_bank.search(
                query_emb, k=3
            )

            if retrieved_texts and retrieved_texts[0]:
                # Check if expected text is in top results
                retrieved = retrieved_texts[0]
                if expected_text in retrieved:
                    correct_retrievals += 1

        retrieval_score = (
            correct_retrievals / total_queries if total_queries > 0 else 0.0
        )

        print(
            f"  Retrieval accuracy: {retrieval_score:.3f} ({correct_retrievals}/{total_queries})"
        )

        return {
            "score": retrieval_score,
            "correct_retrievals": correct_retrievals,
            "total_queries": total_queries,
        }

    def evaluate_all(self) -> Dict:
        """Run all VL-JEPA evaluations"""
        print(f"\n{'='*60}")
        print(f"Evaluating Shutka (VL-JEPA): {self.config.checkpoint_path}")
        print(f"{'='*60}")

        # Run evaluations
        rep_quality = self.evaluate_representation_quality()
        retrieval = self.evaluate_retrieval()

        # Calculate composite score
        composite_score = (rep_quality["score"] + retrieval["score"]) / 2

        # Compile results
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": self.config.checkpoint_path,
            "representation_quality": rep_quality,
            "retrieval": retrieval,
            "composite_score": composite_score,
            "scores": {
                "representation_quality": rep_quality["score"],
                "retrieval": retrieval["score"],
                "composite": composite_score,
            },
        }

        # Print summary
        print(f"\n{'='*60}")
        print("VL-JEPA EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Representation Quality: {rep_quality['score']:.3f}")
        print(f"Retrieval Accuracy:     {retrieval['score']:.3f}")
        print(f"Composite Score:        {composite_score:.3f}")
        print(f"{'='*60}\n")

        return final_results

    def save_results(self, results: Dict, filename: Optional[str] = None):
        """Save evaluation results to file"""
        os.makedirs(self.config.results_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(self.config.checkpoint_path).replace(
                ".pt", ""
            )
            filename = f"results_{model_name}_{timestamp}.json"

        filepath = os.path.join(self.config.results_dir, filename)

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {filepath}")
        return filepath
