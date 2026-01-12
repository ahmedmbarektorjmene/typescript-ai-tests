"""
Main evaluation script
"""
import torch
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EvaluationConfig
from evaluation.evaluator import VLEPAEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test_suite_dir",
        type=str,
        default="evaluation/test_suites",
        help="Directory containing test suites",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--max_gen_length", type=int, default=512, help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (auto=detect GPU, cpu=force CPU, cuda=force GPU)",
    )

    args = parser.parse_args()

    # Create config
    config = EvaluationConfig()
    config.checkpoint_path = args.checkpoint
    config.test_suite_dir = args.test_suite_dir
    config.results_dir = args.results_dir
    config.max_gen_length = args.max_gen_length
    config.temperature = args.temperature
    config.top_p = args.top_p

    # Determine device
    if args.device == "auto":
        device = None  # Let evaluator auto-detect
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create evaluator
    evaluator = VLEPAEvaluator(config, device=device)

    # Run evaluation
    results = evaluator.evaluate_all()

    # Save results
    evaluator.save_results(results)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
