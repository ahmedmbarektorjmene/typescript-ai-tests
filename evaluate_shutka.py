"""
Script to evaluate Shutka (VL-JEPA) model
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EvaluationConfig
from evaluation.evaluator import VLEPAEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Shutka (VL-JEPA) model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to Shutka checkpoint"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    print(f"{'='*60}")
    print("Evaluating Shutka (VL-JEPA)")
    print(f"{'='*60}")

    config = EvaluationConfig()
    config.checkpoint_path = args.checkpoint
    config.results_dir = args.results_dir

    evaluator = VLEPAEvaluator(config)
    results = evaluator.evaluate_all()

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    evaluator.save_results(results)

    print(f"\nEvaluation completed! Results saved to {args.results_dir}")


if __name__ == "__main__":
    main()
