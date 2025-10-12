#!/usr/bin/env python3
"""
Example script to run expert hyperparameter tuning

This script demonstrates how to use the expert tuning functionality
following the Stage 1 instructions from README.md
"""

import sys
import os

# Add the parent directory to the path so we can import from moe_expert
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_expert.tune_experts import ExpertTuner
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Expert Hyperparameter Tuning")
    parser.add_argument('--dataset', type=str, default='bail', 
                       choices=['bail', 'german', 'math', 'por'],
                       help='Dataset to use for tuning')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--n_trials', type=int, default=30, 
                       help='Number of trials per expert (30-50 recommended)')
    parser.add_argument('--cache_dir', type=str, default='weights/moe_experts',
                       help='Directory to save tuned experts and results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("EXPERT HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Trials per expert: {args.n_trials}")
    print(f"Cache directory: {args.cache_dir}")
    print("="*80)
    
    # Create tuner
    tuner = ExpertTuner(
        dataset=args.dataset,
        seed=args.seed,
        cache_dir=args.cache_dir
    )
    
    # Run tuning for all experts
    results = tuner.tune_all_experts(n_trials=args.n_trials)
    
    print("\n" + "="*80)
    print("TUNING COMPLETE!")
    print("="*80)
    print("Best expert configurations have been saved.")
    print("You can now use these tuned experts for gate training.")
    print(f"Results saved in: {args.cache_dir}")


if __name__ == "__main__":
    main()
