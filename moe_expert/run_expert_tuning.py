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
    
    # CUDA Debug Information
    import torch
    print("="*80)
    print("CUDA DEBUG INFORMATION")
    print("="*80)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("WARNING: CUDA not available - will use CPU (very slow!)")
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
