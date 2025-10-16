#!/usr/bin/env python3
"""
Expert Hyperparameter Tuning Script

This script implements Stage 1 of the MoE training pipeline:
- Tune hyperparameters for each expert independently
- Use appropriate scoring metrics for each expert specialization
- Save best configurations for later use in gate training

Based on the instructions in README.md:
- Expert 1: utility score (accuracy + f1 + auc) / 3
- Expert 2: fairness result (DP + EO) / 2  
- Expert 3: fairness procedure (REF + VEF + ATT_JSD) / 3
"""

import argparse
import json
import os
import random
import time
from typing import Dict, List, Tuple, Any
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Import our MoE components
from .experts import Expert1, Expert2, Expert3
from .trainer import MoETrainer, compute_batch_scores
from utils import load_data_util, normalize_scipy, feature_norm, seed_everything
from torch_geometric.utils import convert
from explanation_metrics import Interpreter
from attention_fairness_utils import compute_attention_fairness_loss


class ExpertTuner:
    """Hyperparameter tuning for individual experts"""
    
    def __init__(self, dataset: str, seed: int = 0, cuda_device: int = 0, 
                 cache_dir: str = "weights/moe_experts"):
        self.dataset = dataset
        self.seed = seed
        self.cuda_device = cuda_device
        self.cache_dir = cache_dir
        
        # Set up device and data
        seed_everything(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(cuda_device)
            print(f"[DEBUG] Using CUDA device {cuda_device}: {torch.cuda.get_device_name(cuda_device)}")
            print(f"[DEBUG] CUDA memory before data loading: {torch.cuda.memory_allocated(cuda_device) / 1024**3:.2f} GB")
        else:
            print("[DEBUG] WARNING: Using CPU - this will be very slow!")
            
        # Load dataset
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_data_util(dataset)
        adj = normalize_scipy(adj)
        X = feature_norm(features).float()
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        
        self.X = X.to(self.device)
        self.labels = labels.to(self.device)
        self.idx_train = idx_train.to(self.device)
        self.idx_val = idx_val.to(self.device)
        self.idx_test = idx_test.to(self.device)
        self.sens = sens.to(self.device)
        self.edge_index = edge_index.to(self.device)
        
        # Debug: Check device and memory after data loading
        if torch.cuda.is_available():
            print(f"[DEBUG] Data loaded to device: {self.X.device}")
            print(f"[DEBUG] CUDA memory after data loading: {torch.cuda.memory_allocated(cuda_device) / 1024**3:.2f} GB")
            print(f"[DEBUG] CUDA memory cached: {torch.cuda.memory_reserved(cuda_device) / 1024**3:.2f} GB")
        
        input_dim = self.X.shape[1]
        self.hidden_dim = 8 if dataset != "german" else 4
        
        # Initialize interpreter for procedural fairness metrics
        self.interpreter = Interpreter(
            features=self.X, 
            edge_index=self.edge_index,
            utility_labels=self.labels, 
            sensitive_labels=self.sens, 
            top_ratio=0.2, 
            topK=1
        )
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Define search spaces for each expert based on README instructions"""
        return {
            "expert1": {
                "lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-3},
                "weight_decay": {"type": "log_uniform", "low": 1e-6, "high": 1e-3},
            },
            "expert2": {
                "lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-3},
                "weight_decay": {"type": "log_uniform", "low": 1e-6, "high": 1e-3},
                "lambda_rep": {"type": "log_uniform", "low": 0.01, "high": 10.0},
                "lambda_fair": {"type": "log_uniform", "low": 0.01, "high": 10.0},
            },
            "expert3": {
                "lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-3},
                "weight_decay": {"type": "log_uniform", "low": 1e-6, "high": 1e-3},
                "lambda_attention": {"type": "log_uniform", "low": 0.01, "high": 10.0},
                "lambda_adv": {"type": "log_uniform", "low": 0.01, "high": 10.0},
            }
        }
    
    def sample_hyperparameters(self, expert_type: str) -> Dict[str, float]:
        """Sample hyperparameters from the search space"""
        search_spaces = self.get_search_spaces()
        params = {}
        
        for param_name, param_config in search_spaces[expert_type].items():
            if param_config["type"] == "log_uniform":
                low = param_config["low"]
                high = param_config["high"]
                # Sample in log space
                log_low = np.log(low)
                log_high = np.log(high)
                log_val = random.uniform(log_low, log_high)
                params[param_name] = np.exp(log_val)
            else:
                raise ValueError(f"Unknown parameter type: {param_config['type']}")
                
        return params
    
    def create_expert(self, expert_type: str, params: Dict[str, float]) -> nn.Module:
        """Create expert with given hyperparameters"""
        input_dim = self.X.shape[1]
        
        if expert_type == "expert1":
            return Expert1(input_dim, self.hidden_dim).to(self.device)
        elif expert_type == "expert2":
            return Expert2(
                input_dim, self.hidden_dim, 
                lambda_rep=params["lambda_rep"], 
                lambda_fair=params["lambda_fair"]
            ).to(self.device)
        elif expert_type == "expert3":
            return Expert3(
                input_dim, self.hidden_dim,
                lambda_attention=params["lambda_attention"],
                lambda_adv=params["lambda_adv"]
            ).to(self.device)
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")
    
    def attention_loss_fn(self, model, X_batch, sens_batch):
        """Attention fairness loss function for Expert3"""
        try:
            return compute_attention_fairness_loss(
                model, X_batch, sens_batch, 
                method='integrated_gradients', 
                background_X=X_batch[:min(50, X_batch.shape[0])]
            )
        except Exception:
            return torch.tensor(0.0, device=X_batch.device)
    
    def train_expert(self, expert: nn.Module, expert_type: str, params: Dict[str, float], 
                    epochs: int = 2000) -> Tuple[float, Dict[str, float]]:
        """Train a single expert and return validation score and metrics"""
        
        # Set up optimizer with gradient clipping for stability
        optimizer = optim.Adam(expert.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        
        # Training data
        X_train = self.X[self.idx_train]
        y_train = self.labels[self.idx_train]
        s_train = self.sens[self.idx_train]
        
        # Validation data
        X_val = self.X[self.idx_val]
        y_val = self.labels[self.idx_val]
        s_val = self.sens[self.idx_val]
        
        best_score = -float('inf') if expert_type == "expert1" else float('inf')
        best_state = None
        best_metrics = None
        
        # Early stopping
        patience = 3
        no_improvement = 0
        
        # Warmup for Expert2 to prevent early collapse
        warmup_epochs = 1000 if expert_type == "expert2" else 0
        
        for epoch in range(epochs):
            expert.train()
            optimizer.zero_grad()
            
            # Compute loss based on expert type
            if expert_type == "expert1":
                loss_dict = expert.compute_loss(X_train, y_train)
            elif expert_type == "expert2":
                # For Expert2, use only CE loss during warmup to prevent collapse
                if epoch < warmup_epochs:
                    _, probs = expert(X_train)
                    loss_dict = {"loss": F.cross_entropy(probs, y_train.long()), "ce": F.cross_entropy(probs, y_train.long())}
                else:
                    loss_dict = expert.compute_loss(X_train, y_train, s_train)
            elif expert_type == "expert3":
                loss_dict = expert.compute_loss(X_train, y_train, s_train, self.attention_loss_fn)
            
            # Check for NaN/Inf losses
            if torch.isnan(loss_dict["loss"]) or torch.isinf(loss_dict["loss"]):
                print(f"[{expert_type}] Warning: NaN/Inf loss at epoch {epoch+1}, skipping...")
                continue
                
            loss_dict["loss"].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Validation evaluation every 100 epochs
            if (epoch + 1) % 100 == 0:
                expert.eval()
                with torch.no_grad():
                    _, probs = expert(X_val)
                    
                    # Check for NaN predictions
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        print(f"[{expert_type}] Warning: NaN/Inf predictions at epoch {epoch+1}")
                        continue
                    
                    # Compute validation metrics
                    score, metrics = compute_batch_scores(
                        probs, y_val, s_val, self.interpreter, X_val, expert, 
                        att_method='integrated_gradients'
                    )
                    
                    # Expert-specific scoring
                    if expert_type == "expert1":
                        # Expert1: maximize utility score
                        current_score = metrics["utility"]
                        is_better = current_score > best_score
                    elif expert_type == "expert2":
                        # Expert2: minimize fairness result (lower is better)
                        # But ensure we don't get degenerate solutions
                        current_score = metrics["fair_res"]
                        # Only consider it better if utility is reasonable (>0.3)
                        is_better = (current_score < best_score) and (metrics["utility"] > 0.3)
                    elif expert_type == "expert3":
                        # Expert3: minimize fairness procedure (lower is better)
                        current_score = metrics["fair_proc"]
                        is_better = current_score < best_score
                    
                    if is_better:
                        best_score = current_score
                        best_state = copy.deepcopy(expert.state_dict())
                        best_metrics = metrics.copy()
                        no_improvement = 0
                    else:
                        no_improvement += 1
                    
                    # Early stopping
                    if no_improvement >= patience:
                        print(f"[{expert_type}] Early stopping at epoch {epoch+1}")
                        break
        
        # Load best model
        if best_state is not None:
            expert.load_state_dict(best_state)
        
        return best_score, best_metrics
    
    def tune_expert(self, expert_type: str, n_trials: int = 50) -> Dict[str, Any]:
        """Tune hyperparameters for a single expert"""
        print(f"\n{'='*60}")
        print(f"Tuning {expert_type.upper()}")
        print(f"{'='*60}")
        
        best_score = -float('inf') if expert_type == "expert1" else float('inf')
        best_params = None
        best_metrics = None
        trial_results = []
        
        for trial in tqdm(range(n_trials), desc=f"Tuning {expert_type}"):
            # Sample hyperparameters
            params = self.sample_hyperparameters(expert_type)
            
            # Create and train expert
            expert = self.create_expert(expert_type, params)
            score, metrics = self.train_expert(expert, expert_type, params)
            
            # Store trial results
            trial_result = {
                "trial": trial,
                "params": params,
                "score": score,
                "metrics": metrics
            }
            trial_results.append(trial_result)
            
            # Check if this is the best configuration
            is_better = False
            if expert_type == "expert1":
                is_better = score > best_score
            else:
                is_better = score < best_score
                
            if is_better:
                best_score = score
                best_params = params.copy()
                best_metrics = metrics.copy()
                
                print(f"\n[{expert_type}] Trial {trial+1}: New best score = {score:.4f}")
                print(f"  Params: {params}")
                print(f"  Metrics: {metrics}")
        
        # Save best expert
        best_expert = self.create_expert(expert_type, best_params)
        self.train_expert(best_expert, expert_type, best_params)  # Train with best params
        
        expert_path = os.path.join(self.cache_dir, f"{self.dataset}_{expert_type}_best.pt")
        torch.save(best_expert.state_dict(), expert_path)
        
        # Save tuning results
        results = {
            "expert_type": expert_type,
            "best_params": best_params,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "trial_results": trial_results,
            "n_trials": n_trials
        }
        
        results_path = os.path.join(self.cache_dir, f"{self.dataset}_{expert_type}_tuning_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n[{expert_type}] Best configuration:")
        print(f"  Score: {best_score:.4f}")
        print(f"  Params: {best_params}")
        print(f"  Metrics: {best_metrics}")
        print(f"  Saved to: {expert_path}")
        
        return results
    
    def tune_all_experts(self, n_trials: int = 50) -> Dict[str, Any]:
        """Tune all three experts"""
        print("Starting Expert Hyperparameter Tuning")
        print(f"Dataset: {self.dataset}")
        print(f"Trials per expert: {n_trials}")
        print(f"Cache directory: {self.cache_dir}")
        
        all_results = {}
        
        # Tune each expert
        for expert_type in ["expert1", "expert2", "expert3"]:
            results = self.tune_expert(expert_type, n_trials)
            all_results[expert_type] = results
        
        # Save summary
        summary = {
            "dataset": self.dataset,
            "seed": self.seed,
            "n_trials": n_trials,
            "results": all_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_path = os.path.join(self.cache_dir, f"{self.dataset}_expert_tuning_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print("EXPERT TUNING COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {summary_path}")
        
        # Print final summary
        for expert_type, results in all_results.items():
            print(f"\n{expert_type.upper()}:")
            print(f"  Best Score: {results['best_score']:.4f}")
            print(f"  Best Params: {results['best_params']}")
        
        return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Expert Hyperparameter Tuning")
    parser.add_argument('--dataset', type=str, default='bail', 
                       choices=['bail', 'german', 'math', 'por'],
                       help='Dataset to use for tuning')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device')
    parser.add_argument('--n_trials', type=int, default=50, 
                       help='Number of trials per expert')
    parser.add_argument('--cache_dir', type=str, default='weights/moe_experts',
                       help='Directory to save tuned experts and results')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create tuner
    tuner = ExpertTuner(
        dataset=args.dataset,
        seed=args.seed,
        cuda_device=args.cuda_device,
        cache_dir=args.cache_dir
    )
    
    # Run tuning
    results = tuner.tune_all_experts(n_trials=args.n_trials)
    
    print("\nExpert tuning completed successfully!")
    print(f"Best expert configurations saved to: {args.cache_dir}")


if __name__ == "__main__":
    main()

