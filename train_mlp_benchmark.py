import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils import load_data_util, normalize_scipy, feature_norm, seed_everything
from torch_geometric.utils import convert
from mlp import MLP


def train_mlp(dataset, epochs=10000, lr=1e-3, weight_decay=1e-5, dropout=0.3, seed=0):
    """Train a simple MLP benchmark on the dataset"""
    seed_everything(seed)
    
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_data_util(dataset)
    adj = normalize_scipy(adj)
    X = feature_norm(features).float()
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    sens = sens.to(device)
    edge_index = edge_index.to(device)
    
    # Model
    input_dim = X.shape[1]
    hidden_dim = 8 if dataset != "german" else 4
    arch = [input_dim, hidden_dim, 2]
    model = MLP(arch, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop with early stopping
    model.train()
    best_score = -1.0
    best_epoch = 0
    patience = 1000
    epochs_without_improvement = 0
    best_model_state = None
    
    for epoch in tqdm(range(epochs), desc="MLP Training"):
        optimizer.zero_grad()
        
        # Forward pass
        hidden, logits = model(X[idx_train])
        loss = F.cross_entropy(logits, labels[idx_train].long())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation every 100 epochs
        if (epoch + 1) % 100 == 0:
            # Helper to compute split metrics
            def eval_split(split: str):
                if split == 'train':
                    idx = idx_train
                elif split == 'val':
                    idx = idx_val
                else:
                    idx = idx_test
                _, logits = model(X[idx])
                preds = logits.argmax(dim=1)
                acc = accuracy_score(labels[idx].cpu().numpy(), preds.cpu().numpy())
                f1 = f1_score(labels[idx].cpu().numpy(), preds.cpu().numpy())
                one_hot = torch.zeros((len(idx), 2), device=device)
                one_hot[torch.arange(len(idx)), labels[idx].long()] = 1
                auc = roc_auc_score(one_hot.cpu().numpy(), logits.detach().cpu().numpy())
                return acc, f1, auc

            model.eval()
            with torch.no_grad():
                val_acc, val_f1, val_auc = eval_split('val')
                utility_score = (val_acc + val_f1 + val_auc) / 3.0

                if utility_score > best_score:
                    best_score = utility_score
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    best_model_state = model.state_dict().copy()
                    print(f"Epoch {epoch+1}: New best! Val Score={utility_score:.4f} (Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f})")
                else:
                    epochs_without_improvement += 100
                    print(f"Epoch {epoch+1}: Val Score={utility_score:.4f} (Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}) - No improvement for {epochs_without_improvement} epochs")

                # Additionally print Train/Val/Test metrics every 100 epochs
                tr_acc, tr_f1, tr_auc = eval_split('train')
                te_acc, te_f1, te_auc = eval_split('test')
                print(
                    f"Epoch {epoch+1}: Train Acc={tr_acc:.4f}, F1={tr_f1:.4f}, AUC={tr_auc:.4f} | "
                    f"Val Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f} | "
                    f"Test Acc={te_acc:.4f}, F1={te_f1:.4f}, AUC={te_auc:.4f}"
                )

                # Early stopping check
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}. Best score {best_score:.4f} achieved at epoch {best_epoch}")
                    break

            model.train()
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch} with score {best_score:.4f}")
    
    # Save best model
    import os
    if not os.path.exists("./weights"):
        os.makedirs("./weights")
        print("Created weights directory")
    
    model_filename = f"./weights/{dataset}_mlp_best_seed_{seed}.pth"
    torch.save(best_model_state, model_filename)
    print(f"Saved best model to {model_filename}")
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_hidden, test_logits = model(X[idx_test])
        test_preds = test_logits.argmax(dim=1)
        
        # Test metrics
        test_acc = accuracy_score(labels[idx_test].cpu().numpy(), test_preds.cpu().numpy())
        test_f1 = f1_score(labels[idx_test].cpu().numpy(), test_preds.cpu().numpy())
        
        one_hot_test = torch.zeros((len(idx_test), 2), device=device)
        one_hot_test[torch.arange(len(idx_test)), labels[idx_test].long()] = 1
        test_auc = roc_auc_score(one_hot_test.cpu().numpy(), test_logits.detach().cpu().numpy())
        
        print(f"\n=== MLP Benchmark Results ({dataset}) ===")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Average Score: {(test_acc + test_f1 + test_auc)/3.0:.4f}")
        print("=" * 50)
        
        return {
            "acc": test_acc, 
            "f1": test_f1, 
            "auc": test_auc
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP benchmark')
    parser.add_argument('--dataset', type=str, default='por', choices=['bail', 'german', 'math', 'por'])
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results = train_mlp(
        dataset=args.dataset,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=args.seed
    )