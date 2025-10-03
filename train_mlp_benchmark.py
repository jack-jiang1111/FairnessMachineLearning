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


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def train_mlp(dataset, epochs=200, lr=1e-3, weight_decay=1e-5, dropout=0.0, seed=0):
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
    model = SimpleMLP(input_dim, hidden_dim, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    model.train()
    for epoch in tqdm(range(epochs), desc="MLP Training"):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(X[idx_train])
        loss = F.cross_entropy(logits, labels[idx_train].long())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation every 10 epochs
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                # Validation metrics
                val_logits = model(X[idx_val])
                val_preds = val_logits.argmax(dim=1)
                val_acc = accuracy_score(labels[idx_val].cpu().numpy(), val_preds.cpu().numpy())
                val_f1 = f1_score(labels[idx_val].cpu().numpy(), val_preds.cpu().numpy())
                
                # One-hot for AUC
                one_hot_val = torch.zeros((len(idx_val), 2), device=device)
                one_hot_val[torch.arange(len(idx_val)), labels[idx_val].long()] = 1
                val_auc = roc_auc_score(one_hot_val.cpu().numpy(), val_logits.detach().cpu().numpy())
                
                print(f"Epoch {epoch+1}: Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, Val AUC={val_auc:.4f}")
            
            model.train()
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X[idx_test])
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
        print("=" * 50)
        
        return {
            "acc": test_acc, 
            "f1": test_f1, 
            "auc": test_auc
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP benchmark')
    parser.add_argument('--dataset', type=str, default='por', choices=['bail', 'german', 'math', 'por'])
    parser.add_argument('--epochs', type=int, default=2000)
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