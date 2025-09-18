import time
import argparse
import numpy as np
import torch
import sys
import os
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import scipy.sparse as sp
from utils import load_data_util, fair_metric, seed_everything, feature_norm, normalize_scipy, group_distance
from mlp import *
from torch_geometric.utils import dropout_adj, convert
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import ctypes
import matplotlib.pyplot as plt
import pickle
import random
from explanation_metrics import *
from parse import *
from attention_fairness_utils import compute_attention_fairness_loss, evaluate_attention_fairness
import math


def train(epoch, model):
    model.train()
    optimizer.zero_grad()

    hidden, output = model(x=X)
    l_classification = F.cross_entropy(output[idx_train], labels[idx_train].long())
    
    # Initialize losses
    l_attention_fairness = torch.tensor(0.0, device=X.device)
    
    if epoch < args.opt_start_epoch: # warm start
        loss_train = l_classification
    else: # apply fairness losses
        
        # New attention-based fairness loss
        if args.use_attention_fairness:
            try:
                l_attention_fairness = compute_attention_fairness_loss(
                    model, X[idx_train], sens[idx_train], 
                    method=args.attention_method,
                    background_X=X[idx_train][:min(50, len(idx_train))]  # Smaller background for efficiency
                )
            except Exception as e:
                print(f"Warning: Attention fairness computation failed at epoch {epoch}: {e}")
                l_attention_fairness = torch.tensor(0.0, device=X.device)
        
        # Combined loss
        loss_train = l_classification + \
                    args.lambda_attention * l_attention_fairness
        #print(f"Loss train: {l_classification.item()}, {l_attention_fairness.item()}")
                    
    loss_train.backward()
    optimizer.step()

    if epoch >= (args.epochs/2):
        if epoch % 10 == 0:
            model.eval()

            hidden, output = model(x=X)
            preds = (output.argmax(axis=1)).type_as(labels)
           
            # accuracy-related metrics (validation)
            acc_val = accuracy_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
            auc_roc_val = roc_auc_score(one_hot_labels[idx_val.cpu().numpy()], output.detach().cpu().numpy()[idx_val.cpu().numpy()])
            f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())
            
            # traditional fairness-related metrics (validation)
            sp_val, eo_val = fair_metric(preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(),\
                sens[idx_val].cpu().numpy())

            # explanation fairness metrics
            p0_val, p1_val, REF_val, v0_val, v1_val, VEF_val \
            = interpretation.interprete(model=model, idx=idx_val.cpu().numpy())

            # attention fairness evaluation
            attention_jsd = 0.0
            if args.use_attention_fairness and epoch % 10 == 0:  # Evaluate less frequently due to computational cost
                try:
                    attention_jsd, _ = evaluate_attention_fairness(
                        model, X[idx_val], sens[idx_val], 
                        method=args.attention_method
                    )
                except Exception as e:
                    print(f"Warning: Attention fairness evaluation failed at epoch {epoch}: {e}")

            # compute comprehensive score
            utility_score = (acc_val + f1_val + auc_roc_val) / 3
            traditional_fairness_score = (sp_val + eo_val) / 2
            explanation_fairness_score = (VEF_val + REF_val) / 2
            
            # Add fairness improvement score (I) - simplified version
            comprehensive_score = utility_score - traditional_fairness_score - explanation_fairness_score 

            # record validation logs
            logs.append([
                l_classification.item(), 
                l_attention_fairness.item(), loss_train.item(), \
                acc_val, auc_roc_val, f1_val, \
                sp_val, eo_val, \
                p0_val, p1_val, REF_val, \
                v0_val, v1_val, VEF_val, \
                attention_jsd, comprehensive_score
                ])
            
            # print U score, trandition fairness score, VEF,REF,explanation fairness score, attention fairness score
            print(f"Epoch {epoch}: U score={utility_score:.4f}, " + 
                  f"Traditional fairness score={traditional_fairness_score:.4f}, " +
                  f"Explanation fairness score={explanation_fairness_score:.4f}, " +
                  f"VEF={VEF_val:.4f}, REF={REF_val:.4f}, " +
                  f"Attention fairness score={attention_jsd:.4f}")


def parse_attention_args():
    """Parse command line arguments for attention-based fairness training"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='cuda device running on.')
    parser.add_argument('--dataset', type=str, default='bail',
                        help='a dataset from bail, german, math, por.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed.')
    parser.add_argument('--model', type=str, default='mlp',
                        help='mlp model.')
    parser.add_argument('--topK', type=int, default=1,
                        help='features to be masked when computing fidelity.')
    
    # Distance-based fairness parameters
    parser.add_argument('--lambda_distance', type=float, default=0.0,
                        help='lambda for distance-based fairness loss')
    parser.add_argument('--use_distance_loss', action='store_true', default=False,
                        help='Use original distance-based fairness loss')
    
    # Attention-based fairness parameters
    parser.add_argument('--lambda_attention', type=float, default=1.0,
                        help='lambda for attention-based fairness loss')
    parser.add_argument('--use_attention_fairness', action='store_true', default=True,
                        help='Use attention-based fairness loss')
    parser.add_argument('--attention_method', type=str, default='integrated_gradients',
                        choices=['shap', 'integrated_gradients', 'lime', 'gradient_shap'],
                        help='Method for extracting attention weights')
    
    parser.add_argument('--top_ratio', type=float, default=0.2,
                        help='top ratio of features to be masked')
    parser.add_argument('--opt_start_epoch', type=int, default=50,
                        help='epoch to start optimizing fairness')
    
    # Similarity-based splitting parameters
    parser.add_argument('--use_similarity_split', action='store_true', default=False,
                        help='Use similarity-based data splitting')
    parser.add_argument('--t1', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--t2', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--t3', type=float, default=0.2,
                        help='Test set ratio')
    parser.add_argument('--fair_noise', type=float, default=0.1,
                        help='Fair noise ratio for similarity-based splitting')
    parser.add_argument('--max_split_seed_tries', type=int, default=10,
                        help='Maximum number of seed tries for similarity-based splitting')
    parser.add_argument('--split_seed_base', type=int, default=42,
                        help='Base seed for similarity-based splitting')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_attention_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    seed_everything(args.seed)

    train_log_path = "./train_logs/attention_fairness/" + args.dataset
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)
        print("The new directory is created for saving the training logs.")

    print(f"Starting attention-based fairness training with method: {args.attention_method}")
    print(f"Lambda attention: {args.lambda_attention}")
    print(f"Use attention fairness: {args.use_attention_fairness}")


    adj, features, labels, idx_train, idx_val, idx_test, sens = load_data_util(args.dataset)
    
    adj_ori = adj
    adj = normalize_scipy(adj)
    features = feature_norm(features)
    X = features.float()
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    one_hot_labels = np.zeros((len(labels), labels.max()+1))
    one_hot_labels[np.arange(len(labels)), labels] = 1

    if args.dataset == "german":
        arch = [X.shape[1], 4, 2]
    else:
        arch = [X.shape[1], 8, 2]

    # create model
    model = MLP(arch, dropout=args.dropout).float()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        torch.cuda.set_device(args.cuda_device)
        model.cuda()
        edge_index = edge_index.cuda()
        X = X.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        sens = sens.cuda()

    interpretation = Interpreter(features=X, edge_index=edge_index,\
        utility_labels=labels, sensitive_labels=sens, top_ratio=args.top_ratio, \
        topK=args.topK)

    logs = []

    print(f"Dataset: {args.dataset}, Features: {X.shape[1]}, Samples: {len(labels)}")
    print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    print(f"Sensitive group distribution: {sens.sum().item()}/{len(sens) - sens.sum().item()}")

    # Train model
    print("\nStarting training...")
    for epoch in tqdm(range(args.epochs)):
        train(epoch, model)

    logs = np.array(logs)
    filename = train_log_path + "/att_" + args.attention_method + "_lambda_att_" + str(args.lambda_attention) + \
              "_seed_" + str(args.seed) + "_" + str(args.lr) + \
              "_" + str(args.weight_decay) + "_" + str(args.dropout) + ".npy"
    np.save(open(filename, 'wb'), logs)
    print(f"Training logs saved to: {filename}")

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        hidden, output = model(x=X)
        preds = (output.argmax(axis=1)).type_as(labels)
        
        # Test accuracy, F1, and AUC
        acc_test = accuracy_score(labels[idx_test].cpu().numpy(), preds[idx_test].cpu().numpy())
        f1_test = f1_score(labels[idx_test].cpu().numpy(), preds[idx_test].cpu().numpy())
        auc_test = roc_auc_score(one_hot_labels[idx_test.cpu().numpy()], output.detach().cpu().numpy()[idx_test.cpu().numpy()])
        
        # Traditional fairness metrics
        sp_test, eo_test = fair_metric(preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].cpu().numpy())
        
        # Test explanation fairness metrics
        p0_test, p1_test, REF_test, v0_test, v1_test, VEF_test = interpretation.interprete(model=model, idx=idx_test.cpu().numpy())
        
        # Attention fairness evaluation
        try:
            attention_jsd_test, attention_stats = evaluate_attention_fairness(
                model, X[idx_test], sens[idx_test], 
                method=args.attention_method
            )
            print(f"Attention group sizes: {attention_stats['group_0_size']}, {attention_stats['group_1_size']}")
        except Exception as e:
            print(f"Warning: Final attention fairness evaluation failed: {e}")
            attention_jsd_test = float('inf')

        # Comprehensive score
        utility_score = (acc_test + f1_test + auc_test) / 3
        traditional_fairness_score = (sp_test + eo_test) / 2
        explanation_fairness_score = (VEF_test + REF_test) / 2
        comprehensive_score = utility_score - traditional_fairness_score - explanation_fairness_score 
        
        print(f"\n=== Final Test Results ===")
        print(f"Utility Metrics:")
        print(f"  Accuracy: {acc_test:.4f}")
        print(f"  F1 Score: {f1_test:.4f}")
        print(f"  AUC Score: {auc_test:.4f}")
        print(f"  Utility Score: {utility_score:.4f}")
        print(f"Traditional Fairness:")
        print(f"  Demographic Parity: {sp_test:.4f}")
        print(f"  Equal Opportunity: {eo_test:.4f}")
        print(f"  Traditional Fairness Score: {traditional_fairness_score:.4f}")
        print(f"Explanation Fairness:")
        print(f"  REF: {REF_test:.4f}")
        print(f"  VEF: {VEF_test:.4f}")
        print(f"  Explanation Fairness Score: {explanation_fairness_score:.4f}")
        print(f"Attention Fairness:")
        print(f"  Jensen-Shannon Divergence: {attention_jsd_test:.4f}")
        print(f"Overall:")
        print(f"  Comprehensive Score: {comprehensive_score:.4f}")
        print(f"==========================\n")