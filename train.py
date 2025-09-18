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
import math


def train(epoch, model):
    model.train()
    optimizer.zero_grad()

    hidden, output = model(x=X)
    l_classification = F.cross_entropy(output[idx_train], labels[idx_train].long())
    
    # Initialize distance losses
    l_distance = torch.tensor(0.0)
    l_distance_masked = torch.tensor(0.0)
    
    if epoch < args.opt_start_epoch: # warm start
        loss_train = l_classification
    else: # apply distance-based loss
        X_masked, _ = interpretation.generate_masked_X(idx=idx_train.cpu().numpy(), model=model)
        hidden_masked, output_masked = model(x=X_masked)
        l_distance = group_distance(hidden, idx_train.cpu().numpy(), sens.cpu().numpy(), \
            labels.cpu().numpy())
        l_distance_masked = group_distance(hidden_masked, range(len(idx_train)), sens[idx_train].cpu().numpy(), \
            labels[idx_train].cpu().numpy())
        loss_train = l_classification + args.lambda_*(l_distance+l_distance_masked)
    loss_train.backward()
    optimizer.step()

    if epoch >= (args.epochs/2):
        if epoch % 10 == 0:
            model.eval()

            hidden, output = model(x=X)
            preds = (output.argmax(axis=1)).type_as(labels)
           
            # accuracy-related metrics (validation)
            acc_val = accuracy_score(labels[idx_val.cpu().numpy()].cpu().numpy(), preds[idx_val.cpu().numpy()].cpu().numpy())
            auc_roc_val = roc_auc_score(one_hot_labels[idx_val.cpu().numpy()], output.detach().cpu().numpy()[idx_val.cpu().numpy()])
            f1_val = f1_score(labels[idx_val.cpu().numpy()].cpu().numpy(), preds[idx_val.cpu().numpy()].cpu().numpy())
            
            # traditional fairness-related metrics (validation)
            sp_val, eo_val = fair_metric(preds[idx_val.cpu().numpy()].cpu().numpy(), labels[idx_val.cpu().numpy()].cpu().numpy(),\
                sens[idx_val.cpu().numpy()].cpu().numpy())

            p0_val, p1_val, REF_val, v0_val, v1_val, VEF_val \
            = interpretation.interprete(model=model, idx=idx_val.cpu().numpy())

            # record validation logs
            logs.append([
                l_classification.item(), l_distance.item(), l_distance_masked.item(), \
                loss_train.item(), \
                acc_val, auc_roc_val, f1_val, \
                sp_val, eo_val, \
                p0_val, p1_val, REF_val, \
                v0_val, v1_val, VEF_val
                ])
            
            # Print validation metrics
            print(f"Epoch {epoch}: Val Acc={acc_val:.4f}, Val F1={f1_val:.4f}, Val AUC={auc_roc_val:.4f}, Val SP={sp_val:.4f}, Val EO={eo_val:.4f}")

if __name__ == '__main__':
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    seed_everything(args.seed)

    train_log_path = "./train_logs/"+args.dataset
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)
        print("The new directory is created for saving the training logs.")

    # load dataset
    if args.use_similarity_split:
        from utils import load_data_util_similarity_based, fair_metric
        chosen_seed = None
        adj = features = labels = idx_train = idx_val = idx_test = sens = None
        for attempt in range(args.max_split_seed_tries):
            split_seed = args.split_seed_base + attempt
            adj_tmp, features_tmp, labels_tmp, idx_train_tmp, idx_val_tmp, idx_test_tmp, sens_tmp = load_data_util_similarity_based(
                args.dataset, t1=args.t1, t2=args.t2, t3=args.t3, fair_noise=args.fair_noise, verbose=False, seed=split_seed
            )
            if adj_tmp is None:
                continue
            sp_tr, eo_tr = fair_metric(labels_tmp[idx_train_tmp].cpu().numpy(), labels_tmp[idx_train_tmp].cpu().numpy(), sens_tmp[idx_train_tmp].cpu().numpy())
            sp_va, eo_va = fair_metric(labels_tmp[idx_val_tmp].cpu().numpy(), labels_tmp[idx_val_tmp].cpu().numpy(), sens_tmp[idx_val_tmp].cpu().numpy())
            fair_tr = (sp_tr + eo_tr) / 2
            fair_va = (sp_va + eo_va) / 2
            # ensure training fairness score larger (more biased) than validation and test
            sp_te, eo_te = fair_metric(labels_tmp[idx_test_tmp].cpu().numpy(), labels_tmp[idx_test_tmp].cpu().numpy(), sens_tmp[idx_test_tmp].cpu().numpy())
            fair_te = (sp_te + eo_te) / 2
            if fair_tr > fair_va and fair_tr > fair_te:
                chosen_seed = split_seed
                adj, features, labels, idx_train, idx_val, idx_test, sens = adj_tmp, features_tmp, labels_tmp, idx_train_tmp, idx_val_tmp, idx_test_tmp, sens_tmp
                break
        if chosen_seed is None:
            # fallback to last attempt
            adj, features, labels, idx_train, idx_val, idx_test, sens = adj_tmp, features_tmp, labels_tmp, idx_train_tmp, idx_val_tmp, idx_test_tmp, sens_tmp
            print(f"[Info] Fairness condition not met within {args.max_split_seed_tries} seeds. Proceeding with seed {split_seed}.")
        else:
            print(f"[Info] Using similarity-based split with seed {chosen_seed}.")
    else:
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
        edge_index.cuda()
        X = X.cuda()
        labels = labels.cuda()
        idx_train = idx_train
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        sens = sens.cuda()

    interpretation = Interpreter(features=X, edge_index=edge_index,\
        utility_labels=labels, sensitive_labels=sens, top_ratio=args.top_ratio, \
        topK=args.topK)

    logs = []

    # Train model
    for epoch in tqdm(range(args.epochs)):
        train(epoch, model)

    logs = np.array(logs)
    filename = "./train_logs/"+str(args.dataset)+"/lambda_"+str(args.lambda_)+\
    "_seed_"+str(args.seed)+"_"+str(args.lr)+"_"+str(args.weight_decay)\
    +"_"+str(args.dropout)+".npy"
    np.save(open(filename, 'wb'), logs)

    # Print final test results
    model.eval()
    with torch.no_grad():
        hidden, output = model(x=X)
        preds = (output.argmax(axis=1)).type_as(labels)
        
        # Test accuracy, F1, and AUC
        acc_test = accuracy_score(labels[idx_test.cpu().numpy()].cpu().numpy(), preds[idx_test.cpu().numpy()].cpu().numpy())
        f1_test = f1_score(labels[idx_test.cpu().numpy()].cpu().numpy(), preds[idx_test.cpu().numpy()].cpu().numpy())
        auc_test = roc_auc_score(one_hot_labels[idx_test.cpu().numpy()], output.detach().cpu().numpy()[idx_test.cpu().numpy()])
        
        
        # Test explanation fairness metrics
        p0_test, p1_test, REF_test, v0_test, v1_test, VEF_test = interpretation.interprete(model=model, idx=idx_test.cpu().numpy())
        
        print(f"\n=== Final Test Results ===")
        print(f"Y Accuracy: {acc_test:.4f}")
        print(f"F1 Score: {f1_test:.4f}")
        print(f"AUC Score: {auc_test:.4f}")
        print(f"REF (Explanation Fairness): {REF_test:.4f}")
        print(f"VEF (Explanation Fairness): {VEF_test:.4f}")
        print(f"==========================\n")

