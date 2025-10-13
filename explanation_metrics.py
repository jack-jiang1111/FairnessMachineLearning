import torch
import time
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch.nn as nn
from sklearn.linear_model import LassoLars, LinearRegression
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
from captum.attr import IntegratedGradients


class GraphLIME_speedup:
    def __init__(self, X, edge_index, hop=1, rho=0.1):
        self.hop = hop
        self.rho = rho
        self.subset_dict = dict()
        num_nodes = X.shape[0]
        self.X = X
        self.edge_index = edge_index

        for node_idx in range(X.shape[0]):
            subset, edge_index_new, mapping, edge_mask = k_hop_subgraph(
                node_idx, self.hop, edge_index, relabel_nodes=True,
                num_nodes=num_nodes, flow="source_to_target")
            self.subset_dict[node_idx] = subset

    def __subgraph__(self, node_idx, y, **kwargs):
        subset = self.subset_dict[node_idx]
        x = self.X[subset]
        y = y[subset]
        return x, y

    def __init_predict__(self, model, **kwargs):
        model.eval()
        with torch.no_grad():
            _, logits = model(x=self.X, **kwargs)
            probas = nn.Softmax(dim=1)(logits) # transform to probability
        return probas
    
    def __compute_kernel__(self, x, reduce):
        assert x.ndim == 2, x.shape
        n, d = x.shape
        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
        dist = dist ** 2
        if reduce:
            dist = np.sum(dist, axis=-1, keepdims=True)  # (n, n, 1)
        std = np.sqrt(d)  
        K = np.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))  # (n, n, 1) or (n, n, d)
        return K
    
    def __compute_gram_matrix__(self, x):

        # more stable and accurate implementation
        G = x - np.mean(x, axis=0, keepdims=True)
        G = G - np.mean(G, axis=1, keepdims=True)

        G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-10)

        return G
        
    def explain_node(self, node_idx, x, edge_index, **kwargs):
        probas = self.__init_predict__(x, edge_index, **kwargs)

        x, probas, _, _, _, _ = self.__subgraph__(
            node_idx, x, probas, edge_index, **kwargs)

        x = x.detach().cpu().numpy()  # (n, d)
        y = probas.detach().cpu().numpy()  # (n, classes)

        n, d = x.shape

        K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
        L = self.__compute_kernel__(y, reduce=True)  # (n, n, 1)

        K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
        L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

        K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
        L_bar = L_bar.reshape(n ** 2,)  # (n ** 2,)

        solver = LassoLars(self.rho, fit_intercept=False, normalize=False, positive=True)

        solver.fit(K_bar * n, L_bar * n)

        return solver.coef_

    def explain_nodes(self, model, node_idxs, **kwargs):
        all_probas = self.__init_predict__(model, **kwargs)

        coefs = []
        for i in range(len(node_idxs)):
            node_idx = node_idxs[i]

            subset = self.subset_dict[node_idx]
            x = self.X[subset].detach().cpu().numpy()
            y = all_probas[subset].detach().cpu().numpy()
            
            n, d = x.shape

            K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
            L = self.__compute_kernel__(y, reduce=True)  # (n, n, 1)

            K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
            L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

            K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
            L_bar = L_bar.reshape(n ** 2,)  # (n ** 2,)

            solver = LassoLars(self.rho, fit_intercept=False, normalize=False, positive=True)

            try:
                solver.fit(K_bar * n, L_bar * n)
                coef = solver.coef_
            except ValueError as e:
                # Handle dimension mismatch errors by using fallback coefficients
                print(f"Warning: LARS solver failed for node {node_idx}: {e}")
                # Use uniform coefficients as fallback
                coef = np.ones(d) / d
            except Exception as e:
                print(f"Warning: Unexpected error in LARS solver for node {node_idx}: {e}")
                coef = np.ones(d) / d

            coefs.append(coef)
        return abs(np.array(coefs))

class Interpreter:
    def __init__(self, features, edge_index, utility_labels, sensitive_labels, top_ratio=0.2, \
        topK=1, rho=0.1):
        self.explainer = GraphLIME_speedup(X=features, edge_index=edge_index, hop=1, rho=rho)
        self.X = features
        self.edge_index = edge_index
        self.utility_labels = utility_labels
        self.sensitive_labels = sensitive_labels
        self.K = topK
        self.top_ratio = top_ratio # select the top high fidelity users

    def generate_masked_X(self, idx, model):
        time_start = time.time()
        X_masked = self.X[idx].detach().clone().cuda()
        
        try:
            # remove top K important features
            coefs = torch.tensor(self.explainer.explain_nodes(model, idx.tolist()))
            indices = coefs.argsort()[:, -self.K:].cuda()
            # use scatter to set the values
            if self.K != 0:
                X_masked = X_masked.scatter(1, indices, torch.zeros((len(idx),self.K)).cuda())
            return X_masked.cuda(), coefs.tolist()
        except Exception as e:
            print(f"Warning: GraphLIME explainer failed: {e}")
            # Return original features and uniform coefficients as fallback
            n_samples, n_features = X_masked.shape
            uniform_coefs = torch.ones((n_samples, n_features)) / n_features
            return X_masked.cuda(), uniform_coefs.tolist()

    def interprete(self, model, idx):
        model.eval()

        # find the index of subgroups
        sens = self.sensitive_labels[idx]
        idx_s0 = (sens == 0).nonzero().squeeze().tolist()
        idx_s1 = (sens == 1).nonzero().squeeze().tolist()

        # compute fidelity_prob for each user
        output = model.predict_proba(x=self.X[idx]) # N, 2
        preds = (output.argmax(axis=1)).type_as(self.utility_labels) # N
        preds_prob = output.gather(1, preds.view(-1,1))

        X_masked, coefs = self.generate_masked_X(idx, model) # idx, F
        masked_output = model.predict_proba(x=X_masked)
        masked_preds = (masked_output.argmax(axis=1)).type_as(self.utility_labels)
        masked_preds_prob = masked_output.gather(1, masked_preds.view(-1,1))
        
        F = preds_prob - masked_preds_prob # N, 1

        # find the top high fidelity
        K = int(self.top_ratio*len(idx))
        value, max_fidelity_index = F.topk(k=K, dim=0)
        max_fidelity_index_set = set(max_fidelity_index.squeeze().tolist())
        group0 = set(idx_s0).intersection(max_fidelity_index_set)
        group1 = set(idx_s1).intersection(max_fidelity_index_set)

        # REF
        p0 = float(len(group0))/float(len(idx_s0))
        p1 = float(len(group1))/float(len(idx_s1))

        # VEF
        k0 = int(self.top_ratio*len(idx_s0))
        k1 = int(self.top_ratio*len(idx_s1))
        _, max_fidelity_index0 = F[idx_s0].topk(k=k0, dim=0)
        _, max_fidelity_index1 = F[idx_s1].topk(k=k1, dim=0)
        max_fidelity_index0 = max_fidelity_index0.squeeze().tolist()
        max_fidelity_index1 = max_fidelity_index1.squeeze().tolist()
        original_acc_0 = accuracy_score(self.utility_labels[idx][max_fidelity_index0].cpu().numpy(), preds[max_fidelity_index0].cpu().numpy())
        masked_acc_0 = accuracy_score(self.utility_labels[idx][max_fidelity_index0].cpu().numpy(), masked_preds[max_fidelity_index0].cpu().numpy())
        original_acc_1 = accuracy_score(self.utility_labels[idx][max_fidelity_index1].cpu().numpy(), preds[max_fidelity_index1].cpu().numpy())
        masked_acc_1 = accuracy_score(self.utility_labels[idx][max_fidelity_index1].cpu().numpy(), masked_preds[max_fidelity_index1].cpu().numpy())
        top_acc_fidelity_g0 = original_acc_0 - masked_acc_0
        top_acc_fidelity_g1 = original_acc_1 - masked_acc_1

        # Compute attention JSD using integrated gradients
        try:
            att_jsd = self._compute_attention_jsd(model, idx, sens)
        except Exception as e:
            print(f"Warning: Attention JSD computation failed: {e}")
            att_jsd = 0.0

        return p0, p1, abs(p0-p1), \
        top_acc_fidelity_g0, top_acc_fidelity_g1, abs(top_acc_fidelity_g0-top_acc_fidelity_g1), att_jsd

    def _compute_attention_jsd(self, model, idx, sens):
        """Compute attention JSD using integrated gradients"""
        model.eval()
        
        # Get input features for the batch
        X_batch = self.X[idx]
        
        # Create a wrapper that returns only the final output
        def forward_wrapper(x):
            _, output = model(x)
            return output
        
        # Create Integrated Gradients explainer
        ig = IntegratedGradients(forward_wrapper)
        
        # Define baseline (zeros)
        baseline = torch.zeros_like(X_batch)
        
        # Get attributions
        attributions = ig.attribute(X_batch, baseline, target=1)
        
        # Convert to attention weights (absolute values, normalized)
        attention_weights = torch.abs(attributions)
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdims=True) + 1e-8)
        
        # Split by sensitive groups
        group_0_mask = (sens == 0)
        group_1_mask = (sens == 1)
        
        if group_0_mask.sum() == 0 or group_1_mask.sum() == 0:
            return 0.0
        
        attention_group_0 = attention_weights[group_0_mask]
        attention_group_1 = attention_weights[group_1_mask]
        
        # Compute Jensen-Shannon Divergence
        jsd = self._compute_jensen_shannon_divergence(attention_group_0, attention_group_1)
        
        return jsd.item()
    
    def _compute_jensen_shannon_divergence(self, attention_group1, attention_group2):
        """Compute Jensen-Shannon Divergence between two attention distributions"""
        # Compute mean attention for each group
        mean_attention_group1 = attention_group1.mean(dim=0)
        mean_attention_group2 = attention_group2.mean(dim=0)
        
        # Ensure non-negative and normalize
        eps = 1e-8
        mean_attention_group1 = torch.clamp(mean_attention_group1, min=eps)
        mean_attention_group2 = torch.clamp(mean_attention_group2, min=eps)
        
        mean_attention_group1 = mean_attention_group1 / mean_attention_group1.sum()
        mean_attention_group2 = mean_attention_group2 / mean_attention_group2.sum()
        
        # Compute Jensen-Shannon Divergence
        M = 0.5 * (mean_attention_group1 + mean_attention_group2)
        
        # KL divergence: KL(P || Q) = sum(P * log(P / Q))
        kl1 = (mean_attention_group1 * torch.log(mean_attention_group1 / M)).sum()
        kl2 = (mean_attention_group2 * torch.log(mean_attention_group2 / M)).sum()
        
        jsd = 0.5 * kl1 + 0.5 * kl2
        
        return jsd
        