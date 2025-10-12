import os
import math
from typing import Dict, Tuple
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

# Suppress noisy sklearn warnings during explanation and LARS solves
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from .experts import Expert1, Expert2, Expert3
from .gate import GatingNetwork
from utils import load_data_util, normalize_scipy, feature_norm, fair_metric, seed_everything
from torch_geometric.utils import convert
from explanation_metrics import Interpreter
from attention_fairness_utils import compute_attention_fairness_loss, evaluate_attention_fairness


def compute_batch_scores(y_prob: torch.Tensor, y_true: torch.Tensor, sens: torch.Tensor, interpreter: Interpreter, X_batch: torch.Tensor, model_for_att, att_method: str = 'integrated_gradients'):
    with torch.no_grad():
        y_pred = y_prob.argmax(dim=1)
        acc = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        # AUC using one-hot labels
        one_hot = torch.zeros((len(y_true), 2), device=y_true.device)
        one_hot[torch.arange(len(y_true)), y_true.long()] = 1
        auc = roc_auc_score(one_hot.cpu().numpy(), y_prob.detach().cpu().numpy())

        sp, eo = fair_metric(y_pred.cpu().numpy(), y_true.cpu().numpy(), sens.cpu().numpy())

        # explanation fairness metrics on this batch indices
        # build a temporary index tensor [0..N-1]
        idx_local = torch.arange(X_batch.shape[0], device=X_batch.device)
        interpreter.utility_labels = y_true
        interpreter.sensitive_labels = sens
        p0, p1, REF, v0, v1, VEF, att_jsd = interpreter.interprete(model=model_for_att, idx=idx_local.cpu().numpy())

        utility = (acc + f1 + auc) / 3.0
        fairness_result = (sp + eo) / 2.0
        # Normalize ATT_JSD to [0,1] scale for consistent averaging
        att_jsd_norm = att_jsd / math.log(2) if att_jsd > 0 else att_jsd
        fairness_procedure = (REF + VEF + att_jsd_norm) / 3.0
        final_score = utility - fairness_result - fairness_procedure

    return final_score, {
        "acc": acc, "f1": f1, "auc": auc,
        "sp": sp, "eo": eo,
        "REF": REF, "VEF": VEF, "att_jsd": att_jsd_norm,
        "utility": utility, "fair_res": fairness_result, "fair_proc": fairness_procedure,
    }


class MoETrainer:
    def __init__(self, dataset: str = "bail", seed: int = 0, cuda_device: int = 0,
                 epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-5,
                 lambda_rep: float = 1.0, lambda_fair: float = 1.0,
                 lambda_attention: float = 1.0, lambda_adv: float = 1.0,
                 gate_lr: float = 1e-3, entropy_coeff: float = 1e-3, lb_coeff: float = 1e-3,
                 use_cached_experts: bool = False, cache_dir: str = "weights/moe_experts",
                 use_cached_gate: bool = False, skip_gate: bool = False):
        self.dataset = dataset
        self.seed = seed
        self.cuda_device = cuda_device
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_rep = lambda_rep
        self.lambda_fair = lambda_fair
        self.lambda_attention = lambda_attention
        self.lambda_adv = lambda_adv
        self.gate_lr = gate_lr
        self.entropy_coeff = entropy_coeff
        self.lb_coeff = lb_coeff
        self.use_cached_experts = use_cached_experts
        self.cache_dir = cache_dir
        self.use_cached_gate = use_cached_gate
        self.skip_gate = skip_gate

        seed_everything(seed)

        adj, features, labels, idx_train, idx_val, idx_test, sens = load_data_util(dataset)
        adj = normalize_scipy(adj)
        X = feature_norm(features).float()
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(cuda_device)

        self.X = X.to(self.device)
        self.labels = labels.to(self.device)
        self.idx_train = idx_train.to(self.device)
        self.idx_val = idx_val.to(self.device)
        self.idx_test = idx_test.to(self.device)
        self.sens = sens.to(self.device)
        self.edge_index = edge_index.to(self.device)

        input_dim = self.X.shape[1]
        hidden_dim = 8 if dataset != "german" else 4

        # Experts
        self.expert1 = Expert1(input_dim, hidden_dim).to(self.device)
        self.expert2 = Expert2(input_dim, hidden_dim, lambda_rep=lambda_rep, lambda_fair=lambda_fair).to(self.device)
        self.expert3 = Expert3(input_dim, hidden_dim, lambda_attention=lambda_attention, lambda_adv=lambda_adv).to(self.device)

        # Optionally load cached experts
        if self.use_cached_experts:
            loaded = self.load_experts()
            if loaded:
                print(f"[Cache] Loaded experts from {self.cache_dir}, skipping pretraining.")
            else:
                print(f"[Cache] No cached experts found for dataset={self.dataset}, will pretrain and save.")

        # Gate: build with augmented state dimension (x + experts' preds/confidences + disagreement)
        with torch.no_grad():
            X_probe = self.X[self.idx_train][:8]
            _, p1_probe = self.expert1(X_probe)
            _, p2_probe = self.expert2(X_probe)
            _, p3_probe = self.expert3(X_probe)
            num_classes = p1_probe.shape[1]
            # features: x + p1+p2+p3 + confidences(3) + disagreement(1)
            gate_input_dim = input_dim + 3 * num_classes + 3 + 1
        self.gate = GatingNetwork(input_dim=gate_input_dim, hidden_dim=16).to(self.device)

        # Interpretor for procedural metrics
        self.interpreter = Interpreter(features=self.X, edge_index=self.edge_index,
                                       utility_labels=self.labels, sensitive_labels=self.sens, top_ratio=0.2, topK=1)

        # Optimizers
        self.opt_e = optim.Adam(list(self.expert1.parameters()) +
                                list(self.expert2.parameters()) +
                                list(self.expert3.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        self.opt_g = optim.Adam(self.gate.parameters(), lr=self.gate_lr, weight_decay=0.0)

        self.baseline = 0.0  # moving average baseline for REINFORCE
        self.baseline_momentum = 0.9

    def cache_paths(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        base = f"{self.dataset}"
        return (
            os.path.join(self.cache_dir, f"{base}_expert1.pt"),
            os.path.join(self.cache_dir, f"{base}_expert2.pt"),
            os.path.join(self.cache_dir, f"{base}_expert3.pt"),
        )

    def gate_cache_path(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        base = f"{self.dataset}"
        return os.path.join(self.cache_dir, f"{base}_gate.pt")


    def save_experts(self):
        p1, p2, p3 = self.cache_paths()
        torch.save(self.expert1.state_dict(), p1)
        torch.save(self.expert2.state_dict(), p2)
        torch.save(self.expert3.state_dict(), p3)
        print(f"[Cache] Experts saved to {self.cache_dir}")

    def save_gate(self):
        gp = self.gate_cache_path()
        torch.save(self.gate.state_dict(), gp)
        print(f"[Cache] Gate saved to {gp}")

    def save_best_models(self):
        """Save best models with 'best' suffix"""
        os.makedirs(self.cache_dir, exist_ok=True)
        base = f"{self.dataset}"
        best_paths = (
            os.path.join(self.cache_dir, f"{base}_expert1_best.pt"),
            os.path.join(self.cache_dir, f"{base}_expert2_best.pt"),
            os.path.join(self.cache_dir, f"{base}_expert3_best.pt"),
            os.path.join(self.cache_dir, f"{base}_gate_best.pt"),
        )
        torch.save(self.expert1.state_dict(), best_paths[0])
        torch.save(self.expert2.state_dict(), best_paths[1])
        torch.save(self.expert3.state_dict(), best_paths[2])
        torch.save(self.gate.state_dict(), best_paths[3])
        print(f"[Cache] Best models saved to {self.cache_dir}")


    def load_experts(self) -> bool:
        p1, p2, p3 = self.cache_paths()
        if not (os.path.exists(p1) and os.path.exists(p2) and os.path.exists(p3)):
            return False
        try:
            self.expert1.load_state_dict(torch.load(p1, map_location=self.device))
            self.expert2.load_state_dict(torch.load(p2, map_location=self.device))
            self.expert3.load_state_dict(torch.load(p3, map_location=self.device))
            return True
        except Exception:
            return False

    def load_gate(self) -> bool:
        gp = self.gate_cache_path()
        if not os.path.exists(gp):
            return False
        try:
            self.gate.load_state_dict(torch.load(gp, map_location=self.device))
            return True
        except Exception:
            return False

    def attention_loss_fn(self, model, X_batch, sens_batch):
        return compute_attention_fairness_loss(model, X_batch, sens_batch, method='integrated_gradients', background_X=X_batch[:min(50, X_batch.shape[0])])

    def pretrain_experts(self, progress_update=None):
        warm_epochs = self.epochs // 2
        best_scores = {1: -float('inf'), 2: float('inf'), 3: float('inf')}  # E1: max utility, E2/E3: min fairness
        best_states = {1: None, 2: None, 3: None}

        for epoch in range(self.epochs):
            self.expert1.train(); self.expert2.train(); self.expert3.train()
            self.opt_e.zero_grad()

            Xb = self.X[self.idx_train]
            yb = self.labels[self.idx_train]
            sb = self.sens[self.idx_train]

            # Expert1 CE only
            out1 = self.expert1.compute_loss(Xb, yb)

            # Expert2 CE during warmup (fairness later)
            out2 = self.expert2.compute_loss(Xb, yb, sb) if epoch >= warm_epochs else {"loss": F.cross_entropy(self.expert2(Xb)[1], yb)}

            # Expert3 CE during warmup (attention+adv later)
            if epoch >= warm_epochs:
                out3 = self.expert3.compute_loss(Xb, yb, sb, self.attention_loss_fn)
            else:
                _, p3 = self.expert3(Xb)
                out3 = {"loss": F.cross_entropy(p3, yb)}

            loss = out1["loss"] + out2["loss"] + out3["loss"]
            loss.backward()
            self.opt_e.step()

            # Progress update per epoch (overall bar). During warm start, only bar is shown
            if progress_update is not None:
                progress_update(1)

            # Logging after warm start: every 100 epochs
            if epoch + 1 > warm_epochs and (epoch + 1) % 100 == 0:
                self.expert1.eval(); self.expert2.eval(); self.expert3.eval()
                with torch.no_grad():
                    _, p1 = self.expert1(Xb)
                    _, p2 = self.expert2(Xb)
                    _, p3 = self.expert3(Xb)
                    y_prob = (p1 + p2 + p3) / 3.0
                score, stats = compute_batch_scores(y_prob, yb, sb, self.interpreter, Xb, self.expert3)
                print(
                    f"[Pretrain][Epoch {epoch+1}] Final={score:.4f} | "
                    f"U={stats['utility']:.4f} (ACC={stats['acc']:.3f}, F1={stats['f1']:.3f}, AUC={stats['auc']:.3f}) | "
                    f"FR={stats['fair_res']:.4f} (DP={stats['sp']:.3f}, EO={stats['eo']:.3f}) | "
                    f"FP={stats['fair_proc']:.4f} (REF={stats['REF']:.5f}, VEF={stats['VEF']:.5f}, ATT={stats['att_jsd']:.5f})"
                )

            # Validation monitoring every 100 epochs with model selection
            if ((epoch + 1) % 100 == 0) and (epoch + 1) > warm_epochs:
                self.expert1.eval(); self.expert2.eval(); self.expert3.eval()
                Xv = self.X[self.idx_val]
                yv = self.labels[self.idx_val]
                sv = self.sens[self.idx_val]
                with torch.no_grad():
                    _, p1_val = self.expert1(Xv)
                    _, p2_val = self.expert2(Xv)
                    _, p3_val = self.expert3(Xv)
                score1, stats1 = compute_batch_scores(p1_val, yv, sv, self.interpreter, Xv, self.expert1, att_method='integrated_gradients')
                score2, stats2 = compute_batch_scores(p2_val, yv, sv, self.interpreter, Xv, self.expert2, att_method='integrated_gradients')
                score3, stats3 = compute_batch_scores(p3_val, yv, sv, self.interpreter, Xv, self.expert3, att_method='integrated_gradients')
                
                # Model selection: each expert optimized for its specialization
                # Expert1: utility (higher is better)
                if stats1['utility'] > best_scores[1]:
                    best_scores[1] = stats1['utility']
                    best_states[1] = copy.deepcopy(self.expert1.state_dict())
                
                # Expert2: result fairness (lower is better, so we minimize)
                if stats2['fair_res'] < best_scores[2]:
                    best_scores[2] = stats2['fair_res']
                    best_states[2] = copy.deepcopy(self.expert2.state_dict())
                
                # Expert3: procedural fairness (lower is better, so we minimize)
                if stats3['fair_proc'] < best_scores[3]:
                    best_scores[3] = stats3['fair_proc']
                    best_states[3] = copy.deepcopy(self.expert3.state_dict())
                
                print(f"Epoch {epoch+1}: Val scores - E1(utility): {stats1['utility']:.4f}, E2(fair_res): {stats2['fair_res']:.4f}, E3(fair_proc): {stats3['fair_proc']:.4f}")
                print(f"Best scores so far - E1: {best_scores[1]:.4f}, E2: {best_scores[2]:.4f}, E3: {best_scores[3]:.4f}")

        # Load best models from validation
        if best_states[1] is not None:
            self.expert1.load_state_dict(best_states[1])
            print(f"[Model Selection] Loaded best Expert1 (utility: {best_scores[1]:.4f})")
        if best_states[2] is not None:
            self.expert2.load_state_dict(best_states[2])
            print(f"[Model Selection] Loaded best Expert2 (fair_res: {best_scores[2]:.4f})")
        if best_states[3] is not None:
            self.expert3.load_state_dict(best_states[3])
            print(f"[Model Selection] Loaded best Expert3 (fair_proc: {best_scores[3]:.4f})")
        
        # Save cached experts
        self.save_experts()

    def train_gate(self, progress_update=None):
        self.expert1.eval(); self.expert2.eval(); self.expert3.eval()
        self.gate.train()

        gate_epochs = self.epochs
        best_gate_score = -float('inf')
        best_gate_state = None
        
        # Early stopping for gate
        no_improvement_count = 0
        early_stop_patience = 3 # if no improvement for 300 epochs, stop
        
        for epoch in range(gate_epochs):
            Xb = self.X[self.idx_train]
            yb = self.labels[self.idx_train]
            sb = self.sens[self.idx_train]

            # Experts produce probabilities
            _, p1 = self.expert1(Xb)
            _, p2 = self.expert2(Xb)
            _, p3 = self.expert3(Xb)
            # Build gate state: x + experts' predictions + confidences + disagreement
            with torch.no_grad():
                conf1 = p1.max(dim=1, keepdim=True).values
                conf2 = p2.max(dim=1, keepdim=True).values
                conf3 = p3.max(dim=1, keepdim=True).values
                # disagreement: mean pairwise L1 distance between prob vectors
                d12 = (p1 - p2).abs().mean(dim=1, keepdim=True)
                d13 = (p1 - p3).abs().mean(dim=1, keepdim=True)
                d23 = (p2 - p3).abs().mean(dim=1, keepdim=True)
                disagree = (d12 + d13 + d23) / 3.0
            state = torch.cat([Xb, p1, p2, p3, conf1, conf2, conf3, disagree], dim=1)

            # Gate policy: sample one expert (categorical)
            actions, log_probs, probs_gate = self.gate.sample_action(state)
            self.gate.update_usage_ma(probs_gate)

            # Route each sample to its chosen expert
            #expert_stack = torch.stack([p1, p2, p3], dim=1)  # [N, 3, C]
            #actions_expanded = actions.view(-1, 1, 1).expand(-1, 1, expert_stack.size(2))
            #y_prob_selected = expert_stack.gather(dim=1, index=actions_expanded).squeeze(1)

            # Use unified evaluation for consistent reward components
            eval_stats = self.compute_eval_stats(Xb, yb, sb)
            u1 = eval_stats["baseline_utility"]
            f1 = (eval_stats["baseline_fair_res"] + eval_stats["baseline_fair_proc"]) / 2.0
            u2 = eval_stats["utility"]
            f2 = (eval_stats["fair_res"] + eval_stats["fair_proc"]) / 2.0
            reward_scalar = (u2 - u1) - (f2 - f1)
            reward = torch.tensor(reward_scalar, device=self.device, dtype=torch.float32)

            # Baseline and advantage
            self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * reward.item()
            advantage = reward - self.baseline

            # Optional disagreement weighting (focus learning where experts differ)
            with torch.no_grad():
                w = disagree.squeeze(1)
                w = w / (w.mean() + 1e-8)
            policy_loss = - (w * log_probs).mean() * advantage

            entropy = self.gate.entropy(probs_gate)
            lb = self.gate.load_balance_loss()
            loss_gate = policy_loss - self.entropy_coeff * entropy + self.lb_coeff * lb

            self.opt_g.zero_grad()
            loss_gate.backward()
            self.opt_g.step()

            # Progress update per epoch on overall bar
            if progress_update is not None:
                progress_update(1)

            # Print stats every 100 epochs
            if (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    g_avg = probs_gate.mean(dim=0)
                # Additional delta-based reward diagnostics
                print(f"[Gate][Tra][Epoch {epoch+1}] R={(reward_scalar):.4f} | u1={u1:.4f}, f1={f1:.4f} || u2={u2:.4f}, f2={f2:.4f}")

                # Validation reward-aligned checkpointing (R_val = (u2-u1) - (f2-f1))
                with torch.no_grad():
                    Xv = self.X[self.idx_val]
                    yv = self.labels[self.idx_val]
                    sv = self.sens[self.idx_val]
                    _, p1_v = self.expert1(Xv)
                # Baseline utility/fairness from Expert1

                # Also print unified evaluation stats on validation split
                val_stats = self.evaluate("val")

                u1_v, f1_v_res,f1_v_proc = val_stats["baseline_utility"], val_stats["baseline_fair_res"], val_stats["baseline_fair_proc"]
                f1_v = (f1_v_res + f1_v_proc) / 2.0
                u2_v, f2_v_res,f2_v_proc = val_stats["utility"], val_stats["fair_res"], val_stats["fair_proc"]
                f2_v = (f2_v_res + f2_v_proc) / 2.0
                val_reward = (u2_v - u1_v) - (f2_v - f1_v)
                print(f"[Gate][Val][Epoch {epoch+1}] R_val={val_reward:.4f} | u1={u1_v:.4f}, f1={f1_v:.4f} || u2={u2_v:.4f}, f2={f2_v:.4f}")

                test_stats = self.evaluate("test")
                u1_t, f1_t_res,f1_t_proc = test_stats["baseline_utility"], test_stats["baseline_fair_res"], test_stats["baseline_fair_proc"]
                f1_t = (f1_t_res + f1_t_proc) / 2.0
                u2_t, f2_t_res,f2_t_proc = test_stats["utility"], test_stats["fair_res"], test_stats["fair_proc"]
                f2_t = (f2_t_res + f2_t_proc) / 2.0
                test_reward = (u2_t - u1_t) - (f2_t - f1_t)
                print(f"[Gate][Test][Epoch {epoch+1}] R_test={test_reward:.4f} | u1={u1_t:.4f}, f1={f1_t:.4f} || u2={u2_t:.4f}, f2={f2_t:.4f}")

                # Model selection for gate: save best based on validation final score
                if val_stats['final_score'] > best_gate_score:
                    best_gate_score = val_stats['final_score']
                    best_gate_state = copy.deepcopy(self.gate.state_dict())
                    no_improvement_count = 0  # Reset counter
                    print(f"[Gate Selection] New best gate (final_score: {best_gate_score:.4f})")
                else:
                    no_improvement_count += 1
                
                print(
                    f"[Eval][Val][Epoch {epoch+1}] Final={val_stats['final_score']:.4f} | "
                    f"U={val_stats['utility']:.4f} (ACC={val_stats['acc']:.3f}, F1={val_stats['f1']:.3f}, AUC={val_stats['auc']:.3f}) | "
                    f"FR={val_stats['fair_res']:.4f} (DP={val_stats['sp']:.3f}, EO={val_stats['eo']:.3f}) | "
                    f"FP={val_stats['fair_proc']:.4f} (REF={val_stats['REF']:.3f}, VEF={val_stats['VEF']:.3f}, ATT={val_stats['att_jsd']:.3f})"
                )
                print(f"[Gate] No improvement count: {no_improvement_count}/{early_stop_patience}")
                
                # Early stopping check for gate
                if no_improvement_count >= early_stop_patience:
                    print(f"[Early Stop] Gate training: No improvement for {early_stop_patience} epochs. Stopping at epoch {epoch+1}")
                    break

            # Light expert fine-tuning every 100 epochs
            if (epoch + 1) % 100 == 0:
                self.expert1.train(); self.expert2.train(); self.expert3.train()
                self.opt_e.zero_grad()
                out1 = self.expert1.compute_loss(Xb, yb)
                out2 = self.expert2.compute_loss(Xb, yb, sb)
                out3 = self.expert3.compute_loss(Xb, yb, sb, self.attention_loss_fn)
                (out1["loss"] + out2["loss"] + out3["loss"]).backward()
                self.opt_e.step()
                self.expert1.eval(); self.expert2.eval(); self.expert3.eval()

        # Load best gate model from validation
        if best_gate_state is not None:
            self.gate.load_state_dict(best_gate_state)
            print(f"[Model Selection] Loaded best gate (final_score: {best_gate_score:.4f})")

    def evaluate(self, split: str = "test") -> Dict[str, float]:
        idx = {"train": self.idx_train, "val": self.idx_val, "test": self.idx_test}[split]
        Xb = self.X[idx]
        yb = self.labels[idx]
        sb = self.sens[idx]
        return self.compute_eval_stats(Xb, yb, sb)

    def compute_eval_stats(self, Xb: torch.Tensor, yb: torch.Tensor, sb: torch.Tensor) -> Dict[str, float]:
        self.expert1.eval(); self.expert2.eval(); self.expert3.eval(); self.gate.eval()
        with torch.no_grad():
            # Expert probabilities
            _, p1 = self.expert1(Xb)
            _, p2 = self.expert2(Xb)
            _, p3 = self.expert3(Xb)

            

            # Gate state features
            conf1 = p1.max(dim=1, keepdim=True).values
            conf2 = p2.max(dim=1, keepdim=True).values
            conf3 = p3.max(dim=1, keepdim=True).values
            d12 = (p1 - p2).abs().mean(dim=1, keepdim=True)
            d13 = (p1 - p3).abs().mean(dim=1, keepdim=True)
            d23 = (p2 - p3).abs().mean(dim=1, keepdim=True)
            disagree = (d12 + d13 + d23) / 3.0
            state = torch.cat([Xb, p1, p2, p3, conf1, conf2, conf3, disagree], dim=1)
            probs_gate = self.gate(state)
            # Mixture output for utility/result fairness
            y_prob = probs_gate[:, 0:1] * p1 + probs_gate[:, 1:2] * p2 + probs_gate[:, 2:3] * p3

            # Utility metrics from mixture
            y_pred = y_prob.argmax(dim=1)
            acc = accuracy_score(yb.cpu().numpy(), y_pred.cpu().numpy())
            f1 = f1_score(yb.cpu().numpy(), y_pred.cpu().numpy())
            one_hot = torch.zeros((len(yb), y_prob.shape[1]), device=yb.device)
            one_hot[torch.arange(len(yb)), yb.long()] = 1
            auc = roc_auc_score(one_hot.cpu().numpy(), y_prob.detach().cpu().numpy())

            # Result fairness from mixture predictions
            sp, eo = fair_metric(y_pred.cpu().numpy(), yb.cpu().numpy(), sb.cpu().numpy())

            # Procedural fairness from experts, weighted by gate usage
            g_avg = probs_gate.mean(dim=0)
            # Compute REF/VEF/ATT for each expert
            idx_local = torch.arange(Xb.shape[0], device=Xb.device)
            self.interpreter.utility_labels = yb
            self.interpreter.sensitive_labels = sb
            _p0, _p1v, REF1, _v0, _v1, VEF1, ATT1 = self.interpreter.interprete(model=self.expert1, idx=idx_local.cpu().numpy())
            _p0, _p1v, REF2, _v0, _v1, VEF2, ATT2 = self.interpreter.interprete(model=self.expert2, idx=idx_local.cpu().numpy())
            _p0, _p1v, REF3, _v0, _v1, VEF3, ATT3 = self.interpreter.interprete(model=self.expert3, idx=idx_local.cpu().numpy())

            # Normalize ATT_JSD to [0,1] scale (JSD max is log(2))
            ATT1_norm = ATT1 / math.log(2) if ATT1 > 0 else ATT1
            ATT2_norm = ATT2 / math.log(2) if ATT2 > 0 else ATT2
            ATT3_norm = ATT3 / math.log(2) if ATT3 > 0 else ATT3
            
            fair_proc1 = (REF1 + VEF1 + ATT1_norm) / 3.0
            fair_proc2 = (REF2 + VEF2 + ATT2_norm) / 3.0
            fair_proc3 = (REF3 + VEF3 + ATT3_norm) / 3.0
            fair_proc = (
                g_avg[0].item() * float(fair_proc1)
                + g_avg[1].item() * float(fair_proc2)
                + g_avg[2].item() * float(fair_proc3)
            )
            # Also report weighted components for transparency (use normalized ATT)
            REF_w = g_avg[0].item() * float(REF1) + g_avg[1].item() * float(REF2) + g_avg[2].item() * float(REF3)
            VEF_w = g_avg[0].item() * float(VEF1) + g_avg[1].item() * float(VEF2) + g_avg[2].item() * float(VEF3)
            ATT_w = g_avg[0].item() * float(ATT1_norm) + g_avg[1].item() * float(ATT2_norm) + g_avg[2].item() * float(ATT3_norm)

        utility = (acc + f1 + auc) / 3.0
        fairness_result = (sp + eo) / 2.0
        final_score = utility - fairness_result - fair_proc


        # Baseline (Expert1) utility and fairness (result fairness)
        y_pred_e1 = p1.argmax(dim=1)
        acc_e1 = accuracy_score(yb.cpu().numpy(), y_pred_e1.cpu().numpy())
        f1_e1 = f1_score(yb.cpu().numpy(), y_pred_e1.cpu().numpy())
        one_hot_e1 = torch.zeros((len(yb), p1.shape[1]), device=yb.device)
        one_hot_e1[torch.arange(len(yb)), yb.long()] = 1
        auc_e1 = roc_auc_score(one_hot_e1.cpu().numpy(), p1.detach().cpu().numpy())
        baseline_utility = (acc_e1 + f1_e1 + auc_e1) / 3.0
        sp_e1, eo_e1 = fair_metric(y_pred_e1.cpu().numpy(), yb.cpu().numpy(), sb.cpu().numpy())
        baseline_fair_res = (sp_e1 + eo_e1) / 2.0
        baseline_fair_proc = fair_proc1

        stats = {
            "acc": acc,
            "f1": f1,
            "auc": auc,
            "sp": sp,
            "eo": eo,
            "REF": REF_w,
            "VEF": VEF_w,
            "att_jsd": ATT_w,
            "utility": utility,
            "fair_res": fairness_result,
            "fair_proc": fair_proc,
            "final_score": final_score,
            "baseline_utility": baseline_utility,
            "baseline_fair_res": baseline_fair_res,
            "baseline_fair_proc": baseline_fair_proc,
        }
        return stats

    def evaluate_individual_experts(self, split: str = "test") -> Dict[str, float]:
        """Evaluate individual experts and return their scores for hyperparameter tuning"""
        idx = {"train": self.idx_train, "val": self.idx_val, "test": self.idx_test}[split]
        Xb = self.X[idx]
        yb = self.labels[idx]
        sb = self.sens[idx]
        
        self.expert1.eval(); self.expert2.eval(); self.expert3.eval()
        
        with torch.no_grad():
            # Get individual expert predictions
            _, p1 = self.expert1(Xb)
            _, p2 = self.expert2(Xb)
            _, p3 = self.expert3(Xb)
            
            # Evaluate each expert individually
            score1, stats1 = compute_batch_scores(p1, yb, sb, self.interpreter, Xb, self.expert1, att_method='integrated_gradients')
            score2, stats2 = compute_batch_scores(p2, yb, sb, self.interpreter, Xb, self.expert2, att_method='integrated_gradients')
            score3, stats3 = compute_batch_scores(p3, yb, sb, self.interpreter, Xb, self.expert3, att_method='integrated_gradients')
        
        return {
            "expert1_score": score1,
            "expert1_stats": stats1,
            "expert2_score": score2,
            "expert2_stats": stats2,
            "expert3_score": score3,
            "expert3_stats": stats3,
            "best_expert1_utility": stats1['utility'],
            "best_expert2_fair_res": stats2['fair_res'],
            "best_expert3_fair_proc": stats3['fair_proc'],
        }

    def run(self):
        if self.skip_gate:
            # Expert-only training for hyperparameter tuning
            print("[Expert Training] Skipping gate training - experts only mode")
            total_steps = self.epochs
            with tqdm(total=total_steps, desc="Expert Training", leave=True) as pbar:
                if self.use_cached_experts and self.load_experts():
                    # Skip pretraining; advance progress bar by pretrain steps
                    pbar.update(self.epochs)
                    print(f"[Cache] Using cached experts from {self.cache_dir}; skipping expert training.")
                else:
                    self.pretrain_experts(progress_update=pbar.update)
            
            # Save best models
            self.save_best_models()
            print(f"[Expert Evaluation] Evaluating individual experts")
            return self.evaluate_individual_experts("test")
        else:
            # Full MoE training with gate
            total_steps = self.epochs*2
            with tqdm(total=total_steps, desc="MoE Training", leave=True) as pbar:
                if self.use_cached_experts and self.load_experts():
                    # Skip pretraining; advance progress bar by pretrain steps
                    pbar.update(self.epochs)
                    print(f"[Cache] Using cached experts from {self.cache_dir}; skipping expert training.")
                else:
                    self.pretrain_experts(progress_update=pbar.update)
                if self.use_cached_gate and self.load_gate():
                    # Skip gate training; advance progress bar by gate steps
                    pbar.update(self.epochs)
                    print(f"[Cache] Loaded cached gate from {self.cache_dir}; skipping gate training.")
                else:
                    self.train_gate(progress_update=pbar.update)
                    self.save_gate()
            # Save best models and use for final test evaluation
            self.save_best_models()
            print(f"[Final Evaluation] Using best models for test evaluation")
            return self.evaluate("test")


