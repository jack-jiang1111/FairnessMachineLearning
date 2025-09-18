import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import shap
from captum.attr import IntegratedGradients, Lime, GradientShap
import torch.nn.functional as F


class AttentionExtractor:
    """Extract attention weights using different interpretability methods"""
    
    def __init__(self, model, method='shap'):
        self.model = model
        self.method = method
        self.explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Setup the chosen interpretability method"""
        # Explainers will be created on demand with proper model wrappers
        # This avoids the tuple/shape issue
        pass
    
    def extract_attention_shap(self, X, background_X=None):
        """Extract attention weights using SHAP"""
        if background_X is None:
            # Use a subset of data as background
            background_X = X[:min(100, len(X))]
        
        # Create SHAP explainer
        def model_wrapper(x):
            with torch.no_grad():
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32)
                if x.device != X.device:
                    x = x.to(X.device)
                _, output = self.model(x)
                return F.softmax(output, dim=1).cpu().numpy()
        
        explainer = shap.KernelExplainer(model_wrapper, background_X.cpu().numpy())
        shap_values = explainer.shap_values(X.cpu().numpy())
        
        # Convert to attention weights (absolute values, normalized)
        if isinstance(shap_values, list):
            # Multi-class case - use values for positive class
            attention_weights = np.abs(shap_values[1])
        else:
            attention_weights = np.abs(shap_values)
        
        # Normalize to sum to 1 for each sample
        attention_weights = attention_weights / (attention_weights.sum(axis=1, keepdims=True) + 1e-8)
        
        return torch.tensor(attention_weights, dtype=torch.float32, device=X.device)
    
    def extract_attention_integrated_gradients(self, X):
        """Extract attention weights using Integrated Gradients"""
        self.model.eval()
        
        # Create a wrapper that returns only the final output
        def forward_wrapper(x):
            _, output = self.model(x)
            return output
        
        # Create IG explainer with the wrapper
        ig = IntegratedGradients(forward_wrapper)
        
        # Define baseline (zeros)
        baseline = torch.zeros_like(X)
        
        # Get attributions
        attributions = ig.attribute(X, baseline, target=1)
        
        # Convert to attention weights
        attention_weights = torch.abs(attributions)
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdims=True) + 1e-8)
        
        return attention_weights
    
    def extract_attention_lime(self, X):
        """Extract attention weights using LIME"""
        self.model.eval()
        
        # Create a wrapper that returns only the final output
        def forward_wrapper(x):
            _, output = self.model(x)
            return output
        
        # Create LIME explainer with the wrapper
        lime = Lime(forward_wrapper)
        
        # LIME requires more setup for tabular data
        attributions = lime.attribute(X, target=1, n_samples=500)
        
        # Convert to attention weights
        attention_weights = torch.abs(attributions)
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdims=True) + 1e-8)
        
        return attention_weights
    
    def extract_attention_gradient_shap(self, X, background_X=None):
        """Extract attention weights using Gradient SHAP"""
        if background_X is None:
            background_X = X[:min(100, len(X))]
        
        self.model.eval()
        
        # Create a wrapper that returns only the final output
        def forward_wrapper(x):
            _, output = self.model(x)
            return output
        
        # Create GradientShap explainer with the wrapper
        grad_shap = GradientShap(forward_wrapper)
        
        # Get attributions
        attributions = grad_shap.attribute(X, background_X, target=1)
        
        # Convert to attention weights
        attention_weights = torch.abs(attributions)
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdims=True) + 1e-8)
        
        return attention_weights
    
    def extract_attention(self, X, background_X=None):
        """Extract attention weights using the chosen method"""
        if self.method == 'shap':
            return self.extract_attention_shap(X, background_X)
        elif self.method == 'integrated_gradients':
            return self.extract_attention_integrated_gradients(X)
        elif self.method == 'lime':
            return self.extract_attention_lime(X)
        elif self.method == 'gradient_shap':
            return self.extract_attention_gradient_shap(X, background_X)


def compute_jensen_shannon_divergence(attention_group1, attention_group2):
    """
    Compute Jensen-Shannon Divergence between two attention distributions
    
    Args:
        attention_group1: Tensor of shape (n_samples1, n_features) - attention weights for group 1
        attention_group2: Tensor of shape (n_samples2, n_features) - attention weights for group 2
    
    Returns:
        jsd: Jensen-Shannon Divergence between the two groups
    """
    # Compute mean attention for each group
    mean_attention_group1 = attention_group1.mean(dim=0)
    mean_attention_group2 = attention_group2.mean(dim=0)
    
    # Ensure non-negative and normalize (keep as tensors for gradient flow)
    eps = 1e-8
    mean_attention_group1 = torch.clamp(mean_attention_group1, min=eps)
    mean_attention_group2 = torch.clamp(mean_attention_group2, min=eps)
    
    mean_attention_group1 = mean_attention_group1 / mean_attention_group1.sum()
    mean_attention_group2 = mean_attention_group2 / mean_attention_group2.sum()
    
    # Compute Jensen-Shannon Divergence using PyTorch operations
    # JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q)
    M = 0.5 * (mean_attention_group1 + mean_attention_group2)
    
    # KL divergence: KL(P || Q) = sum(P * log(P / Q))
    kl1 = (mean_attention_group1 * torch.log(mean_attention_group1 / M)).sum()
    kl2 = (mean_attention_group2 * torch.log(mean_attention_group2 / M)).sum()
    
    jsd = 0.5 * kl1 + 0.5 * kl2
    
    return jsd


def compute_attention_fairness_loss(model, X, sens, method='integrated_gradients', background_X=None):
    """
    Compute attention-based fairness loss using Jensen-Shannon Divergence
    
    Args:
        model: The neural network model
        X: Input features
        sens: Sensitive attributes (0 or 1)
        method: Interpretability method ('shap', 'integrated_gradients', 'lime', 'gradient_shap')
        background_X: Background data for SHAP (optional)
    
    Returns:
        fairness_loss: Jensen-Shannon Divergence between attention distributions
    """
    # Create attention extractor
    extractor = AttentionExtractor(model, method=method)
    
    # Extract attention weights
    attention_weights = extractor.extract_attention(X, background_X)
    
    # Split by sensitive groups
    group_0_mask = (sens == 0)
    group_1_mask = (sens == 1)
    
    if group_0_mask.sum() == 0 or group_1_mask.sum() == 0:
        # If one group is empty, return zero loss
        return torch.tensor(0.0, device=X.device)
    
    attention_group_0 = attention_weights[group_0_mask]
    attention_group_1 = attention_weights[group_1_mask]
    
    # Compute Jensen-Shannon Divergence
    jsd = compute_jensen_shannon_divergence(attention_group_0, attention_group_1)
    
    return jsd


def evaluate_attention_fairness(model, X, sens, method='shap'):
    """
    Evaluate attention fairness using Jensen-Shannon Divergence
    
    Args:
        model: The neural network model
        X: Input features
        sens: Sensitive attributes
        method: Interpretability method
    
    Returns:
        jsd: Jensen-Shannon Divergence score
        group_stats: Dictionary with attention statistics for each group
    """
    with torch.no_grad():
        extractor = AttentionExtractor(model, method=method)
        attention_weights = extractor.extract_attention(X)
        
        # Split by sensitive groups
        group_0_mask = (sens == 0)
        group_1_mask = (sens == 1)
        
        attention_group_0 = attention_weights[group_0_mask]
        attention_group_1 = attention_weights[group_1_mask]
        
        # Compute statistics
        group_stats = {
            'group_0_mean': attention_group_0.mean(dim=0),
            'group_1_mean': attention_group_1.mean(dim=0),
            'group_0_std': attention_group_0.std(dim=0),
            'group_1_std': attention_group_1.std(dim=0),
            'group_0_size': len(attention_group_0),
            'group_1_size': len(attention_group_1)
        }
        
        # Compute Jensen-Shannon Divergence
        jsd = compute_jensen_shannon_divergence(attention_group_0, attention_group_1)
        
    return jsd.item(), group_stats