"""
Uncertainty-Aware SegFormer using Evidential Deep Learning
Based on: "Evidential Deep Learning to Quantify Classification Uncertainty" (Sensoy et al., 2018)
Adapted for semantic segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class UncertainSegFormer(nn.Module):
    """
    SegFormer with Evidential Deep Learning for uncertainty estimation
    
    Outputs:
        - Segmentation predictions
        - Uncertainty estimates (epistemic + aleatoric)
        - Evidence/alpha parameters for analysis
    """
    
    def __init__(self, encoder_name='mit_b2', num_classes=7, pretrained=True):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Base SegFormer model
        self.segformer = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            classes=num_classes,
            activation=None  # We'll apply our own activation
        )
        
        print(f"[UncertainSegFormer] Created with encoder: {encoder_name}")
        print(f"[UncertainSegFormer] Number of classes: {num_classes}")
    
    def forward(self, x, return_uncertainty=True):
        """
        Forward pass with uncertainty quantification
        
        Args:
            x: Input images (B, C, H, W)
            return_uncertainty: Whether to compute and return uncertainty
        
        Returns:
            Dictionary with:
                - 'logits': Raw logits (B, num_classes, H, W)
                - 'evidence': Evidence for each class (B, num_classes, H, W)
                - 'alpha': Dirichlet parameters (B, num_classes, H, W)
                - 'prob': Predicted probabilities (B, num_classes, H, W)
                - 'uncertainty': Uncertainty map (B, H, W) [if return_uncertainty=True]
                - 'pred': Predicted class (B, H, W)
        """
        # Get logits from SegFormer
        logits = self.segformer(x)
        
        # Convert logits to evidence (must be non-negative)
        # Using ReLU instead of softplus for better gradients and numerical stability
        evidence = F.relu(logits)
        
        # Dirichlet parameters: alpha = evidence + 1
        alpha = evidence + 1.0
        
        # Total evidence (Dirichlet strength)
        S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Expected probability (mean of Dirichlet distribution)
        prob = alpha / S
        
        # Predicted class
        pred = torch.argmax(prob, dim=1)  # (B, H, W)
        
        output = {
            'logits': logits,
            'evidence': evidence,
            'alpha': alpha,
            'prob': prob,
            'pred': pred,
        }
        
        if return_uncertainty:
            # Uncertainty quantification
            uncertainty = self.compute_uncertainty(alpha, S)
            output['uncertainty'] = uncertainty
        
        return output
    
    def compute_uncertainty(self, alpha, S):
        """
        Compute uncertainty metrics from Dirichlet parameters
        
        Types of uncertainty:
            1. Epistemic (model uncertainty): high when evidence is low
            2. Aleatoric (data uncertainty): high when distribution is uniform
            3. Total uncertainty: combination of both
        
        Args:
            alpha: Dirichlet parameters (B, num_classes, H, W)
            S: Sum of alpha (B, 1, H, W)
        
        Returns:
            uncertainty: Total uncertainty map (B, H, W)
        """
        # Epistemic uncertainty (vacuity): K / S
        # High when total evidence is low
        epistemic = self.num_classes / S.squeeze(1)  # (B, H, W)
        
        # Aleatoric uncertainty (dissonance): based on variance of Dirichlet
        # Variance = alpha_k * (S - alpha_k) / (S^2 * (S + 1))
        prob = alpha / S  # (B, num_classes, H, W)
        aleatoric = (prob * (1 - prob)).sum(dim=1)  # (B, H, W)
        
        # Total uncertainty (we'll primarily use epistemic for active learning)
        # But we compute both for analysis
        total_uncertainty = epistemic
        
        return total_uncertainty
    
    def get_uncertainty_decomposition(self, alpha, S):
        """
        Get detailed uncertainty decomposition for analysis
        
        Returns dict with epistemic, aleatoric, and total uncertainty
        """
        epistemic = self.num_classes / S.squeeze(1)
        
        prob = alpha / S
        aleatoric = (prob * (1 - prob)).sum(dim=1)
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': epistemic  # We use epistemic as primary uncertainty
        }


class EnsembleUncertainSegFormer(nn.Module):
    """
    Ensemble version for comparison with MC Dropout
    Uses multiple forward passes to estimate uncertainty
    """
    
    def __init__(self, base_model, num_samples=10):
        super().__init__()
        self.base_model = base_model
        self.num_samples = num_samples
    
    def forward(self, x):
        """
        Forward pass with MC sampling
        
        Returns mean prediction and variance-based uncertainty
        """
        predictions = []
        
        # Multiple forward passes
        for _ in range(self.num_samples):
            with torch.no_grad():
                output = self.base_model(x, return_uncertainty=False)
                predictions.append(output['prob'])
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_samples, B, C, H, W)
        
        # Mean prediction
        mean_prob = predictions.mean(dim=0)
        
        # Variance-based uncertainty
        variance = predictions.var(dim=0).sum(dim=1)  # Sum over classes
        
        pred = torch.argmax(mean_prob, dim=1)
        
        return {
            'prob': mean_prob,
            'pred': pred,
            'uncertainty': variance
        }


def get_uncertain_model(config):
    """
    Factory function to create uncertainty-aware model
    
    Args:
        config: Configuration object
    
    Returns:
        UncertainSegFormer model
    """
    model = UncertainSegFormer(
        encoder_name=config.model.encoder,
        num_classes=config.data.num_classes,
        pretrained=config.model.pretrained
    )
    
    return model


def test_model():
    """Test model creation and forward pass"""
    print("Testing UncertainSegFormer...")
    
    # Create model
    model = UncertainSegFormer(encoder_name='mit_b0', num_classes=7)
    model.eval()
    
    # Dummy input
    x = torch.randn(2, 3, 512, 512)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, return_uncertainty=True)
    
    # Check outputs
    print(f"✓ Logits shape: {output['logits'].shape}")
    print(f"✓ Evidence shape: {output['evidence'].shape}")
    print(f"✓ Alpha shape: {output['alpha'].shape}")
    print(f"✓ Prob shape: {output['prob'].shape}")
    print(f"✓ Pred shape: {output['pred'].shape}")
    print(f"✓ Uncertainty shape: {output['uncertainty'].shape}")
    
    # Check values
    assert output['logits'].shape == (2, 7, 512, 512)
    assert output['evidence'].min() >= 0, "Evidence must be non-negative"
    assert output['alpha'].min() >= 1.0, "Alpha must be >= 1"
    assert torch.allclose(output['prob'].sum(dim=1), torch.ones(2, 512, 512)), "Probs must sum to 1"
    assert output['uncertainty'].min() >= 0, "Uncertainty must be non-negative"
    
    print("✅ All tests passed!")


if __name__ == '__main__':
    test_model()