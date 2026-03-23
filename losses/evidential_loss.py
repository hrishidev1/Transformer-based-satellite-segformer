"""
Evidential Loss Functions for Uncertainty-Aware Segmentation

Based on:
- "Evidential Deep Learning to Quantify Classification Uncertainty" (Sensoy et al., 2018)
- "Improving Evidential Deep Learning via Multi-task Learning" (Zhao et al., 2021)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp  # FIXED: Moved to module level


class EvidentialLoss(nn.Module):
    """
    Corrected Evidential Loss with KL Annealing and Proper Math
    
    Loss = MSE (Type II ML) + Annealed KL Divergence
    """
    
    def __init__(self, num_classes, lambda_kl=0.1, max_epoch=10, ignore_index=None):
        """
        Args:
            num_classes: Number of segmentation classes
            lambda_kl: Weight for KL divergence regularization
            max_epoch: Number of epochs to anneal KL from 0 to lambda_kl
            ignore_index: Class index to ignore
        """
        super().__init__()
        self.num_classes = num_classes
        self.lambda_kl = lambda_kl
        self.max_epoch = max_epoch
        self.ignore_index = ignore_index
        
        print(f"[EvidentialLoss] num_classes: {num_classes}")
        print(f"[EvidentialLoss] lambda_kl: {lambda_kl} (Annealing over {max_epoch} epochs)")
    
    def kl_divergence(self, alpha, num_classes):
        """
        Calculates correct KL(Dir(alpha) || Dir(1))
        
        This forces the distribution towards a uniform distribution (high uncertainty)
        unless the data provides strong evidence otherwise.
        """
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=alpha.device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        first_term = (
            torch.lgamma(S)
            - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
            + torch.sum(torch.lgamma(ones), dim=1, keepdim=True)
            - torch.lgamma(torch.sum(ones, dim=1, keepdim=True))
        )
        
        second_term = torch.sum(
            (alpha - ones) * (torch.digamma(alpha) - torch.digamma(S)),
            dim=1, keepdim=True
        )
        
        return first_term + second_term

    def forward(self, output_dict, target, current_epoch=None):
        """
        Compute evidential loss with annealing
        """
        alpha = output_dict['alpha']
        prob = output_dict['prob']
        
        # 1. Flatten spatial dimensions (B, C, H, W) -> (N, C)
        B, C, H, W = alpha.shape
        alpha_flat = alpha.permute(0, 2, 3, 1).reshape(-1, C)
        prob_flat = prob.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target.reshape(-1)
        
        # 2. Masking (Ignore Index)
        if self.ignore_index is not None:
            valid_mask = target_flat != self.ignore_index
            alpha_flat = alpha_flat[valid_mask]
            prob_flat = prob_flat[valid_mask]
            target_flat = target_flat[valid_mask]
            
        target_one_hot = F.one_hot(target_flat, num_classes=self.num_classes).float()
        
        # 3. MSE Loss (Type II Maximum Likelihood)
        # Eq. 12 in Sensoy et al.
        mse_loss = torch.sum((prob_flat - target_one_hot) ** 2, dim=1).mean()
        
        # 4. KL Divergence Regularization
        # We penalize evidence on INCORRECT classes.
        # alpha_tilde = target + (1-target) * alpha
        alpha_tilde = target_one_hot + (1 - target_one_hot) * alpha_flat
        
        kl_div = self.kl_divergence(alpha_tilde, self.num_classes)
        kl_loss = kl_div.squeeze().mean()  # FIXED: Added squeeze() for clarity
        
        # 5. Annealing Coefficient
        # Grows linearly from 0 to 1 over max_epoch
        if current_epoch is not None and current_epoch < self.max_epoch:
            annealing_coef = torch.tensor(current_epoch / self.max_epoch, device=alpha.device)
        else:
            annealing_coef = torch.tensor(1.0, device=alpha.device)
            
        total_loss = mse_loss + (self.lambda_kl * annealing_coef * kl_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'kl': kl_loss.item(),
            'annealing_coef': annealing_coef.item()
        }
        
        return total_loss, loss_dict


class EvidentialDiceLoss(nn.Module):
    """
    Hybrid: Evidential Loss + Dice Loss
    """
    
    def __init__(self, num_classes, lambda_kl=0.1, lambda_dice=0.5, max_epoch=10, ignore_index=None):
        super().__init__()
        # Initialize corrected Evidential Loss
        self.evidential = EvidentialLoss(num_classes, lambda_kl, max_epoch, ignore_index)
        
        # FIXED: Removed local import, using module level import
        self.dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
        
        self.lambda_dice = lambda_dice
        print(f"[EvidentialDiceLoss] lambda_dice: {lambda_dice}")
    
    def forward(self, output_dict, target, current_epoch=None):
        """
        Compute combined loss, passing current_epoch down to evidential loss
        """
        # Evidential loss (handles annealing internally)
        evid_loss, evid_dict = self.evidential(output_dict, target, current_epoch)
        
        # Dice loss (use probabilities)
        dice_loss = self.dice(output_dict['prob'], target)
        
        # Combined loss
        total_loss = evid_loss + self.lambda_dice * dice_loss
        
        # Loss dict
        loss_dict = {
            'total': total_loss.item(),
            'evidential': evid_loss.item(),
            'dice': dice_loss.item(),
            **{f'evid_{k}': v for k, v in evid_dict.items() if k != 'total'}
        }
        
        return total_loss, loss_dict


def get_evidential_loss(config):
    """
    Factory function to create evidential loss
    """
    loss_config = config.loss
    
    # Prioritize 'annealing_step' (from your new config), fallback to 'annealing_epochs', default to 10
    if hasattr(loss_config, 'annealing_step'):
        max_epoch = loss_config.annealing_step
    elif hasattr(loss_config, 'annealing_epochs'):
        max_epoch = loss_config.annealing_epochs
    else:
        max_epoch = 10
    print(f"[Loss Factory] Using max_epoch={max_epoch} for KL annealing.")
    
    if hasattr(loss_config, 'use_dice') and loss_config.use_dice:
        return EvidentialDiceLoss(
            num_classes=config.data.num_classes,
            lambda_kl=loss_config.lambda_kl if hasattr(loss_config, 'lambda_kl') else 0.1,
            lambda_dice=loss_config.lambda_dice if hasattr(loss_config, 'lambda_dice') else 0.5,
            max_epoch=max_epoch,
            ignore_index=config.data.ignore_index if hasattr(config.data, 'ignore_index') else None
        )
    else:
        return EvidentialLoss(
            num_classes=config.data.num_classes,
            lambda_kl=loss_config.lambda_kl if hasattr(loss_config, 'lambda_kl') else 0.1,
            max_epoch=max_epoch,
            ignore_index=config.data.ignore_index if hasattr(config.data, 'ignore_index') else None
        )
    
def test_loss():
    """Test loss computation with annealing"""
    print("Testing EvidentialLoss...")
    
    # Create dummy data WITH GRADIENTS
    B, C, H, W = 2, 7, 64, 64
    
    # Simulate model output - IMPORTANT: requires_grad=True
    # alpha must be >= 1 for Dirichlet
    alpha = torch.rand(B, C, H, W, requires_grad=True) * 10 + 1.0  
    S = alpha.sum(dim=1, keepdim=True)
    prob = alpha / S
    
    output_dict = {
        'alpha': alpha,
        'prob': prob
    }
    
    target = torch.randint(0, C, (B, H, W))
    
    # 1. Test Initial Epoch (Epoch 0 - Annealing Coef = 0)
    print("\n[Test 1] Epoch 0 (KL should be 0)")
    loss_fn = EvidentialLoss(num_classes=C, lambda_kl=0.1, max_epoch=10, ignore_index=None)
    loss, loss_dict = loss_fn(output_dict, target, current_epoch=0)
    
    print(f"✓ Total Loss: {loss.item():.4f}")
    print(f"✓ MSE: {loss_dict['mse']:.4f}")
    print(f"✓ KL Raw: {loss_dict['kl']:.4f}")
    print(f"✓ Annealing Coef: {loss_dict['annealing_coef']:.4f}")
    
    # Verify KL is effectively 0 in total loss (Total should equals MSE)
    assert abs(loss.item() - loss_dict['mse']) < 1e-5, "At epoch 0, Loss should equal MSE"
    
    # 2. Test Max Epoch (Epoch 10 - Annealing Coef = 1.0)
    print("\n[Test 2] Epoch 10 (KL should be full weight)")
    loss_full, loss_dict_full = loss_fn(output_dict, target, current_epoch=10)
    
    print(f"✓ Total Loss: {loss_full.item():.4f}")
    print(f"✓ KL Component: {loss_dict_full['kl'] * 0.1:.4f}") # 0.1 is lambda_kl
    print(f"✓ Annealing Coef: {loss_dict_full['annealing_coef']:.4f}")
    
    # 3. Test Backward Pass
    print("\n[Test 3] Backward Pass")
    # Create fresh tensors
    alpha_test = torch.rand(B, C, H, W, requires_grad=True) * 10 + 1.0
    prob_test = alpha_test / alpha_test.sum(dim=1, keepdim=True)
    output_test = {'alpha': alpha_test, 'prob': prob_test}
    
    loss_test, _ = loss_fn(output_test, target, current_epoch=5)
    loss_test.backward()
    
    print("✓ Backward pass successful (Gradients computed)")
    print("✅ All loss tests passed!")

if __name__ == '__main__':
    test_loss()