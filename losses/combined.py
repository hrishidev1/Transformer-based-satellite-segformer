"""
Combined loss functions for semantic segmentation
Supports both standard and uncertainty-aware training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):
    """
    Combined Cross Entropy + Dice Loss
    Standard loss for semantic segmentation
    """
    
    def __init__(self, config, class_weights=None):
        super().__init__()
        self.config = config
        
        # Get loss configuration
        loss_config = config.loss
        ignore_index = config.data.ignore_index if hasattr(config.data, 'ignore_index') else None
        
        # Cross Entropy Loss
        if class_weights is not None:
            self.ce = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights),
                ignore_index=ignore_index if ignore_index is not None else -100
            )
        else:
            self.ce = nn.CrossEntropyLoss(
                ignore_index=ignore_index if ignore_index is not None else -100
            )
        
        # Dice Loss
        self.dice = smp.losses.DiceLoss(
            mode='multiclass',
            ignore_index=ignore_index
        )
        
        # Loss weights
        self.ce_weight = loss_config.ce_weight if hasattr(loss_config, 'ce_weight') else 0.5
        self.dice_weight = loss_config.dice_weight if hasattr(loss_config, 'dice_weight') else 0.5
        
        print(f"[CombinedLoss] CE weight: {self.ce_weight}, Dice weight: {self.dice_weight}")
        if class_weights is not None:
            print(f"[CombinedLoss] Using class weights")
    
    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        dice_loss = self.dice(preds, targets)
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'ce': ce_loss.item(),
            'dice': dice_loss.item()
        }
        
        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    From "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        print(f"[FocalLoss] alpha: {alpha}, gamma: {gamma}")
    
    def forward(self, preds, targets):
        # FIXED: Removed the expensive and unused target one-hot encoding
        
        # Compute focal loss directly
        ce_loss = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Handle ignore index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            focal_loss = focal_loss * mask
            loss = focal_loss.sum() / mask.sum()
        else:
            loss = focal_loss.mean()
        
        return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice Loss
    Better for handling class imbalance
    """
    
    def __init__(self, alpha=0.5, beta=0.5, ignore_index=None):
        super().__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.ignore_index = ignore_index
        
        print(f"[TverskyLoss] alpha: {alpha}, beta: {beta}")
    
    def forward(self, preds, targets):
        # Get probabilities
        probs = F.softmax(preds, dim=1)
        
        # One-hot encode targets (Required here for Tversky calculations)
        B, C, H, W = preds.shape
        targets_one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
        
        # Handle ignore index
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask
        
        # Compute Tversky index for each class
        tp = (probs * targets_one_hot).sum(dim=(2, 3))
        fp = (probs * (1 - targets_one_hot)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets_one_hot).sum(dim=(2, 3))
        
        tversky_index = tp / (tp + self.alpha * fp + self.beta * fn + 1e-7)
        
        # Average over batch and classes
        loss = 1 - tversky_index.mean()
        
        return loss


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovasz-Softmax Loss
    Optimizes IoU directly
    """
    
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        # FIXED: Instantiated the SMP loss object exactly once in __init__
        self.lovasz = smp.losses.LovaszLoss(mode='multiclass', ignore_index=self.ignore_index)
        print(f"[LovaszSoftmaxLoss] Created")
    
    def forward(self, preds, targets):
        return self.lovasz(preds, targets)


def get_loss_fn(config, class_weights=None):
    """
    Factory function to create loss function
    """
    loss_type = config.loss.type.lower() if hasattr(config.loss, 'type') else 'combined'
    ignore_index = config.data.ignore_index if hasattr(config.data, 'ignore_index') else None
    
    if loss_type == 'evidential':
        from .evidential_loss import get_evidential_loss
        return get_evidential_loss(config)
    
    if loss_type == 'ce':
        if class_weights is not None:
            return nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights),
                ignore_index=ignore_index if ignore_index is not None else -100
            )
        return nn.CrossEntropyLoss(
            ignore_index=ignore_index if ignore_index is not None else -100
        )
    
    elif loss_type == 'dice':
        return smp.losses.DiceLoss(
            mode='multiclass',
            ignore_index=ignore_index
        )
    
    elif loss_type == 'focal':
        alpha = config.loss.focal_alpha if hasattr(config.loss, 'focal_alpha') else 0.25
        gamma = config.loss.focal_gamma if hasattr(config.loss, 'focal_gamma') else 2.0
        return FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
    
    elif loss_type == 'tversky':
        alpha = config.loss.tversky_alpha if hasattr(config.loss, 'tversky_alpha') else 0.5
        beta = config.loss.tversky_beta if hasattr(config.loss, 'tversky_beta') else 0.5
        return TverskyLoss(alpha=alpha, beta=beta, ignore_index=ignore_index)
    
    elif loss_type == 'lovasz':
        return LovaszSoftmaxLoss(ignore_index=ignore_index)
    
    elif loss_type == 'combined':
        return CombinedLoss(config, class_weights)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def test_losses():
    """Test loss functions"""
    print("Testing loss functions...")
    
    B, C, H, W = 2, 7, 64, 64
    targets = torch.randint(0, C, (B, H, W))
    
    losses_to_test = {
        'CrossEntropy': nn.CrossEntropyLoss(),
        'Focal': FocalLoss(),
        'Tversky': TverskyLoss(),
        'Dice': smp.losses.DiceLoss(mode='multiclass'),
        'Lovasz': LovaszSoftmaxLoss()
    }
    
    print("\n" + "="*60)
    for name, loss_fn in losses_to_test.items():
        preds_test = torch.randn(B, C, H, W, requires_grad=True)
        
        loss = loss_fn(preds_test, targets)
        print(f"{name:<20} Loss: {loss.item():.4f}")
        
        loss.backward()
    
    print("="*60)
    print("\n✅ All loss tests passed!")

if __name__ == '__main__':
    test_losses()