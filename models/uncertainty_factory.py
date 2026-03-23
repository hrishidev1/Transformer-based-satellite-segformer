import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UncertainWrapper(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        # FIXED: Added num_classes attribute to prevent AttributeErrors in downstream scripts
        self.num_classes = num_classes
        
    def forward(self, x, return_uncertainty=False):
        logits = self.base_model(x)
        # Evidence must be non-negative
        evidence = torch.relu(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        pred_mask = torch.argmax(prob, dim=1)
        
        # FIXED: Removed redundant 'probs' key
        output = {
            'logits': logits,
            'alpha': alpha,
            'pred': pred_mask,
            'prob': prob
        }
        
        # CRITICAL FIX: Only compute uncertainty math if explicitly requested
        if return_uncertainty:
            K = alpha.shape[1]
            uncertainty = K / S
            output['uncertainty'] = uncertainty.squeeze(1)
            
        return output

def get_uncertainty_model(arch_name, encoder_name, num_classes, pretrained=True):
    weights = "imagenet" if pretrained else None
    
    # Factory Logic
    if arch_name == "Unet":
        base_model = smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights=weights, 
            classes=num_classes,
            activation=None 
        )
    elif arch_name == "UnetPlusPlus":
        base_model = smp.UnetPlusPlus(
            encoder_name=encoder_name, 
            encoder_weights=weights, 
            classes=num_classes,
            activation=None 
        )
    elif arch_name == "DeepLabV3Plus":
        base_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name, 
            encoder_weights=weights, 
            classes=num_classes,
            activation=None
        )
    elif arch_name == "SegFormer":
        base_model = smp.Segformer(
            encoder_name=encoder_name, 
            encoder_weights=weights, 
            classes=num_classes, 
            activation=None
        )
    elif arch_name == "FPN":
        base_model = smp.FPN(
            encoder_name=encoder_name, 
            encoder_weights=weights, 
            classes=num_classes, 
            activation=None
        )
    else:
        # Fallback for generic names if library supports them
        raise ValueError(f"Architecture {arch_name} not supported in factory.")
        
    # FIXED: Pass num_classes to the wrapper
    return UncertainWrapper(base_model, num_classes)