"""
Model factory for creating segmentation models
Supports both standard and uncertainty-aware models
"""
import segmentation_models_pytorch as smp
from .uncertain_segformer import UncertainSegFormer


def get_model(config):
    """
    Factory function to create segmentation models
    
    Args:
        config: Configuration object
    
    Returns:
        PyTorch model (standard or uncertainty-aware)
    """
    num_classes = config.data.num_classes
    encoder_weights = 'imagenet' if config.model.pretrained else None
    
    # Check if uncertainty-aware model is requested
    if hasattr(config.model, 'name') and config.model.name == 'uncertain_segformer':
        print("[Factory] Creating Uncertainty-Aware SegFormer")
        model = UncertainSegFormer(
            encoder_name=config.model.encoder if hasattr(config.model, 'encoder') else 'mit_b2',
            num_classes=num_classes,
            pretrained=config.model.pretrained
        )
        return model
    
    # Standard models
    model_name = config.model.name.lower() if hasattr(config.model, 'name') else 'segformer'
    
    if model_name == 'segformer':
        encoder_name = config.model.encoder if hasattr(config.model, 'encoder') else 'mit_b2'
        
        model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None  # Raw logits
        )
        
        print(f"[Factory] Created SegFormer with {encoder_name} encoder")
        return model
    
    elif model_name == 'unet':
        encoder_name = config.model.encoder if hasattr(config.model, 'encoder') else 'resnet50'
        
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None
        )
        
        print(f"[Factory] Created UNet with {encoder_name} encoder")
        return model
    
    elif model_name == 'unetplusplus':
        encoder_name = config.model.encoder if hasattr(config.model, 'encoder') else 'resnet50'
        
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None
        )
        
        print(f"[Factory] Created UNet++ with {encoder_name} encoder")
        return model
    
    elif model_name == 'deeplabv3':
        encoder_name = config.model.encoder if hasattr(config.model, 'encoder') else 'resnet50'
        
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None
        )
        
        print(f"[Factory] Created DeepLabV3 with {encoder_name} encoder")
        return model
    
    elif model_name == 'deeplabv3plus':
        encoder_name = config.model.encoder if hasattr(config.model, 'encoder') else 'resnet50'
        
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None
        )
        
        print(f"[Factory] Created DeepLabV3+ with {encoder_name} encoder")
        return model
    
    elif model_name == 'fpn':
        encoder_name = config.model.encoder if hasattr(config.model, 'encoder') else 'resnet50'
        
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None
        )
        
        print(f"[Factory] Created FPN with {encoder_name} encoder")
        return model
    
    elif model_name == 'pspnet':
        encoder_name = config.model.encoder if hasattr(config.model, 'encoder') else 'resnet50'
        
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None
        )
        
        print(f"[Factory] Created PSPNet with {encoder_name} encoder")
        return model
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model):
    """Print model information"""
    total_params = count_parameters(model)
    print(f"\n{'='*60}")
    print("MODEL INFORMATION")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {total_params:,}")
    print(f"Model size (FP32):    {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"{'='*60}\n")


def get_model_summary(model, input_size=(1, 3, 512, 512)):
    """
    Get detailed model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
    
    Returns:
        Summary string
    """
    import torch
    from torch.utils.hooks import RemovableHandle
    
    summary = []
    hooks = []
    
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            if hasattr(output, 'shape'):
                output_shape = tuple(output.shape)
            elif isinstance(output, (list, tuple)):
                output_shape = [tuple(o.shape) if hasattr(o, 'shape') else 'N/A' for o in output]
            else:
                output_shape = 'N/A'
            
            params = sum([p.numel() for p in module.parameters()])
            
            summary.append({
                'layer': class_name,
                'output_shape': output_shape,
                'params': params
            })
        
        if not isinstance(module, torch.nn.Sequential) and \
           not isinstance(module, torch.nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))
    
    # Register hooks
    model.apply(register_hook)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        x = torch.zeros(input_size)
        model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"{'Layer':<30} {'Output Shape':<30} {'Params':<15}")
    print(f"{'='*80}")
    
    total_params = 0
    for layer in summary:
        print(f"{layer['layer']:<30} {str(layer['output_shape']):<30} {layer['params']:<15,}")
        total_params += layer['params']
    
    print(f"{'='*80}")
    print(f"Total params: {total_params:,}")
    print(f"{'='*80}\n")
    
    return summary


def test_model_creation():
    """Test model creation"""
    print("Testing model factory...")
    
    # Create dummy config
    class DummyConfig:
        class data:
            num_classes = 7
        class model:
            name = 'segformer'
            encoder = 'mit_b0'
            pretrained = True
    
    config = DummyConfig()
    
    # Test standard model
    print("\n[1/2] Testing standard SegFormer...")
    model = get_model(config)
    print_model_info(model)
    
    # Test uncertainty model
    print("\n[2/2] Testing uncertain SegFormer...")
    config.model.name = 'uncertain_segformer'
    uncertain_model = get_model(config)
    print_model_info(uncertain_model)
    
    print("✅ Model factory tests passed!")


if __name__ == '__main__':
    test_model_creation()