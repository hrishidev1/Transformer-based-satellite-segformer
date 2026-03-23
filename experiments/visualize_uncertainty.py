"""
Visualization script for uncertainty-aware predictions (IEEE Publication Ready)
"""
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- PRIORITY PATH FIX ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
# -------------------------

from utils.config import load_config
from datasets import get_dataset
from models.uncertainty_factory import get_uncertainty_model # FIXED IMPORT


# DeepGlobe color map
DEEPGLOBE_COLORS = np.array([
    [0, 255, 255],      # Urban - Cyan
    [255, 255, 0],      # Agriculture - Yellow
    [255, 0, 255],      # Rangeland - Magenta
    [0, 255, 0],        # Forest - Green
    [0, 0, 255],        # Water - Blue
    [255, 255, 255],    # Barren - White
    [0, 0, 0]           # Unknown - Black
]) / 255.0


def denormalize_image(image):
    """Denormalize image for visualization"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    image = image * std[:, None, None] + mean[:, None, None]
    image = np.clip(image, 0, 1)
    image = image.transpose(1, 2, 0)
    
    return image


def decode_mask(mask):
    """Convert class mask to RGB"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3))
    
    for cls in range(len(DEEPGLOBE_COLORS)):
        rgb[mask == cls] = DEEPGLOBE_COLORS[cls]
    
    return rgb


def visualize_sample_with_uncertainty(model, dataset, idx, device, save_path=None):
    """
    Visualize single sample with uncertainty overlay (IEEE format)
    """
    model.eval()
    
    # Get sample
    image, mask = dataset[idx]
    
    # Inference
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        output = model(image_batch, return_uncertainty=True)
        
        pred = output['pred'].squeeze().cpu().numpy()
        uncertainty = output['uncertainty'].squeeze().cpu().numpy()
    
    # Denormalize image
    img_vis = denormalize_image(image)
    
    # Decode masks
    gt_vis = decode_mask(mask.numpy())
    pred_vis = decode_mask(pred)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Image, GT, Prediction
    axes[0, 0].imshow(img_vis)
    axes[0, 0].set_title('Input Image', fontsize=16, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_vis)
    axes[0, 1].set_title('Ground Truth', fontsize=16, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_vis)
    axes[0, 2].set_title('Prediction', fontsize=16, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Uncertainty overlays
    vmax = uncertainty.max() if uncertainty.max() > 0 else 1.0
    
    # Uncertainty heatmap
    im1 = axes[1, 0].imshow(uncertainty, cmap='inferno', vmin=0, vmax=vmax)
    axes[1, 0].set_title('Uncertainty Map', fontsize=16, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Image with uncertainty overlay
    axes[1, 1].imshow(img_vis)
    im2 = axes[1, 1].imshow(uncertainty, cmap='inferno', alpha=0.6, vmin=0, vmax=vmax)
    axes[1, 1].set_title('Image + Uncertainty Overlay', fontsize=16, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Error map with uncertainty
    error_map = (pred != mask.numpy()).astype(float)
    axes[1, 2].imshow(img_vis)
    axes[1, 2].contour(error_map, levels=[0.5], colors='lime', linewidths=2) # Lime green contrasts better than red
    im3 = axes[1, 2].imshow(uncertainty, cmap='inferno', alpha=0.4, vmin=0, vmax=vmax)
    axes[1, 2].set_title('Errors (Lime) + Uncertainty', fontsize=16, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        # FIXED: Bumped DPI to 300 for IEEE publication standards
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Uncertainty Visualizations")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/uncertainty_viz', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Load dataset
    dataset = get_dataset(config, split='val')
    
    # Load checkpoint and extract architecture safely
    print(f"⏳ Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        arch_name = checkpoint['config']['model'].get('arch', 'SegFormer')
        encoder_name = checkpoint['config']['model'].get('encoder', config.model.encoder)
    else:
        arch_name = getattr(config.model, 'arch', 'SegFormer')
        encoder_name = config.model.encoder
        
    model = get_uncertainty_model(
        arch_name=arch_name,
        encoder_name=encoder_name,
        num_classes=config.data.num_classes,
        pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize multiple samples
    indices = np.random.choice(len(dataset), args.num_samples, replace=False)
    
    print(f"Generating {args.num_samples} IEEE-ready visualizations...")
    
    for i, idx in enumerate(indices):
        save_path = output_dir / f'sample_{i:03d}.png'
        visualize_sample_with_uncertainty(model, dataset, idx, device, save_path)
        print(f"  [{i+1}/{args.num_samples}] Saved: {save_path.name}")
    
    print(f"\n✅ Successfully saved {args.num_samples} visualizations to {output_dir}")


if __name__ == '__main__':
    main()