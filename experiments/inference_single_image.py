import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- PATH FIX ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
# ----------------

from utils.config import load_config
from models.uncertainty_factory import get_uncertainty_model # FIXED IMPORT

# DeepGlobe Color Map
COLORS = np.array([
    [0, 255, 255],      # Urban (Cyan)
    [255, 255, 0],      # Agriculture (Yellow)
    [255, 0, 255],      # Rangeland (Magenta)
    [0, 255, 0],        # Forest (Green)
    [0, 0, 255],        # Water (Blue)
    [255, 255, 255],    # Barren (White)
    [0, 0, 0]           # Unknown (Black)
]) / 255.0

def preprocess_image(image_path, image_size):
    # Load image (RGB)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to dynamic size (Model input size)
    original_shape = image.shape[:2]
    resize_transform = A.Compose([
        A.Resize(image_size, image_size), # FIXED: No more hardcoded 512
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    transformed = resize_transform(image=image)
    tensor = transformed['image'].unsqueeze(0) # Add batch dim: [1, 3, H, W]
    
    return tensor, image, original_shape

def decode_prediction(pred_mask):
    h, w = pred_mask.shape
    rgb = np.zeros((h, w, 3))
    for cls in range(len(COLORS)):
        rgb[pred_mask == cls] = COLORS[cls]
    return rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to your test image (jpg/png)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/uncertain_deepglobe.yaml')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    image_size = config.data.image_size # Dynamically grab image size
    
    # 1. Load Model
    print(f"⏳ Loading model from {args.checkpoint}...")
    
    # Safely extract architecture from checkpoint (like we did in calibration)
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
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # 2. Process Image
    input_tensor, original_img, orig_shape = preprocess_image(args.image, image_size)
    input_tensor = input_tensor.to(device)
    
    # 3. Inference
    with torch.no_grad():
        output = model(input_tensor, return_uncertainty=True)
        pred_mask = output['pred'].squeeze().cpu().numpy()
        uncertainty = output['uncertainty'].squeeze().cpu().numpy()
        
    # 4. Visualization
    # Resize raw image back to target model size for display consistency
    display_img = cv2.resize(original_img, (image_size, image_size))
    display_pred = decode_prediction(pred_mask)
    
    # Create Plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ax[0].imshow(display_img)
    ax[0].set_title("Input Image")
    ax[0].axis('off')
    
    ax[1].imshow(display_pred)
    ax[1].set_title("Prediction")
    ax[1].axis('off')
    
    # FIXED: Dynamically scale colorbar to highlight variations in uncertainty
    vmax = uncertainty.max() if uncertainty.max() > 0 else 1.0
    im = ax[2].imshow(uncertainty, cmap='inferno', vmin=0, vmax=vmax)
    ax[2].set_title("Uncertainty Map")
    ax[2].axis('off')
    
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    plt.savefig('ood_test_result.png', bbox_inches='tight', dpi=150)
    print("✅ Result saved to 'ood_test_result.png'")

if __name__ == '__main__':
    main()