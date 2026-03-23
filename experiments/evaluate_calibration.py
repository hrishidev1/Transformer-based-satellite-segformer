import sys
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# --- Path Fix (CRITICAL: Changed to insert(0) to prevent shadowing) ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ----------------

from utils.config import load_config
from datasets import get_dataset
from metrics.calibration import CalibrationMetrics
from metrics.segmentation import SegmentationMetrics

# --- FIXED IMPORTS ---
from models.uncertainty_factory import get_uncertainty_model
from legacy.factory import get_model as get_legacy_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Uncertainty Model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to uncertain model checkpoint')
    parser.add_argument('--baseline_checkpoint', type=str, default=None, help='(Optional) Path to baseline checkpoint')
    return parser.parse_args()

def evaluate_model(model, dataloader, device, is_uncertain=True):
    model.eval()
    
    seg_metrics = SegmentationMetrics(
        num_classes=dataloader.dataset.num_classes,
        ignore_index=dataloader.dataset.ignore_index if hasattr(dataloader.dataset, 'ignore_index') else None
    )
    
    cal_metrics = CalibrationMetrics(
        num_bins=15,
        num_classes=dataloader.dataset.num_classes,
        ignore_index=dataloader.dataset.ignore_index if hasattr(dataloader.dataset, 'ignore_index') else None
    )
    
    print(f"Evaluating (Uncertainty={is_uncertain})...")
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            if is_uncertain:
                output = model(images, return_uncertainty=True)
                probs = output['prob']
                preds = output['pred']
                uncertainty = output['uncertainty']
            else:
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                uncertainty = None
            
            seg_metrics.update(preds, masks)
            cal_metrics.update(probs, preds, masks, uncertainty)
    
    seg_results = seg_metrics.compute(return_per_class=True)
    cal_results = cal_metrics.compute()
    
    return seg_results, cal_results, cal_metrics

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Config
    config = load_config(args.config)
    
    # Load Data
    val_dataset = get_dataset(config, split='val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,  # Sped up from 2 to 4
        pin_memory=True
    )
    
    # === 1. Evaluate Uncertain Model (Target) ===
    print(f"\n[1/1] Loading Uncertain Model: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # --- INFO WARNING FIX: Extract architecture and encoder from saved checkpoint config ---
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        arch_name = checkpoint['config']['model'].get('arch', 'SegFormer')
        encoder_name = checkpoint['config']['model'].get('encoder', config.model.encoder)
        print(f"Loaded architecture specs from checkpoint -> Arch: {arch_name}, Encoder: {encoder_name}")
    else:
        arch_name = getattr(config.model, 'arch', 'SegFormer')
        encoder_name = config.model.encoder
        print(f"Using architecture specs from YAML -> Arch: {arch_name}, Encoder: {encoder_name}")
        
    uncertain_model = get_uncertainty_model(
        arch_name=arch_name,
        encoder_name=encoder_name,
        num_classes=config.data.num_classes,
        pretrained=False
    ).to(device)
    
    # Handle state dict keys
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    uncertain_model.load_state_dict(state_dict)
    
    seg_res, cal_res, cal_metrics_obj = evaluate_model(uncertain_model, val_loader, device, is_uncertain=True)
    
    print("\n" + "="*40)
    print("RESULTS: Uncertain Model")
    print("="*40)
    print(f"mIoU:           {seg_res['miou']:.4f}")
    print(f"ECE:            {cal_res['ece']:.4f}")
    print(f"MCE:            {cal_res['mce']:.4f}")
    print(f"Brier Score:    {cal_res['brier']:.4f}")
    print(f"Unc-Err Corr:   {cal_res.get('uncertainty_error_corr', 0):.4f}")
    print("="*40)
    
    # Save Plots
    os.makedirs('outputs', exist_ok=True)
    cal_metrics_obj.plot_reliability_diagram()
    plt.suptitle("Reliability Diagram (Uncertain Model)", fontsize=16, y = 1.05)
    plt.savefig('outputs/reliability_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ Saved reliability diagram to outputs/reliability_diagram.png")
    
    # === 2. (Optional) Baseline Comparison ===
    if args.baseline_checkpoint and os.path.exists(args.baseline_checkpoint):
        print(f"\n[Optional] Comparing with Baseline: {args.baseline_checkpoint}")
        baseline_model = get_legacy_model(config).to(device)
        base_ckpt = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
        baseline_model.load_state_dict(base_ckpt['model_state_dict'] if 'model_state_dict' in base_ckpt else base_ckpt)
        
        base_seg, base_cal, _ = evaluate_model(baseline_model, val_loader, device, is_uncertain=False)
        
        print(f"Baseline mIoU: {base_seg['miou']:.4f}")
        print(f"Baseline ECE:  {base_cal['ece']:.4f}")

if __name__ == '__main__':
    main()