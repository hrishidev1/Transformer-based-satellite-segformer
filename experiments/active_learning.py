"""
Active Learning Experiment
Shows that uncertainty-based sampling outperforms random sampling
"""
import sys
import copy
sys.path.insert(0, '/content/drive/MyDrive/satellite-segmentation')

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.config import load_config
from datasets import get_dataset
from models.uncertainty_factory import get_uncertainty_model  # FIXED IMPORT
from losses.evidential_loss import get_evidential_loss
from metrics.segmentation import SegmentationMetrics


def compute_sample_uncertainty(model, dataloader, device):
    """Compute average uncertainty for each sample in dataset"""
    model.eval()
    sample_uncertainties = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Computing uncertainties"):
            images = images.to(device)
            output = model(images, return_uncertainty=True)
            uncertainty = output['uncertainty']
            
            # Average uncertainty per sample
            batch_uncertainties = uncertainty.view(images.size(0), -1).mean(dim=1)
            sample_uncertainties.extend(batch_uncertainties.cpu().numpy())
    
    return np.array(sample_uncertainties)


def train_with_subset(model, train_dataset, val_dataset, subset_indices, config, device):
    """Train model on a subset of data"""
    subset_dataset = Subset(train_dataset, subset_indices)
    
    train_loader = DataLoader(
        subset_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,  # Optimized from previous fixes
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = get_evidential_loss(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay
    )
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            output = model(images, return_uncertainty=False)
            
            # FIXED: Added current_epoch to prevent early KL penalty destabilization
            loss, _ = criterion(output, masks, current_epoch=epoch) 
            
            loss.backward()
            optimizer.step()
    
    model.eval()
    metrics = SegmentationMetrics(
        num_classes=config.data.num_classes,
        ignore_index=config.data.ignore_index if hasattr(config.data, 'ignore_index') else None
    )
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images, return_uncertainty=False)
            metrics.update(output['pred'], masks)
    
    results = metrics.compute(return_per_class=False)
    return results['miou']


def run_active_learning_experiment(config, percentages=[10, 25, 50, 100]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = get_dataset(config, split='train')
    val_dataset = get_dataset(config, split='val')
    total_samples = len(train_dataset)
    
    print("="*60)
    print("ACTIVE LEARNING EXPERIMENT")
    print("="*60 + "\n")
    
    results = {'random': {}, 'uncertainty': {}}
    
    # Pre-build a base model to deepcopy from, saving massive download/init time (Warning 2 Fix)
    arch_name = getattr(config.model, 'arch', 'SegFormer')
    base_model = get_uncertainty_model(
        arch_name=arch_name,
        encoder_name=config.model.encoder,
        num_classes=config.data.num_classes,
        pretrained=True
    ).to(device)
    
    for pct in percentages:
        num_samples = int(total_samples * pct / 100)
        print(f"\n{'='*60}")
        print(f"Training with {pct}% data ({num_samples} samples)")
        print(f"{'='*60}")
        
        # === Random Sampling ===
        print(f"\n[1/2] Random sampling...")
        random_indices = np.random.choice(total_samples, num_samples, replace=False)
        
        model_random = copy.deepcopy(base_model)
        miou_random = train_with_subset(model_random, train_dataset, val_dataset, random_indices, config, device)
        
        print(f"Random sampling mIoU: {miou_random:.4f}")
        results['random'][pct] = miou_random
        
        # === Uncertainty-Based Sampling ===
        if pct < 100:
            print(f"\n[2/2] Uncertainty-based sampling...")
            
            init_samples = min(num_samples, int(total_samples * 0.05))
            init_indices = np.random.choice(total_samples, init_samples, replace=False)
            
            model_init = copy.deepcopy(base_model)
            
            # Quick initial training to gauge uncertainties
            train_with_subset(model_init, train_dataset, val_dataset, init_indices, config, device)
            
            full_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4)
            uncertainties = compute_sample_uncertainty(model_init, full_loader, device)
            
            # Select samples with highest uncertainty
            uncertain_indices = np.argsort(uncertainties)[-num_samples:]
            
            model_uncertain = copy.deepcopy(base_model)
            miou_uncertain = train_with_subset(model_uncertain, train_dataset, val_dataset, uncertain_indices, config, device)
            
            print(f"Uncertainty-based mIoU: {miou_uncertain:.4f}")
            print(f"Improvement: +{(miou_uncertain - miou_random) / miou_random * 100:.1f}%")
            results['uncertainty'][pct] = miou_uncertain
        else:
            results['uncertainty'][pct] = miou_random
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'% Data':<10} {'Random':>12} {'Uncertainty':>15} {'Improvement':>15}")
    print("-"*60)
    
    for pct in percentages:
        random_val = results['random'][pct]
        uncertain_val = results['uncertainty'][pct]
        improvement = (uncertain_val - random_val) / random_val * 100
        print(f"{pct:<10} {random_val:>12.4f} {uncertain_val:>15.4f} {improvement:>14.1f}%")
    
    print("="*60 + "\n")
    return results


def main():
    # FIXED: Load the correct, modern configuration file
    config = load_config('configs/uncertain_segformer.yaml') 
    
    # Optionally override to a smaller encoder just for speed in this experiment
    config.model.encoder = 'mit_b0' 
    
    results = run_active_learning_experiment(config, percentages=[10, 25, 50, 100])
    
    import json
    import os
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/active_learning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Active learning experiment complete! Results saved to outputs/active_learning_results.json")

if __name__ == '__main__':
    main()