"""
Training script for Uncertainty-Aware Segmentation
(Optimized for Production & Dynamic Resolution)
"""
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

# --- PRIORITY PATH FIX ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------

from utils.config import load_config, get_args, merge_args_with_config
from datasets import get_dataset
from models.uncertainty_factory import get_uncertainty_model 
from losses.evidential_loss import get_evidential_loss
from metrics.segmentation import SegmentationMetrics
from metrics.calibration import CalibrationMetrics

class UncertainTrainer:
    """Trainer for uncertainty-aware models"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[Trainer] Using device: {self.device}")
        self.set_seed(config.seed)
        
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_loader, self.val_loader = self.build_dataloaders()
        
        arch_name = getattr(config.model, 'arch', 'SegFormer') 
        encoder_name = config.model.encoder
        print(f"[Trainer] Building Model -> Arch: {arch_name}, Encoder: {encoder_name}")
        
        self.model = get_uncertainty_model(
            arch_name=arch_name,
            encoder_name=encoder_name,
            num_classes=config.data.num_classes,
            pretrained=config.model.pretrained
        ).to(self.device)
        
        self.criterion = get_evidential_loss(config)
        if hasattr(self.criterion, 'to'):
            self.criterion = self.criterion.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=config.scheduler.min_lr
        )
        
        class_names = self.train_loader.dataset.get_class_names()
        
        # --- FIXED: Separate metrics for Train and Val to prevent state bleed ---
        ignore_idx = config.data.ignore_index if hasattr(config.data, 'ignore_index') else None
        
        self.train_seg_metrics = SegmentationMetrics(
            config.data.num_classes, ignore_index=ignore_idx, class_names=class_names
        )
        self.val_seg_metrics = SegmentationMetrics(
            config.data.num_classes, ignore_index=ignore_idx, class_names=class_names
        )
        self.val_cal_metrics = CalibrationMetrics(
            num_bins=15, num_classes=config.data.num_classes, ignore_index=ignore_idx
        )
        
        self.current_epoch = 0
        self.best_miou = 0.0
        self.best_ece = float('inf')
        
        if hasattr(config, 'resume') and config.resume:
            self.resume_training()
    
    def set_seed(self, seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = False  
            torch.backends.cudnn.benchmark = True       
    
    def build_dataloaders(self):
        train_dataset = get_dataset(self.config, split='train')
        val_dataset = get_dataset(self.config, split='val')
        
        # --- FIXED: Worker init function to prevent identical random crops ---
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )
        return train_loader, val_loader
    
    def resume_training(self):
        ckpt_path = self.checkpoint_dir / 'last.pth'
        if hasattr(self.config, 'checkpoint_path') and self.config.checkpoint_path:
             ckpt_path = Path(self.config.checkpoint_path)

        if ckpt_path.exists():
            print(f"🔄 Resuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])        
            self.current_epoch = checkpoint['epoch'] + 1
            self.best_miou = checkpoint.get('best_miou', 0.0)
            self.best_ece = checkpoint.get('best_ece', float('inf'))
            print(f"✅ Resumed at Epoch {self.current_epoch}")
        else:
            print(f"⚠️ Checkpoint not found at {ckpt_path}. Starting from scratch.")        

    def train_epoch(self):
        self.model.train()
        self.train_seg_metrics.reset()
        epoch_losses = []
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.training.epochs}")
        
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            
            # FIXED: return_uncertainty=False saves massive compute overhead
            output = self.model(images, return_uncertainty=False)
            loss, loss_dict = self.criterion(output, masks, current_epoch=self.current_epoch)
            
            loss.backward()
            
            if self.config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            self.train_seg_metrics.update(output['pred'], masks)
            pbar.set_postfix({'loss': np.mean(epoch_losses[-10:])})
        
        results = self.train_seg_metrics.compute(return_per_class=False)
        results['loss'] = np.mean(epoch_losses)
        return results
    
    def validate(self):
        self.model.eval()
        self.val_seg_metrics.reset()
        self.val_cal_metrics.reset()
        val_losses = []
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                output = self.model(images, return_uncertainty=True)
                loss, _ = self.criterion(output, masks, current_epoch=self.current_epoch)
                
                val_losses.append(loss.item())
                self.val_seg_metrics.update(output['pred'], masks)
                self.val_cal_metrics.update(output['prob'], output['pred'], masks, output['uncertainty'])
        
        seg_results = self.val_seg_metrics.compute(return_per_class=True)
        seg_results['loss'] = np.mean(val_losses)
        cal_results = self.val_cal_metrics.compute()
        return {**seg_results, **cal_results}
    
    def save_checkpoint(self, is_best=False, is_best_cal=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'best_ece': self.best_ece,
            'config': self.config.to_dict()
        }
        torch.save(checkpoint, self.checkpoint_dir / 'last.pth')
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_miou.pth')
            print(f"💾 Saved best mIoU checkpoint: {self.best_miou:.4f}")
        if is_best_cal:
            torch.save(checkpoint, self.checkpoint_dir / 'best_ece.pth')

    def train(self):
        print(f"\nTraining Model: {self.config.model.get('arch', 'SegFormer')} | Encoder: {self.config.model.encoder}")
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            train_results = self.train_epoch()
            
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")
            print(f"Train Loss: {train_results['loss']:.4f} | Train mIoU: {train_results['miou']:.4f}")
            
            # --- FIXED: Better logging for skipped validation epochs ---
            if (epoch + 1) % 5 == 0 or epoch == self.config.training.epochs - 1:
                val_results = self.validate()
                print(f"Val Loss:   {val_results['loss']:.4f} | Val mIoU:   {val_results['miou']:.4f}")
                print(f"Val ECE:    {val_results['ece']:.4f} | Uncertainty-Error Corr: {val_results.get('uncertainty_error_corr', 0):.4f}")
                
                is_best = val_results['miou'] > self.best_miou
                is_best_cal = val_results['ece'] < self.best_ece
                
                if is_best: self.best_miou = val_results['miou']
                if is_best_cal: self.best_ece = val_results['ece']
                
                self.save_checkpoint(is_best=is_best, is_best_cal=is_best_cal)
            else:
                print(f"Val Loss:   [skipped] | Val mIoU:   [skipped]")
                print(f"Val ECE:    [skipped] | Uncertainty-Error Corr: [skipped]")
                self.save_checkpoint(is_best=False, is_best_cal=False)
                
            self.scheduler.step()

def main():
    args = get_args()
    config = load_config(args.config)
    config = merge_args_with_config(config, args)
    if args.resume: config.resume = True
    if args.checkpoint: config.checkpoint_path = args.checkpoint
    trainer = UncertainTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()