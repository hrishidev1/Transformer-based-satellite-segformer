"""
HITL Script 3 of 4 — Fine-Tune on Human Labels
================================================

WHAT THIS DOES:
  This is the "learning" step. You've labelled the images the model was most
  confused about. Now we teach the model from your corrections.

  The key engineering challenge: if you just train on the 50 new images,
  the model will FORGET everything it learned from the original 642 images.
  This is called "catastrophic forgetting" and it's a real problem.

  We solve it with two techniques:

  TECHNIQUE 1 — REPLAY BUFFER:
    We don't train only on your 50 new images.
    We mix them with a random sample of the original training data.
    Ratio: for every 1 new image, include 3 original images.
    This means the model practices both old AND new knowledge every epoch.

  TECHNIQUE 2 — LOW LEARNING RATE + ENCODER FREEZE:
    We use a much lower learning rate than initial training (1/10th).
    We FREEZE the encoder (MiT-b3 backbone) for the first few epochs
    and only train the evidential head. This forces the model to adapt
    gently, not overwrite its core feature extraction.
    After a few warmup epochs, we unfreeze everything and train end-to-end.

  TECHNIQUE 3 — ELASTIC WEIGHT CONSOLIDATION (optional, --use_ewc):
    Mathematical protection against forgetting. Adds a penalty to the loss
    that punishes large changes to weights that were important for the
    original dataset. This is the state-of-the-art continual learning method.

HOW TO RUN:
  python hitl_pipeline/03_finetune_hitl.py \\
      --config       configs/uncertain_segformer.yaml \\
      --checkpoint   /path/to/best_miou.pth \\
      --deepglobe_dir /path/to/DeepGlobe \\
      --round_id     01 \\
      --ft_epochs    20 \\
      --replay_ratio 3

ARGUMENTS:
  --config          YAML config (same as training)
  --checkpoint      Previous best checkpoint to fine-tune from
  --deepglobe_dir   Root of DeepGlobe dataset
  --round_id        Which HITL round this is (01, 02, 03...)
  --ft_epochs       Fine-tuning epochs (default: 20)
  --replay_ratio    Old samples per 1 new sample (default: 3)
  --freeze_epochs   Epochs to freeze encoder before unfreezing (default: 5)
  --ft_lr           Fine-tuning learning rate (default: 6e-6, 1/10 of training)
  --use_ewc         Enable Elastic Weight Consolidation (slower but safer)
  --output_dir      Where to save the fine-tuned checkpoint

OUTPUT:
  checkpoint/hitl_round_XX_best.pth    ← best fine-tuned model
  checkpoint/hitl_round_XX_last.pth    ← last epoch model
  hitl_round_XX_training_log.json      ← loss and mIoU per epoch
"""

import sys
import os
import json
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Path fix ---
FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ----------------

from utils.config import load_config
from models.uncertainty_factory import get_uncertainty_model
from losses.evidential_loss import get_evidential_loss
from metrics.segmentation import SegmentationMetrics
from datasets import get_dataset


# ============================================================================
# DEEPGLOBE COLOURMAP
# ============================================================================

DEEPGLOBE_COLORMAP = {
    (0,   255, 255): 0,
    (255, 255,   0): 1,
    (255,   0, 255): 2,
    (0,   255,   0): 3,
    (0,     0, 255): 4,
    (255, 255, 255): 5,
    (0,     0,   0): 6,
}

CLASS_NAMES = [
    'Urban', 'Agriculture', 'Rangeland',
    'Forest', 'Water', 'Barren', 'Unknown'
]


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune model on new HITL labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--config',        required=True)
    p.add_argument('--checkpoint',    required=True,
                   help='Best checkpoint from previous round')
    p.add_argument('--deepglobe_dir', required=True,
                   help='Root of DeepGlobe dataset')
    p.add_argument('--round_id',      default='01',
                   help='HITL round number (01, 02, 03...)')
    p.add_argument('--ft_epochs',     type=int, default=20,
                   help='Fine-tuning epochs')
    p.add_argument('--replay_ratio',  type=int, default=3,
                   help='Ratio of old samples to new samples per batch')
    p.add_argument('--freeze_epochs', type=int, default=5,
                   help='Epochs to freeze encoder backbone')
    p.add_argument('--ft_lr',         type=float, default=6e-6,
                   help='Fine-tuning learning rate (1/10 of original)')
    p.add_argument('--use_ewc',       action='store_true',
                   help='Use Elastic Weight Consolidation to prevent forgetting')
    p.add_argument('--ewc_lambda',    type=float, default=1000.0,
                   help='EWC regularisation strength')
    p.add_argument('--output_dir',    default='checkpoints/hitl',
                   help='Where to save fine-tuned checkpoints')
    return p.parse_args()


# ============================================================================
# DATASET — NEW HITL LABELS ONLY
# ============================================================================

class HITLDataset(Dataset):
    """
    Dataset for the new human-labelled images added this round.
    Reads directly from DeepGlobe/Train/ but only the hitl_rXX_ prefixed files.
    """

    def __init__(self, deepglobe_dir: Path, round_id: str,
                 image_size: int, ignore_index: int = 6):
        self.img_dir   = deepglobe_dir / 'Train' / 'images'
        self.mask_dir  = deepglobe_dir / 'Train' / 'masks'
        self.ignore_index = ignore_index
        self.num_classes  = 7

        prefix = f"hitl_r{round_id}_"
        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.startswith(prefix) and f.endswith('_sat.jpg')
        ])

        if len(self.images) == 0:
            raise ValueError(
                f"No HITL images found with prefix '{prefix}' in {self.img_dir}\n"
                f"Did you run 02_prepare_new_labels.py first?"
            )

        self.transform = A.Compose([
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        print(f"  [HITLDataset] Round {round_id}: {len(self.images)} new images")

    def __len__(self):
        return len(self.images)

    def _encode_mask(self, mask_rgb: np.ndarray) -> np.ndarray:
        h, w, _ = mask_rgb.shape
        mask     = np.zeros((h, w), dtype=np.uint8)
        for rgb, cls_id in DEEPGLOBE_COLORMAP.items():
            matches      = np.all(mask_rgb == np.array(rgb), axis=-1)
            mask[matches] = cls_id
        return mask

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')

        img  = np.array(Image.open(self.img_dir / img_name).convert('RGB'))
        mask_rgb = np.array(Image.open(self.mask_dir / mask_name).convert('RGB'))
        mask = self._encode_mask(mask_rgb)

        aug  = self.transform(image=img, mask=mask)
        return aug['image'], aug['mask'].long()

    def get_class_names(self):
        return CLASS_NAMES


# ============================================================================
# REPLAY BUFFER DATASET
# ============================================================================

class ReplayMixDataset(Dataset):
    """
    Interleaves new HITL samples with old original samples.

    For every 1 new sample:
      - We add `replay_ratio` randomly selected original samples.

    This ensures the model sees enough old data to not forget,
    while still learning heavily from the new hard examples.
    """

    def __init__(self, hitl_dataset: HITLDataset,
                 original_dataset, replay_ratio: int):
        self.hitl      = hitl_dataset
        self.original  = original_dataset
        self.ratio     = replay_ratio

        n_new   = len(hitl_dataset)
        n_old   = len(original_dataset)
        n_replay = min(n_new * replay_ratio, n_old)

        # Randomly sample original indices for this epoch's replay buffer
        self.replay_indices = np.random.choice(n_old, n_replay, replace=False).tolist()

        total = n_new + n_replay
        print(f"  [ReplayMix] {n_new} new + {n_replay} replay = {total} total samples")

    def __len__(self):
        return len(self.hitl) + len(self.replay_indices)

    def __getitem__(self, idx):
        if idx < len(self.hitl):
            return self.hitl[idx]
        else:
            old_idx = self.replay_indices[idx - len(self.hitl)]
            return self.original[old_idx]


# ============================================================================
# ELASTIC WEIGHT CONSOLIDATION (EWC)
# ============================================================================

class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Computes the Fisher Information Matrix on the original training data.
    Adds a quadratic penalty to the loss for parameters that were important
    for the original task.

    Loss_EWC = Loss_task + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

    Where:
      F_i    = Fisher information (importance) of parameter i
      theta* = weights from the old checkpoint
      theta  = current weights
    """

    def __init__(self, model, dataloader, device, criterion, num_samples=200):
        print(f"  [EWC] Computing Fisher Information Matrix "
              f"({num_samples} samples)...")
        self.device   = device
        self.params   = {n: p.clone().detach()
                         for n, p in model.named_parameters()
                         if p.requires_grad}
        self.fisher   = self._compute_fisher(
            model, dataloader, criterion, num_samples
        )
        print(f"  [EWC] Fisher computed for {len(self.fisher)} parameter tensors")

    def _compute_fisher(self, model, dataloader, criterion, num_samples):
        fisher = {n: torch.zeros_like(p)
                  for n, p in model.named_parameters() if p.requires_grad}

        model.eval()
        count = 0

        for images, masks in dataloader:
            if count >= num_samples:
                break

            images = images.to(self.device)
            masks  = masks.to(self.device)

            model.zero_grad()
            output = model(images, return_uncertainty=True)
            loss, _ = criterion(output, masks, current_epoch=999)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

            count += images.size(0)

        # Normalise
        for n in fisher:
            fisher[n] /= max(count, 1)

        model.zero_grad()
        return fisher

    def penalty(self, model):
        """Compute EWC penalty term to add to the loss."""
        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss


# ============================================================================
# ENCODER FREEZE / UNFREEZE
# ============================================================================

def freeze_encoder(model):
    """
    Freeze the MiT encoder backbone.
    Only the decoder/evidential head weights will be updated.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if 'encoder' in name or 'backbone' in name:
            param.requires_grad = False
            frozen += 1
    print(f"  ✓ Frozen {frozen} encoder parameter tensors")


def unfreeze_all(model):
    """Unfreeze all parameters for full fine-tuning."""
    unfrozen = 0
    for param in model.parameters():
        param.requires_grad = True
        unfrozen += 1
    print(f"  ✓ Unfrozen all {unfrozen} parameter tensors")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def fine_tune(model, train_loader, val_loader, config, device,
              ft_epochs, freeze_epochs, ft_lr, ewc=None, ewc_lambda=1000.0,
              round_id='01', output_dir=Path('checkpoints/hitl')):
    """
    Fine-tune the model with:
      - Encoder freeze for first `freeze_epochs`
      - Low learning rate
      - EWC penalty (optional)
      - Best checkpoint saving
    """
    criterion = get_evidential_loss(config)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ft_lr,
        weight_decay=config.optimizer.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ft_epochs,
        eta_min=ft_lr / 10
    )

    seg_metrics = SegmentationMetrics(
        num_classes  = config.data.num_classes,
        ignore_index = getattr(config.data, 'ignore_index', None)
    )

    best_miou  = 0.0
    best_state = None
    log        = []

    # Start with encoder frozen
    freeze_encoder(model)
    print(f"\n  Starting fine-tuning: {ft_epochs} epochs | "
          f"Encoder frozen for first {freeze_epochs} epochs\n")

    for epoch in range(ft_epochs):

        # Unfreeze encoder after warmup
        if epoch == freeze_epochs:
            print(f"\n  Epoch {epoch+1}: Unfreezing encoder for full fine-tuning")
            unfreeze_all(model)
            # Rebuild optimizer with all parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=ft_lr / 3,    # Even lower LR for full fine-tuning
                weight_decay=config.optimizer.weight_decay,
                betas=(0.9, 0.999), eps=1e-8
            )

        # ---- Train ----
        model.train()
        epoch_losses = []
        ewc_losses   = []

        pbar = tqdm(train_loader,
                    desc=f"  [R{round_id}] Epoch {epoch+1:>2}/{ft_epochs}",
                    leave=False)

        for images, masks in pbar:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            output = model(images, return_uncertainty=True)

            # Main evidential loss
            loss, _ = criterion(output, masks, current_epoch=epoch)

            # EWC penalty (prevents forgetting)
            if ewc is not None:
                ewc_loss = ewc_lambda * ewc.penalty(model)
                loss     = loss + ewc_loss
                ewc_losses.append(ewc_loss.item())

            loss.backward()

            if config.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.grad_clip
                )

            optimizer.step()
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{np.mean(epoch_losses[-10:]):.4f}"})

        scheduler.step()

        # ---- Validate ----
        model.eval()
        seg_metrics.reset()

        with torch.no_grad():
            for images, masks in val_loader:
                output = model(images.to(device), return_uncertainty=False)
                seg_metrics.update(output['pred'], masks.to(device))

        results   = seg_metrics.compute(return_per_class=True)
        val_miou  = results['miou']
        val_acc   = results['accuracy']

        # Log
        log_entry = {
            'epoch':    epoch + 1,
            'loss':     float(np.mean(epoch_losses)),
            'val_miou': float(val_miou),
            'val_acc':  float(val_acc),
            'frozen':   epoch < freeze_epochs,
        }
        if ewc_losses:
            log_entry['ewc_loss'] = float(np.mean(ewc_losses))
        log.append(log_entry)

        # Save best
        if val_miou > best_miou:
            best_miou  = val_miou
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {'model_state_dict': best_state,
                 'epoch': epoch + 1,
                 'miou':  best_miou,
                 'round': round_id},
                output_dir / f'hitl_round_{round_id}_best.pth'
            )
            marker = " ← best"
        else:
            marker = ""

        frozen_str = " [encoder frozen]" if epoch < freeze_epochs else ""
        print(f"  [R{round_id}] Epoch {epoch+1:>2}/{ft_epochs} | "
              f"Loss: {np.mean(epoch_losses):.4f} | "
              f"mIoU: {val_miou:.4f} | "
              f"Acc: {val_acc:.4f}"
              f"{frozen_str}{marker}")

    # Save final checkpoint
    torch.save(
        {'model_state_dict': model.state_dict(),
         'epoch': ft_epochs,
         'miou':  val_miou,
         'round': round_id},
        output_dir / f'hitl_round_{round_id}_last.pth'
    )

    return best_miou, log


# ============================================================================
# MAIN
# ============================================================================

def main():
    args      = parse_args()
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config    = load_config(args.config)
    dg_dir    = Path(args.deepglobe_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fix seed
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    print(f"\n{'='*60}")
    print(f"  HITL STEP 3 — FINE-TUNING ON NEW LABELS")
    print(f"  Device:         {device}")
    print(f"  Round:          {args.round_id}")
    print(f"  Fine-tune LR:   {args.ft_lr}")
    print(f"  Replay ratio:   1 new : {args.replay_ratio} old")
    print(f"  Freeze epochs:  {args.freeze_epochs}")
    print(f"  EWC:            {args.use_ewc}")
    print(f"{'='*60}\n")

    # ---- Load model ----
    print("  Loading model...")
    model = get_uncertainty_model(
        arch_name    = getattr(config.model, 'arch', 'SegFormer'),
        encoder_name = config.model.encoder,
        num_classes  = config.data.num_classes,
        pretrained   = False
    ).to(device)

    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    prev_miou = ckpt.get('miou', 0.0)
    print(f"  ✓ Loaded checkpoint (previous mIoU: {prev_miou:.4f})")

    # ---- Build datasets ----
    print("\n  Building datasets...")
    hitl_dataset     = HITLDataset(dg_dir, args.round_id,
                                    config.data.image_size,
                                    getattr(config.data, 'ignore_index', 6))
    original_dataset = get_dataset(config, split='train')
    val_dataset      = get_dataset(config, split='val')

    # Mix new labels with replay buffer
    mixed_dataset = ReplayMixDataset(
        hitl_dataset, original_dataset, args.replay_ratio
    )

    train_loader = DataLoader(
        mixed_dataset,
        batch_size  = config.training.batch_size,
        shuffle     = True,
        num_workers = config.training.num_workers,
        pin_memory  = True,
        drop_last   = True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.training.batch_size,
        shuffle     = False,
        num_workers = config.training.num_workers,
        pin_memory  = True
    )

    # ---- EWC (optional) ----
    ewc = None
    if args.use_ewc:
        print("\n  Computing EWC Fisher Information (this takes a few minutes)...")
        criterion_for_ewc = get_evidential_loss(config)
        # Use a small subset of original data for Fisher computation
        fisher_size    = min(200, len(original_dataset))
        fisher_indices = np.random.choice(len(original_dataset),
                                           fisher_size, replace=False)
        fisher_loader  = DataLoader(
            Subset(original_dataset, fisher_indices),
            batch_size=4, shuffle=False, num_workers=2
        )
        ewc = EWC(model, fisher_loader, device, criterion_for_ewc,
                  num_samples=fisher_size)

    # ---- Fine-tune ----
    print(f"\n  Fine-tuning for {args.ft_epochs} epochs...")
    best_miou, training_log = fine_tune(
        model         = model,
        train_loader  = train_loader,
        val_loader    = val_loader,
        config        = config,
        device        = device,
        ft_epochs     = args.ft_epochs,
        freeze_epochs = args.freeze_epochs,
        ft_lr         = args.ft_lr,
        ewc           = ewc,
        ewc_lambda    = args.ewc_lambda,
        round_id      = args.round_id,
        output_dir    = out_dir
    )

    # ---- Save training log ----
    log_path = out_dir / f'hitl_round_{args.round_id}_training_log.json'
    with open(log_path, 'w') as f:
        json.dump({
            'round':          args.round_id,
            'prev_miou':      prev_miou,
            'best_miou':      best_miou,
            'improvement':    best_miou - prev_miou,
            'ft_epochs':      args.ft_epochs,
            'ft_lr':          args.ft_lr,
            'replay_ratio':   args.replay_ratio,
            'use_ewc':        args.use_ewc,
            'training_log':   training_log,
        }, f, indent=2)

    # ---- Summary ----
    improvement = best_miou - prev_miou
    print(f"\n{'='*60}")
    print(f"  FINE-TUNING COMPLETE — Round {args.round_id}")
    print(f"  Previous mIoU:  {prev_miou:.4f}")
    print(f"  New best mIoU:  {best_miou:.4f}")
    print(f"  Improvement:    {improvement:+.4f} "
          f"({'↑ better' if improvement > 0 else '↓ worse — check replay ratio'})")
    print(f"\n  Best checkpoint: "
          f"{out_dir}/hitl_round_{args.round_id}_best.pth")
    print(f"  Training log:    {log_path}")
    print(f"\n  NEXT STEP:")
    print(f"  python hitl_pipeline/04_evaluate_improvement.py \\")
    print(f"      --config {args.config} \\")
    print(f"      --checkpoint_before {args.checkpoint} \\")
    print(f"      --checkpoint_after  "
          f"{out_dir}/hitl_round_{args.round_id}_best.pth \\")
    print(f"      --round_id {args.round_id}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()