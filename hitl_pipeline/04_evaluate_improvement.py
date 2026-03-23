"""
HITL Script 4 of 4 — Evaluate Improvement
==========================================

WHAT THIS DOES:
  After each HITL round you want to know:
    1. Did the model actually get better overall?
    2. Which specific classes improved?
    3. Did it get better on the TYPES of images it was confused about?
    4. Did it stay calibrated (ECE)?

  This script runs both the old checkpoint and the new checkpoint on the
  full validation set and produces a side-by-side comparison with:
    - Overall mIoU, accuracy, ECE before vs after
    - Per-class IoU table (which classes improved most)
    - Uncertainty maps before vs after (did the model become more confident
      where it was previously confused?)
    - Sample visualisations of specific improvements

HOW TO RUN:
  python hitl_pipeline/04_evaluate_improvement.py \\
      --config             configs/uncertain_segformer.yaml \\
      --checkpoint_before  /path/to/old_best_miou.pth \\
      --checkpoint_after   checkpoints/hitl/hitl_round_01_best.pth \\
      --round_id           01 \\
      --output_dir         outputs/hitl_evaluation

ARGUMENTS:
  --config              YAML config
  --checkpoint_before   Checkpoint BEFORE this HITL round
  --checkpoint_after    Checkpoint AFTER this HITL round
  --round_id            Which round number this is
  --output_dir          Where to save evaluation results and plots
  --num_viz             Number of sample visualisations to generate (default: 6)
"""

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Path fix ---
FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ----------------

from utils.config import load_config
from models.uncertainty_factory import get_uncertainty_model
from metrics.segmentation import SegmentationMetrics
from metrics.calibration import CalibrationMetrics
from datasets import get_dataset


# ============================================================================
# COLOUR MAP
# ============================================================================

CLASS_COLORS = np.array([
    [0,   255, 255],
    [255, 255,   0],
    [255,   0, 255],
    [0,   255,   0],
    [0,     0, 255],
    [255, 255, 255],
    [0,     0,   0],
], dtype=np.uint8)

CLASS_NAMES = [
    'Urban', 'Agriculture', 'Rangeland',
    'Forest', 'Water', 'Barren', 'Unknown'
]


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare model performance before and after HITL round",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--config',             required=True)
    p.add_argument('--checkpoint_before',  required=True,
                   help='Checkpoint before this HITL round')
    p.add_argument('--checkpoint_after',   required=True,
                   help='Checkpoint after this HITL round')
    p.add_argument('--round_id',           default='01')
    p.add_argument('--output_dir',
                   default='outputs/hitl_evaluation')
    p.add_argument('--num_viz',            type=int, default=6,
                   help='Number of sample visualisations to generate')
    return p.parse_args()


# ============================================================================
# HELPERS
# ============================================================================

def load_model(config, checkpoint_path, device):
    model = get_uncertainty_model(
        arch_name    = getattr(config.model, 'arch', 'SegFormer'),
        encoder_name = config.model.encoder,
        num_classes  = config.data.num_classes,
        pretrained   = False
    ).to(device)

    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def decode_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(CLASS_COLORS):
        rgb[mask == i] = color
    return rgb


def denormalize(tensor) -> np.ndarray:
    """Convert normalised tensor back to displayable RGB image."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy()
    img  = img * std[:, None, None] + mean[:, None, None]
    img  = np.clip(img, 0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


# ============================================================================
# FULL EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, config, device, label: str):
    """
    Evaluate model on full validation set.
    Returns seg_results, cal_results, and raw per-sample data for visualisation.
    """
    ignore = getattr(config.data, 'ignore_index', None)

    seg_metrics = SegmentationMetrics(
        num_classes  = config.data.num_classes,
        ignore_index = ignore,
        class_names  = CLASS_NAMES
    )
    cal_metrics = CalibrationMetrics(
        num_bins   = 15,
        num_classes= config.data.num_classes,
        ignore_index = ignore
    )

    samples_data = []   # For visualisation

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(
            tqdm(dataloader, desc=f"  Evaluating [{label}]")
        ):
            images = images.to(device)
            masks  = masks.to(device)

            output = model(images, return_uncertainty=True)

            seg_metrics.update(output['pred'], masks)
            cal_metrics.update(
                output['prob'], output['pred'], masks, output['uncertainty']
            )

            # Store first few batches for visualisation
            if batch_idx < 3:
                for i in range(min(2, images.size(0))):
                    samples_data.append({
                        'image':       images[i].cpu(),
                        'gt_mask':     masks[i].cpu().numpy(),
                        'pred_mask':   output['pred'][i].cpu().numpy(),
                        'uncertainty': output['uncertainty'][i].cpu().numpy(),
                    })

    seg_results = seg_metrics.compute(return_per_class=True)
    cal_results = cal_metrics.compute()
    return seg_results, cal_results, samples_data


# ============================================================================
# PLOTTING
# ============================================================================

def plot_per_class_comparison(before_iou, after_iou, out_path, round_id):
    """
    Bar chart showing per-class IoU before vs after for each class.
    Green = improved. Red = regressed.
    """
    valid_classes = [
        i for i in range(len(CLASS_NAMES))
        if not (np.isnan(before_iou[i]) or np.isnan(after_iou[i]))
    ]

    names  = [CLASS_NAMES[i] for i in valid_classes]
    before = [before_iou[i]  for i in valid_classes]
    after  = [after_iou[i]   for i in valid_classes]
    deltas = [a - b for a, b in zip(after, before)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    x     = np.arange(len(names))
    width = 0.35

    ax1.bar(x - width/2, before, width, label='Before HITL',
            color='#95a5a6', alpha=0.9, edgecolor='white')
    ax1.bar(x + width/2, after,  width, label=f'After HITL R{round_id}',
            color='#2ecc71', alpha=0.9, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha='right')
    ax1.set_ylabel('IoU Score', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Class IoU: Before vs After', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')

    # Delta bar chart
    colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deltas]
    ax2.bar(x, deltas, color=colors, edgecolor='white', alpha=0.9)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha='right')
    ax2.set_ylabel('ΔIoU (After − Before)', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class Improvement', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Annotate deltas
    for i, d in enumerate(deltas):
        ax2.text(i, d + (0.003 if d >= 0 else -0.008),
                 f'{d:+.3f}', ha='center', va='bottom', fontsize=8,
                 fontweight='bold',
                 color='#27ae60' if d >= 0 else '#c0392b')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sample_comparison(before_samples, after_samples, out_dir, num_viz):
    """
    Side-by-side panels showing what changed for individual images:
      Image | GT | Before Pred | Before Uncertainty | After Pred | After Uncertainty
    """
    n = min(num_viz, len(before_samples), len(after_samples))

    for i in range(n):
        b = before_samples[i]
        a = after_samples[i]

        img_rgb   = denormalize(b['image'])
        gt_rgb    = decode_mask(b['gt_mask'])
        b_pred    = decode_mask(b['pred_mask'])
        a_pred    = decode_mask(a['pred_mask'])

        # Error maps
        b_errors  = (b['pred_mask'] != b['gt_mask']).astype(float)
        a_errors  = (a['pred_mask'] != a['gt_mask']).astype(float)

        fig = plt.figure(figsize=(24, 5))
        gs  = gridspec.GridSpec(1, 6, figure=fig, wspace=0.05)

        panels = [
            (img_rgb,            'Input Image',              None),
            (gt_rgb,             'Ground Truth',             None),
            (b_pred,             'Prediction BEFORE',        None),
            (b['uncertainty'],   'Uncertainty BEFORE\n(red=confused)', 'Reds'),
            (a_pred,             'Prediction AFTER',         None),
            (a['uncertainty'],   'Uncertainty AFTER\n(red=confused)',  'Reds'),
        ]

        for j, (data, title, cmap) in enumerate(panels):
            ax = fig.add_subplot(gs[j])
            if cmap:
                ax.imshow(img_rgb)
                im = ax.imshow(data, cmap=cmap, alpha=0.65,
                               vmin=0, vmax=data.max())
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(data)
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.axis('off')

            # Overlay error contours on prediction panels
            if j in (2, 4):
                errors = b_errors if j == 2 else a_errors
                ax.contour(errors, levels=[0.5],
                           colors='red' if j == 2 else 'lime',
                           linewidths=1.2)

        # Add class legend
        patches = [
            mpatches.Patch(color=np.array(CLASS_COLORS[k]) / 255.0,
                           label=CLASS_NAMES[k])
            for k in range(len(CLASS_NAMES))
        ]
        fig.legend(handles=patches, loc='lower center',
                   ncol=7, fontsize=8, framealpha=0.8)

        # Error stats
        before_err = b_errors.mean() * 100
        after_err  = a_errors.mean() * 100
        fig.suptitle(
            f'Sample {i+1}  |  '
            f'Error Before: {before_err:.1f}%  →  '
            f'Error After: {after_err:.1f}%  '
            f'({"↓ improved" if after_err < before_err else "↑ got worse"})',
            fontsize=11, fontweight='bold'
        )

        save_path = out_dir / f'sample_{i+1:02d}_comparison.png'
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()

    print(f"  ✓ {n} sample comparisons saved")


def plot_training_progress(log_path: Path, out_path: Path, round_id: str):
    """
    Plot mIoU over fine-tuning epochs so you can see the learning curve.
    """
    if not log_path.exists():
        return

    with open(log_path) as f:
        log = json.load(f)

    epochs  = [e['epoch']    for e in log['training_log']]
    mious   = [e['val_miou'] for e in log['training_log']]
    losses  = [e['loss']     for e in log['training_log']]
    frozen  = [e.get('frozen', False) for e in log['training_log']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'HITL Round {round_id} — Training Progress',
                 fontsize=13, fontweight='bold')

    # mIoU curve
    ax1.plot(epochs, mious, 'o-', color='#2ecc71', linewidth=2.5, markersize=5)
    # Shade frozen region
    frozen_end = max((i for i, f in enumerate(frozen) if f), default=-1)
    if frozen_end >= 0:
        ax1.axvspan(1, epochs[frozen_end], alpha=0.1, color='orange',
                    label='Encoder frozen')
    ax1.axhline(log['prev_miou'], color='#95a5a6', linestyle='--',
                linewidth=1.5, label=f'Previous mIoU ({log["prev_miou"]:.4f})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation mIoU', fontsize=12)
    ax1.set_title('mIoU During Fine-Tuning', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Loss curve
    ax2.plot(epochs, losses, 'o-', color='#e74c3c', linewidth=2.5, markersize=5)
    if frozen_end >= 0:
        ax2.axvspan(1, epochs[frozen_end], alpha=0.1, color='orange',
                    label='Encoder frozen')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('Loss During Fine-Tuning', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args    = parse_args()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config  = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  HITL STEP 4 — EVALUATE IMPROVEMENT")
    print(f"  Round:  {args.round_id}")
    print(f"  Before: {Path(args.checkpoint_before).name}")
    print(f"  After:  {Path(args.checkpoint_after).name}")
    print(f"{'='*60}\n")

    # ---- Validation dataloader ----
    val_dataset = get_dataset(config, split='val')
    val_loader  = DataLoader(
        val_dataset,
        batch_size  = config.training.batch_size,
        shuffle     = False,
        num_workers = config.training.num_workers,
        pin_memory  = True
    )

    # ---- Evaluate BEFORE ----
    print("  [1/2] Evaluating BEFORE checkpoint...")
    model_before = load_model(config, args.checkpoint_before, device)
    seg_before, cal_before, samples_before = evaluate_model(
        model_before, val_loader, config, device, "BEFORE"
    )
    del model_before
    torch.cuda.empty_cache()

    # ---- Evaluate AFTER ----
    print("\n  [2/2] Evaluating AFTER checkpoint...")
    model_after = load_model(config, args.checkpoint_after, device)
    seg_after, cal_after, samples_after = evaluate_model(
        model_after, val_loader, config, device, "AFTER"
    )
    del model_after
    torch.cuda.empty_cache()

    # ---- Print comparison table ----
    print(f"\n{'='*60}")
    print(f"  RESULTS — HITL Round {args.round_id}")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"  {'-'*55}")

    metrics_compare = [
        ('mIoU',     seg_before['miou'],     seg_after['miou'],     True),
        ('Accuracy', seg_before['accuracy'], seg_after['accuracy'], True),
        ('F1',       seg_before['f1'],       seg_after['f1'],       True),
        ('ECE ↓',    cal_before['ece'],      cal_after['ece'],      False),
        ('Brier ↓',  cal_before['brier'],    cal_after['brier'],    False),
    ]

    if 'uncertainty_error_corr' in cal_before:
        metrics_compare.append((
            'Unc-Err Corr',
            cal_before['uncertainty_error_corr'],
            cal_after['uncertainty_error_corr'],
            True
        ))

    summary = {}
    for name, before, after, higher_is_better in metrics_compare:
        delta = after - before
        arrow = ('↑' if delta > 0 else '↓') if abs(delta) > 1e-5 else '='
        good  = (delta > 0) == higher_is_better
        sign  = '+' if delta >= 0 else ''
        print(f"  {name:<25} {before:>10.4f} {after:>10.4f} "
              f"  {arrow} {sign}{delta:.4f} "
              f"{'✓' if good else '⚠'}")
        summary[name] = {'before': before, 'after': after, 'delta': delta}

    print(f"{'='*60}")

    # ---- Per-class comparison ----
    print(f"\n  Per-class IoU:")
    print(f"  {'Class':<15} {'Before':>8} {'After':>8} {'Δ':>8}")
    print(f"  {'-'*40}")

    iou_before = seg_before.get('iou_per_class', [])
    iou_after  = seg_after.get('iou_per_class',  [])

    per_class_data = {}
    for i, name in enumerate(CLASS_NAMES):
        if i == getattr(config.data, 'ignore_index', None):
            continue
        b = iou_before[i] if i < len(iou_before) and not np.isnan(iou_before[i]) else 0
        a = iou_after[i]  if i < len(iou_after)  and not np.isnan(iou_after[i])  else 0
        d = a - b
        flag = '✓' if d > 0 else ('=' if abs(d) < 0.001 else '↓')
        print(f"  {name:<15} {b:>8.4f} {a:>8.4f} {d:>+8.4f} {flag}")
        per_class_data[name] = {'before': b, 'after': a, 'delta': d}

    # ---- Generate plots ----
    print(f"\n  Generating plots...")

    # Per-class IoU comparison
    if iou_before and iou_after:
        plot_per_class_comparison(
            iou_before, iou_after,
            out_dir / f'round_{args.round_id}_per_class_iou.png',
            args.round_id
        )
        print(f"  ✓ Per-class IoU chart saved")

    # Sample comparisons
    if samples_before and samples_after:
        plot_sample_comparison(
            samples_before, samples_after, out_dir, args.num_viz
        )

    # Training progress (if log exists)
    log_path = Path(f'checkpoints/hitl/hitl_round_{args.round_id}_training_log.json')
    if log_path.exists():
        plot_training_progress(
            log_path,
            out_dir / f'round_{args.round_id}_training_curve.png',
            args.round_id
        )
        print(f"  ✓ Training curve saved")

    # ---- Save full results JSON ----
    results_path = out_dir / f'round_{args.round_id}_evaluation.json'
    with open(results_path, 'w') as f:
        json.dump({
            'round':         args.round_id,
            'summary':       summary,
            'per_class':     per_class_data,
        }, f, indent=2)

    print(f"  ✓ Full results saved: {results_path}")

    # ---- Final verdict ----
    delta_miou = seg_after['miou'] - seg_before['miou']
    delta_ece  = cal_before['ece'] - cal_after['ece']   # Positive = improved

    print(f"\n{'='*60}")
    if delta_miou > 0.005:
        print(f"  ✅ HITL Round {args.round_id} SUCCEEDED")
        print(f"     mIoU improved by {delta_miou:+.4f}")
        print(f"     Run another round: mine more uncertain images,")
        print(f"     label them, and repeat the pipeline.")
    elif delta_miou > 0:
        print(f"  ⚠ HITL Round {args.round_id}: Minor improvement ({delta_miou:+.4f})")
        print(f"     Consider increasing replay_ratio or ft_epochs next round.")
    else:
        print(f"  ❌ HITL Round {args.round_id}: No improvement ({delta_miou:+.4f})")
        print(f"     Possible causes:")
        print(f"       - Labels might have errors — review them carefully")
        print(f"       - Replay ratio too low — increase --replay_ratio")
        print(f"       - Too few epochs — increase --ft_epochs")
        print(f"       - Learning rate too high — lower --ft_lr")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()