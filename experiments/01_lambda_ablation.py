"""
Experiment 1 — λ (KL Weight) Ablation Study
=============================================
Addresses professor feedback Point 5: "No ablation on λ, no ablation on
annealing duration, no study of KL weight vs mIoU vs ECE."

WHAT THIS DOES:
  Trains the same model architecture 5 times, each with a different λ_KL value.
  For each run it records mIoU, ECE, Brier Score, and Uncertainty-Error
  Correlation. Then it generates a 4-panel plot showing exactly how calibration
  and accuracy trade off as you increase KL regularisation strength.

  This is the single experiment that most directly proves your calibration
  crisis finding with numbers. The expected result is:
    - λ=0.00  → best mIoU, worst ECE (over-confident, no KL pressure)
    - λ=0.10  → your paper's setting (moderate balance)
    - λ=0.50  → best ECE (well-spread), worst mIoU (under-confident)

  That curve IS your paper's main experimental contribution made quantitative.

ALSO RUNS annealing duration ablation:
  Fixes λ=0.1 and varies annealing_epochs ∈ {5, 10, 20, 30}.
  Shows how fast you ramp up KL matters as much as how strong it is.

HOW TO RUN:
  python experiments/01_lambda_ablation.py \\
      --config  configs/uncertain_segformer.yaml \\
      --output_dir outputs/ablation_lambda \\
      --epochs  30

  For a quick test (sanity check):
      python experiments/01_lambda_ablation.py \\
          --config configs/uncertain_segformer.yaml \\
          --epochs 5 --quick

ARGUMENTS:
  --config      YAML config (encoder, dataset paths, batch size are reused)
  --output_dir  Where to save checkpoints, results JSON, and plots
  --epochs      Epochs per run (30 recommended, 50 for final paper numbers)
  --quick       Only run λ ∈ {0.0, 0.1, 0.5} for a fast sanity check
  --lambdas     Override λ values, e.g. --lambdas 0.0 0.05 0.1 0.2 0.5

OUTPUT:
  ablation_lambda/
    lambda_0.000/  best.pth + training_log.json
    lambda_0.010/  ...
    ...
    lambda_ablation_results.json    ← all numbers, paste into paper
    lambda_ablation_plot.png        ← 4-panel figure for paper
    annealing_ablation_results.json
    annealing_ablation_plot.png
"""

import sys
import os
import json
import copy
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config
from datasets import get_dataset
from models.uncertainty_factory import get_uncertainty_model
from losses.evidential_loss import EvidentialLoss
from metrics.segmentation import SegmentationMetrics
from metrics.calibration import CalibrationMetrics


# ============================================================================
# ARGS
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config',      required=True)
    p.add_argument('--output_dir',  default='outputs/ablation_lambda')
    p.add_argument('--epochs',      type=int, default=30)
    p.add_argument('--quick',       action='store_true',
                   help='Fast test: only 3 λ values')
    p.add_argument('--lambdas',     type=float, nargs='+',
                   default=None,
                   help='Override λ values')
    p.add_argument('--skip_annealing', action='store_true',
                   help='Skip the annealing duration ablation')
    return p.parse_args()


# ============================================================================
# SINGLE TRAINING RUN
# ============================================================================

def train_one_run(config, device, train_loader, val_loader,
                  lambda_kl: float, annealing_epochs: int,
                  num_epochs: int, save_dir: Path,
                  run_label: str = ""):
    """
    Train model with given λ_KL and annealing_epochs.
    Returns dict of final metrics.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Fresh model every run — same seed for fair comparison
    torch.manual_seed(42)
    np.random.seed(42)

    model = get_uncertainty_model(
        arch_name    = getattr(config.model, 'arch', 'SegFormer'),
        encoder_name = config.model.encoder,
        num_classes  = config.data.num_classes,
        pretrained   = True
    ).to(device)

    # Build loss with this run's λ and annealing schedule
    criterion = EvidentialLoss(
        num_classes  = config.data.num_classes,
        lambda_kl    = lambda_kl,
        max_epoch    = annealing_epochs,
        ignore_index = getattr(config.data, 'ignore_index', None)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config.optimizer.lr,
        weight_decay = config.optimizer.weight_decay,
        betas        = (0.9, 0.999),
        eps          = 1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs,
        eta_min=config.scheduler.min_lr
    )

    ignore      = getattr(config.data, 'ignore_index', None)
    seg_metrics = SegmentationMetrics(config.data.num_classes,
                                       ignore_index=ignore)
    cal_metrics = CalibrationMetrics(num_bins=15,
                                      num_classes=config.data.num_classes,
                                      ignore_index=ignore)

    best_miou  = 0.0
    best_state = None
    log        = []

    label = run_label or f"λ={lambda_kl}"

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        losses = []
        for images, masks in tqdm(train_loader,
                                   desc=f"  [{label}] E{epoch+1}/{num_epochs}",
                                   leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            out          = model(images, return_uncertainty=True)
            loss, _      = criterion(out, masks, current_epoch=epoch)
            loss.backward()
            if config.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(),
                                          config.training.grad_clip)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        # ---- Validate ----
        model.eval()
        seg_metrics.reset()
        cal_metrics.reset()

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                out = model(images, return_uncertainty=True)
                seg_metrics.update(out['pred'], masks)
                cal_metrics.update(out['prob'], out['pred'],
                                   masks, out['uncertainty'])

        seg_r = seg_metrics.compute(return_per_class=False)
        cal_r = cal_metrics.compute()

        entry = {
            'epoch':    epoch + 1,
            'loss':     float(np.mean(losses)),
            'miou':     float(seg_r['miou']),
            'accuracy': float(seg_r['accuracy']),
            'ece':      float(cal_r['ece']),
            'brier':    float(cal_r['brier']),
            'unc_err_corr': float(cal_r.get('uncertainty_error_corr', 0)),
        }
        log.append(entry)

        if seg_r['miou'] > best_miou:
            best_miou  = seg_r['miou']
            best_state = copy.deepcopy(model.state_dict())

        print(f"  [{label}] E{epoch+1:>2} | loss={np.mean(losses):.4f} "
              f"| mIoU={seg_r['miou']:.4f} | ECE={cal_r['ece']:.4f} "
              f"| Brier={cal_r['brier']:.4f}")

    # Save best checkpoint
    torch.save({'model_state_dict': best_state,
                'lambda_kl': lambda_kl,
                'annealing_epochs': annealing_epochs,
                'miou': best_miou},
               save_dir / 'best.pth')

    # Save log
    with open(save_dir / 'training_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    # Final metrics = best epoch's metrics (find it)
    best_entry = max(log, key=lambda e: e['miou'])

    return {
        'lambda_kl':        lambda_kl,
        'annealing_epochs': annealing_epochs,
        'best_miou':        best_miou,
        'best_ece':         best_entry['ece'],
        'best_brier':       best_entry['brier'],
        'unc_err_corr':     best_entry['unc_err_corr'],
        'log':              log,
    }


# ============================================================================
# PLOTTING
# ============================================================================

def plot_lambda_ablation(results: list, out_path: Path):
    """
    4-panel plot:
      (a) mIoU vs λ
      (b) ECE vs λ
      (c) Brier vs λ
      (d) mIoU vs ECE scatter (efficiency frontier)
    """
    lambdas = [r['lambda_kl']  for r in results]
    mious   = [r['best_miou']  for r in results]
    eces    = [r['best_ece']   for r in results]
    briers  = [r['best_brier'] for r in results]
    corrs   = [r['unc_err_corr'] for r in results]

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Shared x-tick labels
    xlabels = [f'{l:.3f}' for l in lambdas]

    # ---- (a) mIoU vs λ ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(len(lambdas)), mious, 'o-',
             color='#2ecc71', linewidth=2.5, markersize=8)
    for i, (x, y) in enumerate(zip(range(len(lambdas)), mious)):
        ax1.annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax1.set_xticks(range(len(lambdas)))
    ax1.set_xticklabels(xlabels)
    ax1.set_xlabel('λ_KL', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Validation mIoU', fontsize=11, fontweight='bold')
    ax1.set_title('(a) mIoU vs λ_KL', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # ---- (b) ECE vs λ ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(len(lambdas)), eces, 'o-',
             color='#e74c3c', linewidth=2.5, markersize=8)
    for i, (x, y) in enumerate(zip(range(len(lambdas)), eces)):
        ax2.annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax2.set_xticks(range(len(lambdas)))
    ax2.set_xticklabels(xlabels)
    ax2.set_xlabel('λ_KL', fontsize=11, fontweight='bold')
    ax2.set_ylabel('ECE (lower = better)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) ECE vs λ_KL', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # ---- (c) Uncertainty-Error Correlation vs λ ----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(range(len(lambdas)), corrs, 'o-',
             color='#9b59b6', linewidth=2.5, markersize=8)
    for i, (x, y) in enumerate(zip(range(len(lambdas)), corrs)):
        ax3.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax3.set_xticks(range(len(lambdas)))
    ax3.set_xticklabels(xlabels)
    ax3.set_xlabel('λ_KL', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Uncertainty-Error ρ', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Uncertainty Quality vs λ_KL', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # ---- (d) mIoU vs ECE scatter (efficiency frontier) ----
    ax4 = fig.add_subplot(gs[1, 1])
    sc = ax4.scatter(eces, mious, c=lambdas, cmap='RdYlGn_r',
                     s=120, zorder=3, edgecolors='black', linewidths=0.8)
    for i, (x, y, l) in enumerate(zip(eces, mious, lambdas)):
        ax4.annotate(f'λ={l:.3f}', (x, y), textcoords='offset points',
                     xytext=(5, 4), fontsize=8)
    plt.colorbar(sc, ax=ax4, label='λ_KL')
    ax4.set_xlabel('ECE (lower = better)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('mIoU (higher = better)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Accuracy-Calibration Frontier', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    # Ideal corner annotation
    ax4.annotate('← Ideal\n(low ECE, high mIoU)',
                 xy=(min(eces), max(mious)),
                 fontsize=8, color='green',
                 xytext=(min(eces) + 0.01, max(mious) - 0.02))

    fig.suptitle('KL Weight (λ) Ablation Study\nEDL-SegFormer on DeepGlobe',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ λ ablation plot saved: {out_path}")


def plot_annealing_ablation(results: list, out_path: Path):
    """2-panel plot for annealing duration ablation."""
    durations = [r['annealing_epochs'] for r in results]
    mious     = [r['best_miou']        for r in results]
    eces      = [r['best_ece']         for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('KL Annealing Duration Ablation (λ=0.1 fixed)',
                 fontsize=13, fontweight='bold')

    ax1.plot(durations, mious, 'o-', color='#2ecc71', linewidth=2.5, markersize=8)
    ax1.set_xlabel('Annealing Epochs', fontsize=11)
    ax1.set_ylabel('Validation mIoU',  fontsize=11)
    ax1.set_title('mIoU vs Annealing Duration')
    ax1.grid(True, alpha=0.3)

    ax2.plot(durations, eces, 'o-', color='#e74c3c', linewidth=2.5, markersize=8)
    ax2.set_xlabel('Annealing Epochs', fontsize=11)
    ax2.set_ylabel('ECE (lower = better)', fontsize=11)
    ax2.set_title('ECE vs Annealing Duration')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Annealing ablation plot saved: {out_path}")


def print_ablation_table(results: list, title: str, key: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {key:<20} {'mIoU':>8} {'ECE':>8} {'Brier':>8} {'Unc-Err ρ':>12}")
    print(f"  {'-'*60}")
    for r in results:
        val = r.get('lambda_kl', r.get('annealing_epochs', '?'))
        print(f"  {str(val):<20} {r['best_miou']:>8.4f} {r['best_ece']:>8.4f} "
              f"{r['best_brier']:>8.4f} {r['unc_err_corr']:>12.4f}")
    print(f"{'='*70}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    args     = parse_args()
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config   = load_config(args.config)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Lambda values ----
    if args.lambdas:
        lambda_values = args.lambdas
    elif args.quick:
        lambda_values = [0.0, 0.1, 0.5]
    else:
        lambda_values = [0.0, 0.01, 0.05, 0.1, 0.5]

    print(f"\n{'='*60}")
    print(f"  λ ABLATION STUDY")
    print(f"  Device:   {device}")
    print(f"  λ values: {lambda_values}")
    print(f"  Epochs:   {args.epochs}")
    print(f"{'='*60}\n")

    # ---- Build dataloaders once (reused across all runs) ----
    train_dataset = get_dataset(config, split='train')
    val_dataset   = get_dataset(config, split='val')

    train_loader = DataLoader(train_dataset,
                               batch_size=config.training.batch_size,
                               shuffle=True,
                               num_workers=config.training.num_workers,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,
                               batch_size=config.training.batch_size,
                               shuffle=False,
                               num_workers=config.training.num_workers,
                               pin_memory=True)

    # ============================================================
    # PART A — λ ablation (fixed annealing = 10 epochs)
    # ============================================================
    lambda_results = []

    for lam in lambda_values:
        print(f"\n{'#'*50}")
        print(f"  Training λ = {lam}")
        print(f"{'#'*50}")

        run_dir = out_dir / f'lambda_{lam:.3f}'
        result  = train_one_run(
            config           = config,
            device           = device,
            train_loader     = train_loader,
            val_loader       = val_loader,
            lambda_kl        = lam,
            annealing_epochs = 10,       # Fixed for λ ablation
            num_epochs       = args.epochs,
            save_dir         = run_dir,
            run_label        = f"λ={lam:.3f}"
        )
        lambda_results.append(result)

    print_ablation_table(lambda_results, "λ_KL ABLATION RESULTS", "λ_KL")

    # Save results (without full log to keep file small)
    save_results = [{k: v for k, v in r.items() if k != 'log'}
                    for r in lambda_results]
    json_path = out_dir / 'lambda_ablation_results.json'
    with open(json_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    plot_lambda_ablation(lambda_results, out_dir / 'lambda_ablation_plot.png')

    # ============================================================
    # PART B — Annealing duration ablation (fixed λ = 0.1)
    # ============================================================
    if not args.skip_annealing:
        annealing_durations = [5, 10, 20, 30]
        # Clip to max epochs
        annealing_durations = [d for d in annealing_durations if d <= args.epochs]

        if len(annealing_durations) >= 2:
            print(f"\n\n{'='*60}")
            print(f"  ANNEALING DURATION ABLATION (λ=0.1 fixed)")
            print(f"  Durations: {annealing_durations}")
            print(f"{'='*60}\n")

            ann_results = []
            for duration in annealing_durations:
                print(f"\n{'#'*50}")
                print(f"  Training annealing_epochs = {duration}")
                print(f"{'#'*50}")

                run_dir = out_dir / f'anneal_{duration}ep'
                result  = train_one_run(
                    config           = config,
                    device           = device,
                    train_loader     = train_loader,
                    val_loader       = val_loader,
                    lambda_kl        = 0.1,
                    annealing_epochs = duration,
                    num_epochs       = args.epochs,
                    save_dir         = run_dir,
                    run_label        = f"anneal={duration}"
                )
                ann_results.append(result)

            print_ablation_table(
                ann_results, "ANNEALING DURATION ABLATION RESULTS",
                "annealing_epochs"
            )

            with open(out_dir / 'annealing_ablation_results.json', 'w') as f:
                json.dump([{k: v for k, v in r.items() if k != 'log'}
                            for r in ann_results], f, indent=2)

            plot_annealing_ablation(
                ann_results, out_dir / 'annealing_ablation_plot.png'
            )

    print(f"\n{'='*60}")
    print(f"  ABLATION COMPLETE")
    print(f"  Results: {json_path}")
    print(f"  Plot:    {out_dir / 'lambda_ablation_plot.png'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()