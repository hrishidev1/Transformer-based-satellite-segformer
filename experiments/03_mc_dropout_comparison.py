"""
Experiment 3 — MC Dropout Comparison
======================================
Addresses professor feedback Point 4: "No comparison against Bayesian
SegFormer (MC Dropout / Deep Ensemble). Without this, the efficiency-
accuracy-calibration tradeoff is not experimentally validated."

WHAT THIS DOES:
  Builds an MC Dropout version of SegFormer and evaluates it against your
  EDL model on the same validation set. Produces a 3-way comparison table:

    Method          | mIoU | ECE | Brier | Infer time | Memory
    Baseline        |      |     |       |            |
    MC Dropout (T)  |      |     |       |            |
    EDL (ours)      |      |     |       |            |

  MC Dropout works by:
    1. Adding dropout layers to the SegFormer decoder (p=0.1)
    2. At inference, keeping dropout ACTIVE (model.train() for dropout layers)
    3. Running T forward passes (T=10, 20, 50)
    4. Averaging probabilities -> uncertainty = variance across T passes

  The key result your paper needs:
    "EDL achieves comparable ECE to MC Dropout (T=10) while requiring
    only a single forward pass, making it X times faster at inference."

HOW TO RUN:
  python experiments/03_mc_dropout_comparison.py \
      --config            configs/uncertain_segformer.yaml \
      --edl_checkpoint    checkpoints/segformer/best_miou.pth \
      --output_dir        outputs/mc_dropout_comparison \
      --train_mc \
      --mc_epochs         30

ARGUMENTS:
  --config               YAML config
  --edl_checkpoint       Your EDL model checkpoint
  --baseline_checkpoint  Optional: pre-trained MC Dropout checkpoint (skips training)
  --output_dir           Where to save results
  --mc_samples           MC forward passes at inference (default: 10 20 50)
  --train_mc             If set, train MC Dropout model from scratch
  --mc_epochs            Training epochs for MC Dropout model (default: 30)
"""

import sys
import time
import json
import copy
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config
from datasets import get_dataset
from models.uncertainty_factory import get_uncertainty_model
from metrics.segmentation import SegmentationMetrics
from metrics.calibration import CalibrationMetrics


# ============================================================================
# MC DROPOUT MODEL
# ============================================================================

class MCDropoutSegFormer(nn.Module):
    """
    SegFormer with MC Dropout for uncertainty estimation.

    Dropout is added to the decoder MLP layers (p=0.1).
    At inference, we use enable_dropout() to keep dropout active,
    then run T forward passes and measure variance.

    WHY DECODER ONLY:
      Adding dropout to the MiT encoder would destabilise the pre-trained
      attention patterns. Standard practice is decoder-only dropout.
    """

    def __init__(self, encoder_name: str, num_classes: int,
                 pretrained: bool = True, dropout_p: float = 0.1):
        super().__init__()
        self.num_classes = num_classes

        # Standard SegFormer from SMP
        self.segformer = smp.Segformer(
            encoder_name    = encoder_name,
            encoder_weights = 'imagenet' if pretrained else None,
            classes         = num_classes,
            activation      = None,
        )

        # Insert dropout into decoder — two-pass to avoid recursion bug
        num_inserted = self._insert_dropout(dropout_p)
        print(f"[MCDropoutSegFormer] encoder={encoder_name}, "
              f"Dropout2d(p={dropout_p}) inserted into {num_inserted} "
              f"decoder Conv2d(1x1) layers")

    def _insert_dropout(self, p: float) -> int:
        """
        Insert dropout after all meaningful decoder layers — two passes to
        avoid mutating the module tree during named_modules() iteration.

        SMP's SegFormer decoder has two layer types to target:
        1. nn.Linear in mlp_stage (x4) — per-scale MLP projections.
            Use nn.Dropout (1D) — these are (B, N, C) tensors.
        2. nn.Conv2d(1x1) in fuse_stage (x1) — channel fusion.
            Use nn.Dropout2d (spatial) — these are (B, C, H, W) tensors.

        Previously only the Conv2d was caught (1 layer). This version
        catches all 5, giving proper stochasticity across the full decoder.
        """
        replacements = []
        for name, module in self.segformer.named_modules():
            if 'decoder' not in name:
                continue
            for child_name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    replacements.append((module, child_name, child, 'linear'))
                elif isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1):
                    replacements.append((module, child_name, child, 'conv'))

        for parent, child_name, child, kind in replacements:
            drop = nn.Dropout(p=p) if kind == 'linear' else nn.Dropout2d(p=p)
            setattr(parent, child_name, nn.Sequential(child, drop))

        if len(replacements) == 0:
            raise RuntimeError(
                "No Linear or Conv2d(1x1) layers found in SegFormer decoder. "
                "Check your SMP version."
            )

        print(f"  [dropout] {sum(1 for *_, k in replacements if k=='linear')} Linear "
            f"+ {sum(1 for *_, k in replacements if k=='conv')} Conv2d(1x1) layers targeted")
        return len(replacements)

    def enable_dropout(self):
        """
        Set all Dropout layers to training mode even when model.eval() is set.
        This is the MC Dropout trick — dropout stays active at test time.
        """
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Single deterministic forward pass -> softmax probabilities (B, C, H, W)."""
        logits = self.segformer(x)
        return torch.softmax(logits, dim=1)

    def forward(self, x: torch.Tensor,
                num_samples: int = 1,
                return_uncertainty: bool = False) -> dict:
        """
        MC Dropout forward pass.
        If num_samples=1, equivalent to standard deterministic inference.
        If num_samples>1, runs T stochastic passes and computes mean + variance.
        """
        if num_samples == 1:
            probs = self.forward_once(x)
            pred  = probs.argmax(dim=1)
            out   = {'prob': probs, 'probs': probs, 'pred': pred}
            if return_uncertainty:
                out['uncertainty'] = (probs * (1 - probs)).sum(dim=1)
            return out

        # --- MC Dropout: T stochastic passes ---
        self.enable_dropout()

        all_probs = []
        with torch.no_grad():
            for _ in range(num_samples):
                all_probs.append(self.forward_once(x))

        all_probs = torch.stack(all_probs, dim=0)   # (T, B, C, H, W)

        # Sanity check: variance should be non-zero if dropout is working
        max_var = all_probs.var(dim=0).max().item()
        if max_var < 1e-8:
            raise RuntimeError(
                "MC Dropout variance is effectively zero — all T passes are "
                "identical. Dropout layers were not inserted correctly. "
                "Check _insert_dropout found Conv2d(1x1) in your SMP version."
            )

        mean_probs = all_probs.mean(dim=0)           # (B, C, H, W)
        variance   = all_probs.var(dim=0).sum(dim=1) # (B, H, W)
        pred       = mean_probs.argmax(dim=1)

        return {
            'prob':        mean_probs,
            'probs':       mean_probs,
            'pred':        pred,
            'uncertainty': variance if return_uncertainty else None,
        }


# ============================================================================
# TRAINING
# ============================================================================

def train_mc_model(config, device, train_loader, val_loader,
                   num_epochs: int, save_dir: Path) -> str:
    """Train MC Dropout SegFormer with standard CE+Dice loss."""
    save_dir.mkdir(parents=True, exist_ok=True)

    model = MCDropoutSegFormer(
        encoder_name = config.model.encoder,
        num_classes  = config.data.num_classes,
        pretrained   = True
    ).to(device)

    ignore   = getattr(config.data, 'ignore_index', None)
    ce       = nn.CrossEntropyLoss(
        ignore_index = ignore if ignore is not None else -100
    )
    try:
        dice     = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore)
        use_dice = True
    except Exception:
        use_dice = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config.optimizer.lr,
        weight_decay = config.optimizer.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=config.scheduler.min_lr
    )

    seg_metrics = SegmentationMetrics(config.data.num_classes, ignore_index=ignore)
    best_miou   = 0.0
    best_state  = None

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for images, masks in tqdm(train_loader,
                                   desc=f"  [MC-Train] E{epoch+1}/{num_epochs}",
                                   leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model.segformer(images)
            loss   = ce(logits, masks)
            if use_dice:
                loss = loss + dice(logits, masks)
            loss.backward()
            if config.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(),
                                          config.training.grad_clip)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        # Validation
        model.eval()
        seg_metrics.reset()
        with torch.no_grad():
            for images, masks in val_loader:
                out = model(images.to(device), num_samples=1)
                seg_metrics.update(out['pred'], masks.to(device))

        seg_r = seg_metrics.compute(return_per_class=False)
        if seg_r['miou'] > best_miou:
            best_miou  = seg_r['miou']
            best_state = copy.deepcopy(model.state_dict())

        print(f"  [MC-Train] E{epoch+1:>2} | loss={np.mean(losses):.4f} "
              f"| mIoU={seg_r['miou']:.4f} | best={best_miou:.4f}")

    ckpt_path = save_dir / 'mc_dropout_best.pth'
    torch.save({'model_state_dict': best_state, 'miou': best_miou}, ckpt_path)
    print(f"  ✓ MC Dropout model saved: {ckpt_path}")
    return str(ckpt_path)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_mc_model(model, dataloader, device, config,
                      num_samples: int = 10) -> dict:
    """Evaluate MC Dropout model with T stochastic passes."""
    ignore      = getattr(config.data, 'ignore_index', None)
    seg_metrics = SegmentationMetrics(config.data.num_classes, ignore_index=ignore)
    cal_metrics = CalibrationMetrics(num_bins=15,
                                      num_classes=config.data.num_classes,
                                      ignore_index=ignore)
    model.eval()
    t_start = time.time()

    with torch.no_grad():
        for images, masks in tqdm(dataloader,
                                   desc=f"  Eval MC (T={num_samples})",
                                   leave=False):
            images, masks = images.to(device), masks.to(device)
            out = model(images, num_samples=num_samples, return_uncertainty=True)
            seg_metrics.update(out['pred'], masks)
            cal_metrics.update(out['prob'], out['pred'], masks, out['uncertainty'])

    elapsed = time.time() - t_start
    seg_r   = seg_metrics.compute(return_per_class=False)
    cal_r   = cal_metrics.compute()

    return {
        'method':        f'MC Dropout (T={num_samples})',
        'num_samples':   num_samples,
        'miou':          float(seg_r['miou']),
        'accuracy':      float(seg_r['accuracy']),
        'ece':           float(cal_r['ece']),
        'brier':         float(cal_r['brier']),
        'unc_err_corr':  float(cal_r.get('uncertainty_error_corr', 0)),
        'inference_sec': float(elapsed),
    }


def evaluate_edl_model(model, dataloader, device, config) -> dict:
    """Evaluate EDL model (your method)."""
    ignore      = getattr(config.data, 'ignore_index', None)
    seg_metrics = SegmentationMetrics(config.data.num_classes, ignore_index=ignore)
    cal_metrics = CalibrationMetrics(num_bins=15,
                                      num_classes=config.data.num_classes,
                                      ignore_index=ignore)
    model.eval()
    t_start = time.time()

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="  Eval EDL", leave=False):
            images, masks = images.to(device), masks.to(device)
            out = model(images, return_uncertainty=True)
            seg_metrics.update(out['pred'], masks)
            cal_metrics.update(out['prob'], out['pred'], masks, out['uncertainty'])

    elapsed = time.time() - t_start
    seg_r   = seg_metrics.compute(return_per_class=False)
    cal_r   = cal_metrics.compute()

    return {
        'method':        'EDL (ours)',
        'num_samples':   1,
        'miou':          float(seg_r['miou']),
        'accuracy':      float(seg_r['accuracy']),
        'ece':           float(cal_r['ece']),
        'brier':         float(cal_r['brier']),
        'unc_err_corr':  float(cal_r.get('uncertainty_error_corr', 0)),
        'inference_sec': float(elapsed),
    }


# ============================================================================
# PLOTTING
# ============================================================================

def plot_comparison(all_results: list, out_path: Path):
    """Bar chart comparison: mIoU, ECE, Brier, inference time across methods."""
    methods   = [r['method']        for r in all_results]
    mious     = [r['miou']          for r in all_results]
    eces      = [r['ece']           for r in all_results]
    briers    = [r['brier']         for r in all_results]
    inf_times = [r['inference_sec'] for r in all_results]

    edl_time  = next((r['inference_sec'] for r in all_results
                      if 'EDL' in r['method']), 1.0)
    rel_times = [t / edl_time for t in inf_times]

    x          = np.arange(len(methods))
    fig        = plt.figure(figsize=(18, 5))
    gs         = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)

    colors_bar = ['#95a5a6'] * len(methods)
    edl_idx    = next((i for i, r in enumerate(all_results)
                       if 'EDL' in r['method']), 0)
    colors_bar[edl_idx] = '#e74c3c'

    for col, (vals, ylabel, title, higher) in enumerate([
        (mious,     'mIoU',          '(a) mIoU ↑',          True),
        (eces,      'ECE',           '(b) ECE ↓',            False),
        (briers,    'Brier Score',   '(c) Brier Score ↓',    False),
        (rel_times, 'Relative Time', '(d) Inference Time ↑', False),
    ]):
        ax   = fig.add_subplot(gs[col])
        bars = ax.bar(x, vals, color=colors_bar, edgecolor='white', alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        best_idx = np.argmax(vals) if higher else np.argmin(vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2.5)

    fig.suptitle('Method Comparison: EDL vs MC Dropout\nDeepGlobe Validation Set',
                 fontsize=13, fontweight='bold')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Comparison plot saved: {out_path}")


def print_comparison_table(all_results: list):
    edl_time = next((r['inference_sec'] for r in all_results
                     if 'EDL' in r['method']), 1.0)

    print(f"\n{'='*85}")
    print(f"  METHOD COMPARISON")
    print(f"{'='*85}")
    print(f"  {'Method':<28} {'mIoU':>7} {'ECE':>7} {'Brier':>7} "
          f"{'Unc-Err ρ':>11} {'Time':>8} {'Speedup':>10}")
    print(f"  {'-'*80}")
    for r in all_results:
        if 'EDL' in r['method']:
            speedup_str = '1.0× (ref)'
        else:
            speedup_str = f"{r['inference_sec'] / edl_time:.1f}×"
        print(f"  {r['method']:<28} {r['miou']:>7.4f} {r['ece']:>7.4f} "
              f"{r['brier']:>7.4f} {r['unc_err_corr']:>11.4f} "
              f"{r['inference_sec']:>7.1f}s {speedup_str:>10}")
    print(f"{'='*85}")
    print(f"\n  Key claim: EDL is 1 forward pass vs T passes for MC Dropout.")
    print(f"  If ECE values are comparable, EDL wins on efficiency.\n")


# ============================================================================
# ARGS
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config',              required=True)
    p.add_argument('--edl_checkpoint',      required=True,
                   help='Your EDL model best_miou.pth')
    p.add_argument('--baseline_checkpoint', default=None,
                   help='Optional pre-trained MC Dropout checkpoint (skips training)')
    p.add_argument('--output_dir',          default='outputs/mc_comparison')
    p.add_argument('--mc_samples',          type=int, nargs='+', default=[10, 20, 50],
                   help='MC forward pass counts to evaluate')
    p.add_argument('--train_mc',            action='store_true',
                   help='Train MC Dropout model from scratch')
    p.add_argument('--mc_epochs',           type=int, default=30)
    return p.parse_args()


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
    print(f"  MC DROPOUT COMPARISON")
    print(f"  Device:     {device}")
    print(f"  MC samples: {args.mc_samples}")
    print(f"{'='*60}\n")

    # ---- Validation loader (always needed) ----
    val_dataset = get_dataset(config, split='val')
    val_loader  = DataLoader(val_dataset,
                              batch_size  = config.training.batch_size,
                              shuffle     = False,
                              num_workers = config.training.num_workers,
                              pin_memory  = True)

    # ---- Train MC model if needed ----
    mc_ckpt_path = args.baseline_checkpoint

    if args.train_mc or mc_ckpt_path is None:
        print("  Training MC Dropout SegFormer from scratch...")
        train_dataset = get_dataset(config, split='train')
        train_loader  = DataLoader(train_dataset,
                                    batch_size  = config.training.batch_size,
                                    shuffle     = True,
                                    num_workers = config.training.num_workers,
                                    pin_memory  = True,
                                    drop_last   = True)
        mc_ckpt_path = train_mc_model(
            config, device, train_loader, val_loader,
            args.mc_epochs, out_dir / 'mc_dropout_checkpoints'
        )

    # ---- Load MC model ----
    print(f"\n  Loading MC Dropout model from {mc_ckpt_path}...")
    mc_model = MCDropoutSegFormer(
        encoder_name = config.model.encoder,
        num_classes  = config.data.num_classes,
        pretrained   = False
    ).to(device)
    mc_ckpt = torch.load(mc_ckpt_path, map_location=device, weights_only=False)
    mc_model.load_state_dict(mc_ckpt.get('model_state_dict', mc_ckpt))
    print(f"  ✓ MC Dropout model loaded\n")

    # ---- Load EDL model ----
    print(f"  Loading EDL model from {args.edl_checkpoint}...")
    edl_model = get_uncertainty_model(
        arch_name    = getattr(config.model, 'arch', 'SegFormer'),
        encoder_name = config.model.encoder,
        num_classes  = config.data.num_classes,
        pretrained   = False
    ).to(device)
    edl_ckpt = torch.load(args.edl_checkpoint, map_location=device, weights_only=False)
    edl_model.load_state_dict(edl_ckpt.get('model_state_dict', edl_ckpt))
    print(f"  ✓ EDL model loaded\n")

    # ---- Evaluate all methods ----
    all_results = []

    print("  [1] Evaluating EDL (ours)...")
    all_results.append(evaluate_edl_model(edl_model, val_loader, device, config))

    for T in args.mc_samples:
        print(f"  [MC T={T}] Evaluating MC Dropout...")
        all_results.append(
            evaluate_mc_model(mc_model, val_loader, device, config, num_samples=T)
        )

    # ---- Output ----
    print_comparison_table(all_results)

    json_path = out_dir / 'comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✓ Results saved: {json_path}")

    plot_comparison(all_results, out_dir / 'mc_comparison_plot.png')

    print(f"\n{'='*60}")
    print(f"  MC COMPARISON COMPLETE")
    print(f"  Results: {json_path}")
    print(f"  Plot:    {out_dir / 'mc_comparison_plot.png'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()