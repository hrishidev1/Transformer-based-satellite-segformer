"""
Experiment 4 — Extended Calibration Analysis
=============================================
Addresses professor feedback Point 3: "No temperature scaling comparison,
no class-wise calibration, no Adaptive ECE."

WHAT THIS DOES:
  Computes four things that your current paper is missing:

  1. TEMPERATURE SCALING:
     Post-hoc calibration baseline. Tunes a single scalar T on the val set
     such that softmax(logits / T) is better calibrated. If your EDL model
     has lower ECE than a temperature-scaled baseline, that's a strong result.
     If not, it means EDL's calibration is no better than a trivial fix.

  2. ADAPTIVE ECE (ACE):
     Standard ECE uses equal-width confidence bins. Adaptive ECE uses
     equal-mass bins (same number of pixels per bin). This is more reliable
     when confidence is not uniformly distributed, which is always the case
     in segmentation. Reviewers who know calibration will ask for this.

  3. CLASS-WISE CALIBRATION:
     Reports ECE per land cover class. Reveals which classes are over- vs
     under-confident. For satellite segmentation, rare classes (Barren, Water)
     tend to be miscalibrated while dominant classes (Agriculture) are fine.

  4. RELIABILITY DIAGRAMS:
     Side-by-side reliability plots for:
       - Baseline (no uncertainty)
       - EDL (your model)
       - EDL + Temperature Scaling (post-hoc fix)
     Shows visually how much of the calibration gap remains after scaling.

HOW TO RUN:
  python experiments/04_calibration_extended.py \\
      --config             configs/uncertain_segformer.yaml \\
      --edl_checkpoint     checkpoints/segformer/best_miou.pth \\
      --output_dir         outputs/calibration_extended

  With baseline comparison:
      python experiments/04_calibration_extended.py \\
          --config           configs/uncertain_segformer.yaml \\
          --edl_checkpoint   checkpoints/segformer/best_miou.pth \\
          --base_checkpoint  checkpoints/baseline/best.pth \\
          --output_dir       outputs/calibration_extended

ARGUMENTS:
  --config           YAML config
  --edl_checkpoint   EDL model checkpoint
  --base_checkpoint  Optional: deterministic baseline checkpoint
  --output_dir       Where to save results and plots
  --temp_lr          Learning rate for temperature optimisation (default: 0.01)
  --temp_epochs      Epochs to optimise temperature (default: 100)
"""

import sys
import json
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

CLASS_NAMES = [
    'Urban', 'Agriculture', 'Rangeland',
    'Forest', 'Water', 'Barren', 'Unknown'
]


# ============================================================================
# TEMPERATURE SCALING
# ============================================================================

class TemperatureScaler(nn.Module):
    """
    Single-parameter post-hoc calibration.
    Wraps any model and divides its logits by a learned scalar T before softmax.

    T > 1 → softer probabilities → less confident → reduces overconfidence
    T < 1 → sharper probabilities → more confident → reduces underconfidence

    For EDL: we apply temperature to the logits BEFORE the evidential head,
    then recompute alpha and vacuity with the scaled logits.
    """

    def __init__(self, model, is_edl: bool = True):
        super().__init__()
        self.model  = model
        self.is_edl = is_edl
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> dict:
        """Run model and scale logits by temperature."""
        with torch.no_grad():
            if self.is_edl:
                out = self.model(x, return_uncertainty=True)
                logits = out['logits']
            else:
                logits = self.model(x)

        scaled_logits = logits / self.temperature

        if self.is_edl:
            # Recompute evidential outputs from scaled logits
            evidence = torch.relu(scaled_logits)
            alpha    = evidence + 1.0
            S        = alpha.sum(dim=1, keepdim=True)
            prob     = alpha / S
            pred     = prob.argmax(dim=1)
            unc      = (config_K := alpha.shape[1]) / S.squeeze(1)
            return {'logits': scaled_logits, 'prob': prob, 'probs': prob,
                    'pred': pred, 'uncertainty': unc, 'alpha': alpha}
        else:
            prob = torch.softmax(scaled_logits, dim=1)
            pred = prob.argmax(dim=1)
            return {'prob': prob, 'probs': prob, 'pred': pred}

    def calibrate(self, val_loader, device, lr: float = 0.01,
                  num_epochs: int = 100, ignore_index: int = None):
        """
        Find optimal T by minimising NLL on validation set.
        Only optimises self.temperature — model weights are frozen.
        """
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=50
        )
        nll_criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index if ignore_index is not None else -100
        )

        # Collect all logits + labels first (don't run model T times)
        all_logits = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for images, masks in tqdm(val_loader,
                                       desc="  Collecting logits for TS",
                                       leave=False):
                images = images.to(device)
                if self.is_edl:
                    out    = self.model(images, return_uncertainty=False)
                    logits = out['logits']
                else:
                    logits = self.model(images)
                all_logits.append(logits.cpu())
                all_labels.append(masks)

        all_logits = torch.cat(all_logits, dim=0)   # (N, C, H, W)
        all_labels = torch.cat(all_labels, dim=0)   # (N, H, W)

        def eval_nll():
            optimizer.zero_grad()
            scaled = all_logits.to(device) / self.temperature
            loss   = nll_criterion(scaled, all_labels.to(device))
            loss.backward()
            return loss

        best_T   = 1.0
        best_nll = float('inf')

        for epoch in range(num_epochs):
            nll = optimizer.step(eval_nll)
            if nll.item() < best_nll:
                best_nll = nll.item()
                best_T   = self.temperature.item()

            if (epoch + 1) % 20 == 0:
                print(f"    TS epoch {epoch+1}/{num_epochs}: "
                      f"T={self.temperature.item():.4f}, NLL={nll.item():.4f}")

        # Clamp T to reasonable range
        with torch.no_grad():
            self.temperature.clamp_(0.1, 10.0)

        print(f"  ✓ Temperature scaling: T = {self.temperature.item():.4f}")
        return self.temperature.item()


# ============================================================================
# ADAPTIVE ECE
# ============================================================================

def compute_adaptive_ece(confidences: np.ndarray, predictions: np.ndarray,
                          targets: np.ndarray, num_bins: int = 15) -> float:
    """
    Adaptive ECE (ACE) — equal-mass bins.

    Standard ECE uses equal-width bins [0, 1/M], [1/M, 2/M], ...
    This can be misleading when confidence is highly skewed (most pixels
    are very high confidence). ACE fixes this by using quantile-based bins
    so each bin has the same number of samples.

    Reference: Nixon et al., "Measuring Calibration in Deep Learning", CVPR 2019.
    """
    n      = len(confidences)
    # Sort by confidence and split into equal-mass bins
    sort_idx = np.argsort(confidences)
    bin_size = n // num_bins

    ace = 0.0
    for b in range(num_bins):
        start = b * bin_size
        end   = start + bin_size if b < num_bins - 1 else n
        idx   = sort_idx[start:end]

        if len(idx) == 0:
            continue

        bin_conf = confidences[idx].mean()
        bin_acc  = (predictions[idx] == targets[idx]).mean()
        ace      += abs(bin_conf - bin_acc) * (len(idx) / n)

    return float(ace)


# ============================================================================
# CLASS-WISE ECE
# ============================================================================

def compute_classwise_ece(probs: np.ndarray, targets: np.ndarray,
                           num_classes: int, num_bins: int = 15,
                           ignore_index: int = None) -> dict:
    """
    Compute ECE separately for each class using one-vs-rest approach.

    For class c:
      confidence = prob[:, c]
      correct    = (target == c)
    This tells you: when the model says "this pixel is class c with
    probability 0.8", is it actually class c 80% of the time?

    Returns dict mapping class_name → ECE value.
    """
    results = {}

    for c in range(num_classes):
        if c == ignore_index:
            results[CLASS_NAMES[c]] = float('nan')
            continue

        conf    = probs[:, c]           # Confidence for class c
        correct = (targets == c).astype(float)

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0

        for b in range(num_bins):
            lo, hi  = bin_boundaries[b], bin_boundaries[b + 1]
            in_bin  = (conf > lo) & (conf <= hi)
            prop    = in_bin.mean()
            if prop > 0:
                avg_conf = conf[in_bin].mean()
                avg_acc  = correct[in_bin].mean()
                ece     += abs(avg_conf - avg_acc) * prop

        results[CLASS_NAMES[c]] = float(ece)

    return results


# ============================================================================
# FULL EVALUATION WITH COLLECTED PROBS/TARGETS
# ============================================================================

def collect_predictions(model, dataloader, device,
                         is_edl: bool = True, is_ts: bool = False) -> dict:
    """
    Run inference and collect all probabilities, predictions, targets, and
    uncertainty values into flat numpy arrays for calibration analysis.
    Subsamples pixels to avoid RAM issues (1M pixels is enough).
    """
    model.eval()

    all_conf  = []   # Max confidence (scalar per pixel)
    all_pred  = []   # Predicted class
    all_tgt   = []   # Ground truth class
    all_probs = []   # Full probability vectors (C,) per pixel
    all_unc   = []   # Uncertainty (vacuity or variance)

    MAX_PIXELS = 2_000_000  # Cap at 2M pixels to avoid RAM exhaustion

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="  Collecting preds",
                                   leave=False):
            images = images.to(device)

            if is_edl or is_ts:
                if is_ts:
                    out = model(images)          # TemperatureScaler has no return_uncertainty arg
                else:
                    out = model(images, return_uncertainty=True)
                prob = out['prob']
                pred = out['pred']
                unc  = out.get('uncertainty', None)
            else:
                logits = model(images)
                prob   = torch.softmax(logits, dim=1)
                pred   = prob.argmax(dim=1)
                unc    = None

            prob_np = prob.cpu().numpy()     # (B, C, H, W)
            pred_np = pred.cpu().numpy()     # (B, H, W)
            mask_np = masks.numpy()          # (B, H, W)
            unc_np  = unc.cpu().numpy() if unc is not None else None

            B, C, H, W = prob_np.shape

            conf_np = prob_np.max(axis=1)                    # (B, H, W)

            all_conf.append(conf_np.reshape(-1))
            all_pred.append(pred_np.reshape(-1))
            all_tgt.append(mask_np.reshape(-1))
            all_probs.append(prob_np.reshape(B * H * W, C))
            if unc_np is not None:
                all_unc.append(unc_np.reshape(-1))

            if sum(len(a) for a in all_conf) > MAX_PIXELS:
                break

    confidences = np.concatenate(all_conf)[:MAX_PIXELS]
    predictions = np.concatenate(all_pred)[:MAX_PIXELS]
    targets     = np.concatenate(all_tgt)[:MAX_PIXELS]
    probs_flat  = np.concatenate(all_probs, axis=0)[:MAX_PIXELS]
    uncertainties = np.concatenate(all_unc)[:MAX_PIXELS] \
                    if all_unc else None

    return {
        'confidences':   confidences,
        'predictions':   predictions,
        'targets':       targets,
        'probs':         probs_flat,
        'uncertainties': uncertainties,
    }


def compute_standard_ece(confidences, predictions, targets, num_bins=15):
    bins  = np.linspace(0, 1, num_bins + 1)
    ece   = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m    = (confidences > lo) & (confidences <= hi)
        if m.mean() > 0:
            ece += abs(confidences[m].mean() -
                       (predictions[m] == targets[m]).mean()) * m.mean()
    return float(ece)


# ============================================================================
# RELIABILITY DIAGRAM PLOTTING
# ============================================================================

def plot_reliability_diagram(ax, confidences, predictions, targets,
                               title: str, num_bins: int = 15):
    """Plot reliability diagram on given axes."""
    bins  = np.linspace(0, 1, num_bins + 1)
    bin_c, bin_a, bin_w = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confidences > lo) & (confidences <= hi)
        if m.sum() > 0:
            bin_c.append(confidences[m].mean())
            bin_a.append((predictions[m] == targets[m]).mean())
            bin_w.append(m.sum())

    bin_c = np.array(bin_c)
    bin_a = np.array(bin_a)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    ax.bar(bin_c, bin_a, width=1.0 / num_bins, alpha=0.7,
           color='#3498db', align='center', label='Model')
    ax.fill_between(bin_c, bin_c, bin_a, alpha=0.3,
                    where=(bin_a < bin_c), color='#e74c3c', label='Overconfident')
    ax.fill_between(bin_c, bin_c, bin_a, alpha=0.3,
                    where=(bin_a >= bin_c), color='#2ecc71', label='Underconfident')

    ece = compute_standard_ece(confidences, predictions, targets, num_bins)
    ax.set_title(f'{title}\nECE = {ece:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Confidence', fontsize=10)
    ax.set_ylabel('Accuracy',   fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])


def plot_classwise_ece(classwise: dict, title: str, out_path: Path):
    """Bar chart of per-class ECE."""
    names = [k for k, v in classwise.items() if not np.isnan(v)]
    vals  = [v for v in classwise.values() if not np.isnan(v)]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors  = ['#e74c3c' if v > np.mean(vals) else '#2ecc71' for v in vals]
    ax.bar(names, vals, color=colors, edgecolor='white', alpha=0.9)
    ax.axhline(np.mean(vals), color='black', linestyle='--',
               linewidth=1.5, label=f'Mean ECE = {np.mean(vals):.4f}')
    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('ECE', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# ARGS
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config',           required=True)
    p.add_argument('--edl_checkpoint',   required=True)
    p.add_argument('--base_checkpoint',  default=None,
                   help='Optional deterministic baseline checkpoint')
    p.add_argument('--output_dir',       default='outputs/calibration_extended')
    p.add_argument('--temp_lr',          type=float, default=0.01)
    p.add_argument('--temp_epochs',      type=int, default=100)
    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args    = parse_args()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config  = load_config(args.config)
    ignore  = getattr(config.data, 'ignore_index', None)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EXTENDED CALIBRATION ANALYSIS")
    print(f"  Device:   {device}")
    print(f"{'='*60}\n")

    # ---- Dataloaders ----
    val_dataset = get_dataset(config, split='val')
    val_loader  = DataLoader(val_dataset,
                              batch_size=config.training.batch_size,
                              shuffle=False,
                              num_workers=config.training.num_workers,
                              pin_memory=True)

    # ---- Load EDL model ----
    print("  Loading EDL model...")
    edl_model = get_uncertainty_model(
        arch_name    = getattr(config.model, 'arch', 'SegFormer'),
        encoder_name = config.model.encoder,
        num_classes  = config.data.num_classes,
        pretrained   = False
    ).to(device)
    ckpt = torch.load(args.edl_checkpoint, map_location=device,
                       weights_only=False)
    edl_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    print(f"  ✓ EDL model loaded\n")

    # ---- Temperature scaling on EDL ----
    print("  Running temperature scaling on EDL model...")
    ts_model = TemperatureScaler(edl_model, is_edl=True).to(device)
    T_val    = ts_model.calibrate(
        val_loader, device,
        lr=args.temp_lr, num_epochs=args.temp_epochs,
        ignore_index=ignore
    )

    # ---- Collect predictions for all methods ----
    print("\n  Collecting predictions (EDL)...")
    edl_preds = collect_predictions(
        edl_model, val_loader, device, is_edl=True
    )

    print("  Collecting predictions (EDL + Temp Scaling)...")
    ts_preds  = collect_predictions(
        ts_model, val_loader, device, is_edl=True, is_ts=True
    )

    base_preds = None
    if args.base_checkpoint:
        print("  Collecting predictions (Baseline)...")
        base_model = get_uncertainty_model(
            arch_name    = getattr(config.model, 'arch', 'SegFormer'),
            encoder_name = config.model.encoder,
            num_classes  = config.data.num_classes,
            pretrained   = False
        ).to(device)
        base_ckpt = torch.load(args.base_checkpoint, map_location=device,
                                weights_only=False)
        base_model.load_state_dict(base_ckpt.get('model_state_dict', base_ckpt))
        base_preds = collect_predictions(
            base_model, val_loader, device, is_edl=False
        )

    # ---- Apply ignore_index mask ----
    def mask_ignore(preds):
        if ignore is None:
            return preds
        m = preds['targets'] != ignore
        return {k: v[m] if (v is not None and v.ndim == 1)
                else (v[m] if (v is not None and v.ndim == 2) else v)
                for k, v in preds.items()}

    edl_p = mask_ignore(edl_preds)
    ts_p  = mask_ignore(ts_preds)
    base_p = mask_ignore(base_preds) if base_preds else None

    # ---- Compute metrics ----
    print("\n  Computing calibration metrics...")

    def metrics_for(preds, label):
        c, pr, t = preds['confidences'], preds['predictions'], preds['targets']
        ece  = compute_standard_ece(c, pr, t)
        ace  = compute_adaptive_ece(c, pr, t)
        cw   = compute_classwise_ece(preds['probs'], t,
                                      config.data.num_classes,
                                      ignore_index=ignore)
        acc  = (pr == t).mean()
        print(f"\n  [{label}]")
        print(f"    Standard ECE:   {ece:.4f}")
        print(f"    Adaptive ECE:   {ace:.4f}")
        print(f"    Accuracy:       {acc:.4f}")
        print(f"    Class-wise ECE:")
        for cls, val in cw.items():
            if not np.isnan(val):
                print(f"      {cls:<15} {val:.4f}")
        return {'ece': ece, 'ace': ace, 'accuracy': acc, 'classwise_ece': cw}

    results = {}
    results['EDL']              = metrics_for(edl_p,   'EDL (ours)')
    results['EDL+TempScaling']  = metrics_for(ts_p,    'EDL + Temperature Scaling')
    if base_p:
        results['Baseline']     = metrics_for(base_p,  'Baseline')

    results['temperature'] = T_val

    # ---- Print comparison ----
    print(f"\n{'='*60}")
    print(f"  CALIBRATION COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<25} {'ECE':>8} {'ACE':>8} {'Accuracy':>10}")
    print(f"  {'-'*55}")
    for name, r in results.items():
        if isinstance(r, dict) and 'ece' in r:
            print(f"  {name:<25} {r['ece']:>8.4f} {r['ace']:>8.4f} "
                  f"{r['accuracy']:>10.4f}")
    print(f"  {'-'*55}")
    print(f"  Optimal Temperature T = {T_val:.4f}")
    edl_ece = results['EDL']['ece']
    ts_ece  = results['EDL+TempScaling']['ece']
    if ts_ece < edl_ece:
        print(f"  ⚠ Temperature scaling reduces ECE further "
              f"({edl_ece:.4f} → {ts_ece:.4f})")
        print(f"    Note for paper: EDL alone is insufficient for full calibration.")
    else:
        print(f"  ✓ EDL achieves lower ECE than temperature scaling alone.")
    print(f"{'='*60}\n")

    # ---- Reliability diagrams ----
    print("  Generating reliability diagrams...")
    n_panels  = 2 + (1 if base_p else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    plot_reliability_diagram(axes[0], edl_p['confidences'],
                              edl_p['predictions'], edl_p['targets'],
                              'EDL (Ours)')
    plot_reliability_diagram(axes[1], ts_p['confidences'],
                              ts_p['predictions'], ts_p['targets'],
                              f'EDL + Temp Scaling (T={T_val:.3f})')
    if base_p:
        plot_reliability_diagram(axes[2], base_p['confidences'],
                                  base_p['predictions'], base_p['targets'],
                                  'Baseline (Deterministic)')

    fig.suptitle('Reliability Diagrams — DeepGlobe Validation Set',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.savefig(out_dir / 'reliability_diagrams.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  ✓ Reliability diagrams saved")

    # ---- Class-wise ECE plots ----
    plot_classwise_ece(
        results['EDL']['classwise_ece'],
        'Per-Class ECE — EDL (Ours)',
        out_dir / 'classwise_ece_edl.png'
    )
    plot_classwise_ece(
        results['EDL+TempScaling']['classwise_ece'],
        'Per-Class ECE — EDL + Temperature Scaling',
        out_dir / 'classwise_ece_ts.png'
    )
    print(f"  ✓ Class-wise ECE plots saved")

    # ---- Save all results ----
    # Convert classwise dicts for JSON serialisation
    json_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            entry = {kk: vv for kk, vv in v.items()
                     if kk != 'classwise_ece'}
            if 'classwise_ece' in v:
                entry['classwise_ece'] = {
                    cls: (float(val) if not np.isnan(val) else None)
                    for cls, val in v['classwise_ece'].items()
                }
            json_results[k] = entry
        else:
            json_results[k] = v

    json_path = out_dir / 'calibration_extended_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  CALIBRATION ANALYSIS COMPLETE")
    print(f"  Results:  {json_path}")
    print(f"  Plots:    {out_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()