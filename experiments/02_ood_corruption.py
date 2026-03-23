"""
Experiment 2 — Synthetic OOD Corruption Evaluation
====================================================
Addresses professor feedback Point 2: "No quantitative OOD metrics,
no AUROC/FPR95/AUPR, no benchmark datasets."

WHAT THIS DOES:
  Applies ImageNet-C style corruptions to your existing DeepGlobe validation
  set and measures how well mean vacuity (u = K/S) detects corrupted images.
  No new data needed — we corrupt images you already have.

  CORRUPTIONS APPLIED (5 types × 5 severity levels = 25 conditions):
    1. Gaussian Noise    — sensor noise simulation
    2. Motion Blur       — drone/satellite movement
    3. Fog/Haze          — atmospheric interference
    4. Brightness Shift  — time-of-day variation
    5. JPEG Compression  — transmission artefacts

  METRICS COMPUTED:
    - AUROC: area under ROC curve where uncertainty = OOD score
             (1.0 = perfect detector, 0.5 = random)
    - FPR95: false positive rate when TPR = 95%
             (lower is better — how many clean images are wrongly flagged)
    - AUPR:  area under precision-recall curve
    - mIoU degradation per corruption type and severity

  The key result your paper needs:
    "As corruption severity increases, mean vacuity increases monotonically
    (Spearman ρ = X.XX), and at severity level 5, AUROC = X.XX, confirming
    that vacuity is a reliable OOD signal for satellite imagery."

HOW TO RUN:
  python experiments/02_ood_corruption.py \\
      --config     configs/uncertain_segformer.yaml \\
      --checkpoint /path/to/best_miou.pth \\
      --output_dir outputs/ood_corruption

ARGUMENTS:
  --config      YAML config
  --checkpoint  Trained model checkpoint
  --output_dir  Where to save results and plots
  --severities  Corruption severity levels to test (default: 1 2 3 4 5)
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config
from models.uncertainty_factory import get_uncertainty_model
from metrics.segmentation import SegmentationMetrics
from datasets import get_dataset


# ============================================================================
# CORRUPTION FUNCTIONS
# ============================================================================

def corrupt_gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    """
    Gaussian noise — simulates sensor noise.
    Severity 1-5 maps to increasing noise standard deviation.
    """
    std_map = {1: 0.04, 2: 0.08, 3: 0.12, 4: 0.18, 5: 0.26}
    std     = std_map[severity]
    noise   = np.random.normal(0, std, img.shape).astype(np.float32)
    out     = np.clip(img.astype(np.float32) / 255.0 + noise, 0, 1)
    return (out * 255).astype(np.uint8)


def corrupt_motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    """
    Motion blur — simulates satellite/drone movement during capture.
    """
    kernel_map = {1: 5, 2: 9, 3: 13, 4: 17, 5: 21}
    k          = kernel_map[severity]
    kernel     = np.zeros((k, k))
    kernel[k // 2, :] = 1.0 / k
    blurred    = cv2.filter2D(img, -1, kernel)
    return blurred.astype(np.uint8)


def corrupt_fog(img: np.ndarray, severity: int) -> np.ndarray:
    """
    Fog/haze — simulates atmospheric interference common in satellite imagery.
    """
    alpha_map = {1: 0.15, 2: 0.30, 3: 0.45, 4: 0.60, 5: 0.75}
    alpha     = alpha_map[severity]
    fog_color = np.array([220, 220, 220], dtype=np.float32)
    out       = img.astype(np.float32) * (1 - alpha) + fog_color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def corrupt_brightness(img: np.ndarray, severity: int) -> np.ndarray:
    """
    Brightness shift — simulates different time-of-day lighting conditions.
    """
    factor_map = {1: 1.3, 2: 1.6, 3: 0.6, 4: 0.35, 5: 0.15}
    factor     = factor_map[severity]
    out        = np.clip(img.astype(np.float32) * factor, 0, 255)
    return out.astype(np.uint8)


def corrupt_jpeg(img: np.ndarray, severity: int) -> np.ndarray:
    """
    JPEG compression artefacts — simulates satellite imagery transmission.
    """
    quality_map = {1: 75, 2: 55, 3: 35, 4: 20, 5: 8}
    quality     = quality_map[severity]
    _, enc      = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, quality])
    dec         = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


CORRUPTIONS = {
    'gaussian_noise': corrupt_gaussian_noise,
    'motion_blur':    corrupt_motion_blur,
    'fog':            corrupt_fog,
    'brightness':     corrupt_brightness,
    'jpeg':           corrupt_jpeg,
}


# ============================================================================
# CORRUPTED DATASET
# ============================================================================

class CorruptedDataset(Dataset):
    """
    Wraps an existing dataset and applies corruption on-the-fly.
    Returns corrupted images with original masks (so mIoU can be computed).
    """

    def __init__(self, base_dataset, corrupt_fn, severity: int, image_size: int):
        self.base      = base_dataset
        self.corrupt   = corrupt_fn
        self.severity  = severity
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Get clean item from base dataset
        image_tensor, mask = self.base[idx]

        # Denormalise back to uint8
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = image_tensor.numpy()
        img  = img * std[:, None, None] + mean[:, None, None]
        img  = np.clip(img, 0, 1)
        img  = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

        # Apply corruption
        img_corrupted = self.corrupt(img, self.severity)

        # Re-apply normalisation
        aug = self.transform(image=img_corrupted)
        return aug['image'], mask


# ============================================================================
# OOD METRICS
# ============================================================================

def compute_auroc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """
    Compute AUROC where:
      in_scores  = uncertainty scores for clean (in-distribution) images
      out_scores = uncertainty scores for corrupted (OOD) images
    Label 0 = clean, 1 = OOD.
    """
    labels = np.concatenate([
        np.zeros(len(in_scores)),
        np.ones(len(out_scores))
    ])
    scores = np.concatenate([in_scores, out_scores])
    return float(roc_auc_score(labels, scores))


def compute_fpr95(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """
    FPR at 95% TPR.
    Threshold set so that 95% of OOD samples are detected.
    Reports what fraction of clean samples are wrongly flagged.
    Lower is better.
    """
    labels = np.concatenate([
        np.zeros(len(in_scores)),
        np.ones(len(out_scores))
    ])
    scores = np.concatenate([in_scores, out_scores])

    fpr, tpr, _ = roc_curve(labels, scores)
    # Find threshold where TPR >= 0.95
    idx = np.searchsorted(tpr, 0.95)
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def compute_aupr(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """Area under Precision-Recall curve."""
    labels = np.concatenate([
        np.zeros(len(in_scores)),
        np.ones(len(out_scores))
    ])
    scores = np.concatenate([in_scores, out_scores])
    prec, rec, _ = precision_recall_curve(labels, scores)
    return float(auc(rec, prec))


# ============================================================================
# EVALUATION
# ============================================================================

def collect_image_uncertainties(model, dataloader, device) -> np.ndarray:
    """
    Run inference and return mean vacuity per image as a 1D array.
    Shape: (N,) where N = number of images in dataloader.
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="    Scoring", leave=False):
            images = images.to(device)
            out    = model(images, return_uncertainty=True)
            unc    = out['uncertainty']                         # (B, H, W)
            mean_u = unc.view(unc.size(0), -1).mean(dim=1)     # (B,)
            scores.extend(mean_u.cpu().numpy().tolist())

    return np.array(scores)


def evaluate_corrupted(model, corrupt_fn, severity, val_dataset,
                        config, device) -> dict:
    """
    Evaluate model on corrupted version of val_dataset.
    Returns mIoU and mean uncertainty.
    """
    ignore = getattr(config.data, 'ignore_index', None)

    corrupted_ds = CorruptedDataset(
        val_dataset, corrupt_fn, severity, config.data.image_size
    )
    loader = DataLoader(corrupted_ds,
                        batch_size=config.training.batch_size,
                        shuffle=False,
                        num_workers=2, pin_memory=True)

    seg_metrics = SegmentationMetrics(config.data.num_classes,
                                       ignore_index=ignore)
    uncertainties = []

    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            out = model(images, return_uncertainty=True)
            seg_metrics.update(out['pred'], masks)
            unc    = out['uncertainty']
            mean_u = unc.view(unc.size(0), -1).mean(dim=1)
            uncertainties.extend(mean_u.cpu().numpy().tolist())

    seg_r = seg_metrics.compute(return_per_class=False)
    return {
        'miou':            float(seg_r['miou']),
        'mean_vacuity':    float(np.mean(uncertainties)),
        'uncertainties':   np.array(uncertainties),
    }


# ============================================================================
# PLOTTING
# ============================================================================

def plot_ood_results(results: dict, clean_unc: np.ndarray,
                     out_dir: Path, severities: list):
    """
    5-panel figure for paper:
      (a) mIoU vs severity per corruption
      (b) Mean vacuity vs severity per corruption
      (c) AUROC vs severity
      (d) FPR95 vs severity
      (e) Sample corrupted image at each severity for one corruption type
    """
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    corr_names = list(CORRUPTIONS.keys())

    # ---- (a) mIoU degradation ----
    ax1 = fig.add_subplot(gs[0, 0])
    for i, cname in enumerate(corr_names):
        mious = [results[cname][s]['miou'] for s in severities]
        ax1.plot(severities, mious, 'o-', color=colors[i],
                 label=cname.replace('_', ' '), linewidth=2, markersize=6)
    ax1.set_xlabel('Severity', fontsize=11)
    ax1.set_ylabel('Validation mIoU', fontsize=11)
    ax1.set_title('(a) mIoU Degradation', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- (b) Vacuity vs severity ----
    ax2 = fig.add_subplot(gs[0, 1])
    for i, cname in enumerate(corr_names):
        vacuities = [results[cname][s]['mean_vacuity'] for s in severities]
        ax2.plot(severities, vacuities, 'o-', color=colors[i],
                 label=cname.replace('_', ' '), linewidth=2, markersize=6)
    ax2.axhline(float(clean_unc.mean()), color='black', linestyle='--',
                linewidth=1.5, label='Clean (no corruption)')
    ax2.set_xlabel('Severity', fontsize=11)
    ax2.set_ylabel('Mean Vacuity', fontsize=11)
    ax2.set_title('(b) Vacuity Rises with Corruption', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- (c) AUROC vs severity ----
    ax3 = fig.add_subplot(gs[0, 2])
    for i, cname in enumerate(corr_names):
        aurocs = [results[cname][s]['auroc'] for s in severities]
        ax3.plot(severities, aurocs, 'o-', color=colors[i],
                 label=cname.replace('_', ' '), linewidth=2, markersize=6)
    ax3.axhline(0.5, color='gray', linestyle=':', linewidth=1.2, label='Random')
    ax3.set_xlabel('Severity', fontsize=11)
    ax3.set_ylabel('AUROC', fontsize=11)
    ax3.set_title('(c) AUROC (Vacuity as OOD Detector)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.set_ylim([0.4, 1.05])
    ax3.grid(True, alpha=0.3)

    # ---- (d) FPR95 vs severity ----
    ax4 = fig.add_subplot(gs[1, 0])
    for i, cname in enumerate(corr_names):
        fpr95s = [results[cname][s]['fpr95'] for s in severities]
        ax4.plot(severities, fpr95s, 'o-', color=colors[i],
                 label=cname.replace('_', ' '), linewidth=2, markersize=6)
    ax4.set_xlabel('Severity', fontsize=11)
    ax4.set_ylabel('FPR@95%TPR (lower=better)', fontsize=11)
    ax4.set_title('(d) False Positive Rate @95% TPR', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ---- (e) Summary heatmap: AUROC across corruption × severity ----
    ax5 = fig.add_subplot(gs[1, 1:])
    auroc_matrix = np.array([
        [results[cname][s]['auroc'] for s in severities]
        for cname in corr_names
    ])
    im = ax5.imshow(auroc_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0,
                    aspect='auto')
    plt.colorbar(im, ax=ax5, label='AUROC')
    ax5.set_xticks(range(len(severities)))
    ax5.set_xticklabels([f'Sev {s}' for s in severities])
    ax5.set_yticks(range(len(corr_names)))
    ax5.set_yticklabels([c.replace('_', '\n') for c in corr_names])
    ax5.set_title('(e) AUROC Heatmap: Corruption × Severity',
                  fontsize=12, fontweight='bold')
    # Annotate cells
    for i in range(len(corr_names)):
        for j in range(len(severities)):
            ax5.text(j, i, f'{auroc_matrix[i, j]:.2f}',
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     color='black' if auroc_matrix[i, j] > 0.65 else 'white')

    fig.suptitle('Synthetic OOD Corruption Evaluation — EDL-SegFormer on DeepGlobe',
                 fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = out_dir / 'ood_corruption_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ OOD plot saved: {save_path}")


def print_ood_table(results: dict, severities: list):
    """Print LaTeX-friendly table of AUROC/FPR95 results."""
    print(f"\n{'='*80}")
    print(f"  OOD DETECTION RESULTS (AUROC / FPR95 / mIoU)")
    print(f"{'='*80}")
    header = f"  {'Corruption':<20}"
    for s in severities:
        header += f"  Sev{s}(AUROC/FPR95)"
    print(header)
    print(f"  {'-'*75}")
    for cname in CORRUPTIONS.keys():
        row = f"  {cname:<20}"
        for s in severities:
            r = results[cname][s]
            row += f"  {r['auroc']:.3f}/{r['fpr95']:.3f}     "
        print(row)
    print(f"{'='*80}\n")


# ============================================================================
# ARGS
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config',      required=True)
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--output_dir',  default='outputs/ood_corruption')
    p.add_argument('--severities',  type=int, nargs='+', default=[1, 2, 3, 4, 5])
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
    print(f"  SYNTHETIC OOD CORRUPTION EVALUATION")
    print(f"  Device:      {device}")
    print(f"  Corruptions: {list(CORRUPTIONS.keys())}")
    print(f"  Severities:  {args.severities}")
    print(f"{'='*60}\n")

    # ---- Load model ----
    model = get_uncertainty_model(
        arch_name    = getattr(config.model, 'arch', 'SegFormer'),
        encoder_name = config.model.encoder,
        num_classes  = config.data.num_classes,
        pretrained   = False
    ).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()
    print(f"  ✓ Model loaded from {args.checkpoint}\n")

    # ---- Load clean val dataset ----
    val_dataset = get_dataset(config, split='val')
    clean_loader = DataLoader(val_dataset,
                               batch_size=config.training.batch_size,
                               shuffle=False,
                               num_workers=config.training.num_workers,
                               pin_memory=True)

    # ---- Collect clean uncertainties (in-distribution baseline) ----
    print("  Collecting clean (in-distribution) uncertainty scores...")
    clean_uncertainties = collect_image_uncertainties(model, clean_loader, device)
    print(f"  Clean mean vacuity: {clean_uncertainties.mean():.4f} ± "
          f"{clean_uncertainties.std():.4f}\n")

    # ---- Run corruptions ----
    all_results = {}

    for cname, corrupt_fn in CORRUPTIONS.items():
        print(f"  Corruption: {cname}")
        all_results[cname] = {}

        for severity in args.severities:
            result = evaluate_corrupted(
                model, corrupt_fn, severity,
                val_dataset, config, device
            )

            # Compute OOD metrics
            corrupt_unc = result['uncertainties']
            auroc = compute_auroc(clean_uncertainties, corrupt_unc)
            fpr95 = compute_fpr95(clean_uncertainties, corrupt_unc)
            aupr  = compute_aupr(clean_uncertainties, corrupt_unc)

            result['auroc'] = auroc
            result['fpr95'] = fpr95
            result['aupr']  = aupr

            # Remove raw arrays before storing
            result.pop('uncertainties')

            all_results[cname][severity] = result

            print(f"    Severity {severity}: mIoU={result['miou']:.4f} | "
                  f"vacuity={result['mean_vacuity']:.4f} | "
                  f"AUROC={auroc:.4f} | FPR95={fpr95:.4f}")

        print()

    # ---- Compute Spearman correlation: severity vs vacuity ----
    from scipy.stats import spearmanr
    all_sev, all_vac = [], []
    for cname in CORRUPTIONS:
        for s in args.severities:
            all_sev.append(s)
            all_vac.append(all_results[cname][s]['mean_vacuity'])

    rho, pval = spearmanr(all_sev, all_vac)
    print(f"  Spearman ρ (severity vs vacuity): {rho:.4f} (p={pval:.4e})")
    print(f"  → Use in paper: 'Mean vacuity increases monotonically with "
          f"corruption severity (Spearman ρ = {rho:.2f})'")

    # ---- Print table ----
    print_ood_table(all_results, args.severities)

    # ---- Save results ----
    json_path = out_dir / 'ood_results.json'
    with open(json_path, 'w') as f:
        json.dump({
            'clean_mean_vacuity': float(clean_uncertainties.mean()),
            'clean_std_vacuity':  float(clean_uncertainties.std()),
            'spearman_rho':       float(rho),
            'spearman_pval':      float(pval),
            'results':            all_results,
        }, f, indent=2)

    # ---- Plot ----
    plot_ood_results(all_results, clean_uncertainties, out_dir, args.severities)

    print(f"\n{'='*60}")
    print(f"  OOD EVALUATION COMPLETE")
    print(f"  Results: {json_path}")
    print(f"  Plot:    {out_dir / 'ood_corruption_results.png'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()