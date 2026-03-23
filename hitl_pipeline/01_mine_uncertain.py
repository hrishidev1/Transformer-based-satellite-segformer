"""
HITL Script 1 of 4 — Uncertainty Mining
=========================================

WHAT THIS DOES:
  You have a folder of raw, unlabelled satellite images the model has never seen.
  This script runs every image through your trained model and computes the mean
  vacuity (epistemic uncertainty) for each one.

  It then ranks all images from most confused → least confused and saves the
  top-K hardest images into an output folder, alongside:
    - The original satellite image (for you to label)
    - A colour uncertainty heatmap (so you know EXACTLY which pixels confused it)
    - The model's current best-guess prediction (as a coloured mask)
    - A side-by-side summary panel (image | prediction | uncertainty)

  After running this script, you open the output folder, look at the images the
  model is most confused about, label them in CVAT or LabelMe, and then run
  Script 3 to teach the model from your corrections.

FOLDER STRUCTURE EXPECTED:
  unlabeled_images/
      any_name_1.png
      any_name_2.jpg
      ...

  Images can be any size. They will be resized to 512×512 for inference
  (same as training). The ORIGINAL image is saved for labelling.

HOW TO RUN:
  python hitl_pipeline/01_mine_uncertain.py \\
      --config    configs/uncertain_segformer.yaml \\
      --checkpoint /path/to/best_miou.pth \\
      --input_dir  /path/to/unlabeled_images \\
      --output_dir /path/to/save/hard_samples \\
      --top_k      50

ARGUMENTS:
  --config      Your existing YAML config
  --checkpoint  Trained model checkpoint (best_miou.pth)
  --input_dir   Folder of raw unlabelled satellite images
  --output_dir  Where to save ranked uncertain images + heatmaps
  --top_k       How many hard images to extract (default: 50)
  --batch_size  Inference batch size (default: 8, safe for A100)

OUTPUT FILES (per image saved):
  rank_001_score_0.8432_imagename.png        ← original image (label this one)
  rank_001_heatmap_imagename.png             ← uncertainty heatmap (red = confused)
  rank_001_prediction_imagename.png          ← model's current guess
  rank_001_summary_imagename.png             ← side-by-side panel
  uncertainty_scores.json                    ← full ranked list with scores
  mining_report.txt                          ← human-readable summary
"""

import sys
import os
import json
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# --- Path fix (works on Colab and local) ---
FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent   # hitl_pipeline/ → project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------

from utils.config import load_config
from models.uncertainty_factory import get_uncertainty_model


# ============================================================================
# DEEPGLOBE COLOUR MAP  (matches your training data exactly)
# ============================================================================

CLASS_COLORS = np.array([
    [0,   255, 255],   # 0 Urban       - Cyan
    [255, 255,   0],   # 1 Agriculture - Yellow
    [255,   0, 255],   # 2 Rangeland   - Magenta
    [0,   255,   0],   # 3 Forest      - Green
    [0,     0, 255],   # 4 Water       - Blue
    [255, 255, 255],   # 5 Barren      - White
    [0,     0,   0],   # 6 Unknown     - Black
], dtype=np.uint8)

CLASS_NAMES = [
    'Urban', 'Agriculture', 'Rangeland',
    'Forest', 'Water', 'Barren', 'Unknown'
]

# Supported image extensions
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Mine most uncertain images for human labelling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--config',      required=True,
                   help='Path to YAML config')
    p.add_argument('--checkpoint',  required=True,
                   help='Path to trained model checkpoint')
    p.add_argument('--input_dir',   required=True,
                   help='Folder of unlabelled satellite images')
    p.add_argument('--output_dir',  default='hitl_outputs/round_01/to_label',
                   help='Where to save hard samples')
    p.add_argument('--top_k',       type=int, default=50,
                   help='Number of hardest images to extract')
    p.add_argument('--batch_size',  type=int, default=8,
                   help='Inference batch size')
    return p.parse_args()


# ============================================================================
# IMAGE PREPROCESSING
# Matches your training pipeline exactly (same normalisation values)
# ============================================================================

def get_inference_transform(image_size: int = 512):
    """
    Inference transform — resize + normalise.
    Must match training normalisation or uncertainty scores are meaningless.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def load_image_paths(input_dir: str):
    """Collect all image paths from input directory."""
    paths = []
    for f in sorted(Path(input_dir).iterdir()):
        if f.suffix.lower() in IMG_EXTENSIONS:
            paths.append(f)
    return paths


# ============================================================================
# INFERENCE — run batched inference and collect uncertainty scores
# ============================================================================

def run_inference_on_all(model, image_paths, transform, device, batch_size):
    """
    Run inference on every image. Returns a list of dicts:
      [
        {
          'path':        Path object,
          'mean_vacuity': float,       ← THE SCORE we rank by
          'vacuity_map':  np.array (H, W),
          'pred_mask':    np.array (H, W),
        },
        ...
      ]

    WHY WE USE MEAN VACUITY:
      Vacuity u = K / S where K=num_classes and S = sum of Dirichlet alphas.
      High S means the model collected lots of evidence → confident.
      Low  S means barely any evidence → confused → high vacuity.
      Mean over all pixels = how confused the model is about the whole image.
    """
    model.eval()
    results = []

    # Process in batches for speed
    for batch_start in tqdm(
        range(0, len(image_paths), batch_size),
        desc="  Running inference"
    ):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        batch_tensors = []
        valid_paths   = []

        for path in batch_paths:
            try:
                img = np.array(Image.open(path).convert('RGB'))
                aug = transform(image=img)
                batch_tensors.append(aug['image'])
                valid_paths.append(path)
            except Exception as e:
                print(f"  ⚠ Skipping {path.name}: {e}")
                continue

        if not batch_tensors:
            continue

        batch_tensor = torch.stack(batch_tensors).to(device)   # (B, 3, H, W)

        with torch.no_grad():
            # ✓ Uses your UncertainWrapper correctly — output is a dict
            output = model(batch_tensor, return_uncertainty=True)

        # uncertainty shape: (B, H, W)
        vacuity   = output['uncertainty'].cpu().numpy()    # (B, H, W)
        pred_mask = output['pred'].cpu().numpy()           # (B, H, W)

        for i, path in enumerate(valid_paths):
            results.append({
                'path':         path,
                'mean_vacuity': float(vacuity[i].mean()),
                'vacuity_map':  vacuity[i],       # (H, W) — values in [0, 1]
                'pred_mask':    pred_mask[i],      # (H, W) — class IDs
            })

    return results


# ============================================================================
# VISUALISATION HELPERS
# ============================================================================

def decode_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert class ID mask (H, W) → RGB image (H, W, 3)."""
    h, w = mask.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[mask == cls_id] = color
    return rgb


def make_uncertainty_heatmap(vacuity_map: np.ndarray) -> np.ndarray:
    """
    Convert vacuity map (H, W) to a JET colourmap heatmap (H, W, 3).
    Red = high uncertainty (model is confused here)
    Blue = low uncertainty (model is confident here)
    """
    # Normalise to 0–255
    v_min, v_max = vacuity_map.min(), vacuity_map.max()
    if v_max > v_min:
        norm = ((vacuity_map - v_min) / (v_max - v_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(vacuity_map, dtype=np.uint8)

    heatmap_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def save_summary_panel(
    original_img: np.ndarray,
    pred_rgb:     np.ndarray,
    heatmap_rgb:  np.ndarray,
    vacuity_map:  np.ndarray,
    rank:         int,
    score:        float,
    image_name:   str,
    save_path:    Path
):
    """
    Save a 3-panel figure: Original | Prediction | Uncertainty Heatmap
    This is the file you look at to understand what the model got wrong.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Rank #{rank} | Mean Vacuity: {score:.4f} | {image_name}',
        fontsize=13, fontweight='bold'
    )

    # Panel 1: Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Input Image (Label This)', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Model prediction
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Model's Current Prediction", fontsize=11, fontweight='bold')
    # Add legend
    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255.0, label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ]
    axes[1].legend(
        handles=patches, loc='lower left', fontsize=7,
        framealpha=0.8, ncol=2
    )
    axes[1].axis('off')

    # Panel 3: Uncertainty heatmap with overlay
    axes[2].imshow(original_img)
    im = axes[2].imshow(
        vacuity_map, cmap='Reds', alpha=0.65,
        vmin=0, vmax=vacuity_map.max()
    )
    axes[2].set_title(
        'Uncertainty Map\n(Red = Confused, Label these areas carefully)',
        fontsize=11, fontweight='bold'
    )
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  HITL STEP 1 — UNCERTAINTY MINING")
    print(f"  Device:     {device}")
    print(f"  Input:      {args.input_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Top-K:      {args.top_k}")
    print(f"{'='*60}\n")

    # ---- Load config and model ----
    config = load_config(args.config)

    print("  Loading model...")
    model = get_uncertainty_model(
        arch_name   = getattr(config.model, 'arch', 'SegFormer'),
        encoder_name= config.model.encoder,
        num_classes = config.data.num_classes,
        pretrained  = False    # We load our own weights below
    ).to(device)

    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state     = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"  ✓ Model loaded from {args.checkpoint}")

    # ---- Collect image paths ----
    image_paths = load_image_paths(args.input_dir)
    if not image_paths:
        print(f"  ✗ No images found in {args.input_dir}")
        print(f"    Supported formats: {IMG_EXTENSIONS}")
        return

    print(f"  Found {len(image_paths)} images to score\n")

    # ---- Run inference ----
    transform = get_inference_transform(config.data.image_size)
    results   = run_inference_on_all(
        model, image_paths, transform, device, args.batch_size
    )

    # ---- Rank by uncertainty ----
    results.sort(key=lambda x: x['mean_vacuity'], reverse=True)

    print(f"\n  Scored {len(results)} images.")
    print(f"  Highest vacuity: {results[0]['mean_vacuity']:.4f}")
    print(f"  Lowest  vacuity: {results[-1]['mean_vacuity']:.4f}")
    print(f"  Mean   vacuity:  {np.mean([r['mean_vacuity'] for r in results]):.4f}")

    # ---- Save top-K ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_k   = min(args.top_k, len(results))
    scores_log = []

    print(f"\n  Saving top {top_k} most uncertain images...\n")

    for rank, entry in enumerate(results[:top_k], start=1):
        path       = entry['path']
        score      = entry['mean_vacuity']
        vac_map    = entry['vacuity_map']
        pred_mask  = entry['pred_mask']
        stem       = path.stem
        tag        = f"rank_{rank:03d}_score_{score:.4f}"

        # Load original image at native resolution for labelling
        orig_img = np.array(Image.open(path).convert('RGB'))

        # Resize prediction and heatmap back to original image size for alignment
        h, w     = orig_img.shape[:2]
        pred_r   = cv2.resize(pred_mask.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST)
        vac_r    = cv2.resize(vac_map, (w, h),
                               interpolation=cv2.INTER_LINEAR)

        pred_rgb   = decode_mask_to_rgb(pred_r)
        heatmap    = make_uncertainty_heatmap(vac_r)

        # 1. Original image  (THIS is the file you label in CVAT)
        shutil.copy(path, out_dir / f"{tag}_LABEL_THIS_{stem}{path.suffix}")

        # 2. Uncertainty heatmap
        Image.fromarray(heatmap).save(out_dir / f"{tag}_heatmap_{stem}.png")

        # 3. Prediction mask
        Image.fromarray(pred_rgb).save(out_dir / f"{tag}_prediction_{stem}.png")

        # 4. Summary panel (the most useful file — open this first)
        save_summary_panel(
            original_img = orig_img,
            pred_rgb     = pred_rgb,
            heatmap_rgb  = heatmap,
            vacuity_map  = vac_r,
            rank         = rank,
            score        = score,
            image_name   = path.name,
            save_path    = out_dir / f"{tag}_SUMMARY_{stem}.png"
        )

        scores_log.append({
            'rank':         rank,
            'filename':     path.name,
            'mean_vacuity': score,
            'max_vacuity':  float(vac_map.max()),
        })

        print(f"  [{rank:>3}/{top_k}] {path.name:<40} vacuity: {score:.4f}")

    # ---- Save full scores JSON ----
    json_path = out_dir / 'uncertainty_scores.json'
    with open(json_path, 'w') as f:
        json.dump({'top_k': scores_log,
                   'all':   [{'filename': r['path'].name,
                               'mean_vacuity': r['mean_vacuity']}
                              for r in results]}, f, indent=2)

    # ---- Write human-readable report ----
    report_path = out_dir / 'mining_report.txt'
    with open(report_path, 'w') as f:
        f.write("UNCERTAINTY MINING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total images scanned:   {len(results)}\n")
        f.write(f"Images selected:        {top_k}\n")
        f.write(f"Mean vacuity (all):     {np.mean([r['mean_vacuity'] for r in results]):.4f}\n")
        f.write(f"Mean vacuity (top-K):   {np.mean([r['mean_vacuity'] for r in results[:top_k]]):.4f}\n\n")
        f.write("WHAT TO DO NEXT:\n")
        f.write("-"*60 + "\n")
        f.write("1. Open the '_SUMMARY_' PNG files in this folder.\n")
        f.write("   Panel 1 = original image\n")
        f.write("   Panel 2 = model's current prediction (check where it's wrong)\n")
        f.write("   Panel 3 = uncertainty map (red = model is confused here)\n\n")
        f.write("2. Open the '_LABEL_THIS_' images in CVAT or LabelMe.\n")
        f.write("   Draw precise masks for each land cover class.\n")
        f.write("   Pay special attention to the RED areas in the heatmap.\n\n")
        f.write("3. Export your labels as PNG masks using the DeepGlobe colour scheme:\n")
        f.write("   Urban=Cyan(0,255,255), Agriculture=Yellow(255,255,0)\n")
        f.write("   Rangeland=Magenta(255,0,255), Forest=Green(0,255,0)\n")
        f.write("   Water=Blue(0,0,255), Barren=White(255,255,255)\n")
        f.write("   Unknown=Black(0,0,0)\n\n")
        f.write("4. Run: python hitl_pipeline/02_prepare_new_labels.py\n")
        f.write("   to validate and integrate your labels.\n\n")
        f.write(f"5. Run: python hitl_pipeline/03_finetune_hitl.py\n")
        f.write("   to teach the model from your corrections.\n\n")
        f.write("RANKED LIST:\n")
        f.write("-"*60 + "\n")
        for entry in scores_log:
            f.write(f"  Rank {entry['rank']:>3}: {entry['filename']:<40} "
                    f"vacuity={entry['mean_vacuity']:.4f}\n")

    print(f"\n{'='*60}")
    print(f"  MINING COMPLETE")
    print(f"  {top_k} images saved to: {out_dir}")
    print(f"  Full scores:    {json_path.name}")
    print(f"  Report:         {report_path.name}")
    print(f"\n  NEXT STEP:")
    print(f"  Open the _SUMMARY_ panels, label the _LABEL_THIS_ images,")
    print(f"  then run 02_prepare_new_labels.py")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()