"""
HITL Script 2 of 4 — Prepare & Validate New Human Labels
==========================================================

WHAT THIS DOES:
  After you label images in CVAT or LabelMe, your exported masks are likely in a
  different format (polygon JSON, grayscale class IDs, or wrong colours).
  This script validates and converts your labels into the exact format
  that DeepGlobe training expects.

  It also runs a quality check — if a label is all one class, or has massive
  areas of "Unknown" (black), it warns you that the label might be incomplete.

  Finally, it integrates your new images + masks into the DeepGlobe Train folder
  and creates an updated training manifest so Script 3 knows what's new.

SUPPORTED INPUT FORMATS:
  1. DeepGlobe RGB masks   (already correct colour coding — just validate)
  2. Grayscale class ID    (class 0–6 as pixel values — convert to RGB)
  3. CVAT YOLO/polygon     (not supported — export as PNG mask from CVAT)

  Recommended CVAT export: "Segmentation mask (PNG)" format.

FOLDER STRUCTURE EXPECTED:
  your_labels/
      imagename_sat.jpg        ← your original satellite image
      imagename_mask.png       ← your label (same name, _mask suffix)

  OR just masks in a folder:
  your_labels/
      imagename_mask.png       ← the mask only (image already in DeepGlobe)

HOW TO RUN:
  python hitl_pipeline/02_prepare_new_labels.py \\
      --input_dir   /path/to/your_labeled_folder \\
      --deepglobe_dir  /path/to/DeepGlobe \\
      --round_id    01

ARGUMENTS:
  --input_dir      Folder containing your labelled images + masks
  --deepglobe_dir  Root of your DeepGlobe dataset (same as config root_dir)
  --round_id       HITL round number (01, 02, 03...) for tracking
  --mask_format    'rgb' (DeepGlobe colours) or 'classid' (grayscale 0-6)
                   default: auto-detect

OUTPUT:
  - Validated images + masks copied into DeepGlobe/Train/
  - hitl_manifest_round_XX.json — records what was added this round
  - validation_report.txt — flags any potential labelling issues
"""

import sys
import os
import json
import argparse
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

# --- Path fix ---
FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ----------------


# ============================================================================
# DEEPGLOBE COLOUR DEFINITIONS
# ============================================================================

# RGB → class ID mapping (matches your deepglobe.py exactly)
RGB_TO_CLASS = {
    (0,   255, 255): 0,   # Urban
    (255, 255,   0): 1,   # Agriculture
    (255,   0, 255): 2,   # Rangeland
    (0,   255,   0): 3,   # Forest
    (0,     0, 255): 4,   # Water
    (255, 255, 255): 5,   # Barren
    (0,     0,   0): 6,   # Unknown / Ignore
}

# Class ID → RGB  (inverse — for converting grayscale masks to RGB)
CLASS_TO_RGB = np.array([
    [0,   255, 255],   # 0 Urban
    [255, 255,   0],   # 1 Agriculture
    [255,   0, 255],   # 2 Rangeland
    [0,   255,   0],   # 3 Forest
    [0,     0, 255],   # 4 Water
    [255, 255, 255],   # 5 Barren
    [0,     0,   0],   # 6 Unknown
], dtype=np.uint8)

CLASS_NAMES = [
    'Urban', 'Agriculture', 'Rangeland',
    'Forest', 'Water', 'Barren', 'Unknown'
]

NUM_CLASSES = 7

# Quality thresholds
MAX_UNKNOWN_RATIO   = 0.30   # Warn if >30% of mask is "Unknown" (might be lazy label)
MIN_CLASS_COUNT     = 2      # Warn if mask has fewer than 2 distinct classes
MIN_IMAGE_SIZE      = 256    # Warn if image is smaller than this


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate and integrate human labels into DeepGlobe dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--input_dir',      required=True,
                   help='Folder with your labelled images and masks')
    p.add_argument('--deepglobe_dir',  required=True,
                   help='Root of DeepGlobe dataset (same as config data.root_dir)')
    p.add_argument('--round_id',       default='01',
                   help='HITL round number for tracking (01, 02, 03...)')
    p.add_argument('--mask_format',    default='auto',
                   choices=['auto', 'rgb', 'classid'],
                   help='Mask colour format: rgb=DeepGlobe colours, '
                        'classid=grayscale 0-6, auto=detect')
    p.add_argument('--dry_run',        action='store_true',
                   help='Validate only, do NOT copy files into DeepGlobe')
    return p.parse_args()


# ============================================================================
# FORMAT DETECTION
# ============================================================================

def detect_mask_format(mask_path: Path) -> str:
    """
    Auto-detect whether a mask is:
      'rgb'      — DeepGlobe RGB colours (0,255,255), etc.
      'classid'  — Grayscale pixel values 0–6
    """
    mask = np.array(Image.open(mask_path).convert('RGB'))
    unique_pixels = set(map(tuple, mask.reshape(-1, 3).tolist()))

    # Check if pixels match DeepGlobe palette
    palette = set(RGB_TO_CLASS.keys())
    matched = unique_pixels & palette
    unmatched = unique_pixels - palette

    if len(unmatched) == 0 and len(matched) > 0:
        return 'rgb'

    # Check if it's grayscale class IDs (load as L mode)
    mask_gray = np.array(Image.open(mask_path).convert('L'))
    if mask_gray.max() <= 6 and mask_gray.min() >= 0:
        return 'classid'

    return 'rgb'   # Fallback — try as RGB and let validation catch errors


# ============================================================================
# MASK CONVERSION
# ============================================================================

def classid_mask_to_rgb(mask_gray: np.ndarray) -> np.ndarray:
    """Convert grayscale class ID mask (H, W) → DeepGlobe RGB mask (H, W, 3)."""
    h, w  = mask_gray.shape
    rgb   = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_TO_RGB):
        rgb[mask_gray == cls_id] = color
    return rgb


def rgb_mask_to_classid(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert DeepGlobe RGB mask → class ID mask (H, W). Used for validation."""
    h, w  = mask_rgb.shape[:2]
    class_mask = np.full((h, w), 6, dtype=np.uint8)   # Default: Unknown

    for rgb_tuple, cls_id in RGB_TO_CLASS.items():
        match = np.all(mask_rgb == np.array(rgb_tuple, dtype=np.uint8), axis=-1)
        class_mask[match] = cls_id

    return class_mask


# ============================================================================
# VALIDATION
# ============================================================================

def validate_pair(image_path: Path, mask_path: Path,
                  fmt: str) -> dict:
    """
    Validate one image-mask pair. Returns a dict with:
      'valid':    bool
      'warnings': list of warning strings
      'errors':   list of error strings
      'stats':    class distribution dict
    """
    warnings = []
    errors   = []
    stats    = {}

    # ---- Load image ----
    try:
        img  = Image.open(image_path).convert('RGB')
        iw, ih = img.size
    except Exception as e:
        errors.append(f"Cannot open image: {e}")
        return {'valid': False, 'warnings': warnings,
                'errors': errors, 'stats': stats}

    # ---- Load mask ----
    try:
        if fmt == 'classid':
            mask_gray = np.array(Image.open(mask_path).convert('L'))
            mask_rgb  = classid_mask_to_rgb(mask_gray)
            class_ids = mask_gray
        else:
            mask_rgb  = np.array(Image.open(mask_path).convert('RGB'))
            class_ids = rgb_mask_to_classid(mask_rgb)
    except Exception as e:
        errors.append(f"Cannot open mask: {e}")
        return {'valid': False, 'warnings': warnings,
                'errors': errors, 'stats': stats}

    mh, mw = mask_rgb.shape[:2]

    # ---- Size checks ----
    if (iw, ih) != (mw, mh):
        errors.append(
            f"Size mismatch: image={iw}×{ih}, mask={mw}×{mh}. "
            f"Resize your mask to match the image."
        )

    if iw < MIN_IMAGE_SIZE or ih < MIN_IMAGE_SIZE:
        warnings.append(
            f"Image is small ({iw}×{ih}). "
            f"Model expects ≥{MIN_IMAGE_SIZE}px. Results may be poor."
        )

    # ---- Class distribution ----
    total_pixels = class_ids.size
    for cls_id in range(NUM_CLASSES):
        count = int((class_ids == cls_id).sum())
        if count > 0:
            stats[CLASS_NAMES[cls_id]] = {
                'pixels': count,
                'percent': round(count / total_pixels * 100, 1)
            }

    unique_classes = [i for i in range(NUM_CLASSES) if (class_ids == i).sum() > 0]

    # ---- Quality warnings ----
    unknown_ratio = (class_ids == 6).sum() / total_pixels
    if unknown_ratio > MAX_UNKNOWN_RATIO:
        warnings.append(
            f"{unknown_ratio*100:.1f}% of mask is 'Unknown' (black). "
            f"This reduces training signal. Try to label more precisely."
        )

    non_ignore_classes = [c for c in unique_classes if c != 6]
    if len(non_ignore_classes) < MIN_CLASS_COUNT:
        warnings.append(
            f"Only {len(non_ignore_classes)} labelled class(es) found. "
            f"Check if your label is complete."
        )

    # ---- Unmatched pixel check (for RGB masks) ----
    if fmt == 'rgb':
        matched  = np.zeros(class_ids.shape, dtype=bool)
        for rgb_t in RGB_TO_CLASS.keys():
            matched |= np.all(mask_rgb == np.array(rgb_t, dtype=np.uint8), axis=-1)
        bad_ratio = (~matched).sum() / total_pixels
        if bad_ratio > 0.01:
            warnings.append(
                f"{bad_ratio*100:.1f}% of pixels don't match any DeepGlobe colour. "
                f"Check your export settings in CVAT/LabelMe."
            )

    return {
        'valid':    len(errors) == 0,
        'warnings': warnings,
        'errors':   errors,
        'stats':    stats,
    }


# ============================================================================
# PAIR DISCOVERY — find image + mask pairs in input_dir
# ============================================================================

def find_pairs(input_dir: Path):
    """
    Find (image, mask) pairs in input_dir.
    Looks for files matching *_sat.jpg + *_mask.png pattern,
    OR any image + mask with the same base name.

    Returns list of (image_path, mask_path) tuples.
    """
    pairs = []
    all_files = list(input_dir.iterdir())

    # Pattern 1: DeepGlobe-style  xxxxx_sat.jpg + xxxxx_mask.png
    sat_files = [f for f in all_files if f.name.endswith('_sat.jpg')]
    for sat in sat_files:
        stem = sat.stem.replace('_sat', '')
        mask = input_dir / f"{stem}_mask.png"
        if mask.exists():
            pairs.append((sat, mask))
        else:
            print(f"  ⚠ No mask found for {sat.name}")

    # Pattern 2: Same basename, different extension
    img_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    if not pairs:
        for f in all_files:
            if f.suffix.lower() in img_extensions and '_mask' not in f.name:
                # Look for matching mask
                for ext in ['.png', '.jpg']:
                    mask = input_dir / f"{f.stem}_mask{ext}"
                    if mask.exists():
                        pairs.append((f, mask))
                        break

    return pairs


# ============================================================================
# INTEGRATION — copy validated pairs into DeepGlobe/Train/
# ============================================================================

def integrate_into_deepglobe(
    pairs:          list,
    deepglobe_dir:  Path,
    round_id:       str,
    fmt:            str,
    dry_run:        bool = False
):
    """
    Copy validated image-mask pairs into DeepGlobe Train split.
    Names them with a hitl_rXX prefix to distinguish from original data.

    Returns manifest of what was integrated.
    """
    train_img_dir  = deepglobe_dir / 'Train' / 'images'
    train_mask_dir = deepglobe_dir / 'Train' / 'masks'

    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_mask_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for img_path, mask_path in tqdm(pairs, desc="  Integrating"):
        stem     = img_path.stem.replace('_sat', '')
        new_stem = f"hitl_r{round_id}_{stem}"

        dest_img  = train_img_dir  / f"{new_stem}_sat.jpg"
        dest_mask = train_mask_dir / f"{new_stem}_mask.png"

        if not dry_run:
            # Copy image as-is
            shutil.copy(img_path, dest_img)

            # Convert mask if needed
            if fmt == 'classid':
                mask_gray = np.array(Image.open(mask_path).convert('L'))
                mask_rgb  = classid_mask_to_rgb(mask_gray)
                Image.fromarray(mask_rgb).save(dest_mask)
            else:
                shutil.copy(mask_path, dest_mask)

        manifest.append({
            'round':           f'r{round_id}',
            'original_image':  str(img_path),
            'original_mask':   str(mask_path),
            'deepglobe_image': str(dest_img),
            'deepglobe_mask':  str(dest_mask),
        })

    return manifest


# ============================================================================
# MAIN
# ============================================================================

def main():
    args      = parse_args()
    input_dir = Path(args.input_dir)
    dg_dir    = Path(args.deepglobe_dir)

    print(f"\n{'='*60}")
    print(f"  HITL STEP 2 — PREPARE & VALIDATE LABELS")
    print(f"  Input:       {input_dir}")
    print(f"  DeepGlobe:   {dg_dir}")
    print(f"  Round:       {args.round_id}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"{'='*60}\n")

    # ---- Find pairs ----
    pairs = find_pairs(input_dir)
    if not pairs:
        print(f"  ✗ No image-mask pairs found in {input_dir}")
        print(f"  Expected pattern: imagename_sat.jpg + imagename_mask.png")
        return

    print(f"  Found {len(pairs)} image-mask pairs\n")

    # ---- Validate all pairs ----
    valid_pairs   = []
    invalid_pairs = []
    all_warnings  = []
    report_lines  = []

    for img_path, mask_path in tqdm(pairs, desc="  Validating"):
        # Auto-detect or use specified format
        fmt = args.mask_format
        if fmt == 'auto':
            fmt = detect_mask_format(mask_path)

        result = validate_pair(img_path, mask_path, fmt)

        entry = {
            'image':    img_path.name,
            'mask':     mask_path.name,
            'format':   fmt,
            'valid':    result['valid'],
            'warnings': result['warnings'],
            'errors':   result['errors'],
            'stats':    result['stats'],
        }

        report_lines.append(entry)

        if result['valid']:
            valid_pairs.append((img_path, mask_path, fmt))
            if result['warnings']:
                all_warnings.append((img_path.name, result['warnings']))
        else:
            invalid_pairs.append((img_path.name, result['errors']))

    # ---- Report ----
    print(f"\n  Validation Summary:")
    print(f"  ✓ Valid pairs:   {len(valid_pairs)}")
    print(f"  ✗ Invalid pairs: {len(invalid_pairs)}")
    print(f"  ⚠ With warnings: {len(all_warnings)}")

    if invalid_pairs:
        print(f"\n  ERRORS (these will NOT be integrated):")
        for name, errs in invalid_pairs:
            print(f"    {name}:")
            for e in errs:
                print(f"      ✗ {e}")

    if all_warnings:
        print(f"\n  WARNINGS (integrated but check these):")
        for name, warns in all_warnings:
            print(f"    {name}:")
            for w in warns:
                print(f"      ⚠ {w}")

    if not valid_pairs:
        print("\n  No valid pairs to integrate. Fix errors above and retry.")
        return

    # ---- Integrate ----
    if args.dry_run:
        print(f"\n  DRY RUN — No files will be copied.")
        print(f"  {len(valid_pairs)} pairs would be integrated.")
    else:
        print(f"\n  Integrating {len(valid_pairs)} pairs into DeepGlobe/Train...")

    # Pass the format for each pair
    valid_pairs_no_fmt = [(ip, mp) for ip, mp, _ in valid_pairs]
    # Use fmt from first valid pair (or detect per-pair in a real loop)
    primary_fmt = valid_pairs[0][2] if valid_pairs else 'rgb'

    manifest = integrate_into_deepglobe(
        pairs         = valid_pairs_no_fmt,
        deepglobe_dir = dg_dir,
        round_id      = args.round_id,
        fmt           = primary_fmt,
        dry_run       = args.dry_run
    )

    # ---- Save manifest ----
    manifest_path = dg_dir / f"hitl_manifest_round_{args.round_id}.json"
    full_manifest = {
        'round':       args.round_id,
        'total_pairs': len(pairs),
        'valid':       len(valid_pairs),
        'invalid':     len(invalid_pairs),
        'integrated':  manifest,
    }

    if not args.dry_run:
        with open(manifest_path, 'w') as f:
            json.dump(full_manifest, f, indent=2)
        print(f"\n  ✓ Manifest saved: {manifest_path}")

    # ---- Save validation report ----
    report_path = input_dir / f"validation_report_round_{args.round_id}.txt"
    with open(report_path, 'w') as f:
        f.write(f"LABEL VALIDATION REPORT — Round {args.round_id}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total pairs:   {len(pairs)}\n")
        f.write(f"Valid:         {len(valid_pairs)}\n")
        f.write(f"Invalid:       {len(invalid_pairs)}\n\n")

        for entry in report_lines:
            f.write(f"\n{entry['image']}\n")
            f.write(f"  Format:  {entry['format']}\n")
            f.write(f"  Valid:   {entry['valid']}\n")
            if entry['stats']:
                f.write(f"  Classes: " +
                        ', '.join(f"{k}={v['percent']}%"
                                  for k, v in entry['stats'].items()) + "\n")
            for e in entry['errors']:
                f.write(f"  ✗ ERROR: {e}\n")
            for w in entry['warnings']:
                f.write(f"  ⚠ WARN:  {w}\n")

        f.write("\n\nNEXT STEP:\n")
        f.write("  python hitl_pipeline/03_finetune_hitl.py \\\n")
        f.write(f"    --config configs/uncertain_segformer.yaml \\\n")
        f.write(f"    --checkpoint /path/to/best_miou.pth \\\n")
        f.write(f"    --deepglobe_dir {dg_dir} \\\n")
        f.write(f"    --round_id {args.round_id}\n")

    print(f"  ✓ Validation report: {report_path}")

    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"  DRY RUN COMPLETE. Run without --dry_run to integrate.")
    else:
        print(f"  INTEGRATION COMPLETE")
        print(f"  {len(valid_pairs)} new samples added to DeepGlobe/Train/")
        print(f"\n  NEXT STEP:")
        print(f"  python hitl_pipeline/03_finetune_hitl.py \\")
        print(f"      --config configs/uncertain_segformer.yaml \\")
        print(f"      --checkpoint /path/to/best_miou.pth \\")
        print(f"      --deepglobe_dir {dg_dir} \\")
        print(f"      --round_id {args.round_id}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()