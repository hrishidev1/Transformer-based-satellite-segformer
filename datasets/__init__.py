"""
Dataset factory and augmentation pipeline
"""
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def get_dataset(config, split='train'):
    dataset_name = config.data.dataset.lower()
    
    if dataset_name == 'deepglobe':
        from .deepglobe import DeepGlobeDataset
        return DeepGlobeDataset(config, split)
    elif dataset_name == 'loveda':
        # Safely stubbed in case you expand your paper later
        raise NotImplementedError("LoveDA dataset not yet implemented in the optimized pipeline")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_geometry_transforms(config, train=True):
    """
    Get spatial/geometry augmentations only.
    Applied to both Image and raw RGB Mask before encoding.
    """
    size = config.data.image_size
    
    if train:
        # Train spatial transforms
        # FIXED: "mask_rgb": "mask" ensures Nearest-Neighbor interpolation 
        # so exact RGB colormap values are never blended/corrupted during rotations.
        return A.Compose([
            A.RandomCrop(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], additional_targets={"mask_rgb": "mask"})
    else:
        # Val/Test spatial transforms
        if hasattr(config.augmentation, 'center_crop_val') and config.augmentation.center_crop_val:
            return A.Compose([A.CenterCrop(size, size)], additional_targets={"mask_rgb": "mask"})
        else:
            return A.Compose([A.Resize(size, size)], additional_targets={"mask_rgb": "mask"})


def get_color_transforms(config, train=True):
    """
    Get color and pixel-level augmentations only.
    Applied ONLY to the image, never to the mask.
    """
    transforms_list = []
    
    if train:
        if hasattr(config.augmentation, 'color_jitter') and config.augmentation.color_jitter:
            transforms_list.append(
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=config.augmentation.color_jitter)
            )
        
        if hasattr(config.augmentation, 'blur') and config.augmentation.blur:
            transforms_list.append(
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=config.augmentation.blur)
            )
        
        if hasattr(config.augmentation, 'brightness_contrast') and config.augmentation.brightness_contrast:
            transforms_list.append(
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=config.augmentation.brightness_contrast)
            )
        
        if hasattr(config.augmentation, 'noise') and config.augmentation.noise:
            transforms_list.append(
                A.OneOf([
                    # FIXED: Albumentations 2.0+ API crash averted
                    A.GaussNoise(noise_scale_factor=0.1, p=0.5),
                    A.ISONoise(p=1.0),
                ], p=config.augmentation.noise)
            )

    # Normalization and ToTensorV2 are always applied (Train and Val)
    transforms_list.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)


def compute_class_distribution(dataset):
    """
    Compute class distribution in dataset for class weights
    """
    print("Computing class distribution...")
    class_counts = np.zeros(dataset.num_classes)
    
    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        mask_np = mask.numpy() if hasattr(mask, 'numpy') else mask
        
        for cls in range(dataset.num_classes):
            class_counts[cls] += (mask_np == cls).sum()
    
    return class_counts


def compute_class_weights(class_counts, method='inverse', ignore_index=None):
    """
    Compute class weights from class counts
    """
    if method == 'inverse':
        weights = 1.0 / (class_counts + 1e-8)
        weights = weights / weights.sum() * len(weights)
    elif method == 'effective_num':
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if ignore_index is not None:
        weights[ignore_index] = 0.0
    
    return weights