"""
DeepGlobe Land Cover Classification Dataset
Dataset: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# DeepGlobe RGB → Class mapping
DEEPGLOBE_COLORMAP = {
    (0, 255, 255): 0,     # Urban - Cyan
    (255, 255, 0): 1,     # Agriculture - Yellow
    (255, 0, 255): 2,     # Rangeland - Magenta
    (0, 255, 0): 3,       # Forest - Green
    (0, 0, 255): 4,       # Water - Blue
    (255, 255, 255): 5,   # Barren - White
    (0, 0, 0): 6          # Unknown / Ignore - Black
}

CLASS_NAMES = [
    'Urban',
    'Agriculture', 
    'Rangeland',
    'Forest',
    'Water',
    'Barren',
    'Unknown'
]


class DeepGlobeDataset(Dataset):
    """
    DeepGlobe Land Cover Classification Dataset
    """
    
    def __init__(self, config, split='train'):
        """
        Args:
            config: Configuration object
            split: 'train', 'val', or 'test'
        """
        self.config = config
        self.split = split.capitalize()  # Train, Val, Test
        self.num_classes = config.data.num_classes
        self.ignore_index = config.data.ignore_index if hasattr(config.data, 'ignore_index') else None
        
        # Paths
        self.root_dir = config.data.root_dir
        self.img_dir = os.path.join(self.root_dir, self.split, "images")
        self.mask_dir = os.path.join(self.root_dir, self.split, "masks")
        
        # Verify directories exist
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        
        # Get image list
        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith("_sat.jpg")
        ])
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
        
        # Verify masks exist (except for test)
        if self.split != "Test":
            if not os.path.exists(self.mask_dir):
                raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
            
            # Count the files, don't store them in memory
            num_masks = sum(1 for f in os.listdir(self.mask_dir) if f.endswith("_mask.png"))
            
            if num_masks == 0:
                raise ValueError(f"No masks found in {self.mask_dir}")
            
            assert len(self.images) == num_masks, \
                f"Mismatch: {len(self.images)} images, {num_masks} masks"
            
        # Get separated transforms
        from . import get_geometry_transforms, get_color_transforms
        is_train = (split.lower() == 'train')
        self.geo_transform = get_geometry_transforms(config, train=is_train)
        self.color_transform = get_color_transforms(config, train=is_train)
        
        print(f"[DeepGlobe] Loaded {len(self.images)} samples from {self.split}")
    
    def __len__(self):
        return len(self.images)
    
    def _encode_mask(self, mask_rgb):
        # Initialize with ignore_index (6) instead of 0 (Urban)
        ignore_val = self.ignore_index if self.ignore_index is not None else 0
        mask = np.full(mask_rgb.shape[:2], ignore_val, dtype=np.uint8)
        
        # Fast channel-wise boolean matching
        r = mask_rgb[:, :, 0]
        g = mask_rgb[:, :, 1]
        b = mask_rgb[:, :, 2]
        
        for rgb, class_id in DEEPGLOBE_COLORMAP.items():
            matches = (r == rgb[0]) & (g == rgb[1]) & (b == rgb[2])
            mask[matches] = class_id
            
        return mask
    
    def __getitem__(self, idx):
        """
        Get sample at index (Optimized: Crop before Encode)
        """
        # 1. Load raw images as numpy arrays
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.split != "Test":
            mask_name = self.images[idx].replace("_sat.jpg", "_mask.png")
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        else:
            # Dummy mask for test set
            mask_rgb = np.zeros_like(image)

        # 2. Geometry Transforms (Crops/Flips on BOTH image and RGB mask)
        if self.geo_transform:
            augmented = self.geo_transform(image=image, mask_rgb=mask_rgb)
            image = augmented["image"]
            mask_rgb = augmented["mask_rgb"]
            
        # 3. Encode ONLY the resulting small crop (e.g., 512x512 instead of 2448x2448)
        mask = self._encode_mask(mask_rgb)
        
        # 4. Color Transforms & Normalization (ONLY on the image)
        if self.color_transform:
            augmented = self.color_transform(image=image)
            image = augmented["image"]
        
        return image, torch.tensor(mask, dtype=torch.long)
    
    def get_class_names(self):
        """Return list of class names"""
        return CLASS_NAMES
    
    def get_image_path(self, idx):
        """Get path to image at index"""
        return os.path.join(self.img_dir, self.images[idx])


def test_deepglobe_dataset():
    """Test DeepGlobe dataset loading"""
    print("Testing DeepGlobe dataset...")
    
    # Create dummy config
    class DummyConfig:
        class data:
            root_dir = "data/DeepGlobe"
            num_classes = 7
            ignore_index = 6
            image_size = 512
        class augmentation:
            color_jitter = 0.3
            blur = 0.2
            brightness_contrast = 0.5
            noise = 0.2
            center_crop_val = True
    
    try:
        config = DummyConfig()
        
        # Try to load dataset
        dataset = DeepGlobeDataset(config, split='train')
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        print(f"✓ Class names: {dataset.get_class_names()}")
        
        # Try to get a sample
        if len(dataset) > 0:
            image, mask = dataset[0]
            print(f"✓ Sample shape - Image: {image.shape}, Mask: {mask.shape}")
            print(f"✓ Unique classes in mask: {mask.unique().tolist()}")
        
        print("✅ DeepGlobe dataset test passed!")
        
    except FileNotFoundError as e:
        print(f"⚠️ Dataset not found (expected if not downloaded yet): {e}")
        print("This is normal - dataset will be downloaded later")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == '__main__':
    test_deepglobe_dataset()