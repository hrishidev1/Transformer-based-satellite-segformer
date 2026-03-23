"""
Comprehensive metrics for semantic segmentation (GPU Optimized)
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator
    Computes IoU, Dice, Precision, Recall, Accuracy, and F1 natively on GPU
    """
    
    def __init__(self, num_classes, ignore_index=None, class_names=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        # Leave as None to initialize on the correct GPU dynamically
        self.confusion_matrix = None
        self.total_samples = 0
    
    def update(self, preds, targets):
        """Pure PyTorch update to avoid GPU-CPU sync bottlenecks"""
        if preds.ndim == 4:
            preds = torch.argmax(preds, dim=1)
            
        if self.confusion_matrix is None:
            self.confusion_matrix = torch.zeros(
                (self.num_classes, self.num_classes), 
                dtype=torch.int64, 
                device=targets.device
            )
            
        preds = preds.flatten()
        targets = targets.flatten()
        
        mask = (targets >= 0) & (targets < self.num_classes)
        if self.ignore_index is not None:
            mask &= (targets != self.ignore_index)
            
        valid_targets = targets[mask].long()
        valid_preds = preds[mask].long()
        
        if len(valid_targets) > 0:
            # Native PyTorch bincount entirely on the GPU
            encoding = valid_targets * self.num_classes + valid_preds
            bincount = torch.bincount(encoding, minlength=self.num_classes**2)
            confusion_update = bincount.reshape(self.num_classes, self.num_classes)
            self.confusion_matrix += confusion_update
        
        self.total_samples += len(targets)
    
    def compute(self, return_per_class=True):
        """Compute all metrics"""
        # CRITICAL FIX: Assign to a local variable `cm` to prevent mutating the persistent GPU state
        if self.confusion_matrix is not None:
            cm = self.confusion_matrix.cpu().numpy()
        else:
            cm = np.zeros((self.num_classes, self.num_classes))
            
        iou_per_class, dice_per_class = [], []
        precision_per_class, recall_per_class, f1_per_class = [], [], []
        
        for i in range(self.num_classes):
            if self.ignore_index is not None and i == self.ignore_index:
                iou_per_class.append(np.nan)
                dice_per_class.append(np.nan)
                precision_per_class.append(np.nan)
                recall_per_class.append(np.nan)
                f1_per_class.append(np.nan)
                continue
            
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            eps = 1e-10
            
            iou_per_class.append(tp / (tp + fp + fn + eps))
            dice_per_class.append((2 * tp) / (2 * tp + fp + fn + eps))
            precision_per_class.append(tp / (tp + fp + eps))
            recall_per_class.append(tp / (tp + fn + eps))
            f1_per_class.append(2 * (tp / (tp + fp + eps) * tp / (tp + fn + eps)) / ((tp / (tp + fp + eps)) + (tp / (tp + fn + eps)) + eps))
        
        results = {
            'miou': np.nanmean(iou_per_class),
            'mdice': np.nanmean(dice_per_class),
            'accuracy': np.trace(cm) / (cm.sum() + 1e-10),
            'precision': np.nanmean(precision_per_class),
            'recall': np.nanmean(recall_per_class),
            'f1': np.nanmean(f1_per_class),
        }
        
        total_pixels = cm.sum()
        if total_pixels > 0:
            freq = cm.sum(axis=1) / total_pixels
            safe_iou = np.array([x if not np.isnan(x) else 0 for x in iou_per_class])
            results['fwiou'] = np.sum(freq * safe_iou)
        else:
            results['fwiou'] = 0.0
            
        if return_per_class:
            results.update({
                'iou_per_class': iou_per_class,
                'dice_per_class': dice_per_class,
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'f1_per_class': f1_per_class
            })
        return results
    
    def get_confusion_matrix(self):
        if self.confusion_matrix is not None:
            return self.confusion_matrix.cpu().numpy()
        return np.zeros((self.num_classes, self.num_classes))
        
    def print_metrics(self, results):
        """Expanded metrics printing"""
        print(f"\nMean IoU: {results['miou']:.4f} | Mean Dice: {results['mdice']:.4f} | Accuracy: {results['accuracy']:.4f}")
        if 'iou_per_class' in results:
            print("Per-class IoU:")
            for name, iou in zip(self.class_names, results['iou_per_class']):
                if not np.isnan(iou):
                    print(f"  - {name}: {iou:.4f}")

    def plot_confusion_matrix(self, save_path=None, normalize=True):
        """WARNING FIX: Implemented actual plotting logic"""
        cm = self.get_confusion_matrix()
        if normalize:
            cm_sums = cm.sum(axis=1)[:, np.newaxis]
            cm = np.divide(cm, cm_sums, out=np.zeros_like(cm, dtype=float), where=cm_sums!=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved confusion matrix to {save_path}")
        return plt.gcf()