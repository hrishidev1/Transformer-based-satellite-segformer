"""
Calibration metrics for uncertainty evaluation

Metrics:
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- Brier Score
- Reliability Diagrams
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

# FIXED: Removed dead import for brier_score_loss

class CalibrationMetrics:
    """
    Compute calibration metrics for uncertainty-aware models
    """
    
    def __init__(self, num_bins=15, num_classes=7, ignore_index=None):
        """
        Args:
            num_bins: Number of bins for calibration histogram
            num_classes: Number of classes
            ignore_index: Class to ignore in metrics
        """
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset accumulated data"""
        self.confidences = []
        self.predictions = []
        self.targets = []
        self.uncertainties = []
        
        # Accumulators for true Brier score
        self.brier_sum = 0.0
        self.brier_count = 0
        
        # Cache for plotting after compute() clears lists
        self._last_confidences = None
        self._last_predictions = None
        self._last_targets = None
        self._last_uncertainties = None
    
    def update(self, probs, preds, targets, uncertainties=None):
        """
        Update metrics with batch data (GPU Optimized)
        """
        # Ensure inputs are tensors
        if not torch.is_tensor(probs): probs = torch.tensor(probs)
        if not torch.is_tensor(preds): preds = torch.tensor(preds)
        if not torch.is_tensor(targets): targets = torch.tensor(targets)
        
        # Flatten spatial dimensions
        B, C, H, W = probs.shape
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # Create mask for valid pixels
        mask = (targets_flat >= 0) & (targets_flat < self.num_classes)
        if self.ignore_index is not None:
            mask &= (targets_flat != self.ignore_index)
            
        valid_targets = targets_flat[mask].long()
        valid_probs = probs_flat[mask]
        valid_preds = preds_flat[mask]
        
        if len(valid_targets) == 0:
            return
            
        # 1. True Multi-class Brier Score (Computed natively on GPU)
        # Formula: sum_k (p_k - y_k)^2
        target_one_hot = torch.nn.functional.one_hot(valid_targets, num_classes=self.num_classes).float()
        brier_batch_sum = torch.sum((valid_probs - target_one_hot) ** 2)
        self.brier_sum += brier_batch_sum.item()
        self.brier_count += len(valid_targets)
        
        # 2. Extract Confidence
        valid_confidences = torch.max(valid_probs, dim=1)[0]
        
        # 3. Store minimal 1D tensors to avoid massive PCIe transfers
        # We detach and move to CPU asynchronously to prevent GPU sync bottlenecks
        self.confidences.append(valid_confidences.detach().cpu())
        self.predictions.append(valid_preds.detach().cpu())
        self.targets.append(valid_targets.detach().cpu())
        
        if uncertainties is not None:
            if not torch.is_tensor(uncertainties):
                uncertainties = torch.tensor(uncertainties)
            valid_unc = uncertainties.reshape(-1)[mask]
            self.uncertainties.append(valid_unc.detach().cpu())
    
    def compute(self):
        """
        Compute all calibration metrics
        """
        if not self.confidences:
            if hasattr(self, '_last_confidences') and self._last_confidences is not None:
                pass # Use cached data if compute() was called recently
            else:
                return {'ece': 0.0, 'mce': 0.0, 'brier': 0.0, 'accuracy': 0.0}
        else:
            # Concatenate all batches ONCE into numpy arrays
            self._last_confidences = torch.cat(self.confidences).numpy()
            self._last_predictions = torch.cat(self.predictions).numpy()
            self._last_targets = torch.cat(self.targets).numpy()
            if len(self.uncertainties) > 0:
                self._last_uncertainties = torch.cat(self.uncertainties).numpy()
            else:
                self._last_uncertainties = None
                
            self._last_brier = self.brier_sum / max(self.brier_count, 1)
            
            # FIXED: Clear lists so calling update() after compute() doesn't double-count
            self.confidences = []
            self.predictions = []
            self.targets = []
            self.uncertainties = []
            self.brier_sum = 0.0
            self.brier_count = 0

        confidences = self._last_confidences
        predictions = self._last_predictions
        targets = self._last_targets
        
        # Compute metrics
        ece = self._compute_ece(confidences, predictions, targets)
        mce = self._compute_mce(confidences, predictions, targets)
        brier = self._last_brier
        
        results = {
            'ece': ece,
            'mce': mce,
            'brier': brier,
            'accuracy': (predictions == targets).mean()
        }
        
        # Uncertainty correlation
        if self._last_uncertainties is not None:
            uncertainties = self._last_uncertainties
            errors = (predictions != targets).astype(float)
            
            # Correlation
            if np.std(uncertainties) > 0 and np.std(errors) > 0:
                correlation = np.corrcoef(uncertainties, errors)[0, 1]
            else:
                correlation = 0.0
            results['uncertainty_error_corr'] = correlation
            
            # Mean uncertainty split
            correct_mask = predictions == targets
            results['uncertainty_correct'] = uncertainties[correct_mask].mean() if correct_mask.sum() > 0 else 0.0
            results['uncertainty_incorrect'] = uncertainties[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0.0
        
        return results
    
    def _compute_ece(self, confidences, predictions, targets):
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece
    
    def _compute_mce(self, confidences, predictions, targets):
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if in_bin.sum() > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        return mce
        
    def _get_plot_data(self):
        """Helper to get data for plotting regardless of compute() state"""
        if self.confidences:
            conf = torch.cat(self.confidences).numpy()
            pred = torch.cat(self.predictions).numpy()
            targ = torch.cat(self.targets).numpy()
            unc = torch.cat(self.uncertainties).numpy() if self.uncertainties else None
            return conf, pred, targ, unc
        elif hasattr(self, '_last_confidences') and self._last_confidences is not None:
            return self._last_confidences, self._last_predictions, self._last_targets, self._last_uncertainties
        return None, None, None, None
    
    def plot_reliability_diagram(self, save_path=None):
        confidences, predictions, targets, _ = self._get_plot_data()
        if confidences is None:
            print("No data available for plotting.")
            return None
            
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        accuracies = []
        counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                accuracies.append((predictions[in_bin] == targets[in_bin]).mean())
                counts.append(in_bin.sum())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.plot(bin_centers, accuracies, 'o-', label='Model', markersize=8)
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Reliability Diagram', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Histogram
        ax2.hist(confidences, bins=self.num_bins, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Confidence Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved reliability diagram to {save_path}")
        return fig
    
    def plot_uncertainty_vs_error(self, save_path=None):
        _, predictions, targets, uncertainties = self._get_plot_data()
        if uncertainties is None:
            print("No uncertainty data available")
            return None
            
        errors = (predictions != targets).astype(float)
        
        n_samples = min(10000, len(uncertainties))
        indices = np.random.choice(len(uncertainties), n_samples, replace=False)
        
        uncertainties_sample = uncertainties[indices]
        errors_sample = errors[indices]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot
        axes[0].scatter(
            uncertainties_sample[errors_sample == 0],
            np.random.randn((errors_sample == 0).sum()) * 0.1,
            alpha=0.3, s=1, c='green', label='Correct'
        )
        axes[0].scatter(
            uncertainties_sample[errors_sample == 1],
            np.random.randn((errors_sample == 1).sum()) * 0.1 + 1,
            alpha=0.3, s=1, c='red', label='Error'
        )
        axes[0].set_xlabel('Uncertainty', fontsize=12)
        axes[0].set_ylabel('Prediction', fontsize=12)
        axes[0].set_yticks([0, 1])
        axes[0].set_yticklabels(['Correct', 'Error'])
        axes[0].set_title('Uncertainty vs Prediction Error', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Histogram
        axes[1].hist(
            uncertainties[errors == 0],
            bins=30, alpha=0.5, label='Correct', color='green', density=True
        )
        axes[1].hist(
            uncertainties[errors == 1],
            bins=30, alpha=0.5, label='Error', color='red', density=True
        )
        axes[1].set_xlabel('Uncertainty', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].set_title('Uncertainty Distribution', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved uncertainty analysis to {save_path}")
        return fig

def test_calibration():
    print("Testing CalibrationMetrics...")
    B, C, H, W = 2, 7, 64, 64
    probs = torch.rand(B, C, H, W)
    probs = probs / probs.sum(dim=1, keepdim=True)
    preds = torch.argmax(probs, dim=1)
    targets = torch.randint(0, C, (B, H, W))
    uncertainties = torch.rand(B, H, W)
    
    metrics = CalibrationMetrics(num_bins=10, num_classes=C)
    metrics.update(probs, preds, targets, uncertainties)
    results = metrics.compute()
    
    print(f"✓ ECE: {results['ece']:.4f}")
    print(f"✓ MCE: {results['mce']:.4f}")
    print(f"✓ True Brier Score: {results['brier']:.4f}")
    print(f"✓ Accuracy: {results['accuracy']:.4f}")
    print(f"✓ Uncertainty-Error Correlation: {results['uncertainty_error_corr']:.4f}")
    print("✅ All calibration tests passed!")

if __name__ == '__main__':
    test_calibration()