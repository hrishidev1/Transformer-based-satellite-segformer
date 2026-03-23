
# Uncertainty-Aware SegFormer for Satellite Imagery

This repository contains the PyTorch implementation of our research on Uncertainty-Aware Semantic Segmentation for high-resolution satellite imagery. 

Instead of relying on computationally expensive ensemble methods like Monte Carlo (MC) Dropout, we integrate **Evidential Deep Learning (EDL)** into a Vision Transformer (SegFormer MiT-b3). By replacing the standard Softmax activation with a custom ReLU evidential head, the model learns to output Dirichlet distribution parameters, allowing it to mathematically quantify its own epistemic uncertainty (vacuity) in a single forward pass.

## 🖥️ Hardware & Training Environment

This architecture was developed and trained using **Google Colab** environments accelerated by **NVIDIA A100 Tensor Core GPUs (40GB VRAM)**. 

Handling native high-resolution satellite imagery (2448x2448 `.tiff` files) presents severe Out-of-Memory (OOM) bottlenecks on standard hardware. To address this, our pipeline features:
* **Dynamic Spatial Cropping:** The `DataLoader` extracts random `512x512` patches during training to maintain VRAM limits while preserving exact image-to-mask spatial alignment.
* **PCIe Optimization:** We utilize `num_workers=4` and `pin_memory=True` to streamline CPU-to-GPU tensor transfers over the PCIe bus, maximizing CUDA utilization on the A100.
* **Cloud Reproducibility:** The configuration system (`configs/`) is fully decoupled from local absolute paths, allowing seamless execution across Colab, AWS, or local workstations.

## 🚀 Key Features

* **Evidential Head:** Outputs non-negative evidence ($e_k \ge 0$) to calculate both class probabilities and pixel-level uncertainty maps.
* **Custom Loss Function:** Combines Bayes Risk (MSE) with a Kullback-Leibler (KL) Divergence penalty. Includes a linear annealing scheduler ($\lambda_t$) to prevent premature regularization of spatial features.
* **OOD Evaluation Pipeline:** Built-in benchmarking suite that applies synthetic ImageNet-C corruptions (motion blur, fog, etc.) to evaluate domain-shift detection natively.
* **YAML-Driven Architecture:** Hyperparameters, backbone selection, and data loading are handled dynamically via modular YAML files.

## 📊 Results Summary

* **Inference Efficiency:** Achieved a **1.5x inference speedup** compared to a 10-pass MC Dropout baseline (12.9s vs 19.9s) while maintaining comparable mean Intersection over Union (mIoU).
* **Failure Detection:** Demonstrated strong out-of-distribution (OOD) failure detection, yielding a Spearman correlation of $\rho = 0.686$ between epistemic uncertainty and model error against increasing corruption severities.

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/hrishidev1/Transformer-based-satellite-segformer.git](https://github.com/hrishidev1/Transformer-based-satellite-segformer.git.git)
   cd Transformer-based-satellite-segformer
   ```

2. Install the required dependencies (Python 3.8+ & CUDA recommended):
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Dataset Setup

We utilize the **DeepGlobe Land Cover Classification** dataset. The dataloader expects RGB `.tiff` images and their corresponding mask files.

Organize your `datasets/` directory as follows:
```text
datasets/
└── deepglobe/
    ├── train/
    │   ├── images/       # .tiff RGB images
    │   └── masks/        # .png or .tiff RGB masks
    └── val/
        ├── images/
        └── masks/
```

## 💻 Usage

### 1. Training the Model
All hyperparameters (learning rate, maximum $\lambda$, epochs, crop size) are defined in `configs/uncertain_segformer.yaml`. To initiate the training loop:

```bash
python experiments/train_uncertain.py --config configs/uncertain_segformer.yaml
```
*(To resume training from a saved checkpoint, append the `--resume` flag).*

### 2. Out-of-Distribution (OOD) Evaluation
To test the model's uncertainty calibration against synthetic domain shifts:

```bash
python experiments/02_ood_corruption.py --weights checkpoints/segformer/best_model.pth --severity 5
```

### 3. MC Dropout Benchmarking
To run the inference latency and calibration comparison against the Monte Carlo Dropout baseline ($p=0.1$):

```bash
python experiments/03_mc_dropout_comparison.py --weights checkpoints/segformer/best_model.pth
```

## 🧠 Repository Structure

* `configs/` - Modular YAML configuration files.
* `datasets/` - Data loading and pre-processing logic (crops, horizontal flips).
* `models/` - Dynamic model factories (`uncertainty_factory.py`), Evidential Head, and SegFormer backbone integration.
* `losses/` - Custom PyTorch loss functions (`evidential_loss.py`).
* `metrics/` - Evaluation scripts for mIoU (ignoring void classes) and Expected Calibration Error (ECE).
* `experiments/` - Core execution scripts for ablation studies, training, and testing.
```
