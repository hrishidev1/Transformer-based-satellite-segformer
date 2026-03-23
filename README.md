Uncertainty-Aware SegFormer for Satellite Imagery

This repository contains the PyTorch implementation of our project/paper on Uncertainty-Aware Semantic Segmentation for high-resolution satellite imagery. 

Instead of relying on computationally expensive ensemble methods like MC Dropout, we integrate Evidential Deep Learning (EDL) into a Vision Transformer (SegFormer MiT-b3). By replacing the standard Softmax activation with a ReLU head, the model learns to output Dirichlet distribution parameters, allowing it to quantify its own epistemic uncertainty (vacuity) in a single forward pass.

Key Features:

* Evidential Head: Outputs non-negative evidence to calculate both class probabilities and pixel-level uncertainty.
* Custom Loss Function: Combines Bayes Risk (MSE) with a Kullback-Leibler (KL) Divergence penalty. Includes a linear annealing scheduler to prevent premature regularization of spatial features.
* OOD Evaluation Pipeline: Built-in benchmarking suite that applies synthetic ImageNet-C corruptions (motion blur, fog, etc.) to evaluate domain-shift detection.
* YAML-Driven Architecture: Hyperparameters, backbone selection, and data loading are decoupled from the training loop via modular `configs/`.

Results Summary:
* Efficiency: Achieves a 1.5x inference speedup compared to a 10-pass MC Dropout baseline (12.9s vs 19.9s) while maintaining comparable mIoU.
* Reliability: Demonstrates strong out-of-distribution (OOD) failure detection, yielding a Spearman correlation of rho = 0.686 against increasing corruption severities.

---

Installation:

1. Clone the repository:
   ```bash
   git clone [https://github.com/hrishidev1/Transformer-based-satellite-segformer.git](https://github.com/hrishidev1/Transformer-based-satellite-segformer.git)
   cd Transformer-based-satellite-segformer
Install the required dependencies (Python 3.8+ & CUDA recommended):Bashpip install -r requirements.txt
Note: Ensure your PyTorch version matches your local CUDA toolkit.📂 Dataset SetupWe use the DeepGlobe Land Cover Classification dataset. The dataloader expects the images to be high-resolution .tiff files, which are dynamically cropped to 512x512 during training to manage GPU VRAM. Organize your datasets/ folder as follows:Plaintextdatasets/
└── deepglobe/
    ├── train/
    │   ├── images/       # .tiff RGB images
    │   └── masks/        # .png or .tiff RGB masks
    └── val/
        ├── images/
        └── masks/
💻 Usage1. Training the ModelHyperparameters (learning rate, $\lambda$ max, epochs, crop size) are handled in configs/uncertain_segformer.yaml. To start training:Bashpython experiments/train_uncertain.py --config configs/uncertain_segformer.yaml
To resume training from a checkpoint, append --resume.2. Out-of-Distribution (OOD) EvaluationTo test the model's uncertainty calibration against synthetic domain shifts (ImageNet-C):Bashpython experiments/02_ood_corruption.py --weights checkpoints/best_model.pth --severity 5
3. MC Dropout BenchmarkingTo run the inference latency and calibration comparison against the Monte Carlo Dropout baseline ($p=0.1$):Bashpython experiments/03_mc_dropout_comparison.py --weights checkpoints/best_model.pth
🧠 Repository Structureconfigs/ - YAML configuration files.datasets/ - Data loading and pre-processing logic (random cropping, augmentations).models/ - Model definitions (uncertainty_factory.py, Evidential Head, SegFormer backbone).losses/ - Custom PyTorch loss functions (evidential_loss.py with KL Annealing).metrics/ - Evaluation scripts for mIoU, Expected Calibration Error (ECE), and Brier Score.experiments/ - Core execution scripts for training, testing, and benchmarking.📝 Citation(If this project helped your research, please consider citing our upcoming IEEE paper once published. Citation details will be added here).
