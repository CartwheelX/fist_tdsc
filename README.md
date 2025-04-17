# APCMIA: Adaptive Perturbation-assisted Contrastive Membership Inference Attack

This repository contains the official implementation of **APCMIA**, a fully differentiable membership inference attack framework designed to operate in black-box settings, especially effective against well-generalized and differentially private (DP-SGD-trained) models.

APCMIA selectively perturbs ambiguous non-member prediction vectors using a contrastive learning framework. It adapts cosine similarity and entropy thresholds based on the target model's generalization gap, achieving state-of-the-art performance under low FPR regimes.

---



## 🧠 Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Directory Structure

```
data/                     # Contains sample datasets (adult, location)
demoloader/              # Trained target/shadow models and attack artifacts
results/                 # ROC curves, TPR-FPR, and entropy visualizations
roc_curves/              # PDF/CSV results for each dataset
threshold_plots/         # Threshold plots for CNN, MLP, VGG16, WRN
main.py                  # Entry point to train models and launch attacks
target_shadow_nn_models.py  # Model architectures and training logic
meminf.py, my_lira.py    # Attack logic for various MIA baselines
```

---

## 🏃 Usage

### 🔧 Train Target and Shadow Models

Example for CIFAR-10 (CNN):

```bash
python main.py --attack_type 0 --dataset_name cifar10 --arch cnn --train_model
```

---

### 🚨 Run Membership Inference Attacks

Baseline attacks:

```bash
python main.py --attack_type 0 --dataset_name cifar10 --attack_name mia --arch cnn
python main.py --attack_type 0 --dataset_name cifar10 --attack_name memia --arch cnn
python main.py --attack_type 0 --dataset_name cifar10 --attack_name nsh --arch cnn
python main.py --attack_type 0 --dataset_name cifar10 --attack_name m_lira --lira_train --arch cnn
```

Run **APCMIA**:

```bash
python main.py --attack_type 0 --dataset_name cifar10 --attack_name apcmia --arch cnn --apcmia_cluster
```

---

### 📊 Plotting ROC and Threshold Curves

```bash
python main.py --plot --plot_results roc --dataset_name cifar10
python main.py --plot --plot_results th --dataset_name cifar10 --attack_name apcmia
```

---

## 📚 Supported Datasets

- Image datasets: CIFAR-10, CIFAR-100, STL10, FMNIST, UTKFace
- Non-image datasets: Purchase-100, Texas-100, Adult, Location

---

## 📈 Attack Architectures

You can run all attacks across:

- `--arch cnn`
- `--arch mlp`
- `--arch vgg16`
- `--arch wrn`

Example:

```bash
python main.py --attack_type 0 --dataset_name adult --attack_name apcmia --arch mlp --apcmia_cluster
```

---

## 🧪 Evaluation

The framework automatically saves:

- ROC curves
- TPR/FPR tables
- Learned thresholds
- Attack prediction vectors

Check `results/` and `roc_curves/` directories.

---

## 📄 Citation

If you use this code, please cite our work:

> Khan et al. "Breaking the Shield of Generalization: Adaptive Perturbation-based Contrastive Membership Inference Attacks." ACM CCS 2025 (under review).
