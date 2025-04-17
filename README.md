# apcMIA: Adaptive Perturbation-assisted Contrastive Membership Inference Attack

This repository contains the official implementation of **apcMIA**, a fully differentiable membership inference attack framework designed to operate in black-box settings, especially effective against well-generalized and differentially private (DP-SGD-trained) models.

A PyTorch-based implementation of apcMIA – a contrastive learning-powered black-box membership inference attack effective even against DP-trained models.

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
meminf.py                # Attack logic for our MIA
requirements.txt         # Python dependencies
```

---

## 🏃 Usage

### 🔧 Train Target and Shadow Models

Example for Location (MLP):

```bash
python main.py --dataset_name location --arch mlp --train_model 
python main.py --dataset_name location --arch mlp --train_shadow
```

---

### 🚨 Train and Test Our Attack Model

Run **apcMIA** attack (this command will train our attack and test it):

```bash
python main.py --dataset_name location --attack_name apcmia --arch mlp --apcmia_cluster
```

**Note:** if you want to plot the cluster results presented in the paper, you need to use `--apcmia_cluster` flag.

---

### 🛡️ Attacking DP-Trained Models

Train the target/shadow models with DP-SGD:

```bash
python main.py --dataset_name location --train_model --use_DP --noise 0.3 --norm 5 --delta 1e-5
python main.py --dataset_name location --train_shadow --use_DP --noise 0.3 --norm 5 --delta 1e-5
```

Here `--norm` represents the clipping value. Adjust DP parameters as needed to fit the desired privacy budget.

Attack the DP-SGD trained models:

```bash
python main.py --dataset_name location --attack_name apcmia --arch mlp --apcmia_cluster
```

---

### 📊 Plotting ROC and Threshold Curves

```bash
python main.py --plot --plot_results roc --dataset_name location --attack_name apcmia
python main.py --plot --plot_results th --dataset_name location --attack_name apcmia
```

---

## 📚 Supported Datasets

- Image datasets: CIFAR-10, CIFAR-100, STL10, FMNIST, UTKFace  
- Non-image datasets: Purchase-100, Texas-100, Adult, Location

> ⚠️ **Note:** This repository provides only two example datasets — `adult` and `location` — in the `data/` directory for demonstration purposes.  
To use other datasets (e.g., CIFAR-10, CIFAR-100, Purchase-100), please download them from their official sources as cited in the paper and place them in the corresponding folder under `data/`.  
Refer to the dataset links in our article or supplementary material for details.

---

## 📈 Attack Architectures

You can run all attacks across:

- `--arch cnn`
- `--arch mlp`
- `--arch vgg16`

Example:

```bash
python main.py --dataset_name adult --attack_name apcmia --arch mlp --apcmia_cluster
```

**Note:** Use `--arch mlp` for non-image datasets (Location, Adult, etc.) and `--arch cnn` / `--arch vgg16` for image-based datasets (CIFAR-10, CIFAR-100, STL10, etc.).

---

## 🧪 Evaluation

The framework automatically saves:

- ROC curves
- TPR/FPR tables
- Learned thresholds
- Attack prediction vectors

Check the `results/` and `roc_curves/` directories.

---

## 📄 Citation

If you use this code, please cite our work:

> Khan et al. "Breaking the Shield of Generalization: Adaptive Perturbation-based Contrastive Membership Inference Attacks." ACM CCS 2025 (under review).