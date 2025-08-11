# Fitting In to Stand Out: Contrastive Membership Inference Attacks via Learned Adversarial Perturbations

This repository contains the official PyTorch implementation of **FiST**, a fully differentiable membership inference attack framework that learns **adversarial perturbations** to make non-member samples “fit in” with members during training.  
By applying these perturbations uniformly at inference, FiST induces asymmetric responses between members and non-members, enabling the attack model to exploit **contrastive signals** (via cosine similarity and entropy) for highly accurate membership inference — even in **strict black-box** and **differentially-private (DP-SGD–trained)** settings.

---

## 📦 Requirements

- Python ≥ 3.8  
- PyTorch  
- NumPy  
- scikit-learn  
- Matplotlib  

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
data/                    
  ├─ adult/               # UCI Census Income
  └─ location/            # Foursquare check-in dataset (Shukri et al.)

demoloader/               # Pretrained target/shadow models and attack artifacts  
results/                  # Evaluation logs and metrics  
roc_curves/               # Saved ROC plots (PDF/CSV) per dataset & architecture  
threshold_plots/          # Learned threshold visualizations  
main.py                   # Main script: training + attack pipeline  
target_shadow_nn_models.py # Model architectures and training  
meminf.py                 # Membership inference attack implementations  
requirements.txt          # Dependencies  
```

---

## 🚀 Usage

### 1️⃣ Train Shadow Models

Example (MLP on Location):

```bash
python main.py --dataset_name location --arch mlp --train_shadow
```

---

### 2️⃣ Train Target Model + Run FiST Attack

```bash
python main.py --dataset_name location --arch mlp --attack_name fist --train_model --attack
```

This will:
1. Train the target model  
2. Save the overfitting gap  
3. Run the FiST attack  

---

### 3️⃣ Attacking DP-Trained Models

Train shadow model with DP-SGD:

```bash
python main.py --dataset_name location --arch mlp --train_shadow --use_DP --noise 0.3 --norm 5 --delta 1e-5
```

Train target model with DP-SGD + attack:

```bash
python main.py --dataset_name location --arch mlp --attack_name fist --train_model --attack
```

---

### 4️⃣ Plotting Results

```bash
# ROC curves
python main.py --plot --plot_results roc --dataset_name location --arch mlp --attack_name fist

# Threshold curves
python main.py --plot --plot_results th --dataset_name location --arch mlp --attack_name fist
```

Add `--fist_cluster` for clustering visualizations from the paper.

---

## 📚 Supported Datasets

**Image datasets**:
- CIFAR-10  
- CIFAR-100  
- Fashion-MNIST  
- STL-10  

**Non-image datasets**:
- Location (Foursquare check-ins)  
- Texas-100  
- Adult (UCI Census Income)  
- Purchase-100  

> Only `adult/` and `location/` are included for demonstration.  
> For other datasets, download from original sources and place under `data/{dataset_name}/`.

---

## 🏗 Supported Architectures

- `--arch mlp` → Non-image datasets  
- `--arch cnn` or `--arch vgg16` → Image datasets  

Example:

```bash
python main.py --dataset_name adult --arch mlp --attack_name fist --train_model --attack
```

---

## 📈 Outputs

When running `--attack` or `--plot`, the following are saved:

- ROC curve plots (`roc_curves/`)  
- TPR/FPR CSVs  
- Learned threshold plots (`threshold_plots/`)  
- Excel summaries (`roc_curves/*.xlsx`)  
- Prediction vectors and logs (`results/`)  

---

## 📄 Citation

If you use FiST in your research, please cite:

```bibtex
@article{FiST2025,
  title     = {Fitting In to Stand Out: Contrastive Membership Inference Attacks via Learned Adversarial Perturbations},
  author    = {Your Name and Coauthors},
  journal   = {Under Review},
  year      = {2025}
}
```
