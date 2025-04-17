import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import auc

# 1. Load each CSV file into a separate DataFrame
df_apcmia_03 = pd.read_csv("apcmia_noise_0.3_10.csv")
df_apcmia_18 = pd.read_csv("apcmia_noise_1.8_10.csv")
df_baseline_03 = pd.read_csv("baseline_noise_0.3_10.csv")
df_baseline_18 = pd.read_csv("baseline_noise_1.8_10.csv")


# 4. Extract FPR/TPR columns and compute AUC
# APCMIA, sigma=0.3
fpr_a03 = df_apcmia_03["FPR"]
tpr_a03 = df_apcmia_03["TPR"]
auc_a03 = auc(fpr_a03, tpr_a03)  # area under curve

# Baseline, sigma=0.3
fpr_b03 = df_baseline_03["FPR"]
tpr_b03 = df_baseline_03["TPR"]
auc_b03 = auc(fpr_b03, tpr_b03)

# APCMIA, sigma=1.8
fpr_a18 = df_apcmia_18["FPR"]
tpr_a18 = df_apcmia_18["TPR"]
auc_a18 = auc(fpr_a18, tpr_a18)

# Baseline, sigma=1.8
fpr_b18 = df_baseline_18["FPR"]
tpr_b18 = df_baseline_18["TPR"]
auc_b18 = auc(fpr_b18, tpr_b18)

params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 14,
   'xtick.labelsize': 14,
   'ytick.labelsize': 14,
   'text.usetex': False,
   'figure.figsize': [8, 5],
   "font.family" : "serif"
   }
plt.rcParams.update(params)
plt.grid(linestyle='dotted')


# 1. Known or reported epsilons for each noise level (per your data)
epsilon_for_sigma = {
    0.3: 256.8,
    1.8: 3.16
}
clipping_bound = 10
# 2. Create a figure
# plt.figure(figsize=(8, 6))

# APCMIA (noise=0.3)
plt.plot(
    fpr_a03, tpr_a03, 
    label=f"APCMIA: σ=0.3, ε={epsilon_for_sigma[0.3]}, AUC={auc_a03:.2f}"
)

# APCMIA (noise=1.8)
plt.plot(
    fpr_a18, tpr_a18, 
    label=f"APCMIA: σ=1.8, ε={epsilon_for_sigma[1.8]}, AUC={auc_a18:.2f}"
)

# Baseline (noise=0.3)
plt.plot(
    fpr_b03, tpr_b03, '--', 
    label=f"Baseline: σ=0.3, ε={epsilon_for_sigma[0.3]}, AUC={auc_b03:.2f}"
)


# Baseline (noise=1.8)
plt.plot(
    fpr_b18, tpr_b18, '--', 
    label=f"Baseline: σ=1.8, ε={epsilon_for_sigma[1.8]} , AUC={auc_b18:.2f}"
)

# 7. Customize plot
plt.xlabel("FPR", fontsize=14)
plt.ylabel("TPR", fontsize=14)

# plt.xlabel("False Positive Rate (FPR)")
# plt.ylabel("True Positive Rate (TPR)")

# plt.title("Membership Inference Attack ROC Curves (C=10)")
plt.legend()

legend = plt.legend(loc="lower right")
frame = legend.get_frame()
frame.set_facecolor('0.95')
frame.set_edgecolor('0.91')
plt.subplots_adjust(left=0.45)

# plt.grid(True)
plt.tight_layout()

plt.semilogx()
plt.semilogy()
plt.xlim(1e-5, 1)
plt.ylim(1e-5, 1.01)

plt.plot([0, 1], [0, 1], ls="--", color="gray", label="Random Classifier")

filename = f"location_roc_curves_.pdf"
    # if save_path:
# plt.savefig(filename, format='pdf', dpi=300)
# print(f"Figure saved to {save_path}")


# 8. Show the figure
plt.show()
