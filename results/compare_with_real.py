import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

real_data = np.load("data/mimic3/mimic_data.npy")
train_indices = np.load("data/mimic3/train_indices.npy")
real_train = real_data[train_indices]

gen_data = np.load("./results/mimic_edm/samples/all_x.npy")

print("=== Real Data vs Generated Data ===")
print(f"Real data shape: {real_train.shape}")
print(f"Generated data shape: {gen_data.shape}")

# 1. 计算 prevalence
real_prevalence = real_train.mean(axis=0)
gen_prevalence = gen_data.mean(axis=0)

# 2. 计算相关性
correlation, p_value = pearsonr(real_prevalence, gen_prevalence)
print(f"\nPrevalence Correlation: {correlation:.4f} (p-value: {p_value:.4f})")

# 3. 非零特征数
real_nzc = np.sum(real_train.sum(axis=0) > 0)
gen_nzc = np.sum(gen_data.sum(axis=0) > 0)
print(f"Real data non-zero features: {real_nzc}")
print(f"Generated data non-zero features: {gen_nzc}")

# 4. 绘制散点图
plt.figure(figsize=(10, 8))
plt.scatter(real_prevalence, gen_prevalence, alpha=0.5, s=5)
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Match', alpha=0.8)
plt.xlabel('Real Prevalence')
plt.ylabel('Generated Prevalence')
plt.title(f'Dimension-wise Prevalence (Correlation: {correlation:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('prevalence_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n Scatter plot saved to prevalence_comparison.png")