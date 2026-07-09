"""
评估合成 EHR 数据质量：计算 corr, NZC, CMD 指标
用法：python evaluate_mimic4.py
"""

import numpy as np
from scipy.stats import pearsonr

# 加载真实数据（测试集）
real_all = np.load("data/mimic4/mimic4_data.npy", mmap_mode='r')
test_idx = np.load("data/mimic4/test_indices.npy")
real = real_all[test_idx]  # 测试集
print(f"真实数据: {real.shape}")

# 加载合成数据
syn = np.load("results/mimic4/samples/all_x.npy")
print(f"合成数据: {syn.shape}")

# 1. Prevalence Correlation + NZC
real_prevalence = np.mean(real, axis=0)
syn_prevalence = np.mean(syn, axis=0)
nzc = int(np.sum(syn_prevalence > 0))
corr = pearsonr(syn_prevalence, real_prevalence)[0]
print(f"\n{'='*40}")
print(f"NZC (non-zero columns): {nzc} / {syn.shape[1]}")
print(f"Prevalence Correlation: {corr:.4f}")

# 2. CMD (Central Moment Discrepancy)
# CMD = sum of L1 differences between central moments (mean, var, skew, kurtosis)
real_samples = min(real.shape[0], 50000)
syn_samples = min(syn.shape[0], 50000)

real_subset = real[:real_samples]
syn_subset = syn[:syn_samples]

# 逐特征计算前4阶中心矩
cmd_scores = []
n_features = syn.shape[1]
for i in range(n_features):
    r = real_subset[:, i]
    s = syn_subset[:, i]

    # 跳过全零特征
    if r.mean() == 0 and s.mean() == 0:
        continue

    # 矩差异
    m1_r, m1_s = r.mean(), s.mean()
    m2_r, m2_s = r.var(), s.var()
    m3_r, m3_s = np.mean((r - m1_r)**3), np.mean((s - m1_s)**3)
    m4_r, m4_s = np.mean((r - m1_r)**4), np.mean((s - m1_s)**4)

    cmd = abs(m1_r - m1_s) + abs(m2_r - m2_s) + abs(m3_r - m3_s) + abs(m4_r - m4_s)
    cmd_scores.append(cmd)

cmd = np.mean(cmd_scores)
print(f"CMD (Central Moment Discrepancy): {cmd:.4f}")
print(f"{'='*40}")

# 稀疏度对比
print(f"\n真实数据稀疏度: {1 - real.mean():.4f}")
print(f"合成数据稀疏度: {1 - syn.mean():.4f}")

# 论文目标参考
print(f"\n{'='*40}")
print("论文目标 (Table 1):")
print("  NZC  ≈ 1770")
print("  CMD  ≈ 7.769")
print("  corr > 0.99")
print(f"{'='*40}")
