"""
下游任务评估：用合成数据训练 LightGBM 分类器，在真实数据上验证
用法：python evaluate_downstream.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# 加载数据
real_all = np.load("data/mimic4/mimic4_data.npy", mmap_mode='r')
test_idx = np.load("data/mimic4/test_indices.npy")
train_idx = np.load("data/mimic4/train_indices.npy")

real_train = real_all[train_idx]
real_test = real_all[test_idx]
syn = np.load("results/mimic4/samples/all_x.npy")

print(f"真实训练集: {real_train.shape}")
print(f"真实测试集: {real_test.shape}")
print(f"合成数据: {syn.shape}")

# 选择高频代码作为预测目标
target_codes = [
    # code, name
    (0, "401: Hypertension"),
    (1, "E87: Fluid/electrolyte"),
    (2, "E78: Lipid disorders"),
    (3, "Z79: Long-term drug"),
    (4, "E11: Type 2 diabetes"),
    (5, "272: Lipid metabolism"),
    (6, "I10: Hypertension"),
    (7, "250: Diabetes"),
    (8, "V58: Aftercare"),
    (9, "Z87: Personal history"),
]

results = []
for col, name in target_codes:
    # 跳过目标列过少的样本
    pos_ratio = real_train[:, col].mean()
    if pos_ratio < 0.01 or pos_ratio > 0.99:
        print(f"  {name}: 跳过 (prevalence={pos_ratio:.4f})")
        continue

    # 特征 = 去掉目标列，标签 = 目标列
    mask = np.ones(real_train.shape[1], dtype=bool)
    mask[col] = False

    # --- 用真实数据训练 ---
    X_real = real_train[:, mask]
    y_real = real_train[:, col]
    X_t = real_test[:, mask]
    y_t = real_test[:, col]

    clf_real = LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, verbose=-1, n_jobs=-1
    )
    clf_real.fit(X_real, y_real)
    y_pred_real = clf_real.predict_proba(X_t)[:, 1]
    auc_real = roc_auc_score(y_t, y_pred_real)
    ap_real = average_precision_score(y_t, y_pred_real)

    # --- 用合成数据训练 ---
    X_syn = syn[:, mask]
    y_syn = syn[:, col]

    clf_syn = LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, verbose=-1, n_jobs=-1
    )
    clf_syn.fit(X_syn, y_syn)
    y_pred_syn = clf_syn.predict_proba(X_t)[:, 1]
    auc_syn = roc_auc_score(y_t, y_pred_syn)
    ap_syn = average_precision_score(y_t, y_pred_syn)

    results.append((name, pos_ratio, auc_real, auc_syn, ap_real, ap_syn))
    print(f"  {name:35s} | real={pos_ratio:.3f} | AUROC: real={auc_real:.4f} syn={auc_syn:.4f} | AP: real={ap_real:.4f} syn={ap_syn:.4f}")

# 汇总
if results:
    print("\n" + "=" * 70)
    avg_auc_real = np.mean([r[2] for r in results])
    avg_auc_syn = np.mean([r[3] for r in results])
    avg_ap_real = np.mean([r[4] for r in results])
    avg_ap_syn = np.mean([r[5] for r in results])
    print(f"平均 AUROC:  真实={avg_auc_real:.4f}  合成={avg_auc_syn:.4f}  (ratio={avg_auc_syn/avg_auc_real:.4f})")
    print(f"平均 AvgPrec: 真实={avg_ap_real:.4f}  合成={avg_ap_syn:.4f}  (ratio={avg_ap_syn/avg_ap_real:.4f})")
    print("=" * 70)
