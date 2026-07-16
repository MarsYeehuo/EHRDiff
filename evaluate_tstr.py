"""
TSTR (Train on Synthetic, Test on Real) 评估

用 Logistic Regression 在多个预测任务上比较真实数据和合成数据的效果。

使用方法：
  1. 先加载 real_data + 生成 syn_data
  2. 运行 python evaluate_tstr.py

数据格式：mimic4_dem_data.npy (453949, 2092)
  - 前 2083 列：ICD 代码（二进制）
  - 后 9 列：人口学特征
"""

import numpy as np
import json
import os
import warnings
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
warnings.filterwarnings('ignore')

DATA_DIR = "data/mimic4"
RESULT_DIR = "results/mimic4_dem/samples"

# 人口学特征列名（后 9 列）
DEM_COLS = ['gender', 'age', 'adm_emerg', 'adm_obs', 'adm_urg', 'adm_other',
            'ins_medicare', 'ins_medicaid', 'ins_other']
N_DEM = len(DEM_COLS)


def load_data():
    """加载真实数据和合成数据"""
    print("=" * 60)
    print("加载数据")
    print("=" * 60)

    real_all = np.load(os.path.join(DATA_DIR, "mimic4_dem_data.npy"), mmap_mode='r')
    train_idx = np.load(os.path.join(DATA_DIR, "train_indices.npy"))
    test_idx = np.load(os.path.join(DATA_DIR, "test_indices.npy"))

    real_train = real_all[train_idx]
    real_test = real_all[test_idx]
    print(f"  真实训练集: {real_train.shape}")
    print(f"  真实测试集: {real_test.shape}")

    # 加载合成数据
    syn_path = os.path.join(RESULT_DIR, "all_x.npy")
    if not os.path.exists(syn_path):
        print(f"\n  ⚠ 合成数据未找到: {syn_path}")
        print(f"  请先生成样本：python main.py --config configs/mimic4/sample_dem_edm_2092.yaml ...")
        return None, None, None, None

    syn = np.load(syn_path)
    if syn.ndim == 3:
        syn = syn.squeeze(0)
    print(f"  合成数据: {syn.shape}")

    return real_train, real_test, syn, real_all.shape[1]


def load_metadata():
    """加载 ICD 代码映射"""
    meta_path = os.path.join(DATA_DIR, "dem_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta['selected_codes'], meta['n_code']
    # fallback: 从已有的 metadata.json 加载（原始 2241 版本）
    fallback = os.path.join(DATA_DIR, "metadata.json")
    if os.path.exists(fallback):
        with open(fallback) as f:
            meta = json.load(f)
        return meta['selected_codes'], meta['feature_dim'] - N_DEM
    return None, None


def pick_targets(real_train, n_code, n_targets=15):
    """选择 prevalence 适中的高频 ICD 代码作为预测目标"""
    prevalence = real_train[:, :n_code].mean(axis=0)
    # 选 prevalence 在 [0.03, 0.97] 之间的 top 代码
    valid = np.where((prevalence > 0.03) & (prevalence < 0.97))[0]
    sorted_idx = valid[np.argsort(-prevalence[valid])]
    return sorted_idx[:n_targets]


def evaluate_task(X_train, y_train, X_test, y_test, name=""):
    """训练并评估一个二分类任务"""
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    return auc, ap


def tstr_evaluation():
    """主评估流程"""
    data = load_data()
    if data[0] is None:
        return
    real_train, real_test, syn, total_dim = data

    codes, n_code = load_metadata()
    if n_code is None:
        n_code = total_dim - N_DEM

    print(f"\nICD 维度: {n_code}, 人口学维度: {N_DEM}, 总计: {total_dim}")

    # =============================================
    # 任务 1: ICD 代码预测（使用所有其他特征）
    # =============================================
    print("\n" + "=" * 60)
    print("任务 1: 逐 ICD 代码预测 (TSTR)")
    print("使用所有其他特征预测目标 ICD 代码")
    print("=" * 60)

    targets = pick_targets(real_train, n_code)
    print(f"  目标代码数: {len(targets)}\n")
    print(f"  {'代码':>6} {'名称':<30} {'prevalence':>10} {'AUROC_real':>10} {'AUROC_syn':>10} {'AP_real':>9} {'AP_syn':>9}")
    print(f"  {'-'*6} {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*9} {'-'*9}")

    results = []
    for col in targets:
        code_name = codes[col] if codes and col < len(codes) else f"col_{col}"
        mask = np.ones(total_dim, dtype=bool)
        mask[col] = False

        # 真实数据
        auc_real, ap_real = evaluate_task(
            real_train[:, mask], real_train[:, col],
            real_test[:, mask], real_test[:, col],
        )
        # 合成数据
        auc_syn, ap_syn = evaluate_task(
            syn[:, mask], syn[:, col],
            real_test[:, mask], real_test[:, col],
        )
        prevalence = real_train[:, col].mean()
        results.append((code_name, prevalence, auc_real, auc_syn, ap_real, ap_syn))

        print(f"  {code_name:>6} {get_code_desc(code_name):<30} {prevalence:>10.3f} {auc_real:>10.4f} {auc_syn:>10.4f} {ap_real:>9.4f} {ap_syn:>9.4f}")

    # 汇总
    avg_auc_r = np.mean([r[2] for r in results])
    avg_auc_s = np.mean([r[3] for r in results])
    avg_ap_r = np.mean([r[4] for r in results])
    avg_ap_s = np.mean([r[5] for r in results])
    print(f"\n  平均 AUROC:  真实={avg_auc_r:.4f}  合成={avg_auc_s:.4f}  (ratio={avg_auc_s/avg_auc_r:.4f})")
    print(f"  平均 AvgPrec: 真实={avg_ap_r:.4f}  合成={avg_ap_s:.4f}  (ratio={avg_ap_s/avg_ap_r:.4f})")

    # =============================================
    # 任务 2: 人口学特征预测（从 ICD 代码预测）
    # =============================================
    print("\n" + "=" * 60)
    print("任务 2: 人口学特征预测 (从 ICD 代码)")
    print("用 ICD 代码预测性别、入院类型、保险")
    print("=" * 60)

    dem_tasks = [
        (0, 'gender (F=1)', 'binary'),
        (2, 'admission: emergency', 'binary'),
        (4, 'admission: urgent', 'binary'),
    ]

    print(f"\n  {'任务':<25} {'prevalence':>10} {'AUROC_real':>10} {'AUROC_syn':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    icd_cols = list(range(n_code))
    dem_results = []
    for dem_idx, task_name, _ in dem_tasks:
        dem_col = n_code + dem_idx

        # 真实
        auc_real, ap_real = evaluate_task(
            real_train[:, icd_cols], real_train[:, dem_col],
            real_test[:, icd_cols], real_test[:, dem_col],
        )
        # 合成
        auc_syn, ap_syn = evaluate_task(
            syn[:, icd_cols], syn[:, dem_col],
            real_test[:, icd_cols], real_test[:, dem_col],
        )
        prevalence = real_train[:, dem_col].mean()
        dem_results.append((task_name, prevalence, auc_real, auc_syn))
        print(f"  {task_name:<25} {prevalence:>10.3f} {auc_real:>10.4f} {auc_syn:>10.4f}")

    avg_dem_auc_r = np.mean([r[2] for r in dem_results])
    avg_dem_auc_s = np.mean([r[3] for r in dem_results])
    print(f"\n  平均 AUROC:  真实={avg_dem_auc_r:.4f}  合成={avg_dem_auc_s:.4f}  (ratio={avg_dem_auc_s/avg_dem_auc_r:.4f})")

    # =============================================
    # 总结
    # =============================================
    print("\n" + "=" * 60)
    print("TSTR 评估总结")
    print("=" * 60)
    retention_auc = avg_auc_s / avg_auc_r * 100
    retention_ap = avg_ap_s / avg_ap_r * 100
    retention_dem = avg_dem_auc_s / avg_dem_auc_r * 100
    print(f"  ICD 预测 AUROC 保留率: {retention_auc:.1f}%")
    print(f"  ICD 预测 AvgPrec 保留率: {retention_ap:.1f}%")
    print(f"  人口学预测 AUROC 保留率: {retention_dem:.1f}%")


def get_code_desc(code):
    desc = {
        '401': 'Hypertension', 'E87': 'Fluid/electrolyte', 'E78': 'Lipid disorders',
        'Z79': 'Long-term drug', 'E11': 'Type 2 diabetes', '272': 'Lipid metab.',
        'I10': 'Hypertension', '250': 'Diabetes', 'V58': 'Aftercare',
        'Z87': 'History of disease', 'I25': 'Chronic ischemic HD', '428': 'Heart failure',
        'V45': 'Post-surgical', 'Y92': 'Place of occurrence', '276': 'Fluid/electrolyte',
        '427': 'Dysrhythmias', 'K21': 'GERD', '530': 'Esophageal disease',
        '414': 'Coronary atherosclerosis', '305': 'Drug abuse',
        'F32': 'Depression', 'I48': 'Atrial fib.', 'I50': 'Heart failure',
        'F41': 'Anxiety', 'N18': 'CKD', 'N17': 'AKI', 'G47': 'Sleep disorders',
        '585': 'CKD', '403': 'Hypertensive CKD', '584': 'AKI',
        'E10': 'Type 1 diabetes', 'E13': 'Other diabetes',
    }
    return desc.get(code, '')


if __name__ == '__main__':
    tstr_evaluation()
