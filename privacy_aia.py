"""
属性推断攻击 (Attribute Inference Attack, AIA) —— 从其余特征推断敏感诊断。

场景：攻击者拿到一条记录的大部分 ICD 特征，尝试推断某个"敏感属性"
（此处为某个敏感诊断代码是否存在）。比较两类攻击者：
  - 在真实数据上训练的攻击器（真实数据的固有属性推断风险，基线）
  - 在合成数据上训练的攻击器（合成数据是否引入额外风险）
若 合成训练 AUROC ≈ 真实训练 AUROC，说明合成数据没有增加重识别风险；
若 合成训练 AUROC 明显更高，则是隐私隐患。

复用 evaluate_tstr.py 的 LogisticRegression (lbfgs, max_iter=1000) 结构。

用法（服务器）：
    python privacy_aia.py \
        --data data/mimic4/mimic4_data.npy \
        --train_idx data/mimic4/train_indices.npy \
        --test_idx data/mimic4/test_indices.npy \
        --syn results/mimic4/samples/all_x_large.npy \
        --n_code 2083 \
        --out results/mimic4/privacy/aia.json
"""
import os
import sys
import json
import argparse
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

# 优先作为敏感属性目标的诊断代码（命中才用；命中不足则用 prevalence top 补足）
SENSITIVE_CODES = [
    '305',   # Drug abuse
    'F32',   # Depression
    'F41',   # Anxiety
    'E10',   # Type 1 diabetes
    'F10',   # Alcohol-related
    '304',   # Drug dependence
    'V15',   # Personal history (incl. risk factors)
    'Z87',   # Personal history of diseases
]

CODE_DESC = {
    '401': 'Hypertension', 'E87': 'Fluid/electrolyte', 'E78': 'Lipid disorders',
    'Z79': 'Long-term drug', 'E11': 'Type 2 diabetes', '272': 'Lipid metab.',
    'I10': 'Hypertension', '250': 'Diabetes', 'V58': 'Aftercare',
    'Z87': 'History of disease', 'I25': 'Chronic ischemic HD', '428': 'Heart failure',
    'V45': 'Post-surgical', '276': 'Fluid/electrolyte', '427': 'Dysrhythmias',
    '530': 'Esophageal disease', '414': 'Coronary atherosclerosis',
    '305': 'Drug abuse', 'F32': 'Depression', 'I48': 'Atrial fib.',
    'F41': 'Anxiety', 'N18': 'CKD', 'E10': 'Type 1 diabetes',
}


def load_data(data_path, train_idx_path, test_idx_path, syn_path, max_syn=100000):
    real_all = np.load(data_path, mmap_mode='r')
    train_idx = np.load(train_idx_path)
    test_idx = np.load(test_idx_path)
    real_train = real_all[train_idx].astype(np.float32)
    real_test = real_all[test_idx].astype(np.float32)

    syn = np.load(syn_path)
    if syn.ndim == 3:
        syn = syn.squeeze(0)
    syn = syn.astype(np.float32)
    if len(syn) > max_syn:
        syn = syn[:max_syn]

    print(f'  real_train: {real_train.shape}')
    print(f'  real_test : {real_test.shape}')
    print(f'  syn       : {syn.shape}')
    return real_train, real_test, syn


def load_codes(codes_path, n_code):
    if codes_path and os.path.exists(codes_path):
        with open(codes_path) as f:
            meta = json.load(f)
        codes = meta.get('selected_codes', meta.get('codes', None))
        if codes is not None and len(codes) == n_code:
            return list(codes)
    return None


def pick_targets(real_train, n_code, codes, n_targets=10):
    """优先选敏感代码，命中不足用 prevalence top 补足。返回 (indices, names)。"""
    prevalence = real_train[:, :n_code].mean(axis=0)
    idx = []
    if codes is not None:
        code_to_idx = {c: i for i, c in enumerate(codes)}
        for c in SENSITIVE_CODES:
            if c in code_to_idx and code_to_idx[c] not in idx:
                idx.append(code_to_idx[c])

    # 用 prevalence 高的补足到 n_targets（避开已选的）
    order = np.argsort(-prevalence)
    for i in order:
        if len(idx) >= n_targets:
            break
        if i not in idx:
            idx.append(int(i))

    idx = idx[:n_targets]
    names = [codes[i] if codes else f'col_{i}' for i in idx]
    return idx, names


def evaluate_task(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, y_prob), average_precision_score(y_te, y_prob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/mimic4/mimic4_data.npy')
    parser.add_argument('--train_idx', default='data/mimic4/train_indices.npy')
    parser.add_argument('--test_idx', default='data/mimic4/test_indices.npy')
    parser.add_argument('--syn', default='results/mimic4/samples/all_x_large.npy')
    parser.add_argument('--codes', default='data/mimic4/dem_metadata.json',
                        help='含 selected_codes 的元数据文件，用于代码标签')
    parser.add_argument('--n_code', type=int, default=2083)
    parser.add_argument('--n_targets', type=int, default=10)
    parser.add_argument('--out', default='results/mimic4/privacy/aia.json')
    opt = parser.parse_args()

    print('加载数据...')
    real_train, real_test, syn = load_data(
        opt.data, opt.train_idx, opt.test_idx, opt.syn)
    n_code = opt.n_code
    codes = load_codes(opt.codes, n_code)

    targets, names = pick_targets(real_train, n_code, codes, opt.n_targets)
    print(f'\n目标属性数: {len(targets)}')

    rows = []
    for col, name in zip(targets, names):
        mask = np.ones(n_code, dtype=bool)
        mask[col] = False
        X_tr, y_tr = real_train[:, mask], real_train[:, col]
        X_te, y_te = real_test[:, mask], real_test[:, col]

        auc_real, ap_real = evaluate_task(X_tr, y_tr, X_te, y_te)   # 真实数据训练的攻击器
        auc_syn, ap_syn = evaluate_task(syn[:, mask], syn[:, col], X_te, y_te)  # 合成训练的攻击器

        prevalence = y_tr.mean()
        rows.append({
            'code': name,
            'desc': CODE_DESC.get(name, ''),
            'prevalence': float(prevalence),
            'auroc_real': float(auc_real),
            'auroc_syn': float(auc_syn),
            'avgprec_real': float(ap_real),
            'avgprec_syn': float(ap_syn),
            'excess_risk': float(auc_syn - auc_real),  # >0 表示合成数据增加攻击风险
        })

    # 汇总
    avg_r = np.mean([r['auroc_real'] for r in rows])
    avg_s = np.mean([r['auroc_syn'] for r in rows])

    print(f"\n  {'代码':>6} {'名称':<22} {'prev':>6} {'AUROC_real':>10} "
          f"{'AUROC_syn':>10} {'excess':>8}")
    print(f"  {'-'*6} {'-'*22} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for r in rows:
        print(f"  {r['code']:>6} {r['desc']:<22} {r['prevalence']:>6.3f} "
              f"{r['auroc_real']:>10.4f} {r['auroc_syn']:>10.4f} "
              f"{r['excess_risk']:>+8.4f}")
    print(f"\n  平均 AUROC: 真实攻击={avg_r:.4f}  合成攻击={avg_s:.4f}  "
          f"超额风险={avg_s - avg_r:+.4f}")

    summary = {
        'n_targets': len(rows),
        'avg_auroc_real': float(avg_r),
        'avg_auroc_syn': float(avg_s),
        'avg_excess_risk': float(avg_s - avg_r),
        'rows': rows,
    }
    os.makedirs(os.path.dirname(opt.out), exist_ok=True)
    with open(opt.out, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'\n结果保存: {opt.out}')


if __name__ == '__main__':
    main()