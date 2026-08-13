"""
下游任务评估（正确代码标签版）：用合成数据训练 LightGBM，在真实数据上验证。

从 metadata.json 读取 selected_codes 作为代码标签（不再硬编码），默认取 top-10
高频代码（indices 0-9，与 tost_equivalence.py 一致），合成数据默认 all_x_large.npy。
输出 AUROC / AvgPrec 及保留率，并打印可直接贴进 LaTeX 的表格。

用法（服务器）：
    python evaluate_downstream.py \
        --syn results/mimic4/samples/all_x_large.npy \
        --out results/mimic4/downstream_table.json
"""
import os
import json
import argparse
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/mimic4/mimic4_data.npy')
    parser.add_argument('--train_idx', default='data/mimic4/train_indices.npy')
    parser.add_argument('--test_idx', default='data/mimic4/test_indices.npy')
    parser.add_argument('--codes', default='data/mimic4/metadata.json')
    parser.add_argument('--syn', default='results/mimic4/samples/all_x_large.npy')
    parser.add_argument('--n_targets', type=int, default=10)
    parser.add_argument('--out', default='results/mimic4/downstream_table.json')
    opt = parser.parse_args()

    real_all = np.load(opt.data, mmap_mode='r')
    train_idx = np.load(opt.train_idx)
    test_idx = np.load(opt.test_idx)
    real_train = real_all[train_idx].astype(np.float32)
    real_test = real_all[test_idx].astype(np.float32)

    syn = np.load(opt.syn)
    if syn.ndim == 3:
        syn = syn.squeeze(0)
    syn = syn.astype(np.float32)

    codes = None
    if os.path.exists(opt.codes):
        with open(opt.codes) as f:
            meta = json.load(f)
        codes = meta.get('selected_codes', meta.get('codes', None))
    targets = list(range(min(opt.n_targets, real_train.shape[1])))

    print(f'  real_train: {real_train.shape}')
    print(f'  real_test : {real_test.shape}')
    print(f'  synthetic : {syn.shape}')

    rows = []
    for col in targets:
        mask = np.ones(real_train.shape[1], dtype=bool)
        mask[col] = False
        y_tr = real_train[:, col]
        y_te = real_test[:, col]
        code = codes[col] if codes else f'col_{col}'
        prev = float(y_tr.mean())

        clf = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                             random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(real_train[:, mask], y_tr)
        p = clf.predict_proba(real_test[:, mask])[:, 1]
        auc_r, ap_r = float(roc_auc_score(y_te, p)), float(average_precision_score(y_te, p))

        clf.fit(syn[:, mask], syn[:, col])
        p = clf.predict_proba(real_test[:, mask])[:, 1]
        auc_s, ap_s = float(roc_auc_score(y_te, p)), float(average_precision_score(y_te, p))

        rows.append({'code': code, 'prevalence': prev,
                     'auc_real': auc_r, 'auc_syn': auc_s,
                     'ap_real': ap_r, 'ap_syn': ap_s})
        print(f'  {code:>5}  prev={prev:.3f}  '
              f'AUROC {auc_r:.4f}/{auc_s:.4f}  AP {ap_r:.4f}/{ap_s:.4f}')

    avg = lambda k: np.mean([r[k] for r in rows])
    auc_r_m, auc_s_m = avg('auc_real'), avg('auc_syn')
    ap_r_m, ap_s_m = avg('ap_real'), avg('ap_syn')
    print(f'\n  平均 AUROC: real={auc_r_m:.4f} syn={auc_s_m:.4f}  '
          f'保留率={auc_s_m/auc_r_m*100:.1f}%')
    print(f'  平均 AvgPrec: real={ap_r_m:.4f} syn={ap_s_m:.4f}  '
          f'保留率={ap_s_m/ap_r_m*100:.1f}%')

    # LaTeX 表格（可直接粘贴到 dissertation）
    print('\n  --- LaTeX 表格 ---')
    print(r'  \begin{tabular}{lcccccc}')
    print(r'    \toprule')
    print(r'    Code & Prev & AUROC$_{R}$ & AUROC$_{S}$ & AP$_{R}$ & AP$_{S}$ \\')
    print(r'    \midrule')
    for r in rows:
        print(f'    {r["code"]:>4} & {r["prevalence"]:.3f} & '
              f'{r["auc_real"]:.4f} & {r["auc_syn"]:.4f} & '
              f'{r["ap_real"]:.4f} & {r["ap_syn"]:.4f} \\\\')
    print(f'    \\textbf{{Mean}} & & {auc_r_m:.4f} & {auc_s_m:.4f} & '
          f'{ap_r_m:.4f} & {ap_s_m:.4f} \\\\')
    print(f'    \\textbf{{Retention}} & & & {auc_s_m/auc_r_m*100:.1f}\\% & & '
          f'{ap_s_m/ap_r_m*100:.1f}\\% \\\\')
    print(r'    \bottomrule')
    print(r'  \end{tabular}')

    os.makedirs(os.path.dirname(opt.out), exist_ok=True)
    with open(opt.out, 'w') as f:
        json.dump({'rows': rows,
                   'mean_auc_real': auc_r_m, 'mean_auc_syn': auc_s_m,
                   'mean_ap_real': ap_r_m, 'mean_ap_syn': ap_s_m,
                   'retention_auc_pct': auc_s_m / auc_r_m * 100,
                   'retention_ap_pct': ap_s_m / ap_r_m * 100}, f, indent=2)
    print(f'\n结果保存: {opt.out}')


if __name__ == '__main__':
    main()
