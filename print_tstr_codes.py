"""打印 logistic TSTR 协议的 15 个 ICD 预测目标（代码 + 训练 prevalence + AUROC retention）。

选取标准与 evaluate_tstr_multi.py 的 pick_targets 完全一致：训练集 prevalence
∈ (0.03, 0.97)，按 prevalence 降序取前 15。retention 从
results/mimic4_dem/tstr_multi_logistic.json 读取（若存在）。

用法（服务器）:
    python print_tstr_codes.py
"""
import json

import numpy as np

DATA_DIR = 'data/mimic4'
SYN_RET_JSON = 'results/mimic4_dem/tstr_multi_logistic.json'


def main():
    meta = json.load(open(f'{DATA_DIR}/dem_metadata.json'))
    codes = meta['selected_codes']
    n_code = meta.get('n_code', 2083)

    real_all = np.load(f'{DATA_DIR}/mimic4_dem_data.npy', mmap_mode='r')
    train_idx = np.load(f'{DATA_DIR}/train_indices.npy')
    X = real_all[train_idx, :n_code].astype(np.float32)

    p = X.mean(axis=0)
    valid = np.where((p > 0.03) & (p < 0.97))[0]
    idx = valid[np.argsort(-p[valid])][:15]

    try:
        j = json.load(open(SYN_RET_JSON))
        ret = {t['code']: 100 * t['retention_mean'] for t in j['tasks']}
        print(f'(retention loaded from {SYN_RET_JSON})')
    except Exception as e:
        ret = {}
        print(f'(retention unavailable: {e})')

    print('--- LaTeX rows (code & prevalence & retention) ---')
    for i in idx:
        c = codes[i]
        r = f'{ret[c]:.1f}' if c in ret else '--'
        print(f'    {c} & {p[i]:.3f} & {r} \\\\')
    print('--- plain (code prevalence) ---')
    for i in idx:
        print(codes[i], round(p[i], 4))


if __name__ == '__main__':
    main()
