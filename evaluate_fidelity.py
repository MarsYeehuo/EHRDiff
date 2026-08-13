"""
保真度评估：Prevalence Correlation / NZC / CMD（逐特征 prevalence 最大偏差）。

参数化版，可对任意合成样本集运行（无条件 2083 模型、人口统计 2092 模型皆可）。
CMD 定义为 max_j |p_syn_j - p_real_j|（与 dissertation 的 Methods 定义一致）。

用法（服务器）：
    # 无条件模型
    python evaluate_fidelity.py \
        --syn results/mimic4/samples/all_x_large.npy \
        --real data/mimic4/mimic4_data.npy

    # 人口统计模型（2092 维，只取前 2083 个 ICD 列与无条件模型对齐比较）
    python evaluate_fidelity.py \
        --syn results/mimic4_dem/samples/all_x.npy \
        --real data/mimic4/mimic4_data.npy \
        --n_code 2083

    python evaluate_fidelity.py --syn ... --out results/mimic4/fidelity.json
"""
import os
import json
import argparse

import numpy as np
from scipy.stats import pearsonr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--syn', required=True, help='合成样本 npy 路径')
    parser.add_argument('--real', default='data/mimic4/mimic4_data.npy')
    parser.add_argument('--train_idx', default='data/mimic4/train_indices.npy')
    parser.add_argument('--test_idx', default='data/mimic4/test_indices.npy')
    parser.add_argument('--n_code', type=int, default=2083,
                        help='取前 N 列作 ICD 特征（人口统计模型为 2083）')
    parser.add_argument('--max_syn', type=int, default=None,
                        help='合成样本截取条数。做公平对比时设为与另一模型相同'
                             '（NZC 对样本量极敏感，10k vs 100k 不可比）')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--out', default=None, help='JSON 输出路径（可选）')
    opt = parser.parse_args()

    real_all = np.load(opt.real, mmap_mode='r')
    test_idx = np.load(opt.test_idx)
    real = real_all[test_idx, :opt.n_code].astype(np.float32)

    syn = np.load(opt.syn)
    if syn.ndim == 3:
        syn = syn.squeeze(0)
    syn = syn[:, :opt.n_code].astype(np.float32)
    if opt.max_syn and len(syn) > opt.max_syn:
        rng = np.random.default_rng(opt.seed)
        syn = syn[rng.choice(len(syn), opt.max_syn, replace=False)]

    print(f'  real (test): {real.shape}')
    print(f'  synthetic  : {syn.shape}')

    p_real = real.mean(axis=0)
    p_syn = syn.mean(axis=0)
    corr = pearsonr(p_syn, p_real)[0]
    nzc = int(np.sum(p_syn > 0))
    cmd = np.abs(p_syn - p_real).max()

    # 覆盖率与缺口分析：未生成的代码里多少是低 prevalence
    missing = p_real[p_syn == 0]
    missing_rare = (missing < 0.001).mean() if len(missing) else float('nan')

    print(f'\n  Prevalence correlation: {corr:.4f}')
    print(f'  NZC: {nzc} / {syn.shape[1]} ({nzc/syn.shape[1]*100:.1f}%)')
    print(f'  CMD (max prevalence dev): {cmd:.4f}')
    print(f'  未生成代码中 prevalence<0.001 的比例: {missing_rare:.1%}')
    print(f'  稀疏度 real={1-p_real.mean():.4f} syn={1-p_syn.mean():.4f}')

    if opt.out:
        result = {
            'corr': float(corr), 'nzc': int(nzc), 'n_dims': int(syn.shape[1]),
            'cmd': float(cmd), 'missing_frac_rare': float(missing_rare),
        }
        os.makedirs(os.path.dirname(opt.out), exist_ok=True)
        with open(opt.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\n结果保存: {opt.out}')


if __name__ == '__main__':
    main()
