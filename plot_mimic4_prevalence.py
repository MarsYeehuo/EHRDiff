"""在服务器上生成 MIMIC-IV 最终 prevalence 图 + corr/NZC/CMD，用于核对 dissertation B2。

注意：必须在服务器上运行 —— 那里的 data/mimic4/mimic4_data.npy 才是 2083 维版本
（本地的是 545,576 x 2241，版本不同，算出来对不上）。

用法（服务器）:
    python plot_mimic4_prevalence.py

期望输出（无条件 2083 模型，见 EHRDiff_MIMICIV_Report.md 4.1）:
    corr 0.9901, NZC 1578/2083, CMD 0.0060
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def find_syn(candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    raise SystemExit(f'找不到合成样本文件，尝试过: {candidates}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/mimic4/mimic4_data.npy')
    ap.add_argument('--train_idx', default='data/mimic4/train_indices.npy')
    ap.add_argument('--syn', default=None,
                    help='最终合成样本 npy；默认自动找 all_x_large.npy / all_x.npy')
    ap.add_argument('--out', default='dissertation/figures/mimic4_prevalence_final.png')
    opt = ap.parse_args()

    syn_path = opt.syn or find_syn([
        'results/mimic4/samples/all_x_large.npy',
        'results/mimic4/samples/all_x.npy',
    ])
    real = np.load(opt.data, mmap_mode='r').astype(np.float32)
    tr = np.load(opt.train_idx)
    syn = np.load(syn_path, mmap_mode='r').astype(np.float32)
    if syn.ndim == 3:
        syn = syn.squeeze(0)

    rp = real[tr].mean(0)          # 真实 prevalence（训练集口径）
    sp = syn.mean(0)               # 合成 prevalence（全量生成样本）
    corr, p = pearsonr(sp, rp)
    nzc = int((sp > 0).sum())
    cmd = float(np.abs(sp - rp).max())

    print(f'real      : {real.shape} (train {real[tr].shape[0]} rows)')
    print(f'synthetic : {syn.shape}  <- {syn_path}')
    print(f'corr = {corr:.4f}   NZC = {nzc}/{rp.shape[0]}   CMD = {cmd:.4f}')
    print()
    print('对照: 无条件 2083 模型期望 corr 0.9901, NZC 1578/2083, CMD 0.0060')
    print('  - 人口学 2092 模型(10k)是 corr 0.9884, NZC 1361')
    print('  - 训练快照(Dimension-Wise Distribution.png)是 corr 0.9898, NZC 1442')

    # 对数刻度散点，加小偏移以容纳未被生成的 0 值特征
    eps = 1e-5
    plt.figure(figsize=(7.5, 6))
    plt.scatter(rp + eps, sp + eps, s=6, alpha=0.5)
    lo = min((rp + eps).min(), (sp + eps).min())
    hi = max((rp + eps).max(), (sp + eps).max())
    plt.plot([lo, hi], [lo, hi], 'r--', lw=1, label='identity')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Real feature prevalence (log)')
    plt.ylabel('Synthetic feature prevalence (log)')
    plt.title(f'MIMIC-IV prevalence (corr {corr:.4f}, NZC {nzc})')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(opt.out), exist_ok=True)
    plt.savefig(opt.out, dpi=200)
    print(f'\n图已保存: {opt.out}')


if __name__ == '__main__':
    main()
