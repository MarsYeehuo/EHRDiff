"""计算原论文 EHRDiff 的 Correlation Matrix Distance (CMD)，用于与本文的 MPD 对照。

背景（论文问题清单 P1-1）：
  论文原文（TMLR 2024, Table 1）把 CMD 定义为 "feature correlation matrices 的
  平均绝对差"，在 MIMIC-III 上报 7.769 ± 0.013。但 7.769 这个量级在 [-1,1] 的
  Pearson 相关矩阵上不可能（mean|diff| 上限为 2），说明其实际实现用的是**未缩放
  的协方差矩阵**。本脚本在 MIMIC-III 复现数据上做了校准：

      n=41000, d=1034（去掉 229 个零方差列后 805）：
      Pearson  corr  mean|Δ| = 0.00609    RMSE = 0.01326
      未缩放 cov (Σ(x-p)(x-p)') mean|Δ| = 2.64    RMSE = 8.34   <-- 最接近 7.769
      raw X^T X  mean|Δ| = 3.71    RMSE = 12.06

  因此最可能复现论文口径的是「未缩放协方差矩阵差的 RMSE（= Frobenius 范数/d）」。
  本脚本在服务器上对 MIMIC-IV 计算同一口径，同时输出 Pearson 定义（字面口径）
  作对照。两者单位不同，论文里要写明取哪个、以及为何与 7.769 不可直接比
  （vocabulary 大小 2083 vs 1782、样本量不同）。

用法（服务器，2083 维数据）:
    python compute_cmd.py \
        --real data/mimic4/mimic4_data.npy \
        --train_idx data/mimic4/train_indices.npy \
        --syn results/mimic4/samples/all_x_large.npy \
        --n_sample 100000 --label MIMIC-IV

本地校准（MIMIC-III 复现数据，应接近论文 7.769 的量级）:
    python compute_cmd.py \
        --real data/mimic3/mimic_data.npy \
        --train_idx data/mimic3/train_indices.npy \
        --syn results/mimic_edm/samples/all_x.npy \
        --n_sample 41000 --label MIMIC-III
"""
import argparse
import json

import numpy as np


def cov_unscaled(X):
    """未缩放中心化协方差矩阵：C = X_c^T X_c = n * covariance(X)。"""
    Xc = X - X.mean(0)
    return Xc.T @ Xc


def second_moment(X):
    """未中心化二阶矩：M = X^T X（原始共现计数）。"""
    return X.T @ X


def pearson_corr(X):
    """Pearson 相关矩阵（标准化的 Gram 矩阵）。调用方须先按 keep 切片。"""
    Xc = X - X.mean(0)
    denom = np.sqrt((Xc ** 2).sum(0))
    Z = Xc / denom
    return Z.T @ Z


def mean_abs(A, B):
    return float(np.abs(A - B).mean())


def rmse(A, B):
    return float(np.sqrt(((A - B) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', default='data/mimic4/mimic4_data.npy')
    ap.add_argument('--train_idx', default='data/mimic4/train_indices.npy')
    ap.add_argument('--syn', required=True, help='合成样本 npy')
    ap.add_argument('--n_sample', type=int, default=100000,
                    help='相关/协方差矩阵用到的样本数上限（real 与 syn 匹配到这个数）')
    ap.add_argument('--label', default='', help='数据集标签，打印用')
    ap.add_argument('--seed', type=int, default=2023)
    ap.add_argument('--out', default=None, help='JSON 输出路径（可选）')
    opt = ap.parse_args()

    real = np.load(opt.real, mmap_mode='r')
    tr = np.load(opt.train_idx)
    syn = np.load(opt.syn, mmap_mode='r')
    if syn.ndim == 3:
        syn = syn.squeeze(0)

    Xr = real[tr].astype(np.float32)   # 从 mmap 直接切片，避免整份物化
    syn = syn.astype(np.float32)
    n = min(len(Xr), len(syn), opt.n_sample)
    rng = np.random.RandomState(opt.seed)
    Xr = Xr[rng.choice(len(Xr), n, replace=False)]
    Xs = syn[rng.choice(len(syn), n, replace=False)]
    d = Xr.shape[1]

    print(f'[{opt.label}] n = {n}  (real train subsampled)   d = {d}'
          f'  (synthetic {syn.shape[0]} rows)')
    print('参考: 原论文 MIMIC-III CMD = 7.769 ± 0.013 (d=1782, n≈41868)')
    print('      我们的 MIMIC-III 校准: 未缩放cov RMSE ≈ 8.3, raw X^T X RMSE ≈ 12.1')
    print()

    # 论文数据（MIMIC-III 1782 维）不存在零方差列；为贴合论文设置，先丢 real/syn
    # 中任一为常量的列（并集），三个口径统一在 keep 列上计算。
    keep = (Xr.std(0) > 1e-12) & (Xs.std(0) > 1e-12)
    Xr, Xs = Xr[:, keep], Xs[:, keep]

    # 未缩放协方差（论文 7.769 所在量级）
    Cr, Cs = cov_unscaled(Xr), cov_unscaled(Xs)
    # 原始二阶矩
    Mr, Ms = second_moment(Xr), second_moment(Xs)
    # Pearson 相关矩阵（字面定义）
    Pr, Ps = pearson_corr(Xr), pearson_corr(Xs)

    res = {
        'label': opt.label,
        'n': n,
        'd': d,
        'n_syn': int(syn.shape[0]),
        'reference_paper_mimic3_cmd': 7.769,
        'unscaled_cov_mean_abs': mean_abs(Cr, Cs),
        'unscaled_cov_rmse': rmse(Cr, Cs),
        'raw_second_moment_mean_abs': mean_abs(Mr, Ms),
        'raw_second_moment_rmse': rmse(Mr, Ms),
        'pearson_mean_abs': mean_abs(Pr, Ps),
        'pearson_rmse': rmse(Pr, Ps),
        'n_nonconst': int(keep.sum()),
    }

    print(f'未缩放协方差 Σ(x-p)(x-p)\'   mean|Δ| = {res["unscaled_cov_mean_abs"]:.4f}'
          f'   RMSE = {res["unscaled_cov_rmse"]:.4f}   <- 论文 7.769 所在量级')
    print(f'原始二阶矩 X^T X            mean|Δ| = {res["raw_second_moment_mean_abs"]:.4f}'
          f'   RMSE = {res["raw_second_moment_rmse"]:.4f}')
    print(f'Pearson 相关矩阵（字面定义）  mean|Δ| = {res["pearson_mean_abs"]:.5f}'
          f'   RMSE = {res["pearson_rmse"]:.5f}   ({res["n_nonconst"]}/{d} 非零方差列)')
    print()
    print('解读: 与 7.769 同量级（可对比）的是 unscaled_cov_rmse；Pearson 定义的单位'
          '不同（有界 [0,2]），只用于报告"按论文定义算出的 CMD"。')

    if opt.out:
        with open(opt.out, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'\n已保存: {opt.out}')


if __name__ == '__main__':
    main()
