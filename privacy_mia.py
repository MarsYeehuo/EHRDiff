"""
成员推断攻击 (Membership Inference Attack, MIA) —— 基于距离的记忆/泄露检测。

原理：
  扩散模型若记住了训练集，生成样本会接近训练记录。因此：
  1. DCR (Distance to Closest Record)：对真实 train / 真实 test 记录，
     计算其到最近合成样本的 Hamming 距离。若模型记忆训练集，train 记录的
     DCR 会系统性小于 test 记录。
  2. 攻击分类器：用 DCR 特征训练 LR 判别 train vs test 成员，
     AUROC ≈ 0.5 表示无泄露；> 0.5 表示可通过距离区分成员（有记忆泄露）。
  3. 模仿检测 (NNDR)：对每个合成样本，比较其最近真实邻居在 train 还是在
     test。若模型复制训练记录，合成样本会倾向落在 train 附近。

距离计算：二值数据 Hamming(q, r) = sum(q) + sum(r) - 2*(q·r)，
用 GPU 矩阵乘分块实现，避免 O(n^2) 内存。

用法（服务器）：
    python privacy_mia.py \
        --data data/mimic4/mimic4_data.npy \
        --train_idx data/mimic4/train_indices.npy \
        --test_idx data/mimic4/test_indices.npy \
        --syn results/mimic4/samples/all_x_large.npy \
        --out results/mimic4/privacy/mia.npz
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DEFAULT_TRAIN_SAMPLE = 45395   # 与 test 数量一致，保证判别器类别均衡、密度可比


def load_data(data_path, train_idx_path, test_idx_path, syn_path,
              n_train_sample, seed):
    real_all = np.load(data_path, mmap_mode='r')
    train_idx = np.load(train_idx_path)
    test_idx = np.load(test_idx_path)
    real_train = real_all[train_idx].astype(np.float32)
    real_test = real_all[test_idx].astype(np.float32)

    rng = np.random.RandomState(seed)
    if n_train_sample and len(real_train) > n_train_sample:
        perm = rng.permutation(len(real_train))[:n_train_sample]
        real_train = real_train[perm]

    syn = np.load(syn_path)
    if syn.ndim == 3:
        syn = syn.squeeze(0)
    syn = syn.astype(np.float32)

    print(f'  real_train: {real_train.shape} (subsampled)')
    print(f'  real_test : {real_test.shape}')
    print(f'  syn       : {syn.shape}')
    return real_train, real_test, syn


def min_hamming_to_ref(queries, ref, device, q_chunk=8192, r_chunk=16384):
    """对每个 query 行，返回其到 ref 的最小 Hamming 距离。

    Hamming(q, r) = sum(q) + sum(r) - 2 * (q dot r)，对二值数据成立。
    分块矩阵乘，控制显存。
    """
    Q = torch.from_numpy(queries).to(device)
    R = torch.from_numpy(ref).to(device)
    qsum = Q.sum(dim=1)
    rsum = R.sum(dim=1)
    out = torch.full((Q.shape[0],), float('inf'), device=device)

    for i in range(0, Q.shape[0], q_chunk):
        qb = Q[i:i + q_chunk]
        qsb = qsum[i:i + q_chunk]
        best = torch.full((qb.shape[0],), float('inf'), device=device)
        for j in range(0, R.shape[0], r_chunk):
            rb = R[j:j + r_chunk]
            rsb = rsum[j:j + r_chunk]
            sim = qb @ rb.t()                       # (qc, rc)
            dist = qsb[:, None] + rsb[None, :] - 2.0 * sim
            best = torch.minimum(best, dist.min(dim=1).values)
        out[i:i + q_chunk] = best
    return out.cpu().numpy()


def run_attack_auroc(dcr_train, dcr_test, n_folds=5, seed=0):
    """用 DCR 特征训练 LR 判别 train(test=1) vs test(0)，5 折交叉验证 AUROC。"""
    X = np.concatenate([dcr_test, dcr_train])[:, None]  # 特征：DCR
    y = np.concatenate([np.ones(len(dcr_test)), np.zeros(len(dcr_train))])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        clf.fit(X[tr_idx], y[tr_idx])
        aucs.append(roc_auc_score(y[va_idx], clf.predict_proba(X[va_idx])[:, 1]))
    aucs = np.array(aucs)
    return aucs.mean(), aucs.std()


def imitation_detection(syn, real_train, real_test, device, n_syn_sample=50000,
                        q_chunk=8192, r_chunk=16384, seed=0):
    """对合成样本，比较最近真实邻居在 train 还是 test，衡量模仿程度。"""
    rng = np.random.RandomState(seed)
    if len(syn) > n_syn_sample:
        syn_q = syn[rng.choice(len(syn), n_syn_sample, replace=False)]
    else:
        syn_q = syn

    d_train = min_hamming_to_ref(syn_q, real_train, device, q_chunk, r_chunk)
    d_test = min_hamming_to_ref(syn_q, real_test, device, q_chunk, r_chunk)

    imitation_frac = float(np.mean(d_train < d_test))
    return {
        'n_syn': int(len(syn_q)),
        'dist_to_train_min_mean': float(d_train.mean()),
        'dist_to_train_min_p50': float(np.median(d_train)),
        'dist_to_test_min_mean': float(d_test.mean()),
        'dist_to_test_min_p50': float(np.median(d_test)),
        'imitation_frac': imitation_frac,   # 最近真实邻居落在 train 的比例
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/mimic4/mimic4_data.npy')
    parser.add_argument('--train_idx', default='data/mimic4/train_indices.npy')
    parser.add_argument('--test_idx', default='data/mimic4/test_indices.npy')
    parser.add_argument('--syn', default='results/mimic4/samples/all_x_large.npy')
    parser.add_argument('--out', default='results/mimic4/privacy/mia.npz')
    parser.add_argument('--n_train_sample', type=int, default=DEFAULT_TRAIN_SAMPLE)
    parser.add_argument('--n_syn_sample', type=int, default=50000,
                        help='模仿检测时用到的合成样本子采样数')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()

    print('加载数据...')
    real_train, real_test, syn = load_data(
        opt.data, opt.train_idx, opt.test_idx, opt.syn,
        opt.n_train_sample, opt.seed)
    device = opt.device

    print('\n计算 DCR (train/test → 最近合成样本)...')
    dcr_train = min_hamming_to_ref(real_train, syn, device)
    dcr_test = min_hamming_to_ref(real_test, syn, device)

    auc_mean, auc_std = run_attack_auroc(dcr_train, dcr_test, seed=opt.seed)

    print('\n模仿检测 (synthetic → 最近真实邻居)...')
    imit = imitation_detection(syn, real_train, real_test, device,
                               n_syn_sample=opt.n_syn_sample, seed=opt.seed)

    results = {
        'n_train': int(len(real_train)),
        'n_test': int(len(real_test)),
        'n_syn_total': int(len(syn)),
        'dcr_train_mean': float(dcr_train.mean()),
        'dcr_train_median': float(np.median(dcr_train)),
        'dcr_test_mean': float(dcr_test.mean()),
        'dcr_test_median': float(np.median(dcr_test)),
        'mia_auroc_mean': float(auc_mean),
        'mia_auroc_std': float(auc_std),
        'mia_auroc': float(auc_mean),
    }
    results.update(imit)

    os.makedirs(os.path.dirname(opt.out), exist_ok=True)
    np.savez(opt.out, dcr_train=dcr_train, dcr_test=dcr_test, **results)

    print('\n========== 成员推断 (MIA) 结果 ==========')
    print(f'  DCR train: mean={results["dcr_train_mean"]:.2f} '
          f'median={results["dcr_train_median"]:.2f}')
    print(f'  DCR test : mean={results["dcr_test_mean"]:.2f} '
          f'median={results["dcr_test_median"]:.2f}')
    print(f'  MIA 攻击 AUROC: {auc_mean:.4f} ± {auc_std:.4f} '
          f'(≈0.5 无泄露, >0.5 有记忆泄露)')
    print(f'  模仿检测: 最近真实邻居在 train 的比例 = {imit["imitation_frac"]:.4f}')

    with open(opt.out.replace('.npz', '.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n结果保存: {opt.out}')


if __name__ == '__main__':
    main()