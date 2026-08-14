"""诊断 MIMIC-IV CMD（max prevalence deviation）的来源。

论文里 CMD 定义为 max|p_syn - p_real|（main.tex Metrics 一节）。用最终 100k
合成样本重算得到 0.0166，而此前 claim 是 0.0060。本脚本打印偏离最大的特征，
判断 0.0166 是否由个别稀有码过度生成主导，以及是否集中在低 prevalence 码上。

用法（服务器，data/mimic4/mimic4_data.npy 必须是 2083 维版本）:
    python diag_cmd.py
"""
import argparse
import json
import os

import numpy as np


def find_syn(candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    raise SystemExit(f'找不到合成样本文件，尝试过: {candidates}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/mimic4/mimic4_data.npy')
    ap.add_argument('--train_idx', default='data/mimic4/train_indices.npy')
    ap.add_argument('--metadata', default='data/mimic4/metadata.json')
    ap.add_argument('--syn', default=None,
                    help='合成样本 npy；默认自动找 all_x_large.npy / all_x.npy')
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

    rp = real[tr].mean(0)   # 真实 prevalence（训练集口径）
    sp = syn.mean(0)        # 合成 prevalence
    d = np.abs(sp - rp)

    codes = json.load(open(opt.metadata))['selected_codes']

    print(f'real (train) {real[tr].shape[0]} x {rp.shape[0]}   syn {syn.shape[0]} x {syn.shape[1]}')
    print(f'overall CMD (max|dev|) : {d.max():.5f}')
    print(f'mean|dev| (all 2083)   : {d.mean():.5f}')
    print(f'mean|dev| (generated)  : {d[sp > 0].mean():.5f}')
    print(f'CMD over real>0.001    : {d[rp > 0.001].max():.5f}')
    print(f'CMD over real>0.01     : {d[rp > 0.01].max():.5f}')
    print(f'features with dev>0.01 : {int((d > 0.01).sum())}')
    print('\ntop-10 largest deviations:')
    print(f'{"code":>6}  {"real":>9}  {"syn":>9}  {"dev":>8}  {"ratio":>6}')
    for i in np.argsort(d)[::-1][:10]:
        ratio = sp[i] / rp[i] if rp[i] > 0 else float('inf')
        print(f'{codes[i]:>6}  {rp[i]:9.5f}  {sp[i]:9.5f}  {d[i]:8.5f}  {ratio:6.2f}')


if __name__ == '__main__':
    main()
