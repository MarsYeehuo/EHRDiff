"""
隐私评估结果汇总：读取 privacy_mia.py 和 privacy_aia.py 的输出，
生成 DCR 分布图、MIA/AIA 表格和汇总摘要，供 dissertation 使用。

用法（服务器）：
    python privacy_report.py \
        --mia results/mimic4/privacy/mia.npz \
        --aia results/mimic4/privacy/aia.json \
        --outdir results/mimic4/privacy
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_dcr(dcr_train, dcr_test, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(dcr_train.min(), dcr_test.min()),
                       max(dcr_train.max(), dcr_test.max()), 80)
    ax.hist(dcr_train, bins=bins, alpha=0.6, label=f'Real train (n={len(dcr_train)})',
            color='tab:blue', density=True)
    ax.hist(dcr_test, bins=bins, alpha=0.6, label=f'Real test (n={len(dcr_test)})',
            color='tab:orange', density=True)
    ax.set_xlabel('DCR: distance to closest synthetic record')
    ax.set_ylabel('density')
    ax.set_title('Membership leakage via distance to synthetic data')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  [figure] {out_path}')


def summarize_mia(mia):
    print('\n========== MIA 汇总 ==========')
    print(f'  数据: train={mia["n_train"]} test={mia["n_test"]} '
          f'syn_total={mia["n_syn_total"]}')
    print(f'  DCR train: mean={mia["dcr_train_mean"]:.2f} '
          f'median={mia["dcr_train_median"]:.2f}')
    print(f'  DCR test : mean={mia["dcr_test_mean"]:.2f} '
          f'median={mia["dcr_test_median"]:.2f}')
    print(f'  成员推断攻击 AUROC: {mia["mia_auroc"]:.4f} ± {mia["mia_auroc_std"]:.4f}')
    print(f'  模仿检测: 最近真实邻居落在 train 比例 = {mia["imitation_frac"]:.4f}')
    verdict = ('无成员泄露（AUROC≈0.5）' if mia['mia_auroc'] < 0.55
               else '存在轻微成员泄露' if mia['mia_auroc'] < 0.65
               else '存在明显成员泄露')
    print(f'  判定: {verdict}')


def summarize_aia(aia):
    print('\n========== AIA 汇总 ==========')
    print(f"  平均 AUROC: 真实攻击={aia['avg_auroc_real']:.4f}  "
          f"合成攻击={aia['avg_auroc_syn']:.4f}  超额风险={aia['avg_excess_risk']:+.4f}")
    print(f"  判定: 合成数据{'增加' if aia['avg_excess_risk'] > 0.01 else '未增加'}属性推断风险")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mia', default='results/mimic4/privacy/mia.npz')
    parser.add_argument('--aia', default='results/mimic4/privacy/aia.json')
    parser.add_argument('--outdir', default='results/mimic4/privacy')
    opt = parser.parse_args()

    os.makedirs(opt.outdir, exist_ok=True)

    d = np.load(opt.mia, allow_pickle=True)
    dcr_train, dcr_test = d['dcr_train'], d['dcr_test']
    mia = {k: d[k].item() if d[k].ndim == 0 else d[k] for k in d.files if k not in ('dcr_train', 'dcr_test')}
    mia = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in mia.items()}

    plot_dcr(dcr_train, dcr_test, os.path.join(opt.outdir, 'mia_dcr_dist.png'))
    summarize_mia(mia)

    with open(opt.aia) as f:
        aia = json.load(f)
    summarize_aia(aia)

    # 汇总 markdown 表格（供 dissertation）
    md = ['# 隐私风险评估结果\n',
          '## 1. 成员推断 (MIA)\n',
          '| 指标 | 值 |',
          '|------|-----|',
          f"| DCR train (mean) | {mia['dcr_train_mean']:.2f} |",
          f"| DCR test (mean) | {mia['dcr_test_mean']:.2f} |",
          f"| MIA 攻击 AUROC | {mia['mia_auroc']:.4f} ± {mia['mia_auroc_std']:.4f} |",
          f"| 模仿率 (nearest-in-train) | {mia['imitation_frac']:.4f} |",
          '',
          '## 2. 属性推断 (AIA)\n',
          '| 属性 | prevalence | AUROC(真实攻击) | AUROC(合成攻击) | 超额风险 |',
          '|------|-----------|----------------|----------------|---------|']
    for r in aia['rows']:
        md.append(f"| {r['code']} ({r['desc']}) | {r['prevalence']:.3f} | "
                  f"{r['auroc_real']:.4f} | {r['auroc_syn']:.4f} | "
                  f"{r['excess_risk']:+.4f} |")
    md.append(f"\n平均: 真实={aia['avg_auroc_real']:.4f}, "
              f"合成={aia['avg_auroc_syn']:.4f}, 超额风险={aia['avg_excess_risk']:+.4f}")

    md_path = os.path.join(opt.outdir, 'privacy_summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))
    print(f'\n[summary] {md_path}')


if __name__ == '__main__':
    main()