"""
TOST 等价性检验 (Two One-Sided Tests) —— 合成 vs 真实数据。

提案承诺了 formal TOST。本脚本对两条轴做等价性检验：

1. 下游效用：对 top-10 高频 ICD 代码，分别用真实训练集和合成数据训练分类器，
   在真实测试集上评估。对测试集做 B 次 bootstrap，每次得到配对 AUROC 差
   δ_b = mean_c(AUROC_syn(c,b) - AUROC_real(c,b))。对 δ 分布做 TOST，
   等价界 Δ（AUROC，默认 0.02）：若合成模型 AUROC 与真实模型之差落在 ±Δ 内
   即认为"实际等价"。
2. 边际保真度：逐特征 prevalence 均值差的 TOST，等价界 Δp（默认 0.003）。

复用 evaluate_downstream.py 的 LightGBM 结构与 top-10 目标（indices 0-9）。

用法（服务器）：
    python tost_equivalence.py \
        --data data/mimic4/mimic4_data.npy \
        --syn results/mimic4/samples/all_x_large.npy \
        --clf lightgbm \
        --margin 0.02 --margin_prevalence 0.003 \
        --n_boot 500 \
        --out results/mimic4/tost_results.json
"""
import os
import json
import argparse
import warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


def load_data(data_path, train_idx_path, test_idx_path, syn_path, max_syn=200000):
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

    print(f'  real_train : {real_train.shape}')
    print(f'  real_test  : {real_test.shape}')
    print(f'  synthetic  : {syn.shape}')
    return real_train, real_test, syn


def load_codes(codes_path, n_code):
    if codes_path and os.path.exists(codes_path):
        with open(codes_path) as f:
            meta = json.load(f)
        codes = meta.get('selected_codes', meta.get('codes', None))
        if codes is not None and len(codes) == n_code:
            return list(codes)
    return None


def fit_predict(clf_name, X, y, X_test):
    """训练一个分类器并返回测试集正类概率。"""
    if clf_name == 'lightgbm' and LGBMClassifier is not None:
        clf = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                             random_state=42, verbose=-1, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
    clf.fit(X, y)
    return clf.predict_proba(X_test)[:, 1]


def tost(deltas, margin, alpha=0.05):
    """对差值样本做 TOST。返回 (verdict, mean, se, p_upper, p_lower)。"""
    d = np.asarray(deltas, dtype=np.float64)
    mean_d = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        return ('equivalent' if abs(mean_d) <= margin else 'not_equivalent',
                mean_d, se, 0.0, 0.0)
    t_upper = (mean_d + margin) / se
    t_lower = (margin - mean_d) / se
    df = len(d) - 1
    p_upper = 1 - stats.t.cdf(t_upper, df)
    p_lower = 1 - stats.t.cdf(t_lower, df)
    verdict = 'equivalent' if (p_upper < alpha and p_lower < alpha) else 'not_equivalent'
    return verdict, mean_d, se, p_upper, p_lower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/mimic4/mimic4_data.npy')
    parser.add_argument('--train_idx', default='data/mimic4/train_indices.npy')
    parser.add_argument('--test_idx', default='data/mimic4/test_indices.npy')
    parser.add_argument('--codes', default='data/mimic4/metadata.json')
    parser.add_argument('--syn', default='results/mimic4/samples/all_x_large.npy')
    parser.add_argument('--clf', default='lightgbm', choices=['lightgbm', 'lr'])
    parser.add_argument('--n_targets', type=int, default=10,
                        help='按频率取前 N 个 ICD 代码（与 evaluate_downstream 一致）')
    parser.add_argument('--margin', type=float, default=0.02,
                        help='下游 AUROC 等价界 Δ')
    parser.add_argument('--margin_prevalence', type=float, default=0.003,
                        help='边际 prevalence 等价界 Δp')
    parser.add_argument('--n_boot', type=int, default=500)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--out', default='results/mimic4/tost_results.json')
    opt = parser.parse_args()

    rng = np.random.default_rng(opt.seed)
    print('加载数据...')
    real_train, real_test, syn = load_data(
        opt.data, opt.train_idx, opt.test_idx, opt.syn)

    codes = load_codes(opt.codes, real_train.shape[1])
    targets = list(range(min(opt.n_targets, real_train.shape[1])))

    # =========================================================
    # 1) 下游效用 TOST
    # =========================================================
    print(f'\n[1] 下游效用 TOST（分类器={opt.clf}，top-{len(targets)} ICD 代码，'
          f'B={opt.n_boot}，Δ={opt.margin}）')
    n_test = real_test.shape[0]
    probs_real, probs_syn, ys, names = [], [], [], []
    auc_full_real, auc_full_syn = [], []

    for col in targets:
        mask = np.ones(real_train.shape[1], dtype=bool)
        mask[col] = False
        y_test = real_test[:, col]
        if y_test.mean() < 0.005 or y_test.mean() > 0.995:
            continue

        p_r = fit_predict(opt.clf, real_train[:, mask], real_train[:, col],
                          real_test[:, mask])
        p_s = fit_predict(opt.clf, syn[:, mask], syn[:, col], real_test[:, mask])

        auc_full_real.append(roc_auc_score(y_test, p_r))
        auc_full_syn.append(roc_auc_score(y_test, p_s))
        ys.append(y_test); probs_real.append(p_r); probs_syn.append(p_s)
        names.append(codes[col] if codes else f'col_{col}')
        print(f'  {names[-1]}: AUROC real={auc_full_real[-1]:.4f} '
              f'syn={auc_full_syn[-1]:.4f}')

    if not ys:
        print('  没有有效目标代码，退出')
        return
    auc_full_real = np.array(auc_full_real)
    auc_full_syn = np.array(auc_full_syn)
    print(f'  全测试集平均 AUROC: real={auc_full_real.mean():.4f} '
          f'syn={auc_full_syn.mean():.4f}  '
          f'保留率={auc_full_syn.mean()/auc_full_real.mean()*100:.1f}%')

    deltas = np.empty(opt.n_boot)
    for b in range(opt.n_boot):
        ridx = rng.integers(0, n_test, n_test)
        d_b = 0.0
        for y, p_r, p_s in zip(ys, probs_real, probs_syn):
            yb = y[ridx]
            # 避免 bootstrap 采样后单类为 0 的退化情况
            if yb.sum() == 0 or yb.sum() == len(yb):
                continue
            a_r = roc_auc_score(yb, p_r[ridx])
            a_s = roc_auc_score(yb, p_s[ridx])
            d_b += (a_s - a_r) / len(ys)
        deltas[b] = d_b

    verdict, mean_d, se, p_up, p_lo = tost(deltas, opt.margin)
    print(f'\n  配对 AUROC 差 δ: mean={mean_d:+.5f}, se={se:.5f}')
    print(f'  TOST (Δ={opt.margin}): upper p={p_up:.4f}, lower p={p_lo:.4f}')
    print(f'  判定: {verdict}')

    # =========================================================
    # 2) 边际 prevalence TOST
    # =========================================================
    print(f'\n[2] 边际 prevalence TOST（Δp={opt.margin_prevalence}）')
    p_real = real_test.mean(axis=0)
    p_syn = syn.mean(axis=0)
    n_syn = syn.shape[0]
    diff_full = (p_syn - p_real).mean()
    print(f'  全样本平均 prevalence 差: {diff_full:+.5f}')

    deltas_p = np.empty(opt.n_boot)
    for b in range(opt.n_boot):
        r_idx = rng.integers(0, n_test, n_test)
        s_idx = rng.integers(0, n_syn, n_syn)
        d_b = (syn[s_idx].mean(axis=0) - real_test[r_idx].mean(axis=0)).mean()
        deltas_p[b] = d_b

    verdict_p, mean_dp, se_p, p_up_p, p_lo_p = tost(deltas_p, opt.margin_prevalence)
    print(f'  bootstrap prevalence 差: mean={mean_dp:+.5f}, se={se_p:.5f}')
    print(f'  TOST (Δp={opt.margin_prevalence}): upper p={p_up_p:.4f}, lower p={p_lo_p:.4f}')
    print(f'  判定: {verdict_p}')

    # =========================================================
    # 汇总输出
    # =========================================================
    summary = {
        'classifier': opt.clf,
        'n_targets': len(names),
        'auc_full_real_mean': float(auc_full_real.mean()),
        'auc_full_syn_mean': float(auc_full_syn.mean()),
        'retention_auc_pct': float(auc_full_syn.mean() / auc_full_real.mean() * 100),
        'per_code': [{'code': n, 'auc_real': float(a), 'auc_syn': float(s)}
                     for n, a, s in zip(names, auc_full_real, auc_full_syn)],
        'downstream_tost': {
            'margin': opt.margin,
            'mean_delta': float(mean_d),
            'se': float(se),
            'p_upper': float(p_up),
            'p_lower': float(p_lo),
            'verdict': verdict,
        },
        'prevalence_tost': {
            'margin': opt.margin_prevalence,
            'mean_delta': float(mean_dp),
            'se': float(se_p),
            'p_upper': float(p_up_p),
            'p_lower': float(p_lo_p),
            'verdict': verdict_p,
        },
    }
    os.makedirs(os.path.dirname(opt.out), exist_ok=True)
    with open(opt.out, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'\n结果保存: {opt.out}')


if __name__ == '__main__':
    main()
