"""
临床结局 TSTR 评估 —— 医院内死亡率 / 长住院 / 30 天再入院。

需要先运行 data_preprocessing/build_outcomes.py 生成 outcomes.npy：
    outcomes.npy  (n, 3): 列0 mortality, 列1 LOS天数, 列2 readmit30

方法（Train-on-Synthetic, Test-on-Real）：
    对每个结局，分别用 真实训练集 和 合成数据 的特征训练分类器，
    在 真实测试集 上评估 AUROC / AvgPrec，报告保留率及 bootstrap 置信区间。
    这检验合成 ICD 画像是否保留 ICD→结局 的关联。

用法（服务器）：
    python evaluate_outcomes.py \
        --data data/mimic4/mimic4_data.npy \
        --outcomes data/mimic4/outcomes.npy \
        --syn results/mimic4/samples/all_x_large.npy \
        --clf lightgbm \
        --n_boot 200 \
        --out results/mimic4/outcomes_tstr.json
"""
import os
import json
import argparse
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

OUTCOME_NAMES = ['mortality', 'prolonged_stay', 'readmit30']


def load_data(data_path, train_idx_path, test_idx_path, outcomes_path, syn_path,
              max_syn=200000, los_quantile=0.5):
    real_all = np.load(data_path, mmap_mode='r')
    train_idx = np.load(train_idx_path)
    test_idx = np.load(test_idx_path)
    real_train = real_all[train_idx].astype(np.float32)
    real_test = real_all[test_idx].astype(np.float32)

    out = np.load(outcomes_path).astype(np.float32)
    y_mort = out[:, 0]
    y_los = out[:, 1]
    y_rm = out[:, 2]

    # LOS 按真实训练集分位数二值化 → "长住院"
    thr = np.nanmedian(y_los[train_idx])
    y_los_bin = (y_los > thr).astype(np.float32)

    ys = {
        'mortality': y_mort,
        'prolonged_stay': y_los_bin,
        'readmit30': y_rm,
    }

    syn = np.load(syn_path)
    if syn.ndim == 3:
        syn = syn.squeeze(0)
    syn = syn.astype(np.float32)
    if len(syn) > max_syn:
        syn = syn[:max_syn]

    print(f'  real_train : {real_train.shape}')
    print(f'  real_test  : {real_test.shape}')
    print(f'  synthetic  : {syn.shape}')
    for k, y in ys.items():
        tr = y[train_idx]
        te = y[test_idx]
        print(f'  {k:<14}: train prev={tr[~np.isnan(tr)].mean():.4f} '
              f'test prev={te[~np.isnan(te)].mean():.4f}')
    return real_train, real_test, syn, ys, train_idx, test_idx


def fit_predict(clf_name, X, y, X_test):
    if clf_name == 'lightgbm' and LGBMClassifier is not None:
        clf = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                             random_state=42, verbose=-1, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
    # 丢弃 NaN 标签行
    valid = ~np.isnan(y)
    clf.fit(X[valid], y[valid])
    return clf.predict_proba(X_test)[:, 1]


def evaluate_task(X_tr, y_tr, X_te, y_te, clf_name):
    p = fit_predict(clf_name, X_tr, y_tr, X_te)
    valid = ~np.isnan(y_te)
    return p[valid], y_te[valid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/mimic4/mimic4_data.npy')
    parser.add_argument('--train_idx', default='data/mimic4/train_indices.npy')
    parser.add_argument('--test_idx', default='data/mimic4/test_indices.npy')
    parser.add_argument('--outcomes', default='data/mimic4/outcomes.npy')
    parser.add_argument('--syn', default='results/mimic4/samples/all_x_large.npy')
    parser.add_argument('--clf', default='lightgbm', choices=['lightgbm', 'lr'])
    parser.add_argument('--n_boot', type=int, default=200)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--out', default='results/mimic4/outcomes_tstr.json')
    opt = parser.parse_args()

    rng = np.random.default_rng(opt.seed)
    print('加载数据...')
    real_train, real_test, syn, ys, train_idx, test_idx = load_data(
        opt.data, opt.train_idx, opt.test_idx, opt.outcomes, opt.syn)

    rows = []
    for name in OUTCOME_NAMES:
        y = ys[name]
        y_tr = y[train_idx]
        y_te = y[test_idx]
        te_valid = ~np.isnan(y_te)
        if te_valid.sum() < 100 or y_te[te_valid].sum() < 30:
            print(f'\n[{name}] 测试集有效样本或阳性过少，跳过')
            continue

        print(f'\n[{name}] ({opt.clf})')
        p_r, yv = evaluate_task(real_train, y_tr, real_test, y_te, opt.clf)
        p_s, _ = evaluate_task(syn, y_tr, real_test, y_te, opt.clf)
        auc_r = roc_auc_score(yv, p_r)
        auc_s = roc_auc_score(yv, p_s)
        ap_r = average_precision_score(yv, p_r)
        ap_s = average_precision_score(yv, p_s)
        print(f'  AUROC: real={auc_r:.4f} syn={auc_s:.4f} '
              f'保留率={auc_s/auc_r*100:.1f}%')
        print(f'  AvgPrec: real={ap_r:.4f} syn={ap_s:.4f}')

        # bootstrap 测试集 → AUROC 保留率 CI
        ret_boot = np.empty(opt.n_boot)
        nv = len(yv)
        for b in range(opt.n_boot):
            ridx = rng.integers(0, nv, nv)
            yb = yv[ridx]
            if yb.sum() == 0 or yb.sum() == len(yb):
                ret_boot[b] = np.nan
                continue
            a_r = roc_auc_score(yb, p_r[ridx])
            a_s = roc_auc_score(yb, p_s[ridx])
            ret_boot[b] = a_s / a_r * 100
        ret_boot = ret_boot[~np.isnan(ret_boot)]
        ci = (np.percentile(ret_boot, 2.5), np.percentile(ret_boot, 97.5))
        print(f'  AUROC 保留率 95% CI: {ret_boot.mean():.1f}% '
              f'[{ci[0]:.1f}%, {ci[1]:.1f}%]')

        rows.append({
            'outcome': name,
            'classifier': opt.clf,
            'auc_real': float(auc_r),
            'auc_syn': float(auc_s),
            'retention_pct': float(auc_s / auc_r * 100),
            'retention_ci_low': float(ci[0]),
            'retention_ci_high': float(ci[1]),
            'ap_real': float(ap_r),
            'ap_syn': float(ap_s),
        })

    os.makedirs(os.path.dirname(opt.out), exist_ok=True)
    with open(opt.out, 'w') as f:
        json.dump({'n_boot': opt.n_boot, 'rows': rows}, f, indent=2)
    print(f'\n结果保存: {opt.out}')


if __name__ == '__main__':
    main()
