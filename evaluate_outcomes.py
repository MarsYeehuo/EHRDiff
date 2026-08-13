"""
临床结局 TSTR 评估 —— 医院内死亡率 / 长住院 / 30 天再入院。

需要先运行 data_preprocessing/build_outcomes.py 生成 outcomes.npy：
    outcomes.npy  (n, 3): 列0 mortality, 列1 LOS天数, 列2 readmit30

方法（Train-on-Synthetic, Test-on-Real）：
    合成数据只生成 ICD 代码、不含结局标签，因此"在合成数据上训练"采用
    pseudo-labeling：先用真实训练集训练一个 teacher 模型，用它为每条合成
    样本预测结局概率并依概率采样伪标签，再用 (合成特征, 伪标签) 训练
    student 模型，最后在真实测试集上评估 AUROC。若合成 ICD 画像保留了
    ICD→结局 的关联，student 的性能应接近直接在真实数据上训练的模型。

    对每个结局，报告 real/syn AUROC、AvgPrec 及保留率，并用 bootstrap
    重采样真实测试集得到 95% 置信区间。

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
              max_syn=200000):
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


def make_clf(clf_name):
    if clf_name == 'lightgbm' and LGBMClassifier is not None:
        return LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                              random_state=42, verbose=-1, n_jobs=-1)
    return LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)


def fit_clf(clf_name, X, y):
    """训练分类器，丢弃 NaN 标签行。"""
    valid = ~np.isnan(y)
    clf = make_clf(clf_name)
    clf.fit(X[valid], y[valid])
    return clf


def fit_pseudo(clf_name, X_syn, y_tr, X_real, seed):
    """Teacher on real -> pseudo-label synthetic -> student on synthetic.

    返回在 X_syn 上训练的 student 分类器。伪标签由 teacher 对每条合成
    样本的结局概率依 Bernoulli 采样得到，使 (X_syn, y_syn) 保留真实的
    P(y|X) 结构，而不是把合成特征与任意真实标签硬配对。
    """
    teacher = fit_clf(clf_name, X_real, y_tr)
    p = teacher.predict_proba(X_syn)[:, 1]
    rng = np.random.default_rng(seed)
    y_syn = (rng.uniform(size=len(X_syn)) < p).astype(np.float32)
    return fit_clf(clf_name, X_syn, y_syn)


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
        clf_r = fit_clf(opt.clf, real_train, y_tr)
        p_r = clf_r.predict_proba(real_test)[:, 1]
        clf_s = fit_pseudo(opt.clf, syn, y_tr, real_train, seed=opt.seed)
        p_s = clf_s.predict_proba(real_test)[:, 1]

        yv = y_te[te_valid]
        p_r, p_s = p_r[te_valid], p_s[te_valid]
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
