"""
Multi-seed TSTR with confidence intervals.

Repeats a TSTR protocol over N repetitions to report mean +/- SD (and 95% CI of
the mean) for per-task AUROC and retention, instead of single point estimates.

Two protocols, matching the dissertation tables exactly:
  --protocol lightgbm   unconditional model: mimic4_data.npy (2083 cols), top-10
                        codes (indices 0-9), synthetic = all_x_large.npy.
                        LightGBM (n_estimators=200, max_depth=5, lr=0.1); each
                        repetition uses random_state=seed (model-seed variance)
                        plus a fresh bootstrap resample of the real test set
                        (data variance).
  --protocol logistic   demographic model: mimic4_dem_data.npy (2092 cols: 2083
                        ICD + 9 demographic), 15 ICD targets (prevalence in
                        [0.03, 0.97]) + 3 demographic targets, synthetic =
                        all_x.npy. Logistic regression (lbfgs) is deterministic,
                        so the model is fit once; only the test-set bootstrap
                        varies across repetitions.

Retention = AUROC_syn / AUROC_real (per task); aggregate = mean over tasks.

Usage (server):
    python evaluate_tstr_multi.py --protocol lightgbm --seeds 5 \
        --out results/mimic4/tstr_multi_lightgbm.json
    python evaluate_tstr_multi.py --protocol logistic --seeds 5 \
        --out results/mimic4_dem/tstr_multi_logistic.json
"""
import os
import json
import argparse
import warnings

import numpy as np
from scipy.stats import t as tdist
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

DATA_DIR = 'data/mimic4'
DEM_COLS = ['gender', 'age', 'adm_emerg', 'adm_obs', 'adm_urg', 'adm_other',
            'ins_medicare', 'ins_medicaid', 'ins_other']
N_DEM = len(DEM_COLS)


def load_data(protocol):
    if protocol == 'lightgbm':
        fname, syn_path = 'mimic4_data.npy', 'results/mimic4/samples/all_x_large.npy'
        n_code = 2083
    else:
        fname, syn_path = 'mimic4_dem_data.npy', 'results/mimic4_dem/samples/all_x.npy'
        n_code = None  # resolved from dem_metadata.json
    real_all = np.load(os.path.join(DATA_DIR, fname), mmap_mode='r')
    train_idx = np.load(os.path.join(DATA_DIR, 'train_indices.npy'))
    test_idx = np.load(os.path.join(DATA_DIR, 'test_indices.npy'))
    real_train = real_all[train_idx].astype(np.float32)
    real_test = real_all[test_idx].astype(np.float32)

    syn = np.load(syn_path)
    if syn.ndim == 3:
        syn = syn.squeeze(0)
    syn = syn.astype(np.float32)
    return real_train, real_test, syn, n_code


def load_code_labels(protocol, total_dim):
    """Return list of code labels aligned with ICD columns, plus n_code."""
    if protocol == 'lightgbm':
        with open(os.path.join(DATA_DIR, 'metadata.json')) as f:
            meta = json.load(f)
        codes = meta['selected_codes']
        n_code = len(codes)
        return codes, n_code
    # logistic: dem_metadata.json
    meta_path = os.path.join(DATA_DIR, 'dem_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta['selected_codes'], meta['n_code']
    return None, total_dim - N_DEM


def pick_targets(real_train, n_code, n_targets=15):
    prevalence = real_train[:, :n_code].mean(axis=0)
    valid = np.where((prevalence > 0.03) & (prevalence < 0.97))[0]
    sorted_idx = valid[np.argsort(-prevalence[valid])]
    return sorted_idx[:n_targets]


def fit_lightgbm(X, y, seed):
    clf = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                         random_state=seed, verbose=-1, n_jobs=-1)
    clf.fit(X, y)
    return clf


def fit_logistic(X, y):
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
    clf.fit(X, y)
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', choices=['lightgbm', 'logistic'],
                        default='lightgbm')
    parser.add_argument('--seeds', type=int, default=5,
                        help='number of repetitions (>=3 recommended)')
    parser.add_argument('--out', default=None)
    opt = parser.parse_args()
    protocol = opt.protocol
    K = opt.seeds

    print(f'protocol: {protocol}, repetitions: {K}')
    real_train, real_test, syn, n_code_arg = load_data(protocol)
    total_dim = real_train.shape[1]
    codes, n_code = load_code_labels(protocol, total_dim)
    if n_code is None:
        n_code = n_code_arg
    print(f'  real_train: {real_train.shape}  real_test: {real_test.shape}  syn: {syn.shape}')
    print(f'  n_code: {n_code}, total_dim: {total_dim}')

    # ---- build the task list ----
    if protocol == 'lightgbm':
        n_targets = min(10, n_code)
        targets = list(range(n_targets))
        tasks = []
        for col in targets:
            code = codes[col] if codes else f'col_{col}'
            mask = np.ones(total_dim, dtype=bool)
            mask[col] = False
            tasks.append({
                'name': f'code {code}', 'code': code,
                'y_tr': real_train[:, col], 'y_tr_syn': syn[:, col],
                'y_te': real_test[:, col],
                'mask': mask, 'X_tr_real': real_train[:, mask],
                'X_te_real': real_test[:, mask], 'X_tr_syn': syn[:, mask],
                'X_te_syn': real_test[:, mask],
            })
    else:
        targets = pick_targets(real_train, n_code)
        tasks = []
        for col in targets:
            code = codes[col] if codes else f'col_{col}'
            mask = np.ones(total_dim, dtype=bool)
            mask[col] = False
            tasks.append({
                'name': f'code {code}', 'code': code,
                'y_tr': real_train[:, col], 'y_tr_syn': syn[:, col],
                'y_te': real_test[:, col],
                'mask': mask, 'X_tr_real': real_train[:, mask],
                'X_te_real': real_test[:, mask], 'X_tr_syn': syn[:, mask],
                'X_te_syn': real_test[:, mask],
            })
        icd_cols = list(range(n_code))
        for dem_idx, tname in [(0, 'gender'), (2, 'adm emergency'), (4, 'adm urgent')]:
            dem_col = n_code + dem_idx
            tasks.append({
                'name': f'dem {tname}', 'code': tname,
                'y_tr': real_train[:, dem_col], 'y_tr_syn': syn[:, dem_col],
                'y_te': real_test[:, dem_col],
                'mask': None, 'X_tr_real': real_train[:, icd_cols],
                'X_te_real': real_test[:, icd_cols], 'X_tr_syn': syn[:, icd_cols],
                'X_te_syn': real_test[:, icd_cols],
            })

    # ---- fit models ----
    # logistic: deterministic -> fit once, reuse predictions across reps.
    cache = {}  # task index -> (p_real, p_syn) on full test
    if protocol == 'logistic':
        for i, t in enumerate(tasks):
            clf_r = fit_logistic(t['X_tr_real'], t['y_tr'])
            clf_s = fit_logistic(t['X_tr_syn'], t['y_tr_syn'])
            cache[i] = (clf_r.predict_proba(t['X_te_real'])[:, 1],
                        clf_s.predict_proba(t['X_te_syn'])[:, 1])
        print('  logistic models fit once (deterministic).')

    # ---- repeated evaluation with bootstrap of the real test set ----
    # per-task results over reps: auc_real[rep], auc_syn[rep]
    n_te = real_test.shape[0]
    per_task = {i: {'auc_real': [], 'auc_syn': []} for i in range(len(tasks))}

    for rep in range(K):
        rng = np.random.default_rng(2023 + rep * 7)
        idx = rng.integers(0, n_te, n_te)  # bootstrap test sample
        for i, t in enumerate(tasks):
            if protocol == 'logistic':
                p_r, p_s = cache[i]
                auc_r = roc_auc_score(t['y_te'][idx], p_r[idx])
                auc_s = roc_auc_score(t['y_te'][idx], p_s[idx])
            else:
                clf_r = fit_lightgbm(t['X_tr_real'], t['y_tr'], rep)
                clf_s = fit_lightgbm(t['X_tr_syn'], t['y_tr_syn'], rep)
                auc_r = roc_auc_score(
                    t['y_te'][idx], clf_r.predict_proba(t['X_te_real'][idx])[:, 1])
                auc_s = roc_auc_score(
                    t['y_te'][idx], clf_s.predict_proba(t['X_te_syn'][idx])[:, 1])
            per_task[i]['auc_real'].append(auc_r)
            per_task[i]['auc_syn'].append(auc_s)
        if protocol == 'lightgbm':
            print(f'  rep {rep+1}/{K} done')

    # ---- aggregate ----
    rows = []
    for i, t in enumerate(tasks):
        ar = np.array(per_task[i]['auc_real'])
        asy = np.array(per_task[i]['auc_syn'])
        retention = asy / ar
        rows.append({
            'task': t['name'], 'code': t['code'],
            'auc_real_mean': float(ar.mean()), 'auc_real_sd': float(ar.std(ddof=1)),
            'auc_syn_mean': float(asy.mean()), 'auc_syn_sd': float(asy.std(ddof=1)),
            'retention_mean': float(retention.mean()),
            'retention_sd': float(retention.std(ddof=1)),
            'prevalence': float(t['y_tr'].mean()),
        })

    def agg_ci(vals):
        v = np.array(vals)
        sd = v.std(ddof=1)
        se = sd / np.sqrt(len(v))
        tcrit = tdist.ppf(0.975, len(v) - 1)
        return v.mean(), sd, se, tcrit * se

    print('\n  per-task AUROC mean +/- SD over %d reps:' % K)
    print(f"  {'task':<22} {'prevalence':>9} {'AUC_real':>16} {'AUC_syn':>16} {'retention':>12}")
    for r in rows:
        print(f"  {r['task']:<22} {r['prevalence']:>9.3f} "
              f"{r['auc_real_mean']:>10.4f}±{r['auc_real_sd']:.4f} "
              f"{r['auc_syn_mean']:>10.4f}±{r['auc_syn_sd']:.4f} "
              f"{r['retention_mean']:>10.4f}±{r['retention_sd']:.4f}")

    # aggregate retention (mean over tasks of per-rep retention)
    n_tasks = len(tasks)
    agg_ret_rep = np.zeros(K)
    agg_auc_r_rep = np.zeros(K)
    agg_auc_s_rep = np.zeros(K)
    for rep in range(K):
        rs, ars, ass = [], [], []
        for i in range(n_tasks):
            ars.append(per_task[i]['auc_real'][rep])
            ass.append(per_task[i]['auc_syn'][rep])
            rs.append(per_task[i]['auc_syn'][rep] / per_task[i]['auc_real'][rep])
        agg_ret_rep[rep] = np.mean(rs)
        agg_auc_r_rep[rep] = np.mean(ars)
        agg_auc_s_rep[rep] = np.mean(ass)

    m_r, sd_r, se_r, ci_r = agg_ci(agg_auc_r_rep)
    m_s, sd_s, se_s, ci_s = agg_ci(agg_auc_s_rep)
    m_t, sd_t, se_t, ci_t = agg_ci(agg_ret_rep)
    print('\n  Aggregate over %d tasks (mean of per-rep values):' % n_tasks)
    print(f'  AUROC real : {m_r:.4f} +/- {sd_r:.4f} (95% CI {m_r-ci_r:.4f}..{m_r+ci_r:.4f})')
    print(f'  AUROC syn  : {m_s:.4f} +/- {sd_s:.4f} (95% CI {m_s-ci_s:.4f}..{m_s+ci_s:.4f})')
    print(f'  Retention  : {100*m_t:.2f}% +/- {100*sd_t:.2f}pp '
          f'(95% CI {100*(m_t-ci_t):.2f}..{100*(m_t+ci_t):.2f}%)')

    # LaTeX table
    print('\n  --- LaTeX table ---')
    print(r'  \begin{tabular}{lcccc}')
    print(r'    \toprule')
    print(r'    Task & Prev & AUROC$_{R}$ & AUROC$_{S}$ & Retention \\')
    print(r'    \midrule')
    for r in rows:
        print(f'    {r["task"]:<20} & {r["prevalence"]:.3f} & '
              f'{r["auc_real_mean"]:.4f} & {r["auc_syn_mean"]:.4f} & '
              f'{100*r["retention_mean"]:.1f}\% \\\\')
    print(f'    \\textbf{{Mean}} & & {m_r:.4f} & {m_s:.4f} & '
          f'{100*m_t:.1f}\% ({100*(m_t-ci_t):.1f}--{100*(m_t+ci_t):.1f}) \\\\')
    print(r'    \bottomrule')
    print(r'  \end{tabular}')

    if opt.out:
        result = {
            'protocol': protocol, 'seeds': K, 'n_tasks': n_tasks,
            'tasks': rows,
            'agg_auc_real_mean': float(m_r), 'agg_auc_real_sd': float(sd_r),
            'agg_auc_syn_mean': float(m_s), 'agg_auc_syn_sd': float(sd_s),
            'retention_mean_pct': float(100 * m_t),
            'retention_sd_pct': float(100 * sd_t),
            'retention_ci95_lo_pct': float(100 * (m_t - ci_t)),
            'retention_ci95_hi_pct': float(100 * (m_t + ci_t)),
        }
        os.makedirs(os.path.dirname(opt.out), exist_ok=True)
        with open(opt.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\n结果保存: {opt.out}')


if __name__ == '__main__':
    main()
