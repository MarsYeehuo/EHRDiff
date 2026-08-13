"""
为 mimic4_data.npy 的每一行重建 hadm_id 映射，并从 admissions.csv 提取临床结局。

背景：process_mimic4_for_ehrdiff.py 构建特征矩阵时记录了 hadm_ids，但没有存盘。
本脚本用完全相同的方式重建 hadm_id 顺序，并通过"重建 X == mimic4_data.npy"
做强校验；若顺序不匹配则报错并提示，绝不静默用错映射。

输出（与 mimic4_data.npy 行一一对齐）：
    data/mimic4/hadm_ids.npy    (n,)      每个 admission 的 hadm_id（str）
    data/mimic4/outcomes.npy    (n, 3)    每行的三个结局：
                                          列 0  in-hospital mortality (0/1)
                                          列 1  LOS 天数（float，可作回归或按阈值分类）
                                          列 2  30 天内非计划再入院 (0/1)

用法（服务器）：
    python data_preprocessing/build_outcomes.py \
        --data_dir data/mimic4 \
        --readmit_days 30
"""
import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


def process_icd_code(code):
    """与 process_mimic4_for_ehrdiff.process_icd_code 完全一致。"""
    if pd.isna(code):
        return None
    code = str(code).strip()
    return code[:3]


def rebuild_hadm_order(data_dir):
    """重建 (hadm_id 排序, hadm->codes)。返回 None 若文件缺失。"""
    for f in ('diagnoses_icd.csv', 'admissions.csv'):
        p = os.path.join(data_dir, f)
        if not os.path.exists(p):
            listing = os.listdir(data_dir) if os.path.isdir(data_dir) else '<dir not found>'
            raise SystemExit(
                f'找不到原始文件: {p}\n'
                f'data_dir 内容: {listing}\n\n'
                '说明: 服务器上可能只有处理后的 npy，没有原始 MIMIC-IV CSV。\n'
                '两个选择:\n'
                '  1) 用 --raw_dir 指向原始 CSV 所在目录（若在别处）\n'
                '  2) 从本地 data/mimic4/ 上传 admissions.csv / diagnoses_icd.csv / '
                'procedures_icd.csv 到服务器（它们是重建 hadm 映射和提取结局的必需输入）')
    diagnoses = pd.read_csv(
        os.path.join(data_dir, 'diagnoses_icd.csv'),
        dtype={'subject_id': str, 'hadm_id': str, 'icd_code': str,
               'icd_version': int})
    diagnoses['icd_code_3d'] = diagnoses.apply(
        lambda r: process_icd_code(r['icd_code']), axis=1)
    diagnoses = diagnoses.dropna(subset=['icd_code_3d'])

    procs_path = os.path.join(data_dir, 'procedures_icd.csv')
    if os.path.exists(procs_path):
        procedures = pd.read_csv(
            procs_path,
            dtype={'subject_id': str, 'hadm_id': str, 'icd_code': str,
                   'icd_version': int})
        procedures['icd_code_3d'] = procedures.apply(
            lambda r: process_icd_code(r['icd_code']), axis=1)
        procedures = procedures.dropna(subset=['icd_code_3d'])
        all_codes = pd.concat([
            diagnoses[['hadm_id', 'icd_code_3d']].rename(
                columns={'icd_code_3d': 'code'}),
            procedures[['hadm_id', 'icd_code_3d']].rename(
                columns={'icd_code_3d': 'code'}),
        ])
    else:
        all_codes = diagnoses[['hadm_id', 'icd_code_3d']].rename(
            columns={'icd_code_3d': 'code'})

    # 与 process_mimic4_for_ehrdiff 相同的 groupby（默认 sort=True → 排序键）
    hadm_to_codes = {}
    for hadm_id, group in all_codes.groupby('hadm_id'):
        hadm_to_codes[hadm_id] = set(group['code'].values)

    return hadm_to_codes, all_codes['hadm_id'].unique()


def verify_mapping(data_dir, hadm_to_codes, codes, n_samples):
    """重建 X 并与 mimic4_data.npy 对比；返回匹配的 hadm 顺序列表。"""
    X_real = np.load(os.path.join(data_dir, 'mimic4_data.npy'), mmap_mode='r')
    if X_real.shape[0] != n_samples:
        raise SystemExit(f'样本数不一致: npy={X_real.shape[0]} vs 重建={n_samples}')

    def build_X(order):
        X = np.zeros((len(order), len(codes)), dtype=np.float32)
        for i, hadm in enumerate(order):
            for c in hadm_to_codes[hadm]:
                if c in codes:
                    X[i, codes[c]] = 1.0
        return X

    # 尝试 1：排序键顺序（groupby 默认行为）
    sorted_order = sorted(hadm_to_codes.keys())
    X = build_X(sorted_order)
    if np.array_equal(X, X_real):
        print(f'  校验通过：mimic4_data.npy 行序 = 排序后的 hadm_id (n={len(sorted_order)})')
        return sorted_order

    # 尝试 2：首次出现顺序
    first_order = list(hadm_to_codes.keys())
    X = build_X(first_order)
    if np.array_equal(X, X_real):
        print(f'  校验通过：mimic4_data.npy 行序 = 首次出现顺序 (n={len(first_order)})')
        return first_order

    diff = int((build_X(sorted_order) != X_real).sum())
    raise SystemExit(f'mapping 校验失败：重建 X 与 mimic4_data.npy 有 {diff} 处不一致。'
                     '请检查预处理脚本版本。')


def build_outcomes(data_dir, hadm_order, readmit_days=30):
    """从 admissions.csv 提取 mortality / LOS / readmission。"""
    adm = pd.read_csv(os.path.join(data_dir, 'admissions.csv'),
                      dtype={'subject_id': str, 'hadm_id': str})
    adm['admittime'] = pd.to_datetime(adm['admittime'])
    adm['dischtime'] = pd.to_datetime(adm['dischtime'])

    hadm_meta = adm.set_index('hadm_id')
    # 按 subject 计算再入院：按 admittime 排序，找下一条在出院 30 天内的入院
    adm_sorted = adm.sort_values(['subject_id', 'admittime'])
    next_time = adm_sorted.groupby('subject_id')['admittime'].shift(-1)
    next_type = adm_sorted.groupby('subject_id')['admitt_type'].shift(-1)
    disch = adm_sorted['dischtime']
    days_to_next = (next_time - disch).dt.total_seconds() / 86400.0
    # 非计划 = 下一条为急诊/观察入院（MIMIC-IV 中 EMERGENCY 为主要非计划类型）
    unplanned = next_type.fillna('').astype(str).str.upper().str.contains(
        'EMERGENCY|OBSERVATION')
    readmit_flag = ((days_to_next >= 0) & (days_to_next <= readmit_days) &
                    unplanned).astype(np.float32)
    # 以 hadm_id 为键（默认 readmit_flag 的索引是 adm 的原整数索引，需对齐）
    readmit = readmit_flag.to_numpy()
    readmit_by_hadm = dict(zip(adm_sorted['hadm_id'].to_numpy(), readmit))

    n = len(hadm_order)
    outcomes = np.zeros((n, 3), dtype=np.float32)
    missing = 0
    for i, hadm in enumerate(hadm_order):
        row = hadm_meta.loc[hadm]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]  # 理论上 hadm_id 唯一，防御性处理
        try:
            outcomes[i, 0] = float(row['hospital_expire_flag'])
            outcomes[i, 1] = float(
                (row['dischtime'] - row['admittime']).total_seconds() / 86400.0)
            outcomes[i, 2] = float(readmit_by_hadm[hadm])
        except (KeyError, ValueError):
            missing += 1
            outcomes[i, :] = np.nan

    if missing:
        print(f'  ⚠ {missing}/{n} 行在 admissions.csv 中未找到对应记录，置 NaN')
    return outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/mimic4',
                        help='处理后的 npy 数据目录')
    parser.add_argument('--raw_dir', default=None,
                        help='原始 MIMIC-IV CSV 目录（admissions/diagnoses/procedures），'
                             '默认同 --data_dir')
    parser.add_argument('--readmit_days', type=int, default=30)
    opt = parser.parse_args()

    raw_dir = opt.raw_dir or opt.data_dir
    print('[1] 重建 hadm_id 顺序...')
    hadm_to_codes, uniq = rebuild_hadm_order(raw_dir)
    n_samples = len(uniq)
    print(f'  有代码的住院记录: {n_samples}')

    print('[2] 校验 mapping 与 mimic4_data.npy...')
    with open(os.path.join(opt.data_dir, 'metadata.json')) as f:
        meta = json.load(f)
    codes = {c: i for i, c in enumerate(meta['selected_codes'])}
    hadm_order = verify_mapping(opt.data_dir, hadm_to_codes, codes, n_samples)

    print('[3] 提取临床结局...')
    outcomes = build_outcomes(raw_dir, hadm_order, opt.readmit_days)

    np.save(os.path.join(opt.data_dir, 'hadm_ids.npy'),
            np.array(hadm_order, dtype=object))
    np.save(os.path.join(opt.data_dir, 'outcomes.npy'), outcomes)

    valid = ~np.isnan(outcomes).any(axis=1)
    print(f'\n  有效行: {valid.sum()}/{n_samples}')
    print(f'  in-hospital mortality: {outcomes[valid,0].mean():.4f}')
    print(f'  LOS 天数: mean={np.nanmean(outcomes[:,1]):.2f}, '
          f'median={np.nanmedian(outcomes[:,1]):.2f}')
    print(f'  {opt.readmit_days}-day readmission: {outcomes[valid,2].mean():.4f}')
    print(f'\n已保存: hadm_ids.npy, outcomes.npy (列: mortality, LOS, readmit)')


if __name__ == '__main__':
    main()
