"""
拒绝采样筛选脚本：对生成的合成数据进行人口学条件筛选

原理：生成大批量样本 → 按人口学/ICD 条件筛选 → 得到符合目标分布的子集

用法：
  # 对已有生成结果进行筛选
  python filter_by_demographics.py --samples results/mimic4_dem/samples/all_x.npy

  # 先采样再筛选（一步完成）
  python filter_by_demographics.py --generate --n_gen 100000

  # 筛选糖尿病患者，平均年龄 ~55，性别比例 F:M ≈ 7:3
  python filter_by_demographics.py --samples results/mimic4_dem/samples/all_x.npy \
      --icd "E11" --age_min 45 --age_max 65 --gender_f_ratio 0.7
"""

import numpy as np
import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

# ==================== 人口学特征配置 ====================
# 与 data_preprocessing/process_mimic4_with_demographics.py 保持一致
DEM_COLS = [
    'dem_gender',           # 0: M, 1: F
    'dem_age',              # 归一化到 [0,1]，对应 18~91 岁
    'dem_adm_emerg',        # 入院类型: Emergency
    'dem_adm_obs',          # 入院类型: Observation
    'dem_adm_urg',          # 入院类型: Urgent
    'dem_adm_other',        # 入院类型: Other
    'dem_ins_medicare',     # 保险: Medicare
    'dem_ins_medicaid',     # 保险: Medicaid
    'dem_ins_other',        # 保险: Other
]
N_DEM = len(DEM_COLS)
AGE_MIN, AGE_MAX = 18.0, 91.0

# ICD-9/10 糖尿病相关代码前缀（用于自动识别）
DIABETES_CODES = {'E11', 'E10', 'E13', 'E14', '250', '249', '3572', '3620'}


def load_metadata(metadata_path="data/mimic4/dem_metadata.json"):
    """加载元数据，获取 ICD 代码到列索引的映射"""
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        n_code = meta['n_code']
        code_to_idx = {code: i for i, code in enumerate(meta['selected_codes'])}
        idx_to_code = {i: code for code, i in code_to_idx.items()}
        print(f"  加载元数据: {n_code} 个 ICD 代码 + {meta['n_dem']} 个人口学特征")
        return meta, code_to_idx, idx_to_code
    else:
        print(f"  {metadata_path} 不存在，尝试从数据维度推断")
        return None, None, None


def denormalize_age(age_norm):
    """将归一化年龄还原为实际年龄"""
    return age_norm * (AGE_MAX - AGE_MIN) + AGE_MIN


def normalize_age(age):
    """将实际年龄转换为归一化年龄"""
    return np.clip((age - AGE_MIN) / (AGE_MAX - AGE_MIN), 0, 1)


def get_code_name(code):
    """获取 ICD 代码的可读名称"""
    code_names = {
        '401': 'Essential hypertension', 'E87': 'Fluid/electrolyte disorders',
        'E78': 'Lipid disorders', 'Z79': 'Long-term drug therapy',
        'E11': 'Type 2 diabetes', '272': 'Lipid metabolism disorders',
        'I10': 'Essential hypertension', '250': 'Diabetes mellitus',
        'V58': 'Aftercare', 'Z87': 'Personal history of diseases',
        'I25': 'Chronic ischemic heart disease', '428': 'Heart failure',
        'V45': 'Post-surgical states', 'Y92': 'Place of occurrence',
        '276': 'Fluid/electrolyte disorders', '427': 'Cardiac dysrhythmias',
        'K21': 'GERD', '530': 'Esophageal diseases',
        '414': 'Coronary atherosclerosis', '305': 'Nondependent drug abuse',
        'F32': 'Major depressive disorder', 'I48': 'Atrial fibrillation',
        'I50': 'Heart failure', 'F41': 'Anxiety disorders',
        'N18': 'Chronic kidney disease', 'Z68': 'BMI',
        'N17': 'Acute kidney failure', 'G47': 'Sleep disorders',
        '585': 'Chronic kidney disease', '403': 'Hypertensive CKD',
        '584': 'Acute kidney failure', '300': 'Anxiety/dissociative disorders',
        'E10': 'Type 1 diabetes', 'E13': 'Other diabetes',
        'E14': 'Unspecified diabetes', '249': 'Secondary diabetes',
    }
    return code_names.get(code, '')


def load_data(data_path):
    """加载合成数据"""
    print(f"  加载数据: {data_path}")
    X = np.load(data_path)
    if X.ndim == 3:
        X = X.squeeze(0)
    print(f"  数据形状: {X.shape}, dtype={X.dtype}")
    return X


def describe_sample(X, n_code, label=""):
    """打印一批样本的基本统计信息"""
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")
    print(f"  样本数: {X.shape[0]}")

    # ICD 部分统计
    icd_part = X[:, :n_code]
    nzc = (icd_part.sum(axis=0) > 0).sum()
    sparsity = 1 - icd_part.mean()
    avg_codes = icd_part.sum(axis=1).mean()
    print(f"  ICD 非零列 (NZC): {nzc}/{n_code}")
    print(f"  ICD 稀疏度: {sparsity:.4f}")
    print(f"  平均每样本代码数: {avg_codes:.2f}")

    # 人口学统计
    dem_part = X[:, n_code:]
    print(f"\n  人口学特征:")
    print(f"    性别 F 比例: {dem_part[:, 0].mean():.3f}")
    print(f"    年龄: {denormalize_age(dem_part[:, 1]).mean():.1f} ± {denormalize_age(dem_part[:, 1]).std():.1f} 岁")
    print(f"    入院类型: emerg={dem_part[:, 2].mean():.3f} obs={dem_part[:, 3].mean():.3f} urg={dem_part[:, 4].mean():.3f} other={dem_part[:, 5].mean():.3f}")
    print(f"    保险: medicare={dem_part[:, 6].mean():.3f} medicaid={dem_part[:, 7].mean():.3f} other={dem_part[:, 8]:.3f}")


def filter_by_gender(X, n_code, target_ratio=None, target_sex=None):
    """
    按性别筛选
    target_ratio: 目标女性比例 (0-1)，例如 0.7 表示 70% 女性
    target_sex: 0=男性, 1=女性, None=不筛选
    """
    dem_gender = X[:, n_code]  # 1=F, 0=M

    if target_sex is not None:
        mask = (dem_gender == target_sex)
        return X[mask]

    if target_ratio is not None:
        female_mask = (dem_gender == 1)
        female_idx = np.where(female_mask)[0]
        male_idx = np.where(~female_mask)[0]

        n_total = len(female_idx) + len(male_idx)
        n_female_needed = int(n_total * target_ratio)

        if len(female_idx) < n_female_needed:
            print(f"  ⚠ 女性样本不足: 需要 {n_female_needed}, 实际 {len(female_idx)}")
            n_female_needed = len(female_idx)

        n_male_needed = n_total - n_female_needed
        if len(male_idx) < n_male_needed:
            print(f"  ⚠ 男性样本不足: 需要 {n_male_needed}, 实际 {len(male_idx)}")
            n_male_needed = len(male_idx)
            n_female_needed = n_total - n_male_needed

        np.random.shuffle(female_idx)
        np.random.shuffle(male_idx)

        selected = np.concatenate([
            female_idx[:n_female_needed],
            male_idx[:n_male_needed]
        ])
        np.random.shuffle(selected)
        return X[selected]

    return X


def filter_by_age(X, n_code, age_min=None, age_max=None, age_target=None, age_tolerance=5):
    """
    按年龄筛选
    age_min/age_max: 年龄范围（实际年龄）
    age_target: 目标年龄，在 ±age_tolerance 范围内
    """
    dem_age = X[:, n_code + 1]

    if age_target is not None:
        age_low = normalize_age(age_target - age_tolerance)
        age_high = normalize_age(age_target + age_tolerance)
        mask = (dem_age >= age_low) & (dem_age <= age_high)
        result = X[mask]
        actual_mean = denormalize_age(result[:, n_code + 1]).mean() if len(result) > 0 else 0
        print(f"  年龄筛选: target={age_target}±{age_tolerance}, 保留 {len(result)}/{len(X)} 样本, 实际均值={actual_mean:.1f}")
        return result

    if age_min is not None or age_max is not None:
        age_low = normalize_age(age_min) if age_min is not None else 0.0
        age_high = normalize_age(age_max) if age_max is not None else 1.0
        mask = (dem_age >= age_low) & (dem_age <= age_high)
        return X[mask]

    return X


def filter_by_icd(X, n_code, codes, require_all=False):
    """
    按 ICD 代码筛选
    codes: ICD 代码列表，如 ['E11', '250']
    require_all: True=必须同时包含所有指定代码, False=包含任一即可
    """
    # 先用代码列表查找列索引
    meta, code_to_idx, idx_to_code = load_metadata()
    if code_to_idx is None:
        print("  ⚠ 无元数据，无法按 ICD 代码筛选")
        return X

    col_indices = []
    for code in codes:
        if code in code_to_idx:
            col_indices.append(code_to_idx[code])
        else:
            print(f"  ⚠ ICD 代码 {code} 不在字典中，跳过")

    if not col_indices:
        print("  ⚠ 没有找到任何有效的 ICD 代码")
        return X

    icd_part = X[:, col_indices]
    if require_all:
        mask = icd_part.min(axis=1) > 0.5  # 所有指定代码都存在
    else:
        mask = icd_part.max(axis=1) > 0.5  # 至少一个存在

    return X[mask]


def filter_by_admission(X, n_code, adm_types=None):
    """
    按入院类型筛选
    adm_types: 列表，可选 'emerg', 'obs', 'urg', 'other'
    """
    if not adm_types:
        return X

    cols = {
        'emerg': n_code + 2,
        'obs': n_code + 3,
        'urg': n_code + 4,
        'other': n_code + 5,
    }
    adm_part = X[:, [cols[t] for t in adm_types]]
    mask = adm_part.max(axis=1) > 0.5
    return X[mask]


def filter_by_insurance(X, n_code, ins_types=None):
    """
    按保险类型筛选
    ins_types: 列表，可选 'medicare', 'medicaid', 'other'
    """
    if not ins_types:
        return X

    cols = {
        'medicare': n_code + 6,
        'medicaid': n_code + 7,
        'other': n_code + 8,
    }
    ins_part = X[:, [cols[t] for t in ins_types]]
    mask = ins_part.max(axis=1) > 0.5
    return X[mask]


def filter_diabetes_patients(X, n_code, idx_to_code):
    """筛选糖尿病患者（包含任何糖尿病相关代码）"""
    diabetes_indices = []
    for i in range(n_code):
        code = idx_to_code.get(i, '')
        if code in DIABETES_CODES:
            diabetes_indices.append(i)

    if not diabetes_indices:
        print("  ⚠ 未找到糖尿病相关 ICD 代码")
        return X

    icd_part = X[:, diabetes_indices]
    mask = icd_part.max(axis=1) > 0.5
    return X[mask]


def rejection_sample(X, n_code, config):
    """
    拒绝采样主函数
    config: dict，包含各项筛选参数
    """
    print("\n" + "=" * 60)
    print(" 开始拒绝采样筛选")
    print("=" * 60)
    print(f"  输入样本数: {X.shape[0]}")

    # 1. ICD 代码筛选
    if config.get('icd_codes'):
        X = filter_by_icd(X, n_code, config['icd_codes'],
                          require_all=config.get('icd_require_all', False))
        print(f"  按 ICD 代码 {config['icd_codes']} 筛选后: {X.shape[0]} 样本")
        if len(X) == 0:
            return X

    # 2. 年龄筛选
    X = filter_by_age(X, n_code,
                      age_min=config.get('age_min'),
                      age_max=config.get('age_max'),
                      age_target=config.get('age_target'),
                      age_tolerance=config.get('age_tolerance', 5))
    print(f"  年龄筛选后: {X.shape[0]} 样本")
    if len(X) == 0:
        return X

    # 3. 性别比例控制（在年龄/ICD 筛选之后）
    if config.get('gender_f_ratio') is not None:
        X = filter_by_gender(X, n_code, target_ratio=config['gender_f_ratio'])
        actual_ratio = (X[:, n_code] == 1).mean()
        print(f"  性别筛选后: {X.shape[0]} 样本, F 比例={actual_ratio:.3f}")
        if len(X) == 0:
            return X

    # 4. 入院类型
    if config.get('admission_types'):
        X = filter_by_admission(X, n_code, config['admission_types'])
        print(f"  入院类型筛选后: {X.shape[0]} 样本")
        if len(X) == 0:
            return X

    # 5. 保险类型
    if config.get('insurance_types'):
        X = filter_by_insurance(X, n_code, config['insurance_types'])
        print(f"  保险筛选后: {X.shape[0]} 样本")
        if len(X) == 0:
            return X

    return X


def generate_and_filter(config, n_code):
    """使用模型生成样本并筛选"""
    import torch
    from omegaconf import OmegaConf

    cfg_path = config.get('config', 'configs/mimic4/sample_dem_edm_2092.yaml')
    workdir = config.get('workdir', 'results/mimic4_dem')
    n_gen = config.get('n_gen', 100000)
    batch_size = min(config.get('batch_size', 10000), n_gen)

    print(f"  加载配置: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    if config.get('model_ckpt'):
        cfg.model.ckpt = config['model_ckpt']

    # 设置 device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.setup.device = device
    print(f"  使用设备: {device}")

    # 构建模型
    from runners.generate_base import get_model
    from samplers import ablation_sampler

    model = get_model(cfg, 0)

    def sampler(x, y=None):
        return ablation_sampler(x, y, model, **cfg.sampler)

    # 分批采样
    all_samples = []
    n_batches = (n_gen + batch_size - 1) // batch_size

    print(f"  生成 {n_gen} 个样本, batch_size={batch_size}, {n_batches} 批")
    for i in range(n_batches):
        current_bs = min(batch_size, n_gen - len(all_samples) * batch_size)
        print(f"    采样批次 {i+1}/{n_batches} (batch_size={current_bs})...")

        x = torch.randn(current_bs, cfg.data.resolution, device=device)
        with torch.no_grad():
            x = sampler(x)

        x = x.cpu().numpy()
        if cfg.test.type == 'binary':
            x = np.rint(np.clip(x, 0, 1))

        all_samples.append(x)

    X = np.concatenate(all_samples, axis=0)
    print(f"  生成完成: {X.shape}")

    # 保存原始生成结果
    out_dir = config.get('out_dir', os.path.join(workdir, 'samples'))
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'all_x_raw.npy'), X)

    # 筛选
    if config.get('filter_after_generate', True):
        X_filtered = rejection_sample(X, n_code, config)
        return X, X_filtered
    return X, None


def main():
    parser = argparse.ArgumentParser(
        description='EHRDiff 拒绝采样筛选 — 按人口学/ICD 条件筛选合成数据'
    )

    # 数据源
    parser.add_argument('--samples', type=str, default=None,
                        help='已有生成数据路径，如 results/mimic4_dem/samples/all_x.npy')
    parser.add_argument('--generate', action='store_true',
                        help='从模型直接生成新样本')
    parser.add_argument('--config', type=str, default='configs/mimic4/sample_dem_edm_2092.yaml',
                        help='采样配置文件')
    parser.add_argument('--workdir', type=str, default='results/mimic4_dem',
                        help='工作目录')
    parser.add_argument('--model_ckpt', type=str, default=None,
                        help='模型 checkpoint 路径（覆盖配置中的路径）')
    parser.add_argument('--n_gen', type=int, default=100000,
                        help='生成样本总数')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='每批采样数量')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='输出目录')

    # ICU/人口学筛选参数
    parser.add_argument('--icd', type=str, default=None, nargs='+',
                        help='筛选目标 ICD 代码，如 --icd E11 250')
    parser.add_argument('--icd_require_all', action='store_true',
                        help='要求同时包含所有指定 ICD 代码（默认包含任一即可）')
    parser.add_argument('--diabetes', action='store_true',
                        help='筛选糖尿病患者（快捷方式，等价于 --icd E11 250 等）')

    parser.add_argument('--age_min', type=float, default=None,
                        help='最小年龄')
    parser.add_argument('--age_max', type=float, default=None,
                        help='最大年龄')
    parser.add_argument('--age_target', type=float, default=None,
                        help='目标年龄（将筛选 ±age_tolerance 范围）')
    parser.add_argument('--age_tolerance', type=float, default=5,
                        help='目标年龄容差（默认 ±5 岁）')

    parser.add_argument('--gender_f_ratio', type=float, default=None,
                        help='目标女性比例，如 0.7 表示 70% 女性')
    parser.add_argument('--gender_m_only', action='store_true',
                        help='仅保留男性')
    parser.add_argument('--gender_f_only', action='store_true',
                        help='仅保留女性')

    parser.add_argument('--admission', type=str, default=None, nargs='+',
                        choices=['emerg', 'obs', 'urg', 'other'],
                        help='入院类型，如 --admission emerg urg')
    parser.add_argument('--insurance', type=str, default=None, nargs='+',
                        choices=['medicare', 'medicaid', 'other'],
                        help='保险类型，如 --insurance medicare')

    parser.add_argument('--n_final', type=int, default=None,
                        help='最终保留样本数（从筛选结果中随机抽取）')

    args = parser.parse_args()

    # 加载元数据
    meta, code_to_idx, idx_to_code = load_metadata()
    n_code = meta['n_code'] if meta else None

    # 构建筛选配置
    filter_config = {}

    if args.icd:
        filter_config['icd_codes'] = args.icd
        filter_config['icd_require_all'] = args.icd_require_all
    if args.diabetes:
        if meta:
            diabetes_codes = [c for c in DIABETES_CODES if c in code_to_idx]
            filter_config['icd_codes'] = diabetes_codes
        else:
            print("  ⚠ 需要元数据才能筛选糖尿病，请先运行预处理脚本生成 dem_metadata.json")

    if args.age_min is not None:
        filter_config['age_min'] = args.age_min
    if args.age_max is not None:
        filter_config['age_max'] = args.age_max
    if args.age_target is not None:
        filter_config['age_target'] = args.age_target
        filter_config['age_tolerance'] = args.age_tolerance

    if args.gender_f_ratio is not None:
        filter_config['gender_f_ratio'] = args.gender_f_ratio
    if args.gender_m_only:
        filter_config['gender_f_ratio'] = 0.0
    if args.gender_f_only:
        filter_config['gender_f_ratio'] = 1.0

    if args.admission:
        filter_config['admission_types'] = args.admission
    if args.insurance:
        filter_config['insurance_types'] = args.insurance

    filter_config['out_dir'] = args.out_dir or (os.path.join(args.workdir, 'samples'))
    filter_config['config'] = args.config
    filter_config['workdir'] = args.workdir
    filter_config['model_ckpt'] = args.model_ckpt
    filter_config['n_gen'] = args.n_gen
    filter_config['batch_size'] = args.batch_size

    # 确定 n_code
    if n_code is None:
        # 从数据推断
        if args.samples:
            X = np.load(args.samples, mmap_mode='r')
            if X.ndim == 3:
                X = X.squeeze(0)
            n_code = X.shape[1] - N_DEM
            del X
        else:
            n_code = 2083  # 默认值（已知配置）

    print(f"\nICD 维度: {n_code}, 人口学维度: {N_DEM}, 总计: {n_code + N_DEM}")

    # 生成或加载数据
    if args.generate:
        X, X_filtered = generate_and_filter(filter_config, n_code)
    elif args.samples:
        X = load_data(args.samples)
        X_filtered = rejection_sample(X, n_code, filter_config)
    else:
        parser.print_help()
        print("\n⚠ 请指定 --samples 或 --generate")
        return

    # 保存筛选结果
    if X_filtered is not None and len(X_filtered) > 0:
        # 最终数量控制
        if args.n_final is not None and len(X_filtered) > args.n_final:
            idx = np.random.choice(len(X_filtered), args.n_final, replace=False)
            X_filtered = X_filtered[idx]

        # 描述筛选结果
        describe_sample(X_filtered, n_code, label=f"筛选结果 ({len(X_filtered)} 样本)")

        out_dir = filter_config['out_dir']
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'filtered_samples.npy')
        np.save(out_path, X_filtered)
        print(f"\n  筛选结果已保存: {out_path}")
        print(f"  最终样本数: {len(X_filtered)}")

        # 保存筛选配置（方便复现）
        config_path = os.path.join(out_dir, 'filter_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'filter_config': {k: str(v) if not isinstance(v, (list, dict, float, int)) else v
                                 for k, v in filter_config.items()},
                'result_shape': list(X_filtered.shape),
            }, f, indent=2)
        print(f"  筛选配置已保存: {config_path}")

    elif X_filtered is not None:
        print("\n  ⚠ 筛选后无剩余样本，请放宽筛选条件或增大生成数量")


if __name__ == '__main__':
    main()
